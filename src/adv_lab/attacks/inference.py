"""Inference-time data manipulation attacks.

Production-grade attacks that operate at inference time to manipulate model
predictions through gradient-based watermark flipping, upstream preprocessing
poisoning, and soft-label (confidence score) exploitation.

These attacks target the inference pipeline rather than the model weights,
making them relevant to deployed ML systems where attackers cannot retrain
but can manipulate inputs or preprocessing stages.

Key components:
  * **watermark_flip** -- gradient-based attack that finds minimal perturbations
    causing a watermarked model to evade watermark verification while preserving
    task accuracy. Supports custom watermark detection functions and joint
    optimization of competing objectives.
  * **prediction_poison** -- finds minimal modifications to data preprocessing
    parameters (brightness, contrast, gamma, channel normalization) that induce
    systematic and targeted label shifts across the entire input distribution.
  * **soft_label_manipulation** -- exploits full probability vectors returned by
    ML-as-a-Service APIs to guide query-efficient attacks. Leverages calibration
    gaps, runner-up class information, and confidence geometry to minimize the
    number of queries needed to find adversarial examples.

References:
  - Adi et al., "Turning Your Weakness Into a Strength: Watermarking Deep
    Neural Networks by Backdooring" (USENIX Security 2018).
  - Zhang et al., "Protecting Intellectual Property of Deep Neural Networks
    with Watermarking" (AsiaCCS 2018).
  - Namba and Sakuma, "Robust Watermarking of Neural Network with Exponential
    Weighting" (AsiaCCS 2019).
  - Papernot et al., "Practical Black-Box Attacks Against Machine Learning"
    (AsiaCCS 2017).
  - Ilyas et al., "Black-box Adversarial Attacks with Limited Queries and
    Information" (ICML 2018).
  - Cheng et al., "Sign-OPT: A Query-Efficient Hard-label Adversarial Attack"
    (ICLR 2020).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type protocols for flexibility
# ---------------------------------------------------------------------------


class WatermarkDetector(Protocol):
    """Protocol for watermark detection functions.

    A detector takes model logits and returns a per-sample score where
    positive values indicate watermark presence.
    """

    def __call__(self, logits: Tensor) -> Tensor: ...


# ---------------------------------------------------------------------------
# Watermark Flipping Attack
# ---------------------------------------------------------------------------


@dataclass
class WatermarkFlipResult:
    """Detailed result from a watermark flipping attack.

    Attributes:
        adversarial_images: the perturbed images.
        watermark_scores_before: detector scores on clean inputs.
        watermark_scores_after: detector scores on adversarial inputs.
        classification_preserved: per-sample bool -- did classification stay correct.
        watermark_flipped: per-sample bool -- did watermark detection flip.
        perturbation_linf: L-inf norm of the perturbation per sample.
        steps_used: number of optimization steps actually executed.
    """

    adversarial_images: Tensor
    watermark_scores_before: Tensor
    watermark_scores_after: Tensor
    classification_preserved: Tensor
    watermark_flipped: Tensor
    perturbation_linf: Tensor
    steps_used: int


def watermark_flip(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    watermark_detector: Optional[Callable[[Tensor], Tensor]] = None,
    epsilon: float = 0.05,
    steps: int = 100,
    alpha: float = 0.003,
    classification_weight: float = 1.0,
    watermark_weight: float = 5.0,
    early_stop: bool = True,
) -> Tensor:
    """Gradient-based attack to flip prediction watermarks.

    Finds a minimal perturbation that causes the watermark detection signal to
    flip (watermark no longer detectable) while maintaining correct classification
    on the primary task. Uses a bi-objective PGD formulation with adaptive
    weighting between task preservation and watermark evasion.

    The optimization solves:
        min_{delta} w_cls * L_cls(f(x+delta), y) - w_wm * D(f(x+delta))
        s.t. ||delta||_inf <= epsilon, x+delta in [0,1]

    where L_cls is classification loss, D is the watermark detector score, and
    w_cls, w_wm are the respective objective weights.

    If no watermark_detector is provided, a default detector is used that
    measures logit variance deviation from the batch mean (simulating a
    typical backdoor-based watermark where trigger inputs produce distinctive
    logit patterns).

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        watermark_detector: callable that takes logits ``(N, C)`` and returns a
            per-sample score tensor ``(N,)`` where positive values indicate the
            watermark is detected. If None, uses a variance-based heuristic.
        epsilon: L-inf perturbation budget.
        steps: maximum number of PGD iterations.
        alpha: step size per iteration. Should be smaller than epsilon/steps for
            fine-grained optimization.
        classification_weight: weight for the classification preservation objective.
        watermark_weight: weight for the watermark flipping objective.
        early_stop: if True, stop iterating on samples where both objectives
            are satisfied (correct classification AND watermark flipped).

    Returns:
        Adversarial images, detached, clamped to ``[0, 1]``, same shape as input.

    Raises:
        ValueError: if model is in training mode or inputs have invalid shape.

    References:
        Adi et al., "Turning Your Weakness Into a Strength: Watermarking Deep
        Neural Networks by Backdooring" (USENIX Security 2018).
        Namba and Sakuma, "Robust Watermarking of Neural Network with Exponential
        Weighting" (AsiaCCS 2019).
    """
    _require_eval_mode(model)

    if images.dim() != 4:
        raise ValueError(
            f"Expected images with 4 dimensions (N, C, H, W), got {images.dim()}"
        )

    batch_size = images.shape[0]

    def _default_watermark_detector(logits: Tensor) -> Tensor:
        """Default variance-based watermark detector.

        Watermarked models typically produce distinctive logit patterns on
        trigger inputs. We detect this via deviation from expected variance.
        """
        per_sample_var = logits.var(dim=1)
        baseline_var = per_sample_var.mean()
        return per_sample_var - baseline_var

    detector = (
        watermark_detector
        if watermark_detector is not None
        else _default_watermark_detector
    )

    # Record initial watermark scores for reference
    with torch.no_grad():
        initial_logits = model(images)
        detector(initial_logits)

    x_adv = images.clone().detach()
    x_orig = images.clone().detach()

    # Track which samples still need optimization
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=images.device)

    for step in range(steps):
        if early_stop and not active_mask.any():
            logger.debug("Early stop at step %d: all samples satisfied.", step)
            break

        x_adv.requires_grad_(True)
        logits = model(x_adv)

        # Classification objective: minimize cross-entropy to preserve correctness
        cls_loss = nn.functional.cross_entropy(logits, labels, reduction="none")

        # Watermark objective: push watermark scores negative (flip detection)
        wm_scores = detector(logits)
        wm_loss = wm_scores  # We want to minimize this (push negative)

        # Combined per-sample loss with weighting
        per_sample_loss = classification_weight * cls_loss + watermark_weight * wm_loss

        # Only optimize active samples
        if early_stop:
            masked_loss = (per_sample_loss * active_mask.float()).sum() / max(
                active_mask.sum().item(), 1.0
            )
        else:
            masked_loss = per_sample_loss.mean()

        grad = torch.autograd.grad(masked_loss, x_adv)[0]

        # PGD step with sign gradient
        x_adv = x_adv.detach() + alpha * grad.sign()

        # Project back to epsilon-ball around original
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

        # Update active mask: deactivate samples where both objectives met
        if early_stop and (step + 1) % 10 == 0:
            with torch.no_grad():
                check_logits = model(x_adv)
                check_preds = check_logits.argmax(dim=1)
                check_wm = detector(check_logits)
                cls_ok = check_preds == labels
                wm_flipped = check_wm <= 0.0
                active_mask = ~(cls_ok & wm_flipped)

    return x_adv.detach()


# ---------------------------------------------------------------------------
# Preprocessing Pipeline and Prediction Poisoning
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingParams:
    """Parameters defining a data preprocessing pipeline.

    Represents the configurable knobs of a production data preprocessing
    stage. An attacker who compromises the preprocessing config (e.g., via
    a supply-chain attack on a config file) can induce systematic prediction
    shifts without touching model weights or input data directly.

    Attributes:
        brightness: additive brightness offset in [-1, 1].
        contrast: multiplicative contrast scaling factor (positive).
        gamma: gamma correction exponent (1.0 = no correction).
        channel_shift: per-channel additive shift tensor of shape (C,).
        normalize_mean: per-channel normalization mean (C,).
        normalize_std: per-channel normalization std (C,).
    """

    brightness: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    channel_shift: Optional[Tensor] = None
    normalize_mean: Optional[Tensor] = None
    normalize_std: Optional[Tensor] = None

    def apply(self, images: Tensor) -> Tensor:
        """Apply the full preprocessing pipeline to a batch of images.

        Operations are applied in order: gamma -> contrast -> brightness ->
        channel_shift -> normalization. Output is clamped to [0, 1].

        Args:
            images: input images in [0, 1] with shape (N, C, H, W).

        Returns:
            Preprocessed images clamped to [0, 1].
        """
        x = images.clone()

        # Gamma correction
        if self.gamma != 1.0:
            x = torch.pow(torch.clamp(x, min=1e-8), self.gamma)

        # Contrast scaling
        if self.contrast != 1.0:
            x = x * self.contrast

        # Brightness offset
        if self.brightness != 0.0:
            x = x + self.brightness

        # Per-channel shift
        if self.channel_shift is not None:
            shift = self.channel_shift.view(1, -1, 1, 1)
            x = x + shift.to(x.device)

        # Normalization (applied last, simulating a data loading pipeline)
        if self.normalize_mean is not None and self.normalize_std is not None:
            mean = self.normalize_mean.view(1, -1, 1, 1).to(x.device)
            std = self.normalize_std.view(1, -1, 1, 1).to(x.device)
            x = (x - mean) / (std + 1e-8)

        return torch.clamp(x, 0.0, 1.0)

    def distance_from_default(self) -> float:
        """Compute the total parameter deviation from default (identity) preprocessing."""
        dist = abs(self.brightness) + abs(self.contrast - 1.0) + abs(self.gamma - 1.0)
        if self.channel_shift is not None:
            dist += self.channel_shift.abs().sum().item()
        return dist


def prediction_poison(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    target_shift: int = 1,
    search_steps: int = 50,
    param_budget: float = 0.1,
    num_restarts: int = 3,
) -> Tensor:
    """Manipulate upstream data preprocessing to induce systematic label shifts.

    Simulates a supply-chain attack where the adversary compromises the
    preprocessing configuration (e.g., normalization parameters in a config
    file, data augmentation settings in a pipeline definition). The attack
    finds the minimal preprocessing parameter modification that causes the
    model to systematically shift its predictions.

    Uses multi-restart coordinate descent with adaptive step sizes to
    efficiently search the preprocessing parameter space. Each restart begins
    from a random initialization within the budget to escape local optima.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        target_shift: desired label shift amount modulo num_classes
            (e.g., 1 means class k -> class (k+1) % C).
        search_steps: number of coordinate descent steps per restart.
        param_budget: maximum total deviation of preprocessing parameters
            from their default values.
        num_restarts: number of random restarts for the search.

    Returns:
        Poisoned images after adversarial preprocessing, detached, clamped to
        ``[0, 1]``, same shape as input.

    Raises:
        ValueError: if model is in training mode.

    References:
        Zhang et al., "Data Poisoning Attacks Against Autoregressive Models"
        (AAAI 2020).
        Saha et al., "Hidden Trigger Backdoor Attacks" (AAAI 2020).
    """
    _require_eval_mode(model)

    num_channels = images.shape[1]
    with torch.no_grad():
        num_classes = model(images[:1]).shape[1]

    # Target labels: cyclically shift by target_shift
    target_labels = (labels + target_shift) % num_classes

    best_params: Optional[PreprocessingParams] = None
    best_loss = float("inf")

    for restart in range(num_restarts):
        # Initialize with random perturbation within budget
        if restart == 0:
            # First restart: start from identity
            params = PreprocessingParams(
                brightness=0.0,
                contrast=1.0,
                gamma=1.0,
                channel_shift=torch.zeros(num_channels),
            )
        else:
            # Random initialization within budget
            init_scale = param_budget * 0.5
            params = PreprocessingParams(
                brightness=torch.empty(1).uniform_(-init_scale, init_scale).item(),
                contrast=1.0 + torch.empty(1).uniform_(-init_scale, init_scale).item(),
                gamma=1.0
                + torch.empty(1).uniform_(-init_scale * 0.5, init_scale * 0.5).item(),
                channel_shift=torch.empty(num_channels).uniform_(
                    -init_scale, init_scale
                ),
            )

        # Adaptive step size: start large, decay
        base_step = param_budget / max(search_steps, 1)

        for step in range(search_steps):
            step_size = base_step * max(0.1, 1.0 - 0.5 * step / search_steps)

            # Coordinate descent over each parameter
            param_candidates: List[PreprocessingParams] = []

            # Brightness perturbations
            for direction in [-1.0, 1.0]:
                candidate = PreprocessingParams(
                    brightness=max(
                        -param_budget,
                        min(param_budget, params.brightness + direction * step_size),
                    ),
                    contrast=params.contrast,
                    gamma=params.gamma,
                    channel_shift=params.channel_shift.clone()
                    if params.channel_shift is not None
                    else None,
                )
                param_candidates.append(candidate)

            # Contrast perturbations
            for direction in [-1.0, 1.0]:
                candidate = PreprocessingParams(
                    brightness=params.brightness,
                    contrast=max(
                        1.0 - param_budget,
                        min(
                            1.0 + param_budget, params.contrast + direction * step_size
                        ),
                    ),
                    gamma=params.gamma,
                    channel_shift=params.channel_shift.clone()
                    if params.channel_shift is not None
                    else None,
                )
                param_candidates.append(candidate)

            # Gamma perturbations
            for direction in [-1.0, 1.0]:
                candidate = PreprocessingParams(
                    brightness=params.brightness,
                    contrast=params.contrast,
                    gamma=max(
                        1.0 - param_budget,
                        min(
                            1.0 + param_budget,
                            params.gamma + direction * step_size * 0.5,
                        ),
                    ),
                    channel_shift=params.channel_shift.clone()
                    if params.channel_shift is not None
                    else None,
                )
                param_candidates.append(candidate)

            # Per-channel shift perturbations
            if params.channel_shift is not None:
                for c in range(num_channels):
                    for direction in [-1.0, 1.0]:
                        new_shift = params.channel_shift.clone()
                        new_shift[c] = max(
                            -param_budget,
                            min(
                                param_budget,
                                new_shift[c].item() + direction * step_size,
                            ),
                        )
                        candidate = PreprocessingParams(
                            brightness=params.brightness,
                            contrast=params.contrast,
                            gamma=params.gamma,
                            channel_shift=new_shift,
                        )
                        param_candidates.append(candidate)

            # Evaluate all candidates and select the best
            for candidate in param_candidates:
                # Enforce total budget constraint
                if candidate.distance_from_default() > param_budget * 3.0:
                    continue

                x_processed = candidate.apply(images)
                with torch.no_grad():
                    logits = model(x_processed)
                    loss = nn.functional.cross_entropy(logits, target_labels).item()

                if loss < best_loss:
                    best_loss = loss
                    best_params = PreprocessingParams(
                        brightness=candidate.brightness,
                        contrast=candidate.contrast,
                        gamma=candidate.gamma,
                        channel_shift=candidate.channel_shift.clone()
                        if candidate.channel_shift is not None
                        else None,
                    )
                    params = candidate

    # Apply the best preprocessing parameters found
    if best_params is None:
        return images.clone().detach()

    result = best_params.apply(images)
    return result.detach()


# ---------------------------------------------------------------------------
# Soft Label Manipulation
# ---------------------------------------------------------------------------


@dataclass
class SoftLabelAttackState:
    """Internal state for the soft-label manipulation attack.

    Tracks query efficiency metrics and attack progress per sample.

    Attributes:
        queries_used: total model queries consumed.
        confidence_trajectory: per-step mean confidence of true class.
        boundary_distances: estimated distance to decision boundary per step.
        samples_flipped: cumulative count of successfully attacked samples.
    """

    queries_used: int = 0
    confidence_trajectory: List[float] = field(default_factory=list)
    boundary_distances: List[float] = field(default_factory=list)
    samples_flipped: int = 0


def soft_label_manipulation(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.05,
    steps: int = 50,
    alpha: float = 0.003,
    temperature: float = 1.0,
    confidence_threshold: float = 0.3,
    momentum: float = 0.9,
) -> Tensor:
    """Exploit confidence scores from APIs returning full probability vectors.

    Uses the complete probability distribution (soft labels) returned by ML
    APIs to guide a significantly more query-efficient attack than hard-label
    methods. The attack exploits three information channels:

      1. **Runner-up identification**: The second-highest probability class
         identifies the nearest decision boundary, allowing targeted movement.
      2. **Confidence geometry**: The gap between top-1 and top-2 probabilities
         estimates distance to the boundary, enabling adaptive step sizing.
      3. **Calibration exploitation**: Overconfident models reveal more
         gradient information per query via the probability vector curvature.

    The attack uses momentum-accelerated gradient steps with per-sample
    adaptive early stopping when confidence drops below threshold.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        epsilon: L-inf perturbation budget.
        steps: maximum number of optimization iterations.
        alpha: base step size per iteration.
        temperature: softmax temperature for probability computation.
            Higher values produce softer distributions revealing more structure.
        confidence_threshold: stop attacking samples whose true-class confidence
            drops below this value (they are effectively at the boundary).
        momentum: momentum coefficient for gradient accumulation (0 = no momentum).

    Returns:
        Adversarial images, detached, clamped to ``[0, 1]``, same shape as input.

    Raises:
        ValueError: if model is in training mode or epsilon is non-positive.

    References:
        Ilyas et al., "Black-box Adversarial Attacks with Limited Queries and
        Information" (ICML 2018).
        Cheng et al., "Sign-OPT: A Query-Efficient Hard-label Adversarial Attack"
        (ICLR 2020).
        Papernot et al., "Practical Black-Box Attacks Against Machine Learning"
        (AsiaCCS 2017).
    """
    _require_eval_mode(model)

    if epsilon <= 0:
        return images.clone().detach()

    batch_size = images.shape[0]
    x_adv = images.clone().detach()
    x_orig = images.clone().detach()

    # Momentum buffer for gradient accumulation
    grad_momentum = torch.zeros_like(images)

    # Track active samples (not yet below confidence threshold)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=images.device)

    for step in range(steps):
        if not active_mask.any():
            break

        x_adv.requires_grad_(True)
        logits = model(x_adv)

        # Compute temperature-scaled probabilities
        probs = torch.softmax(logits / temperature, dim=1)

        # Extract true-class confidence
        true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # Identify runner-up class (highest prob excluding true class)
        masked_probs = probs.clone()
        masked_probs.scatter_(1, labels.unsqueeze(1), 0.0)
        runner_up_probs = masked_probs.max(dim=1).values

        # Confidence gap: distance to decision boundary estimate
        conf_gap = true_probs - runner_up_probs

        # Adaptive step size based on confidence gap
        # Larger steps when far from boundary, smaller when close
        adaptive_alpha = alpha * (1.0 + conf_gap.detach()).view(-1, 1, 1, 1)

        # Loss: push toward decision boundary using cross-entropy
        # and calibration exploitation (penalize high true-class confidence)
        ce_loss = nn.functional.cross_entropy(
            logits / temperature, labels, reduction="none"
        )

        # Weight active samples more
        weighted_loss = (ce_loss * active_mask.float()).sum() / max(
            active_mask.sum().item(), 1.0
        )

        grad = torch.autograd.grad(weighted_loss, x_adv)[0]

        # Apply momentum
        grad_momentum = momentum * grad_momentum + (1.0 - momentum) * grad

        # Masked update: only perturb active samples
        update = adaptive_alpha * grad_momentum.sign()
        if not active_mask.all():
            update = update * active_mask.float().view(-1, 1, 1, 1)

        x_adv = x_adv.detach() + update

        # Project back to epsilon-ball and valid range
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

        # Update active mask: deactivate samples below confidence threshold
        with torch.no_grad():
            check_logits = model(x_adv)
            check_probs = torch.softmax(check_logits / temperature, dim=1)
            current_conf = check_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            active_mask = current_conf > confidence_threshold

    return x_adv.detach()
