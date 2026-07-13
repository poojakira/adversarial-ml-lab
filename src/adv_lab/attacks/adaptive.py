"""Defense-aware adaptive attacks.

This module implements attacks that detect and circumvent gradient masking,
non-differentiable defenses, and stochastic preprocessing layers.

Key components:
  * **GradientMaskingDetector** -- monitors loss trajectory and triggers when a
    plateau is detected within the first 20% of expected optimization steps.
  * **BPDA** (Backward Pass Differentiable Approximation) -- Athalye et al.,
    "Obfuscated Gradients Give a False Sense of Security" (ICML 2018). Wraps a
    non-differentiable defense with a differentiable surrogate for backward pass.
  * **EoT** (Expectation over Transformations) -- Athalye et al., "Synthesizing
    Robust Adversarial Examples" (ICML 2018). Averages gradients computed over
    multiple random transformations.
  * **adaptive_attack** -- orchestrator that starts with PGD, detects masking,
    and auto-switches to BPDA / EoT / random-search-with-momentum.

References:
  - Athalye et al., "Obfuscated Gradients Give a False Sense of Security"
    (ICML 2018).
  - Athalye et al., "Synthesizing Robust Adversarial Examples" (ICML 2018).
  - Tramer et al., "On Adaptive Attacks to Adversarial Example Defenses"
    (NeurIPS 2020).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GradientMaskingDetector
# ---------------------------------------------------------------------------


@dataclass
class GradientMaskingDetector:
    """Detects gradient masking by monitoring the loss trajectory.

    Gradient masking (or obfuscated gradients) causes the loss to plateau
    very early in optimization, typically within the first 20% of expected
    steps.  This detector records the loss at each step and fires when:
      1. The loss has been flat (change < ``tolerance``) for
         ``plateau_window`` consecutive steps, AND
      2. This plateau begins before ``threshold_frac`` of the total
         expected steps have elapsed.

    Attributes:
        expected_steps: total number of optimization steps planned.
        threshold_frac: fraction of expected_steps within which a plateau
            is suspicious (default 0.20 = 20%).
        tolerance: minimum per-step loss change to count as "progress".
        plateau_window: how many consecutive "flat" steps trigger detection.
        losses: recorded loss values.
        detected: whether masking has been flagged.
        detection_step: the step at which masking was detected (or None).
    """

    expected_steps: int
    threshold_frac: float = 0.20
    tolerance: float = 1e-4
    plateau_window: int = 5
    losses: List[float] = field(default_factory=list)
    detected: bool = False
    detection_step: Optional[int] = None

    def record(self, loss_value: float) -> bool:
        """Record a loss value and return True if masking is now detected."""
        self.losses.append(loss_value)
        step = len(self.losses)

        if self.detected:
            return True

        # Need at least plateau_window observations
        if step < self.plateau_window:
            return False

        # Check if we are still within the first threshold_frac of steps
        threshold_step = int(self.expected_steps * self.threshold_frac)
        if step > threshold_step:
            return False

        # Check the last plateau_window losses for flatness
        recent = self.losses[-self.plateau_window:]
        max_change = max(
            abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))
        )
        if max_change < self.tolerance:
            self.detected = True
            self.detection_step = step
            logger.info(
                "Gradient masking detected at step %d/%d "
                "(loss plateau: max_change=%.6f < tolerance=%.6f)",
                step,
                self.expected_steps,
                max_change,
                self.tolerance,
            )
            return True

        return False


# ---------------------------------------------------------------------------
# BPDA
# ---------------------------------------------------------------------------


class BPDA(nn.Module):
    """Backward Pass Differentiable Approximation.

    Wraps a non-differentiable defense ``f`` with a differentiable
    approximation ``f_approx`` so that the forward pass uses the true defense
    but the backward pass uses gradients from the approximation.

    This is the canonical technique from Athalye et al. (ICML 2018) for
    defeating defenses that rely on non-differentiable operations (e.g.,
    JPEG compression, median filtering, thermometer encoding).

    Example::

        defense = lambda x: jpeg_compress(x)
        approx = lambda x: x  # identity approximation
        bpda_defense = BPDA(defense, approx)
        # In attack loop: compute loss on bpda_defense(x), backprop gets
        # gradients from the identity.
    """

    def __init__(
        self,
        defense: Callable[[Tensor], Tensor],
        approximation: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.defense = defense
        self.approximation = approximation

    def forward(self, x: Tensor) -> Tensor:
        """Apply defense in forward, approximation in backward."""
        # Straight-through estimator trick:
        # forward: defense(x), backward: approximation(x)
        approx_out = self.approximation(x)
        with torch.no_grad():
            defense_out = self.defense(x)
        # Detach defense output and add the gradient path from approx
        return defense_out + (approx_out - approx_out.detach())


# ---------------------------------------------------------------------------
# EoT
# ---------------------------------------------------------------------------


class EoT(nn.Module):
    """Expectation over Transformations.

    Averages gradients computed over ``n_samples`` random transformations of
    the input.  This makes gradient-based attacks robust to stochastic
    defenses (e.g., random resizing, random padding, noise injection).

    Reference: Athalye et al., "Synthesizing Robust Adversarial Examples"
    (ICML 2018).

    Args:
        model: the model (or defense + model pipeline) to attack.
        transform_fn: a callable that applies a random transformation to
            the input tensor. Must be differentiable.
        n_samples: number of transformations to average over.
    """

    def __init__(
        self,
        model: nn.Module,
        transform_fn: Callable[[Tensor], Tensor],
        n_samples: int = 20,
    ) -> None:
        super().__init__()
        self.model = model
        self.transform_fn = transform_fn
        self.n_samples = n_samples

    def forward(self, x: Tensor) -> Tensor:
        """Compute mean logits over n_samples random transformations."""
        logits_sum: Optional[Tensor] = None
        for _ in range(self.n_samples):
            x_t = self.transform_fn(x)
            logits = self.model(x_t)
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum = logits_sum + logits
        assert logits_sum is not None
        return logits_sum / self.n_samples


# ---------------------------------------------------------------------------
# Random search with momentum
# ---------------------------------------------------------------------------


def _random_search_with_momentum(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.03,
    steps: int = 100,
    momentum: float = 0.9,
    n_samples: int = 10,
) -> Tensor:
    """Gradient-free random search attack with momentum.

    At each step, samples ``n_samples`` random perturbation directions,
    evaluates the loss for each, selects the best, and applies a momentum
    update.  This is useful when gradients are completely unavailable or
    unreliable (severe masking).

    Args:
        model: classifier in ``eval()`` mode.
        images: clean inputs in ``[0, 1]``.
        labels: ground-truth class indices.
        epsilon: L-inf budget.
        steps: number of search iterations.
        momentum: momentum coefficient for velocity update.
        n_samples: number of random directions per step.

    Returns:
        Detached adversarial images.
    """
    _require_eval_mode(model)

    x_orig = images.clone().detach()
    x_adv = x_orig.clone()
    velocity = torch.zeros_like(x_adv)
    best_loss = torch.full((images.shape[0],), -float("inf"), device=images.device)

    for _ in range(steps):
        best_direction = torch.zeros_like(x_adv)
        step_best_loss = best_loss.clone()

        for _ in range(n_samples):
            direction = torch.empty_like(x_adv).uniform_(-1, 1).sign()
            candidate = torch.clamp(x_adv + 0.01 * direction, 0.0, 1.0)
            delta = torch.clamp(candidate - x_orig, -epsilon, epsilon)
            candidate = torch.clamp(x_orig + delta, 0.0, 1.0)

            with torch.no_grad():
                logits = model(candidate)
                loss = nn.functional.cross_entropy(logits, labels, reduction="none")

            improved = loss > step_best_loss
            step_best_loss = torch.where(improved, loss, step_best_loss)
            best_direction = torch.where(
                improved.view(-1, *([1] * (x_adv.ndim - 1))),
                direction,
                best_direction,
            )

        # Momentum update
        velocity = momentum * velocity + best_direction
        x_adv = x_adv + 0.01 * velocity.sign()
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
        best_loss = step_best_loss

    return x_adv.detach()


# ---------------------------------------------------------------------------
# adaptive_attack orchestrator
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveAttackLog:
    """Log of detection events and strategy switches."""

    events: List[dict] = field(default_factory=list)

    def log_event(self, event_type: str, **kwargs: object) -> None:
        """Record an event with type and extra details."""
        entry = {"event_type": event_type, **kwargs}
        self.events.append(entry)
        logger.info("AdaptiveAttack event: %s", entry)


def adaptive_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.007,
    steps: int = 40,
    defense: Optional[Callable[[Tensor], Tensor]] = None,
    defense_approx: Optional[Callable[[Tensor], Tensor]] = None,
    transform_fn: Optional[Callable[[Tensor], Tensor]] = None,
    eot_samples: int = 20,
    random_search_steps: int = 100,
) -> tuple[Tensor, AdaptiveAttackLog]:
    """Adaptive attack orchestrator.

    Starts with standard PGD, monitors the loss trajectory for gradient
    masking, and auto-switches strategy:

      1. PGD (default) -- if loss improves normally, PGD suffices.
      2. BPDA -- if masking detected AND a defense/approximation pair provided.
      3. EoT -- if a stochastic transform is detected/provided.
      4. Random search with momentum -- fallback when gradients are unusable.

    Detection events and strategy switches are logged to ``AdaptiveAttackLog``.

    Args:
        model: target classifier in ``eval()`` mode.
        images: clean inputs in ``[0, 1]``.
        labels: ground-truth class indices.
        epsilon: L-inf perturbation budget.
        alpha: PGD step size.
        steps: total number of optimization steps.
        defense: optional non-differentiable defense (for BPDA).
        defense_approx: differentiable approximation of the defense.
        transform_fn: optional stochastic transform (for EoT).
        eot_samples: number of EoT samples.
        random_search_steps: steps for random search fallback.

    Returns:
        Tuple of (adversarial images, attack log).
    """
    _require_eval_mode(model)

    attack_log = AdaptiveAttackLog()
    attack_log.log_event("start", strategy="pgd", epsilon=epsilon, steps=steps)

    detector = GradientMaskingDetector(expected_steps=steps)

    x_orig = images.clone().detach()
    x_adv = x_orig.clone().detach()

    # Add random start
    noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

    # Phase 1: PGD with masking detection
    masking_detected = False
    for step_idx in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        loss = nn.functional.cross_entropy(logits, labels)
        loss_val = loss.item()

        if detector.record(loss_val):
            masking_detected = True
            attack_log.log_event(
                "masking_detected",
                step=step_idx,
                loss=loss_val,
                strategy="pgd",
            )
            break

        try:
            grad = torch.autograd.grad(loss, x_adv)[0]
        except RuntimeError:
            # Gradient computation failed — this IS gradient masking
            masking_detected = True
            attack_log.log_event(
                "masking_detected",
                step=step_idx,
                loss=loss_val,
                strategy="pgd",
                reason="grad_computation_failed",
            )
            break
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    if not masking_detected:
        attack_log.log_event("completed", strategy="pgd", final_loss=loss_val)
        return x_adv.detach(), attack_log

    # Phase 2: Try BPDA if defense provided
    if defense is not None and defense_approx is not None:
        attack_log.log_event("switch_strategy", from_strategy="pgd", to_strategy="bpda")
        bpda = BPDA(defense, defense_approx)

        x_adv = x_orig.clone().detach()
        noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

        remaining_steps = steps - (detector.detection_step or 0)
        for _ in range(remaining_steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            defended = bpda(x_adv)
            logits = model(defended)
            loss = nn.functional.cross_entropy(logits, labels)
            try:
                grad = torch.autograd.grad(loss, x_adv)[0]
            except RuntimeError:
                break
            x_adv = x_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

        attack_log.log_event("completed", strategy="bpda", final_loss=loss.item())
        return x_adv.detach(), attack_log

    # Phase 3: Try EoT if transform function provided
    if transform_fn is not None:
        attack_log.log_event("switch_strategy", from_strategy="pgd", to_strategy="eot")
        eot_model = EoT(model, transform_fn, n_samples=eot_samples)

        x_adv = x_orig.clone().detach()
        noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

        remaining_steps = steps - (detector.detection_step or 0)
        for _ in range(remaining_steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            logits = eot_model(x_adv)
            loss = nn.functional.cross_entropy(logits, labels)
            try:
                grad = torch.autograd.grad(loss, x_adv)[0]
            except RuntimeError:
                break
            x_adv = x_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

        attack_log.log_event("completed", strategy="eot", final_loss=loss.item())
        return x_adv.detach(), attack_log

    # Phase 4: Fallback to random search with momentum
    attack_log.log_event(
        "switch_strategy", from_strategy="pgd", to_strategy="random_search_momentum"
    )
    x_adv = _random_search_with_momentum(
        model,
        images,
        labels,
        epsilon=epsilon,
        steps=random_search_steps,
    )
    with torch.no_grad():
        final_loss = nn.functional.cross_entropy(model(x_adv), labels).item()
    attack_log.log_event(
        "completed", strategy="random_search_momentum", final_loss=final_loss
    )
    return x_adv.detach(), attack_log
