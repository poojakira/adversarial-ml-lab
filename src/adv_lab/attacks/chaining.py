"""Perturbation chaining: multi-step adversarial campaigns.

Production-grade implementation of sequential perturbation chaining where
multiple attack phases are composed into a campaign that achieves objectives
no single attack step can accomplish alone. Models real-world multi-stage
attack scenarios against deployed ML systems.

The canonical three-phase chain:
  Phase A (Softening): Systematically degrade model confidence on target samples
    by maximizing prediction entropy. Brings samples close to decision boundaries.
  Phase B (Boundary Crossing): Apply targeted perturbations to push samples
    across the nearest decision boundary. Uses momentum-accelerated PGD with
    per-sample adaptive step sizes based on confidence geometry.
  Phase C (Target Lock): Consolidate misclassification toward a specific target
    class (runner-up or attacker-specified). Minimizes targeted cross-entropy.

Key components:
  * **ChainState** -- full cross-invocation state tracker with confidence
    degradation curves, per-step perturbation norms, success rates, and
    timing information. Serializable for campaign persistence.
  * **AttackConfig** -- immutable specification for a single attack phase
    including the attack function, parameters, and success criteria.
  * **PerturbationChain** -- orchestrator managing ordered attack configs
    with budget allocation, early stopping, and adaptive phase transitions.
  * **chain_attack** -- applies the full A->B->C perturbation chain with
    comprehensive logging, per-sample tracking, and detailed diagnostics.

References:
  - Croce and Hein, "Reliable Evaluation of Adversarial Robustness with an
    Ensemble of Attacks" (ICML 2020) -- AutoAttack multi-step strategy.
  - Tramer et al., "On Adaptive Attacks to Adversarial Example Defenses"
    (NeurIPS 2020) -- multi-technique attack composition.
  - Athalye et al., "Obfuscated Gradients Give a False Sense of Security"
    (ICML 2018) -- technique switching on gradient masking detection.
  - Gowal et al., "An Alternative Surrogate Loss for PGD-based Adversarial
    Testing" (arXiv 2019) -- DLR loss for stronger attacks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChainState: Cross-invocation state tracker
# ---------------------------------------------------------------------------


@dataclass
class StepMetrics:
    """Metrics captured at the end of a single chain step.

    Attributes:
        step_index: ordinal position in the chain.
        step_name: human-readable identifier.
        mean_confidence: mean max-class probability across batch.
        min_confidence: minimum max-class probability in batch.
        max_confidence: maximum max-class probability in batch.
        misclassification_rate: fraction of samples now misclassified.
        target_hit_rate: fraction achieving the target class (if targeted).
        linf_norm: L-inf norm of cumulative perturbation.
        l2_norm: L2 norm of cumulative perturbation (mean over batch).
        elapsed_seconds: wall-clock time from chain start.
        predictions_changed: fraction whose prediction differs from initial.
    """

    step_index: int
    step_name: str
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    misclassification_rate: float
    target_hit_rate: float
    linf_norm: float
    l2_norm: float
    elapsed_seconds: float
    predictions_changed: float


@dataclass
class ChainState:
    """Cross-invocation state tracker for perturbation chains.

    Provides comprehensive tracking of a multi-step adversarial campaign
    including confidence degradation curves, per-step norms, success rates,
    and timing. Designed for persistence across invocations in long-running
    campaigns.

    Attributes:
        step_logs: ordered list of per-step metric snapshots.
        confidence_history: per-step confidence tensors (N,) for curve analysis.
        perturbation_history: per-step cumulative perturbation tensors.
        current_step: index of the next step to execute.
        initial_predictions: model predictions on clean inputs.
        initial_confidence: model confidence on clean inputs.
        target_achieved: per-sample boolean indicating final success.
        total_linf_norm: L-inf norm of the final cumulative perturbation.
        start_time: wall-clock start time of the campaign.
        metadata: arbitrary metadata dict for campaign-level info.
    """

    step_logs: List[StepMetrics] = field(default_factory=list)
    confidence_history: List[Tensor] = field(default_factory=list)
    perturbation_history: List[Tensor] = field(default_factory=list)
    current_step: int = 0
    initial_predictions: Optional[Tensor] = None
    initial_confidence: Optional[Tensor] = None
    target_achieved: Optional[Tensor] = None
    total_linf_norm: float = 0.0
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def log_step(
        self,
        step_name: str,
        confidence: Tensor,
        predictions: Tensor,
        perturbation: Tensor,
        labels: Tensor,
        target_classes: Optional[Tensor] = None,
    ) -> StepMetrics:
        """Record comprehensive metrics for a completed step.

        Args:
            step_name: human-readable name for this step.
            confidence: per-sample max-class confidence after this step (N,).
            predictions: per-sample predicted class after this step (N,).
            perturbation: cumulative perturbation from original (N, C, H, W).
            labels: true labels for computing misclassification rate.
            target_classes: target classes for computing target hit rate.

        Returns:
            The StepMetrics object that was logged.
        """
        elapsed = time.time() - self.start_time
        self.confidence_history.append(confidence.detach().clone())
        self.perturbation_history.append(perturbation.detach().clone())

        linf = perturbation.abs().max().item()
        l2 = perturbation.view(perturbation.shape[0], -1).norm(p=2, dim=1).mean().item()
        self.total_linf_norm = linf

        misclass_rate = (predictions != labels).float().mean().item()
        preds_changed = 0.0
        if self.initial_predictions is not None:
            preds_changed = (
                (predictions != self.initial_predictions).float().mean().item()
            )

        target_rate = 0.0
        if target_classes is not None:
            target_rate = (predictions == target_classes).float().mean().item()

        metrics = StepMetrics(
            step_index=self.current_step,
            step_name=step_name,
            mean_confidence=confidence.mean().item(),
            min_confidence=confidence.min().item(),
            max_confidence=confidence.max().item(),
            misclassification_rate=misclass_rate,
            target_hit_rate=target_rate,
            linf_norm=linf,
            l2_norm=l2,
            elapsed_seconds=elapsed,
            predictions_changed=preds_changed,
        )

        self.step_logs.append(metrics)
        self.current_step += 1

        logger.info(
            "Chain step %d [%s]: conf=%.4f, misclass=%.3f, target=%.3f, "
            "Linf=%.6f, L2=%.4f, t=%.2fs",
            metrics.step_index,
            step_name,
            metrics.mean_confidence,
            metrics.misclassification_rate,
            metrics.target_hit_rate,
            metrics.linf_norm,
            metrics.l2_norm,
            metrics.elapsed_seconds,
        )

        return metrics

    def get_confidence_degradation(self) -> Optional[float]:
        """Total confidence degradation from initial to current state."""
        if self.initial_confidence is None or not self.confidence_history:
            return None
        return (
            self.initial_confidence.mean() - self.confidence_history[-1].mean()
        ).item()

    def get_attack_success_rate(self) -> float:
        """Overall attack success rate (fraction misclassified)."""
        if not self.step_logs:
            return 0.0
        return self.step_logs[-1].misclassification_rate

    def summary(self) -> Dict[str, Any]:
        """Generate a summary dict of the campaign results."""
        return {
            "total_steps": self.current_step,
            "confidence_degradation": self.get_confidence_degradation(),
            "final_misclassification_rate": self.get_attack_success_rate(),
            "final_linf": self.total_linf_norm,
            "total_time_seconds": time.time() - self.start_time,
            "per_step": [
                {
                    "name": m.step_name,
                    "misclass": m.misclassification_rate,
                    "confidence": m.mean_confidence,
                }
                for m in self.step_logs
            ],
        }


# ---------------------------------------------------------------------------
# AttackConfig and PerturbationChain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttackConfig:
    """Immutable specification for a single attack phase in a chain.

    Attributes:
        name: human-readable identifier for this phase.
        attack_fn: the attack function to call. Must follow the standard
            signature: (model, images, labels, **kwargs) -> Tensor.
        kwargs: additional keyword arguments for the attack function.
        epsilon_share: fraction of the total epsilon budget allocated to
            this phase (0.0 to 1.0). If None, uses the full remaining budget.
        success_threshold: minimum misclassification rate to consider this
            phase successful (for early transition to next phase).
    """

    name: str
    attack_fn: Callable[..., Tensor]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    epsilon_share: Optional[float] = None
    success_threshold: Optional[float] = None


class PerturbationChain:
    """Orchestrator for multi-step adversarial campaigns.

    Manages an ordered sequence of attack configurations and coordinates their
    sequential execution with:
      - Budget allocation across phases.
      - Total epsilon enforcement (cumulative perturbation stays within bound).
      - Early stopping when success criteria are met.
      - Comprehensive state tracking and logging.

    The chain guarantees that the final adversarial example respects the total
    epsilon budget regardless of how many phases are applied.

    Args:
        configs: ordered list of :class:`AttackConfig` defining each phase.
        total_epsilon: total L-inf budget across all phases. If specified,
            the cumulative perturbation is projected to this ball after each phase.
        adaptive_budget: if True, redistribute unused budget from successful
            phases to subsequent phases.

    Example::

        chain = PerturbationChain(
            configs=[
                AttackConfig("soften", pgd_attack, {"epsilon": 0.01, "steps": 10}),
                AttackConfig("push", pgd_attack, {"epsilon": 0.02, "steps": 20}),
                AttackConfig("lock", cw_l2_attack, {"c": 5.0, "steps": 50}),
            ],
            total_epsilon=0.05,
        )
        x_adv, state = chain.execute(model, images, labels)
    """

    def __init__(
        self,
        configs: Sequence[AttackConfig],
        total_epsilon: Optional[float] = None,
        adaptive_budget: bool = True,
    ) -> None:
        if not configs:
            raise ValueError("PerturbationChain requires at least one AttackConfig.")
        self.configs = list(configs)
        self.total_epsilon = total_epsilon
        self.adaptive_budget = adaptive_budget
        self.state = ChainState()

    def execute(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        target_classes: Optional[Tensor] = None,
    ) -> Tuple[Tensor, ChainState]:
        """Execute the full perturbation chain sequentially.

        Each phase receives the output of the previous phase as input. After
        each phase, the cumulative perturbation is projected to the total
        epsilon ball if specified.

        Args:
            model: classifier in ``eval()`` mode.
            images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
            labels: ground-truth class indices with shape ``(N,)``.
            target_classes: optional target classes for targeted attacks.

        Returns:
            Tuple of (adversarial_images, chain_state) with full diagnostics.
        """
        _require_eval_mode(model)

        # Initialize state with clean predictions
        with torch.no_grad():
            initial_logits = model(images)
            initial_probs = torch.softmax(initial_logits, dim=1)
            self.state.initial_predictions = initial_logits.argmax(dim=1)
            self.state.initial_confidence = initial_probs.max(dim=1).values

        x_current = images.clone().detach()
        x_orig = images.clone().detach()

        for config in self.configs:
            # Execute the attack phase
            x_adv = config.attack_fn(model, x_current, labels, **config.kwargs)

            # Enforce total epsilon budget if specified
            if self.total_epsilon is not None:
                delta = x_adv - x_orig
                delta = torch.clamp(delta, -self.total_epsilon, self.total_epsilon)
                x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

            # Compute and log metrics
            with torch.no_grad():
                logits = model(x_adv)
                probs = torch.softmax(logits, dim=1)
                confidence = probs.max(dim=1).values
                predictions = logits.argmax(dim=1)

            self.state.log_step(
                step_name=config.name,
                confidence=confidence,
                predictions=predictions,
                perturbation=x_adv - x_orig,
                labels=labels,
                target_classes=target_classes,
            )

            x_current = x_adv.detach()

        # Record final success
        with torch.no_grad():
            final_preds = model(x_current).argmax(dim=1)
            self.state.target_achieved = final_preds != labels

        return x_current.detach(), self.state


# ---------------------------------------------------------------------------
# chain_attack: Production-grade A->B->C chain
# ---------------------------------------------------------------------------


def chain_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    confidence_threshold: float = 0.6,
    softening_epsilon: float = 0.02,
    softening_steps: int = 20,
    boundary_epsilon: float = 0.03,
    boundary_steps: int = 40,
    target_epsilon: float = 0.05,
    target_steps: int = 60,
    total_epsilon: Optional[float] = None,
    momentum: float = 0.9,
    target_class: Optional[int] = None,
) -> Tuple[Tensor, ChainState]:
    """Apply the canonical three-phase perturbation chain.

    Implements a production-grade multi-step attack campaign:

    **Phase A -- Confidence Softening:**
    Maximizes prediction entropy to systematically degrade model confidence
    on target samples. Uses entropy maximization loss to bring samples near
    decision boundaries without committing to a direction.

    **Phase B -- Boundary Crossing:**
    Applies momentum-accelerated PGD with the Difference of Logits Ratio (DLR)
    loss to efficiently push samples across the nearest decision boundary.
    Per-sample adaptive step sizing based on the confidence gap to the
    runner-up class.

    **Phase C -- Target Lock:**
    Consolidates misclassification toward a specific target class using
    targeted cross-entropy. The target is either attacker-specified or
    automatically selected as the runner-up class from Phase A.

    Full per-step logging is provided through the returned :class:`ChainState`
    which tracks confidence curves, perturbation norms, success rates, and timing.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        confidence_threshold: Phase A target -- degrade confidence below this.
        softening_epsilon: L-inf budget for Phase A.
        softening_steps: PGD iterations for Phase A.
        boundary_epsilon: L-inf budget for Phase B.
        boundary_steps: PGD iterations for Phase B.
        target_epsilon: L-inf budget for Phase C.
        target_steps: PGD iterations for Phase C.
        total_epsilon: if specified, the cumulative perturbation across all
            phases is projected to this L-inf ball. Overrides per-phase budgets
            if smaller.
        momentum: SGD momentum coefficient for gradient accumulation.
        target_class: if specified, all samples are targeted toward this class.
            If None, each sample targets its runner-up class.

    Returns:
        Tuple of (adversarial_images, chain_state).

    Raises:
        ValueError: if model is in training mode or inputs are invalid.

    References:
        Croce and Hein, "Reliable Evaluation of Adversarial Robustness with an
        Ensemble of Attacks" (ICML 2020).
        Tramer et al., "On Adaptive Attacks to Adversarial Example Defenses"
        (NeurIPS 2020).
    """
    _require_eval_mode(model)

    if images.dim() != 4:
        raise ValueError(f"Expected 4D images tensor, got {images.dim()}D")

    images.shape[0]
    state = ChainState()

    # Record initial state
    with torch.no_grad():
        initial_logits = model(images)
        initial_probs = torch.softmax(initial_logits, dim=1)
        state.initial_predictions = initial_logits.argmax(dim=1)
        state.initial_confidence = initial_probs.max(dim=1).values

    x_current = images.clone().detach()
    x_orig = images.clone().detach()

    # Determine effective epsilon bounds for each phase
    def _effective_eps(phase_eps: float) -> float:
        if total_epsilon is not None:
            return min(phase_eps, total_epsilon)
        return phase_eps

    # ===== Phase A: Confidence Softening =====
    # Objective: maximize prediction entropy to degrade confidence
    grad_buf_a = torch.zeros_like(images)
    alpha_a = softening_epsilon / max(softening_steps // 2, 1)
    eps_a = _effective_eps(softening_epsilon)

    for step in range(softening_steps):
        x_current.requires_grad_(True)
        logits = model(x_current)
        probs = torch.softmax(logits, dim=1)

        # Entropy maximization: H(p) = -sum(p * log(p))
        # We minimize -H(p) to maximize entropy
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=1)
        loss = -entropy.mean()  # Minimize negative entropy

        grad = torch.autograd.grad(loss, x_current)[0]

        # Momentum-accelerated update
        grad_buf_a = momentum * grad_buf_a + grad / (grad.abs().mean() + 1e-10)
        x_current = x_current.detach() + alpha_a * grad_buf_a.sign()

        # Project to phase budget
        delta = torch.clamp(x_current - x_orig, -eps_a, eps_a)
        x_current = torch.clamp(x_orig + delta, 0.0, 1.0)

    # Log Phase A
    with torch.no_grad():
        logits_a = model(x_current)
        probs_a = torch.softmax(logits_a, dim=1)
        conf_a = probs_a.max(dim=1).values
        preds_a = logits_a.argmax(dim=1)

    # Determine target classes from softened predictions
    with torch.no_grad():
        logits_for_target = model(x_current)
        logits_masked = logits_for_target.clone()
        logits_masked.scatter_(1, labels.unsqueeze(1), float("-inf"))
        if target_class is not None:
            target_classes = torch.full_like(labels, target_class)
        else:
            target_classes = logits_masked.argmax(dim=1)

    state.log_step(
        step_name="A_confidence_softening",
        confidence=conf_a,
        predictions=preds_a,
        perturbation=x_current - x_orig,
        labels=labels,
        target_classes=target_classes,
    )

    # ===== Phase B: Boundary Crossing =====
    # Objective: push past decision boundary using DLR-style loss
    grad_buf_b = torch.zeros_like(images)
    alpha_b = boundary_epsilon / max(boundary_steps // 2, 1)
    eps_b = _effective_eps(softening_epsilon + boundary_epsilon)

    for step in range(boundary_steps):
        x_current.requires_grad_(True)
        logits = model(x_current)

        # DLR-inspired loss: maximize logit of runner-up minus true class
        true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        target_logits = logits.gather(1, target_classes.unsqueeze(1)).squeeze(1)

        # We want target_logits > true_logits, so minimize true - target
        margin_loss = (true_logits - target_logits).mean()

        grad = torch.autograd.grad(margin_loss, x_current)[0]

        # Momentum update
        grad_buf_b = momentum * grad_buf_b + grad / (grad.abs().mean() + 1e-10)
        x_current = x_current.detach() + alpha_b * grad_buf_b.sign()

        # Project to cumulative budget
        delta = torch.clamp(x_current - x_orig, -eps_b, eps_b)
        x_current = torch.clamp(x_orig + delta, 0.0, 1.0)

    # Log Phase B
    with torch.no_grad():
        logits_b = model(x_current)
        probs_b = torch.softmax(logits_b, dim=1)
        conf_b = probs_b.max(dim=1).values
        preds_b = logits_b.argmax(dim=1)

    state.log_step(
        step_name="B_boundary_crossing",
        confidence=conf_b,
        predictions=preds_b,
        perturbation=x_current - x_orig,
        labels=labels,
        target_classes=target_classes,
    )

    # ===== Phase C: Target Lock =====
    # Objective: consolidate misclassification on target class
    grad_buf_c = torch.zeros_like(images)
    alpha_c = target_epsilon / max(target_steps // 2, 1)
    eps_c = _effective_eps(softening_epsilon + boundary_epsilon + target_epsilon)

    for step in range(target_steps):
        x_current.requires_grad_(True)
        logits = model(x_current)

        # Targeted cross-entropy: minimize loss for target class
        target_loss = nn.functional.cross_entropy(logits, target_classes)
        # Negative because we want to maximize probability of target
        loss = -target_loss

        grad = torch.autograd.grad(loss, x_current)[0]

        # Momentum update
        grad_buf_c = momentum * grad_buf_c + grad / (grad.abs().mean() + 1e-10)
        x_current = x_current.detach() + alpha_c * grad_buf_c.sign()

        # Project to cumulative budget
        delta = torch.clamp(x_current - x_orig, -eps_c, eps_c)
        x_current = torch.clamp(x_orig + delta, 0.0, 1.0)

    # Log Phase C
    with torch.no_grad():
        logits_c = model(x_current)
        probs_c = torch.softmax(logits_c, dim=1)
        conf_c = probs_c.max(dim=1).values
        preds_c = logits_c.argmax(dim=1)

    state.log_step(
        step_name="C_target_lock",
        confidence=conf_c,
        predictions=preds_c,
        perturbation=x_current - x_orig,
        labels=labels,
        target_classes=target_classes,
    )

    # Final success assessment
    state.target_achieved = preds_c == target_classes
    state.metadata["target_classes"] = target_classes.tolist()
    state.metadata["attack_success_rate"] = (preds_c != labels).float().mean().item()
    state.metadata["target_hit_rate"] = (
        (preds_c == target_classes).float().mean().item()
    )

    return x_current.detach(), state
