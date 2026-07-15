"""Ensemble and multi-model adversarial attacks.

Attacks that optimize perturbations against multiple models simultaneously,
maximizing transferability and robustness to model uncertainty.

Key components:
  * **ensemble_attack** -- PGD-style attack where the loss is a weighted sum
    of cross-entropy losses across N heterogeneous models.
  * **build_attacker_ensemble** -- constructs an attacker-side ensemble from
    a collection of models (e.g., stolen substitutes from model-stealing).
  * **weighted_ensemble_pgd** -- full PGD implementation with per-model
    weights optimizing the combined cross-entropy loss.

The core insight is that adversarial examples optimized against an ensemble
of diverse models transfer better to unknown target models (Liu et al.,
ICLR 2017). Combined with model stealing (Tramer et al., 2016), this
enables practical black-box attacks without direct gradient access.

References:
  - Liu et al., "Delving into Transferable Adversarial Examples and
    Black-box Attacks" (ICLR 2017).
  - Tramer et al., "Ensemble Adversarial Training: Attacks and Defenses"
    (ICLR 2018).
  - Dong et al., "Boosting Adversarial Attacks with Momentum" (CVPR 2018).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode


# ---------------------------------------------------------------------------
# Ensemble attack
# ---------------------------------------------------------------------------


def ensemble_attack(
    models: Sequence[nn.Module],
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.007,
    steps: int = 40,
    weights: Optional[List[float]] = None,
    random_start: bool = True,
) -> Tensor:
    """PGD-style attack optimizing combined loss across N models.

    Computes ``loss = sum(w_i * CE(model_i(x_adv), y))`` and takes a signed
    gradient step on the weighted sum. This produces adversarial examples
    that fool all models simultaneously, improving transferability to
    unknown targets.

    Reference: Liu et al., "Delving into Transferable Adversarial Examples
    and Black-box Attacks" (ICLR 2017).

    Args:
        models: sequence of N classifiers, all in ``eval()`` mode.
        images: clean inputs in ``[0, 1]``, shape ``(B, C, H, W)``.
        labels: ground-truth class indices, shape ``(B,)``.
        epsilon: L-inf perturbation budget.
        alpha: per-step size.
        steps: number of PGD iterations.
        weights: per-model weights (summing to 1). If None, uniform weights.
        random_start: if True, initialize with uniform noise in epsilon-ball.

    Returns:
        Detached adversarial images, same shape as input.
    """
    for m in models:
        _require_eval_mode(m)

    n_models = len(models)
    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    x_orig = images.clone().detach()
    x_adv = x_orig.clone().detach()

    if random_start:
        noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)

        # Compute weighted ensemble loss
        total_loss = torch.tensor(0.0, device=images.device, requires_grad=True)
        for model, weight in zip(models, weights):
            logits = model(x_adv)
            loss_i = nn.functional.cross_entropy(logits, labels)
            total_loss = total_loss + weight * loss_i

        grad = torch.autograd.grad(total_loss, x_adv)[0]

        # PGD step with projection
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    return x_adv.detach()


# ---------------------------------------------------------------------------
# Build attacker ensemble
# ---------------------------------------------------------------------------


def build_attacker_ensemble(
    model_constructors: List[Callable[[], nn.Module]],
    train_fn: Optional[Callable[[nn.Module], nn.Module]] = None,
    device: Optional[torch.device] = None,
) -> List[nn.Module]:
    """Construct an attacker-side ensemble from model constructors.

    In practical black-box attacks, the attacker builds an ensemble of
    substitute models (possibly via model stealing) to approximate the
    target. This function takes a list of model constructors and optionally
    trains them, returning ready-to-use ensemble members.

    Reference: Tramer et al., "Ensemble Adversarial Training: Attacks and
    Defenses" (ICLR 2018).

    Args:
        model_constructors: list of callables that return fresh nn.Module
            instances (different architectures for diversity).
        train_fn: optional function that trains a model in-place and returns
            it. If None, models are used as-is (assumed pre-trained).
        device: optional device to move models to.

    Returns:
        List of models in eval() mode, ready for ensemble_attack.

    Example::

        constructors = [
            lambda: SmallCNN(num_classes=10),
            lambda: ResNet18(num_classes=10),
            lambda: VGG11(num_classes=10),
        ]
        ensemble = build_attacker_ensemble(constructors, train_fn=my_trainer)
        x_adv = ensemble_attack(ensemble, images, labels)
    """
    ensemble: List[nn.Module] = []

    for constructor in model_constructors:
        model = constructor()
        if device is not None:
            model = model.to(device)
        if train_fn is not None:
            model = train_fn(model)
        model.eval()
        ensemble.append(model)

    return ensemble


# ---------------------------------------------------------------------------
# Weighted ensemble PGD
# ---------------------------------------------------------------------------


def weighted_ensemble_pgd(
    models: Sequence[nn.Module],
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.007,
    steps: int = 40,
    weights: Optional[List[float]] = None,
    random_start: bool = True,
    momentum: float = 0.0,
    targeted: bool = False,
    target_labels: Optional[Tensor] = None,
) -> Tensor:
    """Full PGD implementation with weighted multi-model loss.

    Extends :func:`ensemble_attack` with:
      * Momentum (MI-FGSM, Dong et al., CVPR 2018).
      * Targeted attack support.
      * Per-model gradient accumulation for memory efficiency.

    The combined loss is::

        L = sum_i w_i * CE(model_i(x_adv), y)

    For targeted attacks, the loss is negated to drive predictions toward
    target_labels.

    Args:
        models: sequence of N classifiers in ``eval()`` mode.
        images: clean inputs in ``[0, 1]``.
        labels: ground-truth class indices (untargeted) or ignored (targeted).
        epsilon: L-inf perturbation budget.
        alpha: per-step size.
        steps: number of iterations.
        weights: per-model weights (uniform if None).
        random_start: whether to initialize randomly in epsilon-ball.
        momentum: momentum decay factor (0 = no momentum).
        targeted: if True, minimize loss toward target_labels.
        target_labels: target class indices for targeted attack.

    Returns:
        Detached adversarial images.
    """
    for m in models:
        _require_eval_mode(m)

    n_models = len(models)
    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    attack_labels = target_labels if targeted and target_labels is not None else labels

    x_orig = images.clone().detach()
    x_adv = x_orig.clone().detach()
    velocity = torch.zeros_like(x_adv)

    if random_start:
        noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)

        # Accumulate gradients from all models
        total_loss = torch.tensor(0.0, device=images.device, requires_grad=True)
        for model, weight in zip(models, weights):
            logits = model(x_adv)
            loss_i = nn.functional.cross_entropy(logits, attack_labels)
            total_loss = total_loss + weight * loss_i

        grad = torch.autograd.grad(total_loss, x_adv)[0]

        # For targeted attacks, descend (minimize) instead of ascend
        if targeted:
            grad = -grad

        # Momentum update
        if momentum > 0:
            # Normalize gradient by L1 norm (MI-FGSM)
            grad_norm = (
                grad.abs()
                .mean(dim=list(range(1, grad.ndim)), keepdim=True)
                .clamp(min=1e-8)
            )
            grad = grad / grad_norm
            velocity = momentum * velocity + grad
            step_direction = velocity.sign()
        else:
            step_direction = grad.sign()

        x_adv = x_adv.detach() + alpha * step_direction
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    return x_adv.detach()
