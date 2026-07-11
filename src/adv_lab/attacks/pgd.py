"""Projected Gradient Descent (PGD).

Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"
(ICLR 2018). PGD is FGSM done right: multiple small signed steps, each
followed by a projection back into the epsilon-ball around the clean input.
It is the standard "honest" white-box attack, and the inner loop of
adversarial training.

This module ships BOTH the L-inf and L2 variants on purpose. A model trained
to resist L-inf PGD is frequently *not* robust to L2 PGD (and vice versa) --
the multi-norm / union-robustness gap that remains open in 2025-2026. Shipping
both lets the benchmark harness expose that gap instead of hiding it behind a
single norm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode


def pgd_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.007,
    steps: int = 40,
    random_start: bool = True,
) -> Tensor:
    """PGD under an L-inf constraint.

    Each iteration takes a signed gradient step of size ``alpha`` and then
    projects the perturbation back into the L-inf epsilon-ball and the valid
    ``[0, 1]`` image range::

        x_adv <- clamp(clamp(x_adv, x - eps, x + eps), 0, 1)

    Args:
        model: classifier in ``eval()`` mode.
        images: clean inputs in ``[0, 1]``, shape ``(N, C, H, W)``.
        labels: ground-truth class indices, shape ``(N,)``.
        epsilon: L-inf budget.
        alpha: per-step size.
        steps: number of PGD iterations.
        random_start: if True, initialize inside the epsilon-ball with uniform
            noise in ``[-epsilon, epsilon]`` (recommended; escapes flat points).

    Returns:
        Detached adversarial images, same shape as input.
    """
    _require_eval_mode(model)

    x_orig = images.clone().detach()
    x_adv = x_orig.clone().detach()

    if random_start:
        noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        loss = nn.functional.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]

        # Ascend the loss, then project into the L-inf ball and image range.
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    return x_adv.detach()


def pgd_linf(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.007,
    steps: int = 40,
    random_start: bool = True,
) -> Tensor:
    """Explicit L-inf alias for :func:`pgd_attack` (readability at call sites)."""
    return pgd_attack(
        model,
        images,
        labels,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        random_start=random_start,
    )


def pgd_l2(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.5,
    alpha: float = 0.1,
    steps: int = 40,
    random_start: bool = True,
) -> Tensor:
    """PGD under an L2 constraint.

    Differences from the L-inf variant:
      * the step follows the *normalized* gradient (unit L2 direction) rather
        than its sign, and
      * the projection rescales the whole perturbation so its L2 norm never
        exceeds ``epsilon`` (instead of clamping each coordinate).

    Args:
        model: classifier in ``eval()`` mode.
        images: clean inputs in ``[0, 1]``.
        labels: ground-truth class indices.
        epsilon: L2 budget (radius of the ball).
        alpha: per-step size along the unit gradient direction.
        steps: number of iterations.
        random_start: initialize with random noise inside the L2 ball.

    Returns:
        Detached adversarial images, same shape as input.
    """
    _require_eval_mode(model)

    x_orig = images.clone().detach()
    x_adv = x_orig.clone().detach()
    batch = x_adv.shape[0]
    eps_tol = 1e-12  # guard against divide-by-zero on flat gradients

    if random_start:
        noise = torch.empty_like(x_adv).normal_()
        noise_flat = noise.view(batch, -1)
        noise_norm = noise_flat.norm(p=2, dim=1).clamp_min(eps_tol)
        # Scale random direction to a random radius <= epsilon.
        rand_radius = torch.rand(batch, device=x_adv.device) * epsilon
        noise_flat = noise_flat / noise_norm.unsqueeze(1) * rand_radius.unsqueeze(1)
        x_adv = torch.clamp(x_orig + noise_flat.view_as(x_adv), 0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        loss = nn.functional.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]

        # Step along the unit-L2 gradient direction.
        grad_flat = grad.view(batch, -1)
        grad_norm = grad_flat.norm(p=2, dim=1).clamp_min(eps_tol)
        unit_grad = (grad_flat / grad_norm.unsqueeze(1)).view_as(grad)
        x_adv = x_adv.detach() + alpha * unit_grad

        # Project the perturbation back into the L2 ball of radius epsilon.
        delta = (x_adv - x_orig).view(batch, -1)
        delta_norm = delta.norm(p=2, dim=1).clamp_min(eps_tol)
        factor = (epsilon / delta_norm).clamp(max=1.0)
        delta = delta * factor.unsqueeze(1)
        x_adv = torch.clamp(x_orig + delta.view_as(x_orig), 0.0, 1.0)

    return x_adv.detach()
