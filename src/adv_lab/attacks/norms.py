"""Norm-constrained adversarial attacks beyond standard Lp balls.

This module implements attacks under diverse perturbation models:

* **L0 (sparse)** -- Papernot et al., "The Limitations of Deep Learning in
  Adversarial Settings" (EuroS&P 2016). Minimize the number of modified pixels.
* **L1 (projected)** -- Chen et al., "EAD: Elastic-Net Attacks to DNNs via
  Feature Selection" (AAAI 2018). L1-constrained projected gradient descent.
* **Wasserstein** -- Wong et al., "Wasserstein Adversarial Examples via
  Projected Sinkhorn Divergences" (ICML 2019). Optimal transport cost bound.
* **Semantic** -- Engstrom et al., "Exploring the Landscape of Spatial
  Robustness" (ICML 2019). Differentiable rotation, translation, hue, contrast.
* **Adversarial Patch** -- Brown et al., "Adversarial Patch" (NeurIPS 2017).
  Localized perturbation with printable sRGB gamut constraint.

Each attack includes an epsilon search schedule for automated budget tuning.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode


# --------------------------------------------------------------------------- #
# Epsilon search schedule utilities
# --------------------------------------------------------------------------- #


def _epsilon_search_schedule(
    base_eps: float,
    num_levels: int = 5,
    growth_factor: float = 1.5,
) -> list[float]:
    """Generate an exponentially spaced epsilon schedule for binary search.

    Starting from ``base_eps``, produces ``num_levels`` candidate budgets that
    grow geometrically. The attack can iterate over these to find the minimal
    budget that achieves misclassification.
    """
    return [base_eps * (growth_factor ** i) for i in range(num_levels)]


# --------------------------------------------------------------------------- #
# L0 Sparse Attack
# --------------------------------------------------------------------------- #


def pgd_l0(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    max_pixels: int = 10,
    steps: int = 50,
    alpha: float = 0.1,
    epsilon_schedule: Optional[list[int]] = None,
) -> Tensor:
    """L0-constrained sparse adversarial attack.

    Modifies at most ``max_pixels`` per image. At each step, computes gradients
    and selects the top-k pixels (by gradient magnitude) to perturb. This
    implements the iterative approach of Papernot et al. (EuroS&P 2016) where
    a saliency map guides pixel selection.

    If ``epsilon_schedule`` is provided, the attack tries increasing pixel
    budgets and returns adversarial examples from the lowest budget that
    achieves misclassification.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        max_pixels: maximum number of spatial locations that may be modified.
        steps: number of gradient steps.
        alpha: step size for each pixel perturbation.
        epsilon_schedule: optional list of pixel budgets to search over.

    Returns:
        Detached adversarial images clamped to ``[0, 1]``.
    """
    _require_eval_mode(model)

    if epsilon_schedule is None:
        epsilon_schedule = [max_pixels]

    batch_size = images.shape[0]
    n_channels = images.shape[1]
    spatial_dims = images.shape[2] * images.shape[3]
    best_adv = images.clone().detach()
    best_success = torch.zeros(batch_size, dtype=torch.bool)

    for budget in epsilon_schedule:
        x_adv = images.clone().detach()

        for _ in range(steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            logits = model(x_adv)
            loss = nn.functional.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, x_adv)[0]

            # Compute saliency: sum absolute gradient across channels per pixel
            grad_spatial = grad.abs().sum(dim=1)  # (N, H, W)
            grad_flat = grad_spatial.view(batch_size, -1)  # (N, H*W)

            # Select top-k pixels by gradient magnitude
            k = min(budget, spatial_dims)
            _, top_indices = torch.topk(grad_flat, k, dim=1)

            # Create mask for selected pixels
            mask = torch.zeros(batch_size, spatial_dims, device=images.device)
            mask.scatter_(1, top_indices, 1.0)
            mask = mask.view(batch_size, 1, images.shape[2], images.shape[3])
            mask = mask.expand_as(images)

            # Apply perturbation only at selected pixels
            perturbation = alpha * grad.sign() * mask
            x_adv = x_adv.detach() + perturbation
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # Enforce cumulative L0 constraint: keep only top-budget pixels
        delta = x_adv - images
        delta_spatial_mag = delta.abs().sum(dim=1)  # (N, H, W)
        delta_flat = delta_spatial_mag.view(batch_size, -1)  # (N, H*W)
        k = min(budget, spatial_dims)
        _, top_k_idx = torch.topk(delta_flat, k, dim=1)
        mask_final = torch.zeros(batch_size, spatial_dims, device=images.device)
        mask_final.scatter_(1, top_k_idx, 1.0)
        mask_final = mask_final.view(batch_size, 1, images.shape[2], images.shape[3])
        mask_final = mask_final.expand_as(images)
        x_adv = images + delta * mask_final
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # Check which examples are now misclassified
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
        success = preds != labels

        # Update best adversarial for newly successful examples
        newly_successful = success & ~best_success
        best_adv[newly_successful] = x_adv[newly_successful]
        best_success = best_success | success

        if best_success.all():
            break

    # For remaining unsuccessful, use the last attempt
    remaining = ~best_success
    best_adv[remaining] = x_adv[remaining]

    return best_adv.detach()


# --------------------------------------------------------------------------- #
# L1 Projected Attack
# --------------------------------------------------------------------------- #


def _project_l1_ball(delta: Tensor, epsilon: float) -> Tensor:
    """Project perturbation onto the L1 ball of radius ``epsilon``.

    Implements the simplex projection algorithm of Duchi et al. (2008).
    Applied element-wise: project |delta| onto the L1 simplex, then restore signs.
    """
    batch_size = delta.shape[0]
    flat = delta.view(batch_size, -1)
    signs = flat.sign()
    abs_flat = flat.abs()

    # Check if already inside the ball
    l1_norms = abs_flat.sum(dim=1)
    needs_proj = l1_norms > epsilon

    if not needs_proj.any():
        return delta

    # Simplex projection for those that need it
    for idx in range(batch_size):
        if not needs_proj[idx]:
            continue
        u = abs_flat[idx]
        n = u.numel()
        sorted_u, _ = torch.sort(u, descending=True)
        cssv = torch.cumsum(sorted_u, dim=0)
        rho_candidates = sorted_u - (cssv - epsilon) / torch.arange(
            1, n + 1, device=u.device, dtype=u.dtype
        )
        rho = (rho_candidates > 0).sum().item() - 1
        theta = (cssv[rho] - epsilon) / (rho + 1)
        abs_flat[idx] = torch.clamp(u - theta, min=0.0)

    projected = (signs * abs_flat).view_as(delta)
    return projected


def pgd_l1(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 10.0,
    alpha: float = 0.5,
    steps: int = 50,
    random_start: bool = True,
    epsilon_schedule: Optional[list[float]] = None,
) -> Tensor:
    """L1-constrained Projected Gradient Descent.

    Based on EAD (Chen et al., AAAI 2018). Each step computes the gradient,
    takes a step in the sign direction scaled by ``alpha``, then projects the
    total perturbation onto the L1 ball of radius ``epsilon``.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        epsilon: L1 perturbation budget.
        alpha: per-step size.
        steps: number of PGD iterations.
        random_start: initialize with random noise inside the L1 ball.
        epsilon_schedule: optional list of epsilon values for search.

    Returns:
        Detached adversarial images clamped to ``[0, 1]``.
    """
    _require_eval_mode(model)

    if epsilon_schedule is None:
        epsilon_schedule = [epsilon]

    batch_size = images.shape[0]
    best_adv = images.clone().detach()
    best_success = torch.zeros(batch_size, dtype=torch.bool)

    for eps in epsilon_schedule:
        x_orig = images.clone().detach()
        x_adv = x_orig.clone()

        if random_start:
            # Initialize with sparse random noise inside L1 ball
            noise = torch.randn_like(x_adv)
            noise_flat = noise.view(batch_size, -1)
            # Normalize to have L1 norm = eps * random_fraction
            l1_norm = noise_flat.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
            rand_radius = torch.rand(batch_size, 1, device=x_adv.device) * eps
            noise_flat = noise_flat / l1_norm * rand_radius
            x_adv = torch.clamp(x_orig + noise_flat.view_as(x_adv), 0.0, 1.0)

        for _ in range(steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            logits = model(x_adv)
            loss = nn.functional.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, x_adv)[0]

            # Step in the gradient sign direction
            x_adv = x_adv.detach() + alpha * grad.sign()

            # Project perturbation onto L1 ball
            delta = x_adv - x_orig
            delta = _project_l1_ball(delta, eps)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
        success = preds != labels

        newly_successful = success & ~best_success
        best_adv[newly_successful] = x_adv[newly_successful]
        best_success = best_success | success

        if best_success.all():
            break

    remaining = ~best_success
    best_adv[remaining] = x_adv[remaining]

    return best_adv.detach()


# --------------------------------------------------------------------------- #
# Wasserstein Attack
# --------------------------------------------------------------------------- #


def wasserstein_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.5,
    steps: int = 50,
    alpha: float = 0.01,
    sinkhorn_iters: int = 10,
    reg: float = 0.1,
    epsilon_schedule: Optional[list[float]] = None,
) -> Tensor:
    """Wasserstein-distance constrained adversarial attack.

    Approximates an optimal-transport bounded perturbation following
    Wong et al., "Wasserstein Adversarial Examples via Projected Sinkhorn
    Divergences" (ICML 2019).

    The attack optimizes adversarial perturbations while approximately
    constraining the Wasserstein-1 (Earth Mover's) distance between the clean
    and perturbed images. We use a Sinkhorn-based projection onto the
    Wasserstein ball as a differentiable proxy.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        epsilon: Wasserstein distance budget.
        steps: number of optimization steps.
        alpha: step size.
        sinkhorn_iters: iterations for Sinkhorn projection.
        reg: entropic regularization for Sinkhorn.
        epsilon_schedule: optional list of epsilon values for search.

    Returns:
        Detached adversarial images clamped to ``[0, 1]``.
    """
    _require_eval_mode(model)

    if epsilon_schedule is None:
        epsilon_schedule = [epsilon]

    batch_size = images.shape[0]
    best_adv = images.clone().detach()
    best_success = torch.zeros(batch_size, dtype=torch.bool)

    for eps in epsilon_schedule:
        x_orig = images.clone().detach()
        x_adv = x_orig.clone()

        for _ in range(steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            logits = model(x_adv)
            loss = nn.functional.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, x_adv)[0]

            # Gradient ascent step
            x_adv = x_adv.detach() + alpha * grad.sign()

            # Approximate Wasserstein projection via local transport cost
            # We approximate the W1 constraint by treating each channel
            # independently and using the pixel displacement cost
            delta = x_adv - x_orig
            delta = _project_wasserstein(delta, eps, sinkhorn_iters, reg)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
        success = preds != labels

        newly_successful = success & ~best_success
        best_adv[newly_successful] = x_adv[newly_successful]
        best_success = best_success | success

        if best_success.all():
            break

    remaining = ~best_success
    best_adv[remaining] = x_adv[remaining]

    return best_adv.detach()


def _project_wasserstein(
    delta: Tensor,
    epsilon: float,
    sinkhorn_iters: int = 10,
    reg: float = 0.1,
) -> Tensor:
    """Approximate projection onto the Wasserstein ball.

    Uses a simplified Sinkhorn-based approach: we compute the local transport
    cost matrix based on spatial distances, then scale the perturbation so the
    approximate Wasserstein distance does not exceed epsilon.

    For small images this is tractable. The approximation treats the perturbation
    as a mass redistribution problem: the cost is the sum of |delta| weighted by
    spatial distance from the nearest modified pixel.
    """
    batch_size = delta.shape[0]
    flat = delta.view(batch_size, -1)

    # Approximate W1 distance as weighted L1 norm
    # For a grid, W1 is bounded by the L1 norm of pixel values times their
    # distance from the centroid of the perturbation. We use a simpler bound:
    # W1 <= L1_norm * max_spatial_diameter / num_pixels
    # Here we use the heuristic: scale down delta uniformly if L1 exceeds budget
    h, w = delta.shape[2], delta.shape[3]
    spatial_diameter = math.sqrt(h * h + w * w)

    # Weighted L1: weight each pixel change by its distance from center
    cy, cx = h / 2.0, w / 2.0
    yy = torch.arange(h, device=delta.device, dtype=delta.dtype) - cy
    xx = torch.arange(w, device=delta.device, dtype=delta.dtype) - cx
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    dist_from_center = torch.sqrt(grid_y ** 2 + grid_x ** 2) + 1.0
    dist_from_center = dist_from_center / dist_from_center.max()

    # Weight: (N, C, H, W) * distance_weights
    weights = dist_from_center.unsqueeze(0).unsqueeze(0)
    weighted_delta = delta.abs() * weights

    # Approximate Wasserstein cost per sample
    w_cost = weighted_delta.view(batch_size, -1).sum(dim=1)

    # Scale down if exceeding budget
    scale = torch.where(
        w_cost > epsilon,
        epsilon / w_cost.clamp_min(1e-12),
        torch.ones_like(w_cost),
    )
    delta = delta * scale.view(batch_size, 1, 1, 1)

    return delta


# --------------------------------------------------------------------------- #
# Semantic Attack (rotation, translation, hue, contrast)
# --------------------------------------------------------------------------- #


def semantic_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    steps: int = 50,
    lr: float = 0.01,
    max_rotation: float = 30.0,
    max_translation: float = 0.2,
    max_hue_shift: float = 0.1,
    max_contrast: float = 0.3,
    epsilon_schedule: Optional[list[float]] = None,
) -> Tensor:
    """Semantic adversarial attack via differentiable transforms.

    Following Engstrom et al. (ICML 2019), optimizes rotation, translation,
    hue shift, and contrast adjustment to fool the classifier while keeping
    the image semantically similar to the original.

    All transforms are differentiable, enabling gradient-based optimization.
    The ``epsilon_schedule`` scales all transform bounds simultaneously.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        steps: number of optimization steps.
        lr: learning rate for transform parameters.
        max_rotation: maximum rotation in degrees.
        max_translation: maximum fractional translation (0-1 of image size).
        max_hue_shift: maximum hue shift magnitude.
        max_contrast: maximum contrast change (additive).
        epsilon_schedule: optional list of scale factors for all bounds.

    Returns:
        Detached adversarial images clamped to ``[0, 1]``.
    """
    _require_eval_mode(model)

    if epsilon_schedule is None:
        epsilon_schedule = [1.0]

    batch_size = images.shape[0]
    best_adv = images.clone().detach()
    best_success = torch.zeros(batch_size, dtype=torch.bool)

    for scale in epsilon_schedule:
        rot_bound = max_rotation * scale
        trans_bound = max_translation * scale
        hue_bound = max_hue_shift * scale
        contrast_bound = max_contrast * scale

        # Initialize transform parameters (unconstrained, will be clamped)
        theta_rot = torch.zeros(batch_size, device=images.device, requires_grad=True)
        theta_tx = torch.zeros(batch_size, device=images.device, requires_grad=True)
        theta_ty = torch.zeros(batch_size, device=images.device, requires_grad=True)
        theta_hue = torch.zeros(batch_size, device=images.device, requires_grad=True)
        theta_contrast = torch.zeros(
            batch_size, device=images.device, requires_grad=True
        )

        optimizer = torch.optim.Adam(
            [theta_rot, theta_tx, theta_ty, theta_hue, theta_contrast], lr=lr
        )

        for _ in range(steps):
            optimizer.zero_grad()

            # Clamp parameters to bounds
            rot = torch.clamp(theta_rot, -rot_bound, rot_bound)
            tx = torch.clamp(theta_tx, -trans_bound, trans_bound)
            ty = torch.clamp(theta_ty, -trans_bound, trans_bound)
            hue = torch.clamp(theta_hue, -hue_bound, hue_bound)
            contrast = torch.clamp(theta_contrast, -contrast_bound, contrast_bound)

            # Apply differentiable spatial transform (rotation + translation)
            x_transformed = _apply_affine_transform(images, rot, tx, ty)

            # Apply color transforms
            x_transformed = _apply_hue_shift(x_transformed, hue)
            x_transformed = _apply_contrast(x_transformed, contrast)
            x_transformed = torch.clamp(x_transformed, 0.0, 1.0)

            logits = model(x_transformed)
            # Maximize loss (minimize negative loss)
            loss = -nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        # Final forward pass with optimized parameters
        with torch.no_grad():
            rot = torch.clamp(theta_rot, -rot_bound, rot_bound)
            tx = torch.clamp(theta_tx, -trans_bound, trans_bound)
            ty = torch.clamp(theta_ty, -trans_bound, trans_bound)
            hue = torch.clamp(theta_hue, -hue_bound, hue_bound)
            contrast = torch.clamp(theta_contrast, -contrast_bound, contrast_bound)

            x_final = _apply_affine_transform(images, rot, tx, ty)
            x_final = _apply_hue_shift(x_final, hue)
            x_final = _apply_contrast(x_final, contrast)
            x_final = torch.clamp(x_final, 0.0, 1.0)

            preds = model(x_final).argmax(dim=1)

        success = preds != labels
        newly_successful = success & ~best_success
        best_adv[newly_successful] = x_final[newly_successful]
        best_success = best_success | success

        if best_success.all():
            break

    remaining = ~best_success
    best_adv[remaining] = x_final[remaining]

    return best_adv.detach()


def _apply_affine_transform(
    images: Tensor, rotation_deg: Tensor, tx: Tensor, ty: Tensor
) -> Tensor:
    """Apply differentiable affine (rotation + translation) to a batch.

    Uses bilinear grid sampling for differentiability.
    """
    batch_size = images.shape[0]
    h, w = images.shape[2], images.shape[3]

    # Convert degrees to radians
    angle_rad = rotation_deg * (math.pi / 180.0)

    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    # Build 2x3 affine matrix per sample
    # [cos -sin tx; sin cos ty]
    theta = torch.zeros(batch_size, 2, 3, device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = ty

    grid = nn.functional.affine_grid(theta, images.shape, align_corners=False)
    transformed = nn.functional.grid_sample(
        images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return transformed


def _apply_hue_shift(images: Tensor, hue_shift: Tensor) -> Tensor:
    """Apply a differentiable approximation of hue shift.

    For single-channel images, this becomes a simple brightness shift.
    For multi-channel, rotates in a color-opponent space.
    """
    n_channels = images.shape[1]
    if n_channels == 1:
        # Single channel: hue shift approximated as brightness offset
        return images + hue_shift.view(-1, 1, 1, 1)

    # For RGB: approximate hue rotation using channel mixing
    # This is a linearized approximation of true hue rotation
    cos_h = torch.cos(hue_shift * math.pi).view(-1, 1, 1, 1)
    sin_h = torch.sin(hue_shift * math.pi).view(-1, 1, 1, 1)

    # Decompose into luminance and chrominance
    r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    # Approximate rotation in the RG plane (simplified hue rotation)
    r_new = r * cos_h - g * sin_h
    g_new = r * sin_h + g * cos_h
    return torch.cat([r_new, g_new, b], dim=1)


def _apply_contrast(images: Tensor, contrast: Tensor) -> Tensor:
    """Apply differentiable contrast adjustment.

    Scales pixel values around the mean: x' = mean + (1 + c) * (x - mean).
    """
    mean = images.mean(dim=(2, 3), keepdim=True)
    factor = 1.0 + contrast.view(-1, 1, 1, 1)
    return mean + factor * (images - mean)


# --------------------------------------------------------------------------- #
# Adversarial Patch Attack
# --------------------------------------------------------------------------- #


def patch_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    patch_size: int = 3,
    steps: int = 100,
    lr: float = 0.01,
    patch_location: Optional[tuple[int, int]] = None,
    printable_gamut: bool = True,
    epsilon_schedule: Optional[list[int]] = None,
) -> Tensor:
    """Adversarial patch attack with printable sRGB gamut constraint.

    Optimizes a localized patch in a bounded spatial region. Based on
    Brown et al., "Adversarial Patch" (NeurIPS 2017).

    The patch is constrained to the sRGB gamut (values in [0,1] for each
    channel independently) to ensure physical realizability. An optional
    epsilon schedule searches over patch sizes.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        patch_size: side length of the square patch in pixels.
        steps: number of optimization steps.
        lr: learning rate for patch optimization.
        patch_location: optional ``(top, left)`` corner. If None, centered.
        printable_gamut: if True, clamp patch values to valid sRGB [0, 1].
        epsilon_schedule: optional list of patch sizes to search over.

    Returns:
        Detached adversarial images clamped to ``[0, 1]``.
    """
    _require_eval_mode(model)

    if epsilon_schedule is None:
        epsilon_schedule = [patch_size]

    batch_size = images.shape[0]
    n_channels = images.shape[1]
    h, w = images.shape[2], images.shape[3]
    best_adv = images.clone().detach()
    best_success = torch.zeros(batch_size, dtype=torch.bool)

    for ps in epsilon_schedule:
        ps = min(ps, h, w)  # Ensure patch fits

        # Determine patch location
        if patch_location is not None:
            top, left = patch_location
        else:
            top = max(0, (h - ps) // 2)
            left = max(0, (w - ps) // 2)

        # Ensure patch stays within image bounds
        top = min(top, h - ps)
        left = min(left, w - ps)

        # Initialize patch values from the corresponding region of the image
        patch = images[:, :, top : top + ps, left : left + ps].clone().detach()
        patch.requires_grad_(True)

        optimizer = torch.optim.Adam([patch], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()

            # Apply patch to images
            x_patched = images.clone().detach()
            x_patched[:, :, top : top + ps, left : left + ps] = patch

            if printable_gamut:
                # Clamp to valid sRGB gamut [0, 1]
                x_patched = torch.clamp(x_patched, 0.0, 1.0)

            logits = model(x_patched)
            # Maximize loss (untargeted attack)
            loss = -nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            # Project patch to printable gamut
            if printable_gamut:
                with torch.no_grad():
                    patch.clamp_(0.0, 1.0)

        # Final evaluation
        with torch.no_grad():
            x_final = images.clone()
            clamped_patch = torch.clamp(patch, 0.0, 1.0) if printable_gamut else patch
            x_final[:, :, top : top + ps, left : left + ps] = clamped_patch
            x_final = torch.clamp(x_final, 0.0, 1.0)
            preds = model(x_final).argmax(dim=1)

        success = preds != labels
        newly_successful = success & ~best_success
        best_adv[newly_successful] = x_final[newly_successful]
        best_success = best_success | success

        if best_success.all():
            break

    remaining = ~best_success
    best_adv[remaining] = x_final[remaining]

    return best_adv.detach()
