"""Black-box adversarial attacks that require only query access to the model.

These attacks do NOT use gradients; they rely only on forward-pass queries
(hard labels or logits) to craft adversarial examples. Each attack tracks the
number of model queries consumed, enabling fair comparison under a fixed
query budget.

Attacks implemented:

* **SimBA** -- Guo et al., "Simple Black-box Adversarial Attacks" (ICML 2019).
  Random direction search in pixel or DCT space.
* **Square Attack** -- Andriushchenko et al., "Square Attack: A Query-Efficient
  Black-box Adversarial Attack via Random Search" (ECCV 2020). Square-shaped
  random perturbations with a p-schedule for patch size.
* **HopSkipJump** -- Chen et al., "HopSkipJumpAttack: A Query-Efficient
  Decision-Based Attack" (IEEE S&P 2020). Binary search along the decision
  boundary followed by gradient estimation from sign queries.
* **Boundary Attack** -- Brendel et al., "Decision-Based Adversarial Attacks:
  Reliable Attacks Against Black-Box Machine Learning Models" (ICLR 2018).
  Start from an adversarial point and walk along the decision boundary,
  reducing distance to the original input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode


def simba_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    query_budget: int = 1000,
    epsilon: float = 0.2,
    step_size: float = 0.02,
) -> tuple[Tensor, Tensor]:
    """SimBA: Simple Black-box Adversarial Attack (Guo et al., ICML 2019).

    Iteratively perturbs along random coordinate directions, accepting steps
    that reduce the predicted probability of the true class. Uses pixel-space
    random directions for simplicity.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        query_budget: maximum number of forward passes per example.
        epsilon: L-inf perturbation budget.
        step_size: magnitude of each coordinate perturbation.

    Returns:
        Tuple of ``(adversarial_images, queries_used)`` where queries_used has
        shape ``(N,)`` recording the number of queries consumed per example.
    """
    _require_eval_mode(model)

    batch_size = images.shape[0]
    x_adv = images.clone().detach()
    queries_used = torch.zeros(batch_size, dtype=torch.long)

    # Track which examples are still being attacked (not yet misclassified)
    active = torch.ones(batch_size, dtype=torch.bool)

    # Get initial probabilities for the true class
    with torch.no_grad():
        logits = model(x_adv)
        probs = torch.softmax(logits, dim=1)
        true_probs = probs[torch.arange(batch_size), labels]
    queries_used += 1

    dims = images[0].numel()

    for _ in range(query_budget - 1):
        if not active.any():
            break

        # Generate random direction for each active example
        direction = torch.zeros_like(x_adv)
        for idx in range(batch_size):
            if not active[idx]:
                continue
            # Random coordinate direction
            flat_dir = torch.zeros(dims, device=images.device)
            coord = torch.randint(0, dims, (1,)).item()
            flat_dir[coord] = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            direction[idx] = flat_dir.view(images.shape[1:])

        # Try perturbation in the chosen direction
        x_new = x_adv + step_size * direction
        # Enforce L-inf constraint relative to original
        delta = x_new - images
        delta = torch.clamp(delta, -epsilon, epsilon)
        x_new = torch.clamp(images + delta, 0.0, 1.0)

        with torch.no_grad():
            logits_new = model(x_new)
            probs_new = torch.softmax(logits_new, dim=1)
            true_probs_new = probs_new[torch.arange(batch_size), labels]
            preds_new = logits_new.argmax(dim=1)

        # Update queries for active examples
        queries_used[active] += 1

        # Accept if true-class probability decreased
        improved = true_probs_new < true_probs
        accept = active & improved
        x_adv[accept] = x_new[accept]
        true_probs[accept] = true_probs_new[accept]

        # Deactivate successfully misclassified examples
        misclassified = preds_new != labels
        active = active & ~misclassified

    return x_adv.detach(), queries_used


def square_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    query_budget: int = 1000,
    epsilon: float = 0.05,
    p_init: float = 0.8,
) -> tuple[Tensor, Tensor]:
    """Square Attack (Andriushchenko et al., ECCV 2020).

    A score-based black-box attack that perturbs random square-shaped regions
    of the image. The patch size decreases over iterations following a schedule,
    allowing coarse-to-fine refinement.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        query_budget: maximum number of forward passes per example.
        epsilon: L-inf perturbation budget.
        p_init: initial fraction of image covered by the square patch.

    Returns:
        Tuple of ``(adversarial_images, queries_used)`` where queries_used has
        shape ``(N,)`` recording the number of queries consumed per example.
    """
    _require_eval_mode(model)

    batch_size = images.shape[0]
    n_channels, height, width = images.shape[1], images.shape[2], images.shape[3]
    x_adv = images.clone().detach()
    queries_used = torch.zeros(batch_size, dtype=torch.long)
    active = torch.ones(batch_size, dtype=torch.bool)

    # Initialize with random perturbation within epsilon
    init_delta = (2.0 * torch.randint(0, 2, images.shape).float() - 1.0) * epsilon
    x_adv = torch.clamp(images + init_delta, 0.0, 1.0)

    with torch.no_grad():
        logits = model(x_adv)
        margin = _get_margin(logits, labels)
        preds = logits.argmax(dim=1)
    queries_used += 1
    active = active & (preds == labels)

    for step in range(1, query_budget):
        if not active.any():
            break

        # Patch size schedule: decreases over iterations
        p = p_init * (1.0 - step / query_budget)
        p = max(p, 0.01)
        patch_h = max(1, int(p * height))
        patch_w = max(1, int(p * width))

        # Random patch location for each example
        x_new = x_adv.clone()
        for idx in range(batch_size):
            if not active[idx]:
                continue
            top = torch.randint(0, max(1, height - patch_h + 1), (1,)).item()
            left = torch.randint(0, max(1, width - patch_w + 1), (1,)).item()

            # Random sign perturbation within the patch
            patch_val = (
                2.0 * torch.randint(0, 2, (n_channels, patch_h, patch_w)).float()
                - 1.0
            ) * epsilon
            x_new[idx, :, top : top + patch_h, left : left + patch_w] = torch.clamp(
                images[idx, :, top : top + patch_h, left : left + patch_w] + patch_val,
                0.0,
                1.0,
            )

        # Ensure L-inf constraint
        delta = x_new - images
        delta = torch.clamp(delta, -epsilon, epsilon)
        x_new = torch.clamp(images + delta, 0.0, 1.0)

        with torch.no_grad():
            logits_new = model(x_new)
            margin_new = _get_margin(logits_new, labels)
            preds_new = logits_new.argmax(dim=1)

        queries_used[active] += 1

        # Accept if margin decreased (closer to misclassification)
        improved = margin_new < margin
        accept = active & improved
        x_adv[accept] = x_new[accept]
        margin[accept] = margin_new[accept]

        # Deactivate successfully misclassified examples
        misclassified = preds_new != labels
        newly_done = active & misclassified
        x_adv[newly_done] = x_new[newly_done]
        active = active & ~misclassified

    return x_adv.detach(), queries_used


def hop_skip_jump(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    query_budget: int = 1000,
    num_gradient_estimates: int = 25,
    initial_num_evals: int = 100,
    stepsize_search: str = "geometric",
) -> tuple[Tensor, Tensor]:
    """HopSkipJump Attack (Chen et al., IEEE S&P 2020).

    A decision-based attack that estimates gradients at the decision boundary
    using binary search, then takes steps that minimize distance to the
    original while maintaining adversarial status.

    The attack proceeds in three stages per iteration:
    1. Binary search to find a point on the decision boundary.
    2. Gradient estimation via Monte Carlo sampling of random directions.
    3. Geometric step along the estimated gradient, followed by projection.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        query_budget: maximum number of forward passes per example.
        num_gradient_estimates: number of random directions for gradient
            estimation at the boundary.
        initial_num_evals: number of binary search steps per iteration.
        stepsize_search: step size selection strategy (``"geometric"``).

    Returns:
        Tuple of ``(adversarial_images, queries_used)`` where queries_used has
        shape ``(N,)`` recording the number of queries consumed per example.
    """
    _require_eval_mode(model)

    batch_size = images.shape[0]
    x_adv = images.clone().detach()
    queries_used = torch.zeros(batch_size, dtype=torch.long)

    # Initialize: find an adversarial starting point via random perturbation
    for idx in range(batch_size):
        found = False
        for attempt in range(min(25, query_budget)):
            # Random initialization: uniform noise
            x_rand = torch.rand_like(images[idx : idx + 1])
            with torch.no_grad():
                pred = model(x_rand).argmax(dim=1)
            queries_used[idx] += 1
            if pred.item() != labels[idx].item():
                x_adv[idx] = x_rand[0]
                found = True
                break
        if not found:
            # If no adversarial found, add large perturbation
            x_adv[idx] = torch.clamp(
                images[idx] + 0.5 * torch.randn_like(images[idx]), 0.0, 1.0
            )

    # Iterative refinement
    for idx in range(batch_size):
        remaining = query_budget - queries_used[idx].item()
        if remaining <= 0:
            continue

        x_orig = images[idx : idx + 1]
        x_curr = x_adv[idx : idx + 1]

        max_iters = remaining // max(num_gradient_estimates + 2, 1)
        for iteration in range(min(max_iters, 50)):
            if queries_used[idx].item() >= query_budget:
                break

            # Step 1: Binary search to find boundary point
            x_boundary = _binary_search_boundary(
                model,
                x_orig,
                x_curr,
                labels[idx : idx + 1],
                max_steps=min(10, query_budget - queries_used[idx].item()),
            )
            steps_used = min(10, query_budget - queries_used[idx].item())
            queries_used[idx] += steps_used

            if queries_used[idx].item() >= query_budget:
                x_adv[idx] = x_boundary[0]
                break

            # Step 2: Gradient direction estimation at the boundary
            n_evals = min(
                num_gradient_estimates, query_budget - queries_used[idx].item()
            )
            if n_evals <= 0:
                x_adv[idx] = x_boundary[0]
                break

            grad_dir = _estimate_gradient_direction(
                model, x_boundary, labels[idx : idx + 1], n_evals=n_evals
            )
            queries_used[idx] += n_evals

            # Step 3: Step along estimated gradient, then project
            dist = (x_boundary - x_orig).view(1, -1).norm(p=2, dim=1, keepdim=True)
            step_size = dist.view(1, 1, 1, 1) * 0.1 / max(1, iteration + 1)
            x_new = x_boundary + step_size * grad_dir
            x_new = torch.clamp(x_new, 0.0, 1.0)

            # Verify it is still adversarial
            if queries_used[idx].item() < query_budget:
                with torch.no_grad():
                    pred = model(x_new).argmax(dim=1)
                queries_used[idx] += 1
                if pred.item() != labels[idx].item():
                    x_curr = x_new
                else:
                    x_curr = x_boundary

            x_adv[idx] = x_curr[0]

    return x_adv.detach(), queries_used


def boundary_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    query_budget: int = 1000,
    step_size: float = 0.01,
    orthogonal_step_size: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Boundary Attack (Brendel et al., ICLR 2018).

    A decision-based attack that starts from an adversarial point and iteratively
    moves along the decision boundary, reducing the distance to the original
    image. Each step consists of:
    1. An orthogonal step (random direction perpendicular to the line between
       the current adversarial and the original) to stay near the boundary.
    2. A step toward the original to reduce the perturbation magnitude.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        query_budget: maximum number of forward passes per example.
        step_size: step size toward the original image.
        orthogonal_step_size: step size in the orthogonal direction.

    Returns:
        Tuple of ``(adversarial_images, queries_used)`` where queries_used has
        shape ``(N,)`` recording the number of queries consumed per example.
    """
    _require_eval_mode(model)

    batch_size = images.shape[0]
    x_adv = images.clone().detach()
    queries_used = torch.zeros(batch_size, dtype=torch.long)

    # Initialize: find an adversarial starting point
    for idx in range(batch_size):
        found = False
        for attempt in range(min(25, query_budget)):
            x_rand = torch.rand_like(images[idx : idx + 1])
            with torch.no_grad():
                pred = model(x_rand).argmax(dim=1)
            queries_used[idx] += 1
            if pred.item() != labels[idx].item():
                x_adv[idx] = x_rand[0]
                found = True
                break
        if not found:
            # Start from a point far from the original
            x_adv[idx] = torch.clamp(1.0 - images[idx], 0.0, 1.0)

    # Iterative refinement along the boundary
    for idx in range(batch_size):
        x_orig = images[idx : idx + 1]
        x_curr = x_adv[idx : idx + 1]

        while queries_used[idx].item() < query_budget:
            # Direction from current adversarial toward original
            direction_to_orig = x_orig - x_curr
            dist = direction_to_orig.view(1, -1).norm(p=2, dim=1, keepdim=True)
            if dist.item() < 1e-8:
                break
            direction_to_orig_unit = direction_to_orig / dist.view(1, 1, 1, 1)

            # Orthogonal random step
            noise = torch.randn_like(x_curr)
            # Project out the component along direction_to_orig
            noise_flat = noise.view(1, -1)
            dir_flat = direction_to_orig_unit.view(1, -1)
            proj = (noise_flat * dir_flat).sum(dim=1, keepdim=True)
            noise_flat = noise_flat - proj * dir_flat
            noise_norm = noise_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            noise_flat = noise_flat / noise_norm
            orthogonal_perturbation = noise_flat.view_as(x_curr) * orthogonal_step_size

            # Step toward original
            toward_orig_perturbation = direction_to_orig_unit * step_size

            # Candidate: orthogonal step + toward-original step
            x_candidate = x_curr + orthogonal_perturbation + toward_orig_perturbation
            x_candidate = torch.clamp(x_candidate, 0.0, 1.0)

            # Check if candidate is still adversarial
            with torch.no_grad():
                pred = model(x_candidate).argmax(dim=1)
            queries_used[idx] += 1

            if pred.item() != labels[idx].item():
                x_curr = x_candidate
            else:
                # Try just the toward-original step without orthogonal component
                x_candidate = x_curr + toward_orig_perturbation * 0.5
                x_candidate = torch.clamp(x_candidate, 0.0, 1.0)
                if queries_used[idx].item() < query_budget:
                    with torch.no_grad():
                        pred = model(x_candidate).argmax(dim=1)
                    queries_used[idx] += 1
                    if pred.item() != labels[idx].item():
                        x_curr = x_candidate

        x_adv[idx] = x_curr[0]

    return x_adv.detach(), queries_used


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def _get_margin(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute the margin: true class logit minus best other class logit.

    A positive margin means the model is still correct; a negative margin
    means the attack succeeded.
    """
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
    true_logits = (logits * one_hot).sum(dim=1)
    other_logits = logits - one_hot * 1e9
    best_other = other_logits.max(dim=1)[0]
    return true_logits - best_other


def _binary_search_boundary(
    model: nn.Module,
    x_orig: Tensor,
    x_adv: Tensor,
    labels: Tensor,
    max_steps: int = 10,
) -> Tensor:
    """Binary search between original and adversarial to find boundary point.

    Returns a point close to the decision boundary that is adversarial.
    """
    low = torch.zeros(1, device=x_orig.device)
    high = torch.ones(1, device=x_orig.device)

    x_boundary = x_adv.clone()

    for _ in range(max_steps):
        mid = (low + high) / 2.0
        x_mid = (1.0 - mid) * x_orig + mid * x_adv
        x_mid = torch.clamp(x_mid, 0.0, 1.0)
        with torch.no_grad():
            pred = model(x_mid).argmax(dim=1)
        if pred.item() != labels.item():
            # Still adversarial, move toward original
            high = mid
            x_boundary = x_mid
        else:
            # Not adversarial, move toward adversarial
            low = mid

    return x_boundary


def _estimate_gradient_direction(
    model: nn.Module,
    x_boundary: Tensor,
    labels: Tensor,
    n_evals: int = 25,
) -> Tensor:
    """Estimate gradient direction at boundary via Monte Carlo sign queries.

    Samples random directions and checks which side of the boundary they land
    on, accumulating a gradient estimate from the sign information.
    """
    grad_estimate = torch.zeros_like(x_boundary)

    for _ in range(n_evals):
        noise = torch.randn_like(x_boundary)
        noise_norm = noise.view(1, -1).norm(p=2, dim=1).clamp_min(1e-12)
        noise = noise / noise_norm.view(1, 1, 1, 1)

        # Small step in noise direction
        delta = 0.01
        x_probe = torch.clamp(x_boundary + delta * noise, 0.0, 1.0)
        with torch.no_grad():
            pred = model(x_probe).argmax(dim=1)

        # If probe is adversarial, noise points away from the correct region
        if pred.item() != labels.item():
            grad_estimate += noise
        else:
            grad_estimate -= noise

    # Normalize the gradient estimate
    grad_norm = grad_estimate.view(1, -1).norm(p=2, dim=1).clamp_min(1e-12)
    grad_estimate = grad_estimate / grad_norm.view(1, 1, 1, 1)

    return grad_estimate
