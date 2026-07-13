"""Universal Adversarial Perturbations (Tier 5 -- Infrastructure Integrity).

UAPs are the most dangerous class of adversarial attack for mass deployment
because they require zero per-sample computation at attack time -- one delta,
infinite victims.

Implements:
    Moosavi-Dezfooli et al., "Universal Adversarial Perturbations" (CVPR 2017).
    Mopuri et al., "Fast Feature Fool: A data independent approach to universal
    adversarial perturbations" (BMVC 2017).

A universal perturbation must achieve >= 80% fooling rate across the held-out
validation set using a single fixed delta. Cross-architecture transfer rates
quantify real-world deployment risk.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torch import Tensor


def _require_eval_mode(model: nn.Module) -> None:
    """Attacks must run against a model in eval() mode."""
    if model.training:
        raise ValueError(
            "model must be in eval() mode before attacking; call model.eval(). "
            "Attacking a model in train() mode gives stochastic, unreliable "
            "gradients (a common source of bogus robustness numbers)."
        )


def _project_perturbation(
    delta: Tensor,
    epsilon: float,
    norm: str = "linf",
) -> Tensor:
    """Project perturbation onto the epsilon-ball of the specified norm.

    Args:
        delta: Perturbation tensor.
        epsilon: Maximum perturbation magnitude.
        norm: Norm type -- "linf" or "l2".

    Returns:
        Projected perturbation tensor.
    """
    if norm == "linf":
        return delta.clamp(-epsilon, epsilon)
    elif norm == "l2":
        flat = delta.view(delta.shape[0], -1) if delta.dim() > 3 else delta.view(-1)
        norms = flat.norm(p=2, dim=-1, keepdim=True)
        # Only scale down if norm exceeds epsilon
        scale = torch.clamp(norms / epsilon, min=1.0)
        if delta.dim() > 3:
            return delta / scale.view(-1, 1, 1, 1)
        else:
            return delta * (epsilon / max(float(norms.item()), epsilon))
    else:
        raise ValueError(f"Unsupported norm: {norm}. Use 'linf' or 'l2'.")


def _deepfool_single(
    model: nn.Module,
    image: Tensor,
    max_iter: int = 50,
    overshoot: float = 0.02,
) -> Tensor:
    """Compute minimal perturbation to fool the model on a single image.

    Implements DeepFool (Moosavi-Dezfooli et al., CVPR 2016) for a single
    sample. Used as the inner loop of the UAP algorithm.

    Args:
        model: Classifier in eval() mode.
        image: Single input image, shape (1, C, H, W) or (C, H, W).
        max_iter: Maximum iterations.
        overshoot: Overshoot factor to ensure crossing the decision boundary.

    Returns:
        Minimal perturbation tensor of same spatial shape as input.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    x = image.clone().detach().requires_grad_(True)
    with torch.no_grad():
        logits = model(x)
        pred_label = logits.argmax(dim=1).item()
        num_classes = logits.shape[1]

    perturbation = torch.zeros_like(image)

    for _ in range(max_iter):
        x_var = (image + perturbation).clone().detach().requires_grad_(True)
        logits = model(x_var)
        current_pred = logits.argmax(dim=1).item()

        if current_pred != pred_label:
            break

        # Find closest decision boundary
        logits_orig = logits[0, pred_label]
        min_pert_norm = float("inf")
        best_pert = torch.zeros_like(image)

        for k in range(num_classes):
            if k == pred_label:
                continue

            # Gradient of logit difference
            model.zero_grad()
            if x_var.grad is not None:
                x_var.grad.zero_()

            diff = logits[0, k] - logits_orig
            grad_diff = torch.autograd.grad(diff, x_var, retain_graph=True)[0]

            # Minimal perturbation to cross boundary k
            f_k = float(diff.item())
            grad_norm = float(grad_diff.view(-1).norm(p=2).item())
            if grad_norm < 1e-10:
                continue

            pert_norm = abs(f_k) / grad_norm
            if pert_norm < min_pert_norm:
                min_pert_norm = pert_norm
                best_pert = (abs(f_k) + 1e-8) / (grad_norm**2) * grad_diff

        perturbation = perturbation + (1.0 + overshoot) * best_pert

    return perturbation.squeeze(0) if image.dim() == 3 else perturbation


def uap_generate(
    model: nn.Module,
    dataloader: Iterable[tuple[Tensor, Tensor]],
    *,
    epsilon: float = 0.1,
    norm: str = "linf",
    max_epochs: int = 10,
    target_fooling_rate: float = 0.8,
    max_iter_deepfool: int = 50,
    overshoot: float = 0.02,
    seed: int | None = None,
) -> Tensor:
    """Generate a Universal Adversarial Perturbation using iterative DeepFool.

    Implements the algorithm from Moosavi-Dezfooli et al. (CVPR 2017):
    iterate over training images, compute the minimal per-sample perturbation
    (via DeepFool), accumulate into a universal delta, and project back onto
    the norm ball. Repeat until the target fooling rate is achieved.

    Args:
        model: Classifier in eval() mode.
        dataloader: Yields (images, labels) batches for UAP computation.
        epsilon: Maximum perturbation magnitude (norm budget).
        norm: Constraint norm -- "linf" or "l2".
        max_epochs: Maximum passes over the dataset.
        target_fooling_rate: Stop when this fooling rate is achieved (>=0.8).
        max_iter_deepfool: Max iterations for inner DeepFool loop.
        overshoot: DeepFool overshoot parameter.
        seed: Optional random seed.

    Returns:
        Universal perturbation tensor, shape matching a single input (C, H, W).

    Raises:
        ValueError: If model is in training mode.
    """
    _require_eval_mode(model)

    if seed is not None:
        torch.manual_seed(seed)

    # Collect all data for fooling rate evaluation
    all_images: list[Tensor] = []
    all_labels: list[Tensor] = []
    for images, labels in dataloader:
        all_images.append(images)
        all_labels.append(labels)

    x_all = torch.cat(all_images, dim=0)
    y_all = torch.cat(all_labels, dim=0)
    n_total = x_all.shape[0]

    # Initialize universal perturbation
    input_shape = x_all.shape[1:]  # (C, H, W)
    delta = torch.zeros(input_shape)

    for epoch in range(max_epochs):
        # Shuffle order each epoch
        perm = torch.randperm(n_total)

        for idx in perm:
            xi = x_all[idx : idx + 1]  # (1, C, H, W)

            # Check if already fooled
            with torch.no_grad():
                pred_clean = model(xi).argmax(dim=1).item()
                pred_pert = model((xi + delta.unsqueeze(0)).clamp(0.0, 1.0)).argmax(
                    dim=1
                ).item()

            if pred_pert == pred_clean:
                # Not yet fooled -- compute minimal perturbation
                dr = _deepfool_single(
                    model,
                    (xi + delta.unsqueeze(0)).clamp(0.0, 1.0),
                    max_iter=max_iter_deepfool,
                    overshoot=overshoot,
                )
                # Accumulate and project
                delta = delta + dr.squeeze(0)
                delta = _project_perturbation(delta, epsilon, norm)

        # Evaluate fooling rate
        fooling_rate = evaluate_fooling_rate(model, delta, [(x_all, y_all)])
        if fooling_rate >= target_fooling_rate:
            break

    return delta.detach()


def fast_uap(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    epsilon: float = 0.1,
    norm: str = "linf",
    steps: int = 100,
    lr: float = 0.01,
    seed: int | None = None,
) -> Tensor:
    """Generate a data-independent Universal Adversarial Perturbation.

    Implements the Fast-UAP approach (Mopuri et al., BMVC 2017). Instead of
    iterating over training data, this method optimizes the perturbation to
    maximize feature activation norms, producing a UAP without any data access.

    This is particularly dangerous because an attacker needs zero knowledge of
    the training distribution -- only model access (white-box or query-based).

    Args:
        model: Classifier in eval() mode.
        input_shape: Shape of a single input (C, H, W).
        epsilon: Maximum perturbation magnitude.
        norm: Constraint norm -- "linf" or "l2".
        steps: Number of optimization steps.
        lr: Learning rate for perturbation optimization.
        seed: Optional random seed.

    Returns:
        Universal perturbation tensor, shape (C, H, W).

    Raises:
        ValueError: If model is in training mode.
    """
    _require_eval_mode(model)

    if seed is not None:
        torch.manual_seed(seed)

    # Initialize random perturbation
    delta = torch.zeros(input_shape, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        # Create a batch of random inputs with the perturbation applied
        batch_size = 8
        random_inputs = torch.rand(batch_size, *input_shape)
        perturbed = (random_inputs + delta.unsqueeze(0)).clamp(0.0, 1.0)

        # Maximize logit diversity / activation magnitude
        logits = model(perturbed)
        # Objective: maximize the variance of logits across classes
        # This creates maximum confusion in the classifier
        logit_var = logits.var(dim=1).mean()
        # Also maximize mean activation magnitude
        logit_magnitude = logits.abs().mean()

        # Combined objective (maximize both)
        loss = -(logit_var + logit_magnitude)
        loss.backward()
        optimizer.step()

        # Project back onto norm ball
        with torch.no_grad():
            delta.data = _project_perturbation(delta.data, epsilon, norm)

    return delta.detach()


def evaluate_fooling_rate(
    model: nn.Module,
    uap: Tensor,
    dataloader: Iterable[tuple[Tensor, Tensor]],
) -> float:
    """Compute the fooling rate of a UAP on a dataset.

    Fooling rate = fraction of correctly-classified samples whose prediction
    changes when the UAP is applied. Only samples that are correctly classified
    on clean input are counted (you cannot "fool" an already-wrong prediction).

    Target: >= 80% fooling rate for a production-grade UAP.

    Args:
        model: Classifier in eval() mode.
        uap: Universal perturbation tensor, shape (C, H, W).
        dataloader: Yields (images, labels) batches.

    Returns:
        Fooling rate in [0, 1].

    Raises:
        ValueError: If model is in training mode.
    """
    _require_eval_mode(model)

    n_correct = 0
    n_fooled = 0

    for images, labels in dataloader:
        with torch.no_grad():
            clean_preds = model(images).argmax(dim=1)
            correct_mask = clean_preds == labels

            perturbed = (images + uap.unsqueeze(0)).clamp(0.0, 1.0)
            adv_preds = model(perturbed).argmax(dim=1)

            # Fooled = was correct, now wrong
            fooled = correct_mask & (adv_preds != labels)
            n_correct += int(correct_mask.sum().item())
            n_fooled += int(fooled.sum().item())

    if n_correct == 0:
        return 0.0
    return n_fooled / n_correct


def cross_architecture_transfer(
    models: Sequence[nn.Module],
    uap: Tensor,
    dataloader: Iterable[tuple[Tensor, Tensor]],
) -> dict[str, float]:
    """Evaluate UAP transferability across multiple architectures.

    A highly transferable UAP is dangerous because it means the attacker only
    needs white-box access to ONE model to compromise ALL deployed models of
    similar type. This function quantifies that risk.

    Args:
        models: Sequence of classifiers (each in eval() mode) to test against.
        uap: Universal perturbation tensor, shape (C, H, W).
        dataloader: Yields (images, labels) batches.

    Returns:
        Dictionary mapping model class name (or index) to fooling rate.
        Example: {"ResNet": 0.85, "VGG": 0.72, "DenseNet": 0.68}

    Raises:
        ValueError: If any model is in training mode.
    """
    # Materialize dataloader into a list for repeated iteration
    data_list: list[tuple[Tensor, Tensor]] = []
    for batch in dataloader:
        data_list.append(batch)

    results: dict[str, float] = {}
    for idx, model in enumerate(models):
        _require_eval_mode(model)
        name = f"{model.__class__.__name__}_{idx}"
        rate = evaluate_fooling_rate(model, uap, data_list)
        results[name] = rate

    return results
