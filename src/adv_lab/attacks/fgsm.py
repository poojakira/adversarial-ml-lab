"""Fast Gradient Sign Method (FGSM).

Goodfellow, Shlens & Szegedy, "Explaining and Harnessing Adversarial
Examples" (ICLR 2015). The single-step attack that started the arms race:

    x_adv = x + epsilon * sign(grad_x L(model(x), y))

FGSM is the weakest link in the FGSM < PGD < C&W ladder. It is a single
step in the sign direction of the loss gradient, so it neither follows the
loss surface's curvature nor projects iteratively. We keep it here mostly as
a baseline and a sanity check: if a "defended" model looks robust to PGD but
not FGSM, or reports suspiciously high FGSM robustness, that is a classic
gradient-masking smell (Athalye et al., 2018).
"""
# MITRE ATLAS: AML.T0043 - Craft Adversarial Data | AML.T0015 - Evade ML Model

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _require_eval_mode(model: nn.Module) -> None:
    """Attacks must run against a model in eval() mode.

    Dropout / BatchNorm in training mode make the gradient stochastic and the
    reported robustness meaningless. We fail loud rather than silently produce
    a garbage attack.
    """
    if model.training:
        raise ValueError(
            "model must be in eval() mode before attacking; call model.eval(). "
            "Attacking a model in train() mode gives stochastic, unreliable "
            "gradients (a common source of bogus robustness numbers)."
        )


def _validate_attack_inputs(
    images: Tensor,
    labels: Tensor,
    epsilon: float,
) -> None:
    """Validate the common (images, labels, epsilon) contract shared by all attacks.

    Fails loud on the mistakes that otherwise surface as cryptic autograd or
    broadcasting errors deep in the attack loop:

      * images that are not a 4D ``(N, C, H, W)`` float tensor,
      * a batch-size mismatch between images and labels,
      * labels that are not 1D integer class indices,
      * a negative or non-finite epsilon,
      * inputs outside the ``[0, 1]`` image range the projections assume.

    Raising here (rather than letting torch raise later) turns a garbage
    robustness number into an obvious, actionable error.
    """
    if not isinstance(images, Tensor):
        raise TypeError(f"images must be a torch.Tensor, got {type(images).__name__}")
    if not isinstance(labels, Tensor):
        raise TypeError(f"labels must be a torch.Tensor, got {type(labels).__name__}")

    if images.dim() != 4:
        raise ValueError(
            f"images must be a 4D (N, C, H, W) tensor, got shape {tuple(images.shape)}"
        )
    if not torch.is_floating_point(images):
        raise TypeError(
            f"images must be a floating-point tensor in [0, 1], got dtype {images.dtype}"
        )
    if labels.dim() != 1:
        raise ValueError(
            f"labels must be a 1D tensor of class indices, got shape {tuple(labels.shape)}"
        )
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"batch size mismatch: images has {images.shape[0]} samples "
            f"but labels has {labels.shape[0]}"
        )
    if images.shape[0] == 0:
        raise ValueError("cannot attack an empty batch (0 samples)")

    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    # NaN/inf epsilon would silently poison every perturbation.
    if epsilon != epsilon or epsilon == float("inf"):
        raise ValueError(f"epsilon must be finite, got {epsilon}")

    # The L-inf/L2 projections and the tanh change-of-variables all assume the
    # clean image already lives in [0, 1]. A small tolerance absorbs float noise.
    if images.numel() > 0:
        lo = float(images.min())
        hi = float(images.max())
        if lo < -1e-4 or hi > 1.0 + 1e-4:
            raise ValueError(
                f"images must be in the [0, 1] range, got [{lo:.4f}, {hi:.4f}]. "
                "Normalize/rescale to [0, 1] before attacking."
            )


def fgsm_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.03,
) -> Tensor:
    """Craft FGSM adversarial examples for a batch.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        epsilon: L-inf perturbation budget. ``0.0`` returns the input unchanged.

    Returns:
        Adversarial images, detached, clamped to ``[0, 1]``, same shape as input.
    """
    _require_eval_mode(model)
    _validate_attack_inputs(images, labels, epsilon)

    if epsilon == 0.0:
        # No budget -> no perturbation. Return a detached clone so callers can
        # treat the output uniformly regardless of epsilon.
        return images.clone().detach()

    x = images.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, labels)

    # Fresh gradient wrt the input only.
    grad = torch.autograd.grad(loss, x)[0]

    x_adv = x + epsilon * grad.sign()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def generate(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.03,
) -> Tensor:
    """Backward-compatible FGSM entry point used by older callers."""
    return fgsm_attack(model, images, labels, epsilon=epsilon)


def batch_fgsm(
    model: nn.Module,
    dataloader,
    epsilon: float = 0.03,
) -> tuple[float, list[Tensor]]:
    """Run FGSM over a dataloader and measure how often it flips predictions.

    "Attack success" is measured only on inputs the model originally classified
    correctly: fooling an already-wrong prediction is not a success.

    Args:
        model: classifier in ``eval()`` mode.
        dataloader: yields ``(images, labels)`` batches.
        epsilon: L-inf budget.

    Returns:
        Tuple of ``(attack_success_rate, adversarial_batches)`` where the rate is
        in ``[0, 1]`` and the list holds one adversarial tensor per batch.
    """
    _require_eval_mode(model)

    adv_examples: list[Tensor] = []
    n_correct_clean = 0
    n_flipped = 0

    for images, labels in dataloader:
        with torch.no_grad():
            clean_pred = model(images).argmax(dim=1)
        correct_mask = clean_pred == labels

        x_adv = fgsm_attack(model, images, labels, epsilon=epsilon)
        adv_examples.append(x_adv)

        with torch.no_grad():
            adv_pred = model(x_adv).argmax(dim=1)

        # Success = was correct on the clean input, now wrong on the adv input.
        n_correct_clean += int(correct_mask.sum().item())
        n_flipped += int((correct_mask & (adv_pred != labels)).sum().item())

    success_rate = (n_flipped / n_correct_clean) if n_correct_clean > 0 else 0.0
    return success_rate, adv_examples
