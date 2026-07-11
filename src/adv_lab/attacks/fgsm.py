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
