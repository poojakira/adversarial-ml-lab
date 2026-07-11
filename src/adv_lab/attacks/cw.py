"""Carlini & Wagner L2 attack.

Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks"
(IEEE S&P 2017). The optimization-based attack that broke defensive
distillation and set the standard for honest robustness evaluation.

Two ideas do the heavy lifting:

1. Change of variables. Instead of optimizing ``delta`` under a hard box
   constraint, optimize an unconstrained ``w`` and map through tanh::

       x_adv = 0.5 * (tanh(w) + 1)

   which lives in ``(0, 1)`` by construction, so no clipping is needed and the
   optimizer sees a smooth landscape.

2. A margin ("f") objective rather than cross-entropy::

       f(x) = max( max_{i != t} Z(x)_i  -  Z(x)_t ,  -kappa )

   where ``Z`` are the logits, ``t`` the true label, and ``kappa`` a confidence
   margin. Minimizing ``f`` pushes the true-class logit below the best other
   class by at least ``kappa``. The total objective balances a small L2
   perturbation against a successful misclassification::

       minimize  ||x_adv - x||_2^2  +  c * f(x_adv)

This is an untargeted formulation (drive the input *away* from its true label).
It is the strongest attack in this lab -- it typically finds smaller
perturbations than PGD for the same success, which is exactly why it is the
gold standard for catching gradient masking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode

_TANH_EPS = 1e-6  # keeps atanh() inputs strictly inside (-1, 1)


def _to_tanh_space(x: Tensor) -> Tensor:
    """Map images in ``[0, 1]`` to the unconstrained tanh domain ``w``."""
    x = torch.clamp(x, 0.0, 1.0)
    # x = 0.5*(tanh(w)+1)  =>  w = atanh(2x - 1)
    scaled = (2.0 * x - 1.0) * (1.0 - _TANH_EPS)
    return torch.atanh(scaled)


def _from_tanh_space(w: Tensor) -> Tensor:
    """Map ``w`` back to image space ``[0, 1]``."""
    return 0.5 * (torch.tanh(w) + 1.0)


def cw_l2_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    c: float = 1e-4,
    kappa: float = 0.0,
    steps: int = 1000,
    lr: float = 0.01,
) -> Tensor:
    """Craft C&W L2 adversarial examples (untargeted).

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]``, shape ``(N, C, H, W)``.
        labels: ground-truth class indices, shape ``(N,)``.
        c: trade-off constant between perturbation size and the ``f`` margin.
            A fixed ``c`` is used here; binary search over ``c`` (as in the
            paper) is an optional refinement, not required for the basic attack.
        kappa: confidence margin. Larger ``kappa`` pushes for more confident
            misclassification at the cost of a larger perturbation.
        steps: number of Adam iterations.
        lr: Adam learning rate on ``w``.

    Returns:
        Detached adversarial images in ``[0, 1]``, same shape as input.
    """
    _require_eval_mode(model)

    x_orig = images.clone().detach()
    labels = labels.clone().detach()
    batch = x_orig.shape[0]
    num_classes_onehot = None

    # Optimize w, initialized so that x_adv == x_orig at step 0.
    w = _to_tanh_space(x_orig).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    for _ in range(steps):
        x_adv = _from_tanh_space(w)

        # L2 distortion per example (sum of squares over pixels).
        l2 = ((x_adv - x_orig) ** 2).view(batch, -1).sum(dim=1)

        logits = model(x_adv)
        if num_classes_onehot is None:
            num_classes_onehot = logits.shape[1]
        one_hot = nn.functional.one_hot(labels, num_classes=num_classes_onehot).to(
            logits.dtype
        )

        # Z(x)_t : logit of the true class.
        real = (one_hot * logits).sum(dim=1)
        # max_{i != t} Z(x)_i : best logit among the other classes.
        other = ((1.0 - one_hot) * logits - one_hot * 1e4).max(dim=1)[0]

        # Untargeted margin: push true-class logit below the best other class.
        f = torch.clamp(real - other, min=-kappa)

        loss = (l2 + c * f).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    x_adv = _from_tanh_space(w).detach()
    return torch.clamp(x_adv, 0.0, 1.0)
