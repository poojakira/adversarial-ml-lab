"""C&W L2 tests (3)."""

from __future__ import annotations

import torch

from adv_lab.attacks.cw import cw_l2_attack


def _true_label_confidence(model, x, y) -> torch.Tensor:
    with torch.no_grad():
        probs = model(x).softmax(dim=1)
    return probs.gather(1, y.unsqueeze(1)).squeeze(1)


def test_cw_reduces_confidence_on_true_label(correct_batch):
    model, x, y = correct_batch
    before = _true_label_confidence(model, x, y).mean().item()
    x_adv = cw_l2_attack(model, x, y, c=1.0, kappa=0.0, steps=200, lr=0.01)
    after = _true_label_confidence(model, x_adv, y).mean().item()
    # The whole point of C&W is to suppress the true-class score.
    assert after < before


def test_cw_output_in_valid_range(correct_batch):
    model, x, y = correct_batch
    x_adv = cw_l2_attack(model, x, y, c=1.0, kappa=0.0, steps=100, lr=0.01)
    assert x_adv.min().item() >= 0.0
    assert x_adv.max().item() <= 1.0


def test_cw_with_kappa_zero(correct_batch):
    model, x, y = correct_batch
    x_adv = cw_l2_attack(model, x, y, c=1.0, kappa=0.0, steps=100, lr=0.01)
    # kappa=0 is the default confidence margin; result must be well-formed.
    assert x_adv.shape == x.shape
    assert torch.isfinite(x_adv).all()
