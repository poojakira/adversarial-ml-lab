"""FGSM tests (4)."""

from __future__ import annotations

import pytest
import torch

from adv_lab.attacks.fgsm import fgsm_attack


def test_fgsm_perturbs_input(correct_batch):
    model, x, y = correct_batch
    x_adv = fgsm_attack(model, x, y, epsilon=0.03)
    # A non-zero budget must actually move the input.
    assert not torch.allclose(x_adv, x)


def test_fgsm_stays_in_valid_range(correct_batch):
    model, x, y = correct_batch
    x_adv = fgsm_attack(model, x, y, epsilon=0.1)
    assert x_adv.min().item() >= 0.0
    assert x_adv.max().item() <= 1.0


def test_fgsm_epsilon_zero_no_change(correct_batch):
    model, x, y = correct_batch
    x_adv = fgsm_attack(model, x, y, epsilon=0.0)
    # Zero budget => identical output.
    assert torch.allclose(x_adv, x)


def test_fgsm_raises_if_model_in_train_mode(correct_batch):
    model, x, y = correct_batch
    model.train()
    try:
        with pytest.raises(ValueError):
            fgsm_attack(model, x, y, epsilon=0.03)
    finally:
        model.eval()
