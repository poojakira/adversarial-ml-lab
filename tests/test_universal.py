"""Tests for Universal Adversarial Perturbations (UAP)."""

from __future__ import annotations

import pytest
import torch

from adv_lab.attacks.universal import (
    cross_architecture_transfer,
    evaluate_fooling_rate,
    fast_uap,
    uap_generate,
)


def test_uap_generate_returns_correct_shape(correct_batch):
    """uap_generate returns a perturbation matching input spatial dimensions."""
    model, x, y = correct_batch
    # Use minimal settings for test speed
    dataloader = [(x[:8], y[:8])]
    uap = uap_generate(
        model,
        dataloader,
        epsilon=0.1,
        norm="linf",
        max_epochs=1,
        max_iter_deepfool=3,
        seed=42,
    )
    assert uap.shape == x.shape[1:]  # (C, H, W)
    assert uap.dtype == torch.float32


def test_uap_generate_respects_linf_constraint(correct_batch):
    """uap_generate output is bounded by epsilon in L-inf norm."""
    model, x, y = correct_batch
    epsilon = 0.05
    dataloader = [(x[:4], y[:4])]
    uap = uap_generate(
        model,
        dataloader,
        epsilon=epsilon,
        norm="linf",
        max_epochs=1,
        max_iter_deepfool=3,
        seed=42,
    )
    assert uap.abs().max().item() <= epsilon + 1e-6


def test_uap_generate_raises_on_train_mode(correct_batch):
    """uap_generate raises ValueError if model is in train mode."""
    model, x, y = correct_batch
    model.train()
    try:
        with pytest.raises(ValueError):
            uap_generate(model, [(x[:4], y[:4])], max_epochs=1)
    finally:
        model.eval()


def test_fast_uap_returns_correct_shape(correct_batch):
    """fast_uap returns perturbation of specified input_shape."""
    model, x, y = correct_batch
    input_shape = x.shape[1:]  # (C, H, W)
    uap = fast_uap(
        model,
        input_shape,
        epsilon=0.1,
        norm="linf",
        steps=5,
        lr=0.01,
        seed=42,
    )
    assert uap.shape == input_shape
    assert uap.dtype == torch.float32


def test_fast_uap_respects_linf_constraint(correct_batch):
    """fast_uap output is bounded by epsilon in L-inf norm."""
    model, x, y = correct_batch
    epsilon = 0.08
    uap = fast_uap(
        model,
        x.shape[1:],
        epsilon=epsilon,
        norm="linf",
        steps=5,
        seed=42,
    )
    assert uap.abs().max().item() <= epsilon + 1e-6


def test_fast_uap_raises_on_train_mode(correct_batch):
    """fast_uap raises ValueError if model is in train mode."""
    model, x, y = correct_batch
    model.train()
    try:
        with pytest.raises(ValueError):
            fast_uap(model, x.shape[1:], steps=1)
    finally:
        model.eval()


def test_evaluate_fooling_rate_returns_float(correct_batch):
    """evaluate_fooling_rate returns a float in [0, 1]."""
    model, x, y = correct_batch
    # Random perturbation (may or may not fool)
    uap = torch.randn(x.shape[1:]) * 0.1
    dataloader = [(x, y)]
    rate = evaluate_fooling_rate(model, uap, dataloader)
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0


def test_evaluate_fooling_rate_zero_perturbation(correct_batch):
    """Zero perturbation should give zero fooling rate."""
    model, x, y = correct_batch
    uap = torch.zeros(x.shape[1:])
    dataloader = [(x, y)]
    rate = evaluate_fooling_rate(model, uap, dataloader)
    assert rate == 0.0


def test_evaluate_fooling_rate_raises_on_train_mode(correct_batch):
    """evaluate_fooling_rate raises ValueError if model in train mode."""
    model, x, y = correct_batch
    model.train()
    try:
        with pytest.raises(ValueError):
            evaluate_fooling_rate(model, torch.zeros(x.shape[1:]), [(x, y)])
    finally:
        model.eval()


def test_cross_architecture_transfer_returns_dict(correct_batch):
    """cross_architecture_transfer returns dict of model names to rates."""
    model, x, y = correct_batch
    uap = torch.randn(x.shape[1:]) * 0.05
    dataloader = [(x[:8], y[:8])]
    # Use same model twice to simulate multiple architectures
    results = cross_architecture_transfer([model, model], uap, dataloader)
    assert isinstance(results, dict)
    assert len(results) == 2
    for name, rate in results.items():
        assert isinstance(name, str)
        assert 0.0 <= rate <= 1.0


def test_cross_architecture_transfer_raises_on_train_mode(correct_batch):
    """cross_architecture_transfer raises if any model is in train mode."""
    model, x, y = correct_batch
    from adv_lab.eval.harness import _SmallCNN

    model2 = _SmallCNN(num_classes=3)
    model2.train()  # Left in train mode
    try:
        with pytest.raises(ValueError):
            cross_architecture_transfer(
                [model, model2],
                torch.zeros(x.shape[1:]),
                [(x[:4], y[:4])],
            )
    finally:
        model2.eval()
