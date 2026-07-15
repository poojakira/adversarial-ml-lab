"""Tests for model inversion and membership inference attacks."""

from __future__ import annotations

import pytest
import torch

from adv_lab.attacks.inversion import (
    InversionResult,
    MembershipResult,
    gan_inversion,
    gradient_inversion,
    membership_inference_likelihood,
    membership_inference_shadow,
)


def test_gradient_inversion_returns_valid_result(correct_batch):
    """gradient_inversion returns InversionResult with correct shape and range."""
    model, x, y = correct_batch
    # Compute target gradients from a real forward/backward pass
    x_single = x[:1].clone().detach().requires_grad_(True)
    logits = model(x_single)
    loss = torch.nn.functional.cross_entropy(logits, y[:1])
    target_grads = torch.autograd.grad(loss, list(model.parameters()))
    target_grads = [g.detach() for g in target_grads]

    result = gradient_inversion(
        model,
        target_grads,
        input_shape=x.shape[1:],
        num_samples=1,
        steps=5,
        lr=0.1,
        seed=42,
    )
    assert isinstance(result, InversionResult)
    assert result.reconstructed.shape == (1, *x.shape[1:])
    assert result.reconstructed.min().item() >= 0.0
    assert result.reconstructed.max().item() <= 1.0
    assert result.ssim_scores.shape == (1,)
    assert result.iterations_used == 5


def test_gradient_inversion_raises_on_train_mode(correct_batch):
    """gradient_inversion raises ValueError if model is in train mode."""
    model, x, y = correct_batch
    target_grads = [torch.zeros_like(p) for p in model.parameters()]
    model.train()
    try:
        with pytest.raises(ValueError):
            gradient_inversion(model, target_grads, input_shape=x.shape[1:], steps=1)
    finally:
        model.eval()


def test_gan_inversion_returns_valid_result(correct_batch):
    """gan_inversion returns InversionResult with reconstructed images in [0,1]."""
    model, x, y = correct_batch
    with torch.no_grad():
        target_outputs = model(x[:4])

    result = gan_inversion(
        model,
        target_outputs,
        input_shape=x.shape[1:],
        latent_dim=64,
        steps=5,
        lr=0.01,
        seed=42,
    )
    assert isinstance(result, InversionResult)
    assert result.reconstructed.shape == (4, *x.shape[1:])
    assert result.reconstructed.min().item() >= 0.0
    assert result.reconstructed.max().item() <= 1.0
    assert result.iterations_used == 5


def test_gan_inversion_raises_on_train_mode(correct_batch):
    """gan_inversion raises ValueError if model is in train mode."""
    model, x, y = correct_batch
    with torch.no_grad():
        target_outputs = model(x[:2])
    model.train()
    try:
        with pytest.raises(ValueError):
            gan_inversion(model, target_outputs, input_shape=x.shape[1:], steps=1)
    finally:
        model.eval()


def test_membership_inference_shadow_returns_valid_result(correct_batch):
    """membership_inference_shadow returns MembershipResult with correct shapes."""
    model, x, y = correct_batch
    n = x.shape[0]
    half = n // 2
    # Split data into shadow train/test
    shadow_train = (x[:half], y[:half])
    shadow_test = (x[half:], y[half:])

    result = membership_inference_shadow(
        model,
        x[:8],
        y[:8],
        shadow_train_data=shadow_train,
        shadow_test_data=shadow_test,
        shadow_epochs=2,
    )
    assert isinstance(result, MembershipResult)
    assert result.scores.shape == (8,)
    assert result.predictions.shape == (8,)
    assert 0.0 <= result.auc <= 1.0
    assert 0.0 <= result.tpr_at_low_fpr <= 1.0
    # Scores should be in [0, 1]
    assert result.scores.min().item() >= 0.0
    assert result.scores.max().item() <= 1.0


def test_membership_inference_shadow_raises_on_train_mode(correct_batch):
    """membership_inference_shadow raises ValueError on train mode."""
    model, x, y = correct_batch
    model.train()
    try:
        with pytest.raises(ValueError):
            membership_inference_shadow(
                model,
                x[:4],
                y[:4],
                shadow_train_data=(x[:4], y[:4]),
                shadow_test_data=(x[:4], y[:4]),
            )
    finally:
        model.eval()


def test_membership_inference_likelihood_returns_valid_result(correct_batch):
    """membership_inference_likelihood returns valid MembershipResult."""
    model, x, y = correct_batch
    result = membership_inference_likelihood(
        model,
        x[:8],
        y[:8],
        temperature=1.0,
        threshold=0.0,
    )
    assert isinstance(result, MembershipResult)
    assert result.scores.shape == (8,)
    assert result.predictions.shape == (8,)
    assert 0.0 <= result.auc <= 1.0
    assert result.scores.min().item() >= 0.0
    assert result.scores.max().item() <= 1.0


def test_membership_inference_likelihood_with_reference(correct_batch):
    """membership_inference_likelihood works with a reference model."""
    model, x, y = correct_batch
    # Use the same model as reference (scores should be ~0)
    result = membership_inference_likelihood(
        model,
        x[:4],
        y[:4],
        reference_model=model,
        temperature=1.0,
    )
    assert isinstance(result, MembershipResult)
    assert result.scores.shape == (4,)


def test_membership_inference_likelihood_raises_on_train_mode(correct_batch):
    """membership_inference_likelihood raises ValueError on train mode."""
    model, x, y = correct_batch
    model.train()
    try:
        with pytest.raises(ValueError):
            membership_inference_likelihood(model, x[:4], y[:4])
    finally:
        model.eval()
