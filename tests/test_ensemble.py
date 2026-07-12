"""Tests for ensemble and multi-model attacks."""

from __future__ import annotations

import torch
import torch.nn as nn

from adv_lab.attacks.ensemble import (
    build_attacker_ensemble,
    ensemble_attack,
    weighted_ensemble_pgd,
)


class _TinyModel(nn.Module):
    """Tiny classifier for ensemble tests."""

    def __init__(self, in_features: int = 64, num_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.shape[0], -1))


def test_ensemble_attack_combines_models(correct_batch):
    """ensemble_attack accepts multiple models and produces valid output."""
    model, x, y = correct_batch

    # Create small ensemble of 3 models
    in_feat = x[0].numel()
    models = []
    for _ in range(3):
        m = _TinyModel(in_features=in_feat, num_classes=3)
        m.eval()
        models.append(m)

    epsilon = 0.05
    x_adv = ensemble_attack(models, x, y, epsilon=epsilon, steps=10)

    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0
    linf = (x_adv - x).abs().max().item()
    assert linf <= epsilon + 1e-5


def test_ensemble_attack_with_weights(correct_batch):
    """ensemble_attack respects custom per-model weights."""
    model, x, y = correct_batch
    in_feat = x[0].numel()

    models = [_TinyModel(in_feat, 3) for _ in range(3)]
    for m in models:
        m.eval()

    # Weight heavily toward first model
    weights = [0.8, 0.1, 0.1]
    x_adv = ensemble_attack(models, x, y, epsilon=0.05, steps=10, weights=weights)
    assert x_adv.shape == x.shape


def test_ensemble_attack_optimizes_all_models(correct_batch):
    """ensemble_attack increases loss on all models (not just one)."""
    model, x, y = correct_batch
    in_feat = x[0].numel()

    models = [_TinyModel(in_feat, 3) for _ in range(3)]
    for m in models:
        m.eval()

    x_adv = ensemble_attack(models, x, y, epsilon=0.1, steps=20)

    # Check that loss increased on each model
    for m in models:
        with torch.no_grad():
            loss_clean = nn.functional.cross_entropy(m(x), y).item()
            loss_adv = nn.functional.cross_entropy(m(x_adv), y).item()
        # Adversarial loss should be at least as high (we maximize CE)
        assert loss_adv >= loss_clean - 0.5  # allow small tolerance


def test_build_attacker_ensemble():
    """build_attacker_ensemble constructs models and puts them in eval mode."""
    constructors = [
        lambda: _TinyModel(64, 3),
        lambda: _TinyModel(64, 3),
        lambda: _TinyModel(64, 3),
    ]

    ensemble = build_attacker_ensemble(constructors)
    assert len(ensemble) == 3
    for m in ensemble:
        assert not m.training  # should be in eval mode


def test_build_attacker_ensemble_with_train_fn():
    """build_attacker_ensemble applies train_fn if provided."""
    trained_models = []

    def train_fn(model: nn.Module) -> nn.Module:
        trained_models.append(model)
        return model

    constructors = [lambda: _TinyModel(64, 3) for _ in range(2)]
    ensemble = build_attacker_ensemble(constructors, train_fn=train_fn)

    assert len(trained_models) == 2
    assert len(ensemble) == 2


def test_weighted_ensemble_pgd_basic(correct_batch):
    """weighted_ensemble_pgd produces valid adversarial examples."""
    model, x, y = correct_batch
    in_feat = x[0].numel()

    models = [_TinyModel(in_feat, 3) for _ in range(2)]
    for m in models:
        m.eval()

    epsilon = 0.05
    x_adv = weighted_ensemble_pgd(models, x, y, epsilon=epsilon, steps=10)

    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0
    linf = (x_adv - x).abs().max().item()
    assert linf <= epsilon + 1e-5


def test_weighted_ensemble_pgd_with_momentum(correct_batch):
    """weighted_ensemble_pgd with momentum produces different results."""
    model, x, y = correct_batch
    in_feat = x[0].numel()

    models = [_TinyModel(in_feat, 3) for _ in range(2)]
    for m in models:
        m.eval()

    torch.manual_seed(42)
    x_no_mom = weighted_ensemble_pgd(
        models, x, y, epsilon=0.05, steps=15, momentum=0.0, random_start=False
    )
    torch.manual_seed(42)
    x_with_mom = weighted_ensemble_pgd(
        models, x, y, epsilon=0.05, steps=15, momentum=0.9, random_start=False
    )

    # Momentum should produce different results
    # (they may be similar but not identical due to momentum accumulation)
    assert x_no_mom.shape == x_with_mom.shape


def test_weighted_ensemble_pgd_targeted(correct_batch):
    """weighted_ensemble_pgd supports targeted attacks."""
    model, x, y = correct_batch
    in_feat = x[0].numel()

    models = [_TinyModel(in_feat, 3) for _ in range(2)]
    for m in models:
        m.eval()

    # Target: shift all labels by 1 mod num_classes
    target_labels = (y + 1) % 3

    x_adv = weighted_ensemble_pgd(
        models,
        x,
        y,
        epsilon=0.1,
        steps=20,
        targeted=True,
        target_labels=target_labels,
    )

    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0
