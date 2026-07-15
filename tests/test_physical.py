"""Tests for physical-world adversarial patch attacks."""

from __future__ import annotations

import pytest
import torch

from adv_lab.attacks.physical import (
    PatchResult,
    PhysicalPatchAttack,
    printability_constraint,
)


def test_physical_patch_attack_optimize(correct_batch):
    """PhysicalPatchAttack.optimize returns PatchResult with valid patch."""
    model, x, y = correct_batch
    # Use small patch and few steps for test speed
    attack = PhysicalPatchAttack(
        model,
        patch_size=(3, 3),
        target_class=None,
        angles=[0.0, 10.0],
        lighting_multipliers=[1.0],
        noise_std=0.01,
        lr=0.1,
        steps=3,
    )
    result = attack.optimize(x[:4], y[:4], seed=42)

    assert isinstance(result, PatchResult)
    assert result.patch.shape == (x.shape[1], 3, 3)
    assert result.patch.min().item() >= 0.0
    assert result.patch.max().item() <= 1.0
    assert 0.0 <= result.success_rate <= 1.0
    assert 0.0 <= result.printability_score <= 1.0


def test_physical_patch_attack_raises_on_train_mode(correct_batch):
    """PhysicalPatchAttack raises ValueError if model in train mode."""
    model, x, y = correct_batch
    model.train()
    try:
        with pytest.raises(ValueError):
            PhysicalPatchAttack(model, patch_size=(3, 3))
    finally:
        model.eval()


def test_multi_angle_robustness(correct_batch):
    """multi_angle_robustness returns dict mapping angles to success rates."""
    model, x, y = correct_batch
    attack = PhysicalPatchAttack(
        model,
        patch_size=(3, 3),
        angles=[-10.0, 0.0, 10.0],
        lighting_multipliers=[1.0],
        steps=2,
    )
    patch = torch.rand(x.shape[1], 3, 3)
    result = attack.multi_angle_robustness(patch, x[:4], y[:4])

    assert isinstance(result, dict)
    assert len(result) == 3
    for angle, rate in result.items():
        assert -30.0 <= angle <= 30.0
        assert 0.0 <= rate <= 1.0


def test_lighting_robustness(correct_batch):
    """lighting_robustness returns dict mapping multipliers to success rates."""
    model, x, y = correct_batch
    attack = PhysicalPatchAttack(
        model,
        patch_size=(3, 3),
        angles=[0.0],
        lighting_multipliers=[0.5, 1.0, 2.0],
        steps=2,
    )
    patch = torch.rand(x.shape[1], 3, 3)
    result = attack.lighting_robustness(patch, x[:4], y[:4])

    assert isinstance(result, dict)
    assert len(result) == 3
    for mult, rate in result.items():
        assert 0.5 <= mult <= 2.0
        assert 0.0 <= rate <= 1.0


def test_camera_noise_model(correct_batch):
    """camera_noise_model returns a float success rate in [0, 1]."""
    model, x, y = correct_batch
    attack = PhysicalPatchAttack(
        model,
        patch_size=(3, 3),
        steps=2,
    )
    patch = torch.rand(x.shape[1], 3, 3)
    rate = attack.camera_noise_model(patch, x[:4], y[:4], gaussian_std=0.05, n_trials=3)
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0


def test_printability_constraint_full_gamut():
    """printability_constraint returns 1.0 for patch fully in [0, 1]."""
    patch = torch.rand(3, 8, 8)  # All values in [0, 1]
    score = printability_constraint(patch)
    assert score == 1.0


def test_printability_constraint_out_of_gamut():
    """printability_constraint returns < 1.0 for out-of-gamut pixels."""
    patch = torch.rand(3, 8, 8)
    # Force some out-of-gamut values
    patch[0, 0, 0] = -0.5
    patch[1, 1, 1] = 1.5
    score = printability_constraint(patch)
    assert score < 1.0


def test_patch_result_dataclass():
    """PatchResult dataclass stores expected fields."""
    patch = torch.rand(3, 4, 4)
    result = PatchResult(
        patch=patch,
        success_rate=0.75,
        printability_score=0.95,
        angle_robustness={0.0: 0.8, 10.0: 0.7},
        lighting_robustness={1.0: 0.85},
    )
    assert result.success_rate == 0.75
    assert result.printability_score == 0.95
    assert len(result.angle_robustness) == 2
    assert len(result.lighting_robustness) == 1
