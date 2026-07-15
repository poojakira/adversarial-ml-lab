"""Tests for norm-constrained attacks (L0, L1, Wasserstein, semantic, patch)."""

from __future__ import annotations

import torch

from adv_lab.attacks.norms import (
    patch_attack,
    pgd_l0,
    pgd_l1,
    semantic_attack,
    wasserstein_attack,
    _epsilon_search_schedule,
)


def test_pgd_l0_sparsity_constraint(correct_batch):
    """L0 attack modifies at most max_pixels spatial locations."""
    model, x, y = correct_batch
    max_pixels = 5
    x_adv = pgd_l0(model, x, y, max_pixels=max_pixels, steps=20, alpha=0.2)

    # Count modified pixels per sample (spatial locations, any channel)
    diff = (x_adv - x).abs()
    # A pixel is modified if any channel changed
    pixel_changed = diff.sum(dim=1) > 1e-6  # (N, H, W)
    n_modified = pixel_changed.view(x.shape[0], -1).sum(dim=1)

    # Each sample should have at most max_pixels modified locations
    assert n_modified.max().item() <= max_pixels
    # Output should be in valid range
    assert x_adv.min().item() >= 0.0 and x_adv.max().item() <= 1.0


def test_pgd_l1_ball_projection(correct_batch):
    """L1 attack keeps perturbation within the L1 ball."""
    model, x, y = correct_batch
    epsilon = 5.0
    x_adv = pgd_l1(model, x, y, epsilon=epsilon, alpha=0.3, steps=30)

    # Compute L1 norm of perturbation per sample
    l1_norms = (x_adv - x).abs().view(x.shape[0], -1).sum(dim=1)
    assert l1_norms.max().item() <= epsilon + 1e-3
    assert x_adv.min().item() >= 0.0 and x_adv.max().item() <= 1.0


def test_wasserstein_produces_valid_images(correct_batch):
    """Wasserstein attack produces valid images in [0, 1]."""
    model, x, y = correct_batch
    x_adv = wasserstein_attack(model, x, y, epsilon=1.0, steps=20, alpha=0.02)

    assert x_adv.shape == x.shape
    assert x_adv.min().item() >= 0.0 and x_adv.max().item() <= 1.0


def test_semantic_attack_produces_valid_images(correct_batch):
    """Semantic attack produces valid images via differentiable transforms."""
    model, x, y = correct_batch
    x_adv = semantic_attack(
        model,
        x,
        y,
        steps=20,
        lr=0.05,
        max_rotation=15.0,
        max_translation=0.1,
    )

    assert x_adv.shape == x.shape
    assert x_adv.min().item() >= 0.0 - 1e-6
    assert x_adv.max().item() <= 1.0 + 1e-6


def test_patch_attack_region_bounds(correct_batch):
    """Patch attack only modifies pixels within the patch region."""
    model, x, y = correct_batch
    patch_size = 3
    h, w = x.shape[2], x.shape[3]
    top = (h - patch_size) // 2
    left = (w - patch_size) // 2

    x_adv = patch_attack(
        model,
        x,
        y,
        patch_size=patch_size,
        steps=30,
        lr=0.05,
        patch_location=(top, left),
    )

    # Pixels outside the patch region should be unchanged
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :, top : top + patch_size, left : left + patch_size] = False
    outside_diff = (x_adv[mask] - x[mask]).abs().max().item()
    assert outside_diff < 1e-5

    # Patch values should be in [0, 1] (printable gamut)
    patch_vals = x_adv[:, :, top : top + patch_size, left : left + patch_size]
    assert patch_vals.min().item() >= 0.0 and patch_vals.max().item() <= 1.0


def test_epsilon_search_schedule():
    """Epsilon search schedule generates correct number of levels."""
    schedule = _epsilon_search_schedule(0.1, num_levels=4, growth_factor=2.0)
    assert len(schedule) == 4
    assert abs(schedule[0] - 0.1) < 1e-6
    assert abs(schedule[1] - 0.2) < 1e-6
    assert abs(schedule[2] - 0.4) < 1e-6
    assert abs(schedule[3] - 0.8) < 1e-6
