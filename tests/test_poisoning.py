"""Tests for training-time poisoning attacks and detection."""

from __future__ import annotations

import torch

from adv_lab.attacks.poisoning import (
    badnets_trigger,
    clean_label_poison,
    spectral_backdoor,
    weight_poisoning,
)
from adv_lab.defenses.detection import (
    NeuralCleanse,
    STRIPDetector,
    bypass_neural_cleanse,
    bypass_strip,
)


def test_badnets_trigger_insertion(correct_batch):
    """BadNets correctly inserts triggers and relabels samples."""
    model, x, y = correct_batch
    target_label = 0
    trigger_size = 2

    poisoned_imgs, poisoned_labels = badnets_trigger(
        x, y, target_label=target_label, trigger_size=trigger_size,
        trigger_location="bottom_right", poison_fraction=0.3,
    )

    # Verify shapes preserved
    assert poisoned_imgs.shape == x.shape
    assert poisoned_labels.shape == y.shape

    # Check that some labels were changed to target
    relabeled = (poisoned_labels == target_label) & (y != target_label)
    assert relabeled.sum().item() > 0

    # Check that poisoned images are still in valid range
    assert poisoned_imgs.min().item() >= 0.0
    assert poisoned_imgs.max().item() <= 1.0

    # Check that trigger is actually present in relabeled samples
    h, w = x.shape[2], x.shape[3]
    trigger_region = poisoned_imgs[relabeled][:, :, h - trigger_size:, w - trigger_size:]
    # Trigger region should have the checkerboard pattern (not all zeros)
    assert trigger_region.abs().sum().item() > 0


def test_spectral_backdoor_invisible(correct_batch):
    """Spectral backdoor preserves visual appearance (small perturbation)."""
    model, x, y = correct_batch
    target_label = 0

    poisoned_imgs, poisoned_labels = spectral_backdoor(
        x, y, target_label=target_label,
        trigger_frequency=0.3, trigger_magnitude=0.02, poison_fraction=0.2,
    )

    # Shape preserved
    assert poisoned_imgs.shape == x.shape

    # Perturbation should be small (invisible)
    max_diff = (poisoned_imgs - x).abs().max().item()
    assert max_diff <= 0.05  # trigger_magnitude + potential clipping effects

    # Labels changed for poisoned fraction
    relabeled = (poisoned_labels == target_label) & (y != target_label)
    assert relabeled.sum().item() > 0

    # Valid range
    assert poisoned_imgs.min().item() >= 0.0
    assert poisoned_imgs.max().item() <= 1.0


def test_clean_label_poison_maintains_labels(correct_batch):
    """Clean-label poison keeps the same labels (visually correct)."""
    model, x, y = correct_batch
    # Use first few samples as base, another as target
    base_images = x[:5]
    base_labels = y[:5]
    target_image = x[10:11]
    target_label = y[10].item()

    poisoned = clean_label_poison(
        model, base_images, base_labels, target_image, target_label,
        steps=20, lr=0.05, epsilon=0.1,
    )

    # Shape preserved
    assert poisoned.shape == base_images.shape
    # Valid range
    assert poisoned.min().item() >= 0.0
    assert poisoned.max().item() <= 1.0
    # Perturbation bounded by epsilon
    linf = (poisoned - base_images).abs().max().item()
    assert linf <= 0.1 + 1e-4


def test_weight_poisoning_modifies_weights(correct_batch):
    """Weight poisoning modifies model weights to create a backdoor."""
    model, x, y = correct_batch
    import copy
    model_copy = copy.deepcopy(model)

    # Create a trigger input (e.g., add a patch)
    trigger_input = x[0:1].clone()
    trigger_input[:, :, -2:, -2:] = 1.0  # White patch in corner
    target_label = (y[0].item() + 1) % 3  # Different class

    poisoned_model = weight_poisoning(
        model_copy, trigger_input, target_label,
        poison_strength=1.0, steps=30, lr=0.01,
    )

    # Verify weights have changed
    original_params = dict(model.named_parameters())
    poisoned_params = dict(poisoned_model.named_parameters())

    weight_changed = False
    for name in original_params:
        if not torch.allclose(original_params[name], poisoned_params[name], atol=1e-6):
            weight_changed = True
            break
    assert weight_changed

    # Model should still be in eval mode
    assert not poisoned_model.training


def test_strip_detector_computes_entropy(correct_batch):
    """STRIP detector computes entropy scores for inputs."""
    model, x, y = correct_batch
    # Use a subset as clean reference
    clean_ref = x[:10]
    detector = STRIPDetector(
        model, clean_ref, num_blends=5, blend_alpha=0.5, entropy_threshold=0.5
    )

    entropies = detector.compute_entropy(x[:5])
    assert entropies.shape == (5,)
    # Entropy should be non-negative
    assert (entropies >= 0).all()


def test_neural_cleanse_detects_triggers(correct_batch):
    """Neural Cleanse reverse-engineers triggers for each class."""
    model, x, y = correct_batch
    input_shape = (x.shape[1], x.shape[2], x.shape[3])

    cleanse = NeuralCleanse(
        model, num_classes=3, input_shape=input_shape,
        steps=20, lr=0.1, lambda_reg=0.01,
    )

    masks, patterns, l1_norms = cleanse.reverse_engineer_triggers(x[:8])

    assert len(masks) == 3
    assert len(patterns) == 3
    assert len(l1_norms) == 3

    # Masks should be in [0, 1] (after sigmoid)
    for mask in masks:
        assert mask.min().item() >= 0.0
        assert mask.max().item() <= 1.0

    # L1 norms should be positive
    assert all(n > 0 for n in l1_norms)


def test_bypass_strip_produces_valid_images(correct_batch):
    """STRIP bypass produces triggered images in valid range."""
    model, x, y = correct_batch
    n_channels = x.shape[1]
    h, w = x.shape[2], x.shape[3]

    # Create a simple trigger
    trigger_pattern = torch.ones(n_channels, h, w) * 0.8
    trigger_mask = torch.zeros(1, h, w)
    trigger_mask[:, -2:, -2:] = 1.0

    bypassed = bypass_strip(x[:5], trigger_pattern, trigger_mask, noise_magnitude=0.05)

    assert bypassed.shape == x[:5].shape
    assert bypassed.min().item() >= 0.0
    assert bypassed.max().item() <= 1.0
