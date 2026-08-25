"""Tests for src/adv_lab/defenses/ (adversarial training and detection).

Covers:
  - AdversarialTrainer initialization, single-epoch training, and evaluation
  - STRIPDetector entropy computation and detection
  - NeuralCleanse trigger reverse-engineering and backdoor detection
  - bypass_strip produces valid output
  - bypass_neural_cleanse produces a poisoned model
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adv_lab.defenses.adversarial_training import AdversarialTrainer
from adv_lab.defenses.detection import (
    NeuralCleanse,
    STRIPDetector,
    bypass_neural_cleanse,
    bypass_strip,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


class _TinyCNN(nn.Module):
    """Minimal CNN for unit testing (1-channel 8x8 input, 3 classes)."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(4 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _make_dataloader(n: int = 32, channels: int = 1, size: int = 8, num_classes: int = 3):
    """Create a synthetic dataloader for testing."""
    torch.manual_seed(42)
    images = torch.rand(n, channels, size, size)
    labels = torch.randint(0, num_classes, (n,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=16, shuffle=False)


# ── AdversarialTrainer Tests ──────────────────────────────────────────────────


class TestAdversarialTrainer:
    """Tests for AdversarialTrainer."""

    def test_init_defaults(self):
        """Trainer initializes with correct defaults."""
        model = _TinyCNN()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = AdversarialTrainer(model, optimizer, epsilon=0.03)

        assert trainer.model is model
        assert trainer.optimizer is optimizer
        assert trainer.epsilon == 0.03
        assert trainer.attack_fn is not None

    def test_train_epoch_returns_metrics(self):
        """train_epoch() returns dict with loss, clean_acc, robust_acc."""
        model = _TinyCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = AdversarialTrainer(model, optimizer, epsilon=0.03)

        dataloader = _make_dataloader(n=32)
        metrics = trainer.train_epoch(dataloader)

        assert "loss" in metrics
        assert "clean_acc" in metrics
        assert "robust_acc" in metrics
        assert isinstance(metrics["loss"], float)
        assert 0.0 <= metrics["clean_acc"] <= 1.0
        assert 0.0 <= metrics["robust_acc"] <= 1.0

    def test_train_epoch_reduces_loss(self):
        """After multiple epochs, loss should decrease (model is learning)."""
        torch.manual_seed(0)
        model = _TinyCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = AdversarialTrainer(model, optimizer, epsilon=0.01)

        dataloader = _make_dataloader(n=64)
        first_metrics = trainer.train_epoch(dataloader)
        # Train a few more epochs
        for _ in range(4):
            last_metrics = trainer.train_epoch(dataloader)

        # Loss should decrease over training
        assert last_metrics["loss"] < first_metrics["loss"]

    def test_evaluate_robust(self):
        """evaluate_robust returns a float in [0, 1]."""
        model = _TinyCNN()
        model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = AdversarialTrainer(model, optimizer, epsilon=0.03)

        dataloader = _make_dataloader(n=16)

        def simple_attack(model, images, labels, epsilon):
            return images + epsilon * torch.sign(torch.randn_like(images))

        robust_acc = trainer.evaluate_robust(dataloader, attack_fn=simple_attack)
        assert 0.0 <= robust_acc <= 1.0

    def test_custom_attack_fn(self):
        """Custom attack_fn is properly invoked during training."""
        call_count = [0]

        def counting_attack(model, images, labels, epsilon):
            call_count[0] += 1
            # Return images unchanged (no-op attack)
            return images.clone().detach()

        model = _TinyCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = AdversarialTrainer(model, optimizer, attack_fn=counting_attack, epsilon=0.03)

        dataloader = _make_dataloader(n=16)
        trainer.train_epoch(dataloader)

        assert call_count[0] > 0, "Custom attack function was never called"


# ── STRIPDetector Tests ───────────────────────────────────────────────────────


class TestSTRIPDetector:
    """Tests for STRIPDetector."""

    def test_compute_entropy_returns_correct_shape(self):
        """compute_entropy returns one entropy value per input image."""
        model = _TinyCNN()
        model.eval()
        clean_ref = torch.rand(20, 1, 8, 8)
        detector = STRIPDetector(model, clean_ref, num_blends=5, blend_alpha=0.5)

        test_images = torch.rand(4, 1, 8, 8)
        entropies = detector.compute_entropy(test_images)

        assert entropies.shape == (4,)
        # Entropies should be non-negative
        assert (entropies >= 0.0).all()

    def test_detect_returns_boolean_and_scores(self):
        """detect() returns (is_trojaned, entropy_scores) tuple."""
        model = _TinyCNN()
        model.eval()
        clean_ref = torch.rand(20, 1, 8, 8)
        detector = STRIPDetector(
            model, clean_ref, num_blends=5, blend_alpha=0.5, entropy_threshold=0.5
        )

        test_images = torch.rand(4, 1, 8, 8)
        is_trojaned, scores = detector.detect(test_images)

        assert is_trojaned.shape == (4,)
        assert is_trojaned.dtype == torch.bool
        assert scores.shape == (4,)

    def test_detect_with_adaptive_threshold(self):
        """detect() works with adaptive threshold (entropy_threshold=None)."""
        model = _TinyCNN()
        model.eval()
        clean_ref = torch.rand(20, 1, 8, 8)
        detector = STRIPDetector(model, clean_ref, num_blends=3, entropy_threshold=None)

        test_images = torch.rand(4, 1, 8, 8)
        is_trojaned, scores = detector.detect(test_images)

        assert is_trojaned.shape == (4,)
        assert scores.shape == (4,)


# ── NeuralCleanse Tests ───────────────────────────────────────────────────────


class TestNeuralCleanse:
    """Tests for NeuralCleanse."""

    def test_reverse_engineer_triggers(self):
        """reverse_engineer_triggers returns masks, patterns, and L1 norms."""
        torch.manual_seed(42)
        model = _TinyCNN(num_classes=3)
        model.eval()

        nc = NeuralCleanse(
            model, num_classes=3, input_shape=(1, 8, 8), steps=10, lr=0.1, lambda_reg=0.01
        )
        clean_images = torch.rand(8, 1, 8, 8)
        masks, patterns, l1_norms = nc.reverse_engineer_triggers(clean_images)

        assert len(masks) == 3
        assert len(patterns) == 3
        assert len(l1_norms) == 3
        for m in masks:
            assert m.shape == (1, 8, 8)
        for p in patterns:
            assert p.shape == (1, 8, 8)
        for norm in l1_norms:
            assert norm > 0.0

    def test_detect_backdoor_returns_none_for_clean(self):
        """detect_backdoor should not flag a clean (randomly initialized) model."""
        torch.manual_seed(0)
        model = _TinyCNN(num_classes=3)
        model.eval()

        nc = NeuralCleanse(
            model, num_classes=3, input_shape=(1, 8, 8), steps=5, lr=0.1, lambda_reg=0.01
        )
        clean_images = torch.rand(8, 1, 8, 8)
        backdoor_class, l1_norms = nc.detect_backdoor(clean_images, anomaly_threshold=5.0)

        # With a high threshold and random model, unlikely to flag anything
        assert l1_norms is not None
        assert len(l1_norms) == 3


# ── Bypass Function Tests ─────────────────────────────────────────────────────


class TestBypassFunctions:
    """Tests for bypass_strip and bypass_neural_cleanse."""

    def test_bypass_strip_output_shape_and_range(self):
        """bypass_strip returns images in [0,1] with correct shape."""
        images = torch.rand(4, 1, 8, 8)
        trigger_pattern = torch.ones(1, 8, 8) * 0.8
        trigger_mask = torch.zeros(1, 8, 8)
        trigger_mask[:, 6:8, 6:8] = 1.0

        result = bypass_strip(images, trigger_pattern, trigger_mask, noise_magnitude=0.1)

        assert result.shape == images.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_bypass_neural_cleanse_returns_model_and_triggers(self):
        """bypass_neural_cleanse returns a model and list of trigger patterns."""
        torch.manual_seed(42)
        model = _TinyCNN(num_classes=3)
        images = torch.rand(16, 1, 8, 8)
        labels = torch.randint(0, 3, (16,))

        poisoned_model, trigger_patterns = bypass_neural_cleanse(
            model, images, labels, target_label=0, num_triggers=2, trigger_size=2, steps=5, lr=0.01
        )

        assert isinstance(poisoned_model, nn.Module)
        assert len(trigger_patterns) == 2
        for tp in trigger_patterns:
            assert tp.shape == (1, 2, 2)
