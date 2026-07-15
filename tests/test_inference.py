"""Tests for inference-time data manipulation attacks (Tier 2, Item 12).

Tests watermark_flip, prediction_poison, and soft_label_manipulation from
src/adv_lab/attacks/inference.py.
"""

from __future__ import annotations

import pytest
import torch

from adv_lab.attacks.inference import (
    PreprocessingParams,
    prediction_poison,
    soft_label_manipulation,
    watermark_flip,
)


# ---------------------------------------------------------------------------
# watermark_flip tests
# ---------------------------------------------------------------------------


class TestWatermarkFlip:
    """Tests for gradient-based watermark flipping attack."""

    def test_output_shape_matches_input(self, correct_batch):
        """Output tensor must have the same shape as the input."""
        model, x, y = correct_batch
        x_adv = watermark_flip(model, x, y, epsilon=0.05, steps=10)
        assert x_adv.shape == x.shape

    def test_output_in_valid_range(self, correct_batch):
        """Adversarial images must stay in [0, 1]."""
        model, x, y = correct_batch
        x_adv = watermark_flip(model, x, y, epsilon=0.1, steps=10)
        assert x_adv.min().item() >= 0.0
        assert x_adv.max().item() <= 1.0

    def test_perturbation_respects_epsilon(self, correct_batch):
        """L-inf of the perturbation must not exceed epsilon."""
        model, x, y = correct_batch
        eps = 0.05
        x_adv = watermark_flip(model, x, y, epsilon=eps, steps=20)
        linf = (x_adv - x).abs().max().item()
        assert linf <= eps + 1e-6

    def test_nonzero_perturbation(self, correct_batch):
        """Attack should produce a non-trivial perturbation."""
        model, x, y = correct_batch
        x_adv = watermark_flip(model, x, y, epsilon=0.05, steps=20)
        assert not torch.allclose(x_adv, x)

    def test_custom_watermark_detector(self, correct_batch):
        """Attack should work with a custom watermark detector callable."""
        model, x, y = correct_batch

        def custom_detector(logits: torch.Tensor) -> torch.Tensor:
            # Simple detector: positive means watermark detected
            return logits.mean(dim=1) - 0.5

        x_adv = watermark_flip(
            model,
            x,
            y,
            watermark_detector=custom_detector,
            epsilon=0.05,
            steps=10,
        )
        assert x_adv.shape == x.shape
        assert x_adv.min().item() >= 0.0
        assert x_adv.max().item() <= 1.0

    def test_raises_on_train_mode(self, correct_batch):
        """Must raise ValueError if model is in training mode."""
        model, x, y = correct_batch
        model.train()
        try:
            with pytest.raises(ValueError):
                watermark_flip(model, x, y, epsilon=0.05, steps=5)
        finally:
            model.eval()

    def test_raises_on_invalid_dimensions(self, correct_batch):
        """Must raise ValueError for non-4D input."""
        model, x, y = correct_batch
        x_2d = x.view(x.shape[0], -1)
        with pytest.raises(ValueError):
            watermark_flip(model, x_2d, y, epsilon=0.05, steps=5)

    def test_early_stop_convergence(self, correct_batch):
        """Early stopping should not degrade output quality."""
        model, x, y = correct_batch
        x_adv_es = watermark_flip(model, x, y, epsilon=0.05, steps=50, early_stop=True)
        x_adv_no = watermark_flip(model, x, y, epsilon=0.05, steps=50, early_stop=False)
        # Both should produce valid outputs
        assert x_adv_es.shape == x.shape
        assert x_adv_no.shape == x.shape

    def test_output_is_detached(self, correct_batch):
        """Output tensor must be detached from the computation graph."""
        model, x, y = correct_batch
        x_adv = watermark_flip(model, x, y, epsilon=0.05, steps=5)
        assert not x_adv.requires_grad


# ---------------------------------------------------------------------------
# prediction_poison tests
# ---------------------------------------------------------------------------


class TestPredictionPoison:
    """Tests for preprocessing pipeline poisoning attack."""

    def test_output_shape_matches_input(self, correct_batch):
        """Output tensor must have the same shape as the input."""
        model, x, y = correct_batch
        x_poison = prediction_poison(model, x, y, target_shift=1, search_steps=5)
        assert x_poison.shape == x.shape

    def test_output_in_valid_range(self, correct_batch):
        """Poisoned images must be clamped to [0, 1]."""
        model, x, y = correct_batch
        x_poison = prediction_poison(model, x, y, target_shift=1, search_steps=5)
        assert x_poison.min().item() >= 0.0
        assert x_poison.max().item() <= 1.0

    def test_output_is_detached(self, correct_batch):
        """Output must be detached from the computation graph."""
        model, x, y = correct_batch
        x_poison = prediction_poison(model, x, y, target_shift=1, search_steps=3)
        assert not x_poison.requires_grad

    def test_raises_on_train_mode(self, correct_batch):
        """Must raise ValueError if model is in training mode."""
        model, x, y = correct_batch
        model.train()
        try:
            with pytest.raises(ValueError):
                prediction_poison(model, x, y, target_shift=1, search_steps=3)
        finally:
            model.eval()

    def test_multiple_restarts(self, correct_batch):
        """Multiple restarts should not crash and may find better solutions."""
        model, x, y = correct_batch
        x_poison = prediction_poison(
            model, x, y, target_shift=1, search_steps=3, num_restarts=3
        )
        assert x_poison.shape == x.shape

    def test_target_shift_modular(self, correct_batch):
        """Different target shifts should produce valid outputs."""
        model, x, y = correct_batch
        for shift in [1, 2]:
            x_poison = prediction_poison(
                model, x, y, target_shift=shift, search_steps=3
            )
            assert x_poison.shape == x.shape


# ---------------------------------------------------------------------------
# soft_label_manipulation tests
# ---------------------------------------------------------------------------


class TestSoftLabelManipulation:
    """Tests for soft-label confidence exploitation attack."""

    def test_output_shape_matches_input(self, correct_batch):
        """Output tensor must have the same shape as the input."""
        model, x, y = correct_batch
        x_adv = soft_label_manipulation(model, x, y, epsilon=0.05, steps=10)
        assert x_adv.shape == x.shape

    def test_output_in_valid_range(self, correct_batch):
        """Adversarial images must stay in [0, 1]."""
        model, x, y = correct_batch
        x_adv = soft_label_manipulation(model, x, y, epsilon=0.1, steps=10)
        assert x_adv.min().item() >= 0.0
        assert x_adv.max().item() <= 1.0

    def test_perturbation_respects_epsilon(self, correct_batch):
        """L-inf of the perturbation must not exceed epsilon."""
        model, x, y = correct_batch
        eps = 0.05
        x_adv = soft_label_manipulation(model, x, y, epsilon=eps, steps=20)
        linf = (x_adv - x).abs().max().item()
        assert linf <= eps + 1e-6

    def test_nonzero_perturbation(self, correct_batch):
        """Attack should produce a non-trivial perturbation."""
        model, x, y = correct_batch
        x_adv = soft_label_manipulation(model, x, y, epsilon=0.05, steps=20)
        assert not torch.allclose(x_adv, x)

    def test_zero_epsilon_no_change(self, correct_batch):
        """Zero epsilon should return unmodified input."""
        model, x, y = correct_batch
        x_adv = soft_label_manipulation(model, x, y, epsilon=0.0, steps=10)
        assert torch.allclose(x_adv, x)

    def test_output_is_detached(self, correct_batch):
        """Output tensor must be detached from the computation graph."""
        model, x, y = correct_batch
        x_adv = soft_label_manipulation(model, x, y, epsilon=0.05, steps=5)
        assert not x_adv.requires_grad

    def test_raises_on_train_mode(self, correct_batch):
        """Must raise ValueError if model is in training mode."""
        model, x, y = correct_batch
        model.train()
        try:
            with pytest.raises(ValueError):
                soft_label_manipulation(model, x, y, epsilon=0.05, steps=5)
        finally:
            model.eval()

    def test_momentum_parameter(self, correct_batch):
        """Different momentum values should produce valid outputs."""
        model, x, y = correct_batch
        for mom in [0.0, 0.5, 0.9]:
            x_adv = soft_label_manipulation(
                model, x, y, epsilon=0.05, steps=10, momentum=mom
            )
            assert x_adv.shape == x.shape
            assert x_adv.min().item() >= 0.0

    def test_temperature_scaling(self, correct_batch):
        """Temperature parameter should produce valid outputs."""
        model, x, y = correct_batch
        x_adv = soft_label_manipulation(
            model, x, y, epsilon=0.05, steps=10, temperature=2.0
        )
        assert x_adv.shape == x.shape

    def test_confidence_threshold_early_stop(self, correct_batch):
        """High confidence threshold should still produce valid output."""
        model, x, y = correct_batch
        x_adv = soft_label_manipulation(
            model, x, y, epsilon=0.1, steps=30, confidence_threshold=0.8
        )
        assert x_adv.shape == x.shape


# ---------------------------------------------------------------------------
# PreprocessingParams tests
# ---------------------------------------------------------------------------


class TestPreprocessingParams:
    """Tests for the PreprocessingParams dataclass."""

    def test_identity_preprocessing(self, correct_batch):
        """Default (identity) preprocessing should not change the input."""
        _, x, _ = correct_batch
        params = PreprocessingParams()
        result = params.apply(x)
        assert torch.allclose(result, x, atol=1e-6)

    def test_brightness_shift(self, correct_batch):
        """Brightness shift should offset values."""
        _, x, _ = correct_batch
        params = PreprocessingParams(brightness=0.1)
        result = params.apply(x)
        # Result should be different from original
        assert not torch.allclose(result, x)
        # Output should be clamped to [0, 1]
        assert result.min().item() >= 0.0
        assert result.max().item() <= 1.0

    def test_distance_from_default(self):
        """distance_from_default should report parameter deviation."""
        params = PreprocessingParams(brightness=0.1, contrast=1.2, gamma=0.9)
        dist = params.distance_from_default()
        assert dist > 0.0
        # Should equal |0.1| + |1.2 - 1.0| + |0.9 - 1.0| = 0.1 + 0.2 + 0.1 = 0.4
        assert abs(dist - 0.4) < 1e-6
