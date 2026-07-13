"""Tests for certified defense evaluation (Tier 3, Item 16).

Tests RandomizedSmoothing, lipschitz_eval, ibp_eval, find_certificate_boundary
from src/adv_lab/eval/certified.py.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from adv_lab.eval.certified import (
    IBPBounds,
    LipschitzNetwork,
    RandomizedSmoothing,
    SmoothingResult,
    find_certificate_boundary,
    ibp_eval,
    lipschitz_eval,
)


# ---------------------------------------------------------------------------
# Helper: simple IBP-compatible model (sequential Linear + ReLU)
# ---------------------------------------------------------------------------


def _make_ibp_model(input_dim: int = 64, num_classes: int = 3) -> nn.Module:
    """Create a simple sequential model suitable for IBP evaluation."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# RandomizedSmoothing tests
# ---------------------------------------------------------------------------


class TestRandomizedSmoothing:
    """Tests for the RandomizedSmoothing certified defense evaluator."""

    def test_certify_returns_smoothing_result(self, correct_batch):
        """certify() should return a SmoothingResult dataclass."""
        model, x, y = correct_batch
        smoother = RandomizedSmoothing(model, sigma=0.25, n_samples=10)
        result = smoother.certify(x[:4])
        assert isinstance(result, SmoothingResult)

    def test_predicted_class_shape(self, correct_batch):
        """predicted_class tensor should have shape (N,)."""
        model, x, y = correct_batch
        smoother = RandomizedSmoothing(model, sigma=0.25, n_samples=10)
        result = smoother.certify(x[:4])
        assert result.predicted_class.shape == (4,)

    def test_certified_radius_shape(self, correct_batch):
        """certified_radius tensor should have shape (N,)."""
        model, x, y = correct_batch
        smoother = RandomizedSmoothing(model, sigma=0.25, n_samples=10)
        result = smoother.certify(x[:4])
        assert result.certified_radius.shape == (4,)

    def test_certified_radius_non_negative(self, correct_batch):
        """Certified radii must be non-negative."""
        model, x, y = correct_batch
        smoother = RandomizedSmoothing(model, sigma=0.25, n_samples=50)
        result = smoother.certify(x[:4])
        assert (result.certified_radius >= 0.0).all()

    def test_is_certified_boolean(self, correct_batch):
        """is_certified tensor should be boolean."""
        model, x, y = correct_batch
        smoother = RandomizedSmoothing(model, sigma=0.25, n_samples=10)
        result = smoother.certify(x[:4])
        assert result.is_certified.dtype == torch.bool

    def test_num_samples_recorded(self, correct_batch):
        """num_samples_used should match the constructor argument."""
        model, x, y = correct_batch
        smoother = RandomizedSmoothing(model, sigma=0.25, n_samples=20)
        result = smoother.certify(x[:2])
        assert result.num_samples_used == 20

    def test_confidence_level_recorded(self, correct_batch):
        """confidence_level should match the constructor argument."""
        model, x, y = correct_batch
        smoother = RandomizedSmoothing(model, sigma=0.25, n_samples=10, confidence_level=0.95)
        result = smoother.certify(x[:2])
        assert result.confidence_level == 0.95

    def test_higher_sigma_larger_radius(self, correct_batch):
        """Higher sigma should generally produce larger certified radii."""
        model, x, y = correct_batch
        smoother_low = RandomizedSmoothing(model, sigma=0.1, n_samples=50)
        smoother_high = RandomizedSmoothing(model, sigma=0.5, n_samples=50)
        result_low = smoother_low.certify(x[:4])
        result_high = smoother_high.certify(x[:4])
        # On average, higher sigma should give larger radii (not guaranteed per-sample)
        assert result_high.certified_radius.mean() >= result_low.certified_radius.mean() - 0.1

    def test_inverse_normal_cdf_symmetry(self):
        """The inverse CDF function should satisfy basic properties."""
        # Phi^{-1}(0.5) = 0
        assert abs(RandomizedSmoothing._inverse_normal_cdf(0.5)) < 1e-6
        # Phi^{-1}(p) > 0 for p > 0.5
        assert RandomizedSmoothing._inverse_normal_cdf(0.75) > 0
        # Phi^{-1}(p) < 0 for p < 0.5
        assert RandomizedSmoothing._inverse_normal_cdf(0.25) < 0

    def test_raises_on_train_mode(self, correct_batch):
        """Must raise if model is in training mode."""
        model, x, y = correct_batch
        model.train()
        try:
            with pytest.raises(ValueError):
                RandomizedSmoothing(model, sigma=0.25, n_samples=10)
        finally:
            model.eval()


# ---------------------------------------------------------------------------
# LipschitzNetwork tests
# ---------------------------------------------------------------------------


class TestLipschitzNetwork:
    """Tests for the LipschitzNetwork model."""

    def test_output_shape(self):
        """Output should be (N, num_classes)."""
        net = LipschitzNetwork(input_dim=64, hidden_dim=32, num_classes=3)
        net.eval()
        x = torch.rand(4, 64)
        out = net(x)
        assert out.shape == (4, 3)

    def test_handles_4d_input(self):
        """Should flatten 4D inputs automatically."""
        net = LipschitzNetwork(input_dim=64, hidden_dim=32, num_classes=3)
        net.eval()
        x = torch.rand(4, 1, 8, 8)
        out = net(x)
        assert out.shape == (4, 3)


# ---------------------------------------------------------------------------
# lipschitz_eval tests
# ---------------------------------------------------------------------------


class TestLipschitzEval:
    """Tests for lipschitz_eval function."""

    def test_returns_accuracy_tuple(self):
        """Should return (clean_accuracy, robust_accuracy) tuple."""
        net = LipschitzNetwork(input_dim=64, hidden_dim=32, num_classes=3)
        net.eval()
        x = torch.rand(8, 64)
        y = torch.randint(0, 3, (8,))
        result = lipschitz_eval(net, x, y, epsilon=0.03, attack_steps=5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_accuracies_in_valid_range(self):
        """Both accuracies should be in [0, 1]."""
        net = LipschitzNetwork(input_dim=64, hidden_dim=32, num_classes=3)
        net.eval()
        x = torch.rand(8, 64)
        y = torch.randint(0, 3, (8,))
        clean_acc, robust_acc = lipschitz_eval(net, x, y, epsilon=0.03, attack_steps=5)
        assert 0.0 <= clean_acc <= 1.0
        assert 0.0 <= robust_acc <= 1.0

    def test_robust_leq_clean(self):
        """Robust accuracy should not exceed clean accuracy."""
        net = LipschitzNetwork(input_dim=64, hidden_dim=32, num_classes=3)
        net.eval()
        x = torch.rand(16, 64)
        y = torch.randint(0, 3, (16,))
        clean_acc, robust_acc = lipschitz_eval(net, x, y, epsilon=0.1, attack_steps=20)
        assert robust_acc <= clean_acc + 1e-6

    def test_raises_on_train_mode(self):
        """Must raise if model is in training mode."""
        net = LipschitzNetwork(input_dim=64, hidden_dim=32, num_classes=3)
        net.train()
        x = torch.rand(4, 64)
        y = torch.randint(0, 3, (4,))
        with pytest.raises(ValueError):
            lipschitz_eval(net, x, y, epsilon=0.03, attack_steps=5)


# ---------------------------------------------------------------------------
# ibp_eval tests
# ---------------------------------------------------------------------------


class TestIBPEval:
    """Tests for ibp_eval function."""

    def test_returns_ibp_bounds(self):
        """Should return an IBPBounds dataclass."""
        model = _make_ibp_model(input_dim=64, num_classes=3)
        x = torch.rand(4, 64)
        y = torch.randint(0, 3, (4,))
        result = ibp_eval(model, x, y, epsilon=0.01)
        assert isinstance(result, IBPBounds)

    def test_bounds_shape(self):
        """Lower and upper bounds should have shape (N, num_classes)."""
        model = _make_ibp_model(input_dim=64, num_classes=3)
        x = torch.rand(4, 64)
        y = torch.randint(0, 3, (4,))
        result = ibp_eval(model, x, y, epsilon=0.01)
        assert result.lower.shape == (4, 3)
        assert result.upper.shape == (4, 3)

    def test_lower_leq_upper(self):
        """Lower bounds must not exceed upper bounds."""
        model = _make_ibp_model(input_dim=64, num_classes=3)
        x = torch.rand(4, 64)
        y = torch.randint(0, 3, (4,))
        result = ibp_eval(model, x, y, epsilon=0.01)
        assert (result.lower <= result.upper + 1e-5).all()

    def test_verified_is_boolean(self):
        """verified field should be a boolean tensor."""
        model = _make_ibp_model(input_dim=64, num_classes=3)
        x = torch.rand(4, 64)
        y = torch.randint(0, 3, (4,))
        result = ibp_eval(model, x, y, epsilon=0.01)
        assert result.verified.dtype == torch.bool
        assert result.verified.shape == (4,)

    def test_epsilon_stored(self):
        """epsilon field should match the input argument."""
        model = _make_ibp_model(input_dim=64, num_classes=3)
        x = torch.rand(4, 64)
        y = torch.randint(0, 3, (4,))
        result = ibp_eval(model, x, y, epsilon=0.05)
        assert result.epsilon == 0.05

    def test_handles_4d_input(self):
        """Should handle (N, C, H, W) inputs by flattening."""
        model = _make_ibp_model(input_dim=64, num_classes=3)
        x = torch.rand(4, 1, 8, 8)
        y = torch.randint(0, 3, (4,))
        result = ibp_eval(model, x, y, epsilon=0.01)
        assert result.lower.shape == (4, 3)

    def test_raises_on_train_mode(self):
        """Must raise if model is in training mode."""
        model = _make_ibp_model(input_dim=64, num_classes=3)
        model.train()
        x = torch.rand(4, 64)
        y = torch.randint(0, 3, (4,))
        with pytest.raises(ValueError):
            ibp_eval(model, x, y, epsilon=0.01)


# ---------------------------------------------------------------------------
# find_certificate_boundary tests
# ---------------------------------------------------------------------------


class TestFindCertificateBoundary:
    """Tests for the binary search certificate boundary finder."""

    def test_returns_tuple(self, correct_batch):
        """Should return (critical_epsilon, search_history)."""
        model, x, y = correct_batch
        result = find_certificate_boundary(
            model, x[:4], y[:4],
            method="smoothing",
            sigma=0.25,
            n_samples=10,
            max_iterations=3,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_critical_epsilon_positive(self, correct_batch):
        """Critical epsilon should be positive."""
        model, x, y = correct_batch
        critical_eps, _ = find_certificate_boundary(
            model, x[:4], y[:4],
            method="smoothing",
            sigma=0.25,
            n_samples=10,
            max_iterations=5,
        )
        assert critical_eps >= 0.0

    def test_search_history_non_empty(self, correct_batch):
        """Search history should contain at least one entry."""
        model, x, y = correct_batch
        _, history = find_certificate_boundary(
            model, x[:4], y[:4],
            method="smoothing",
            sigma=0.25,
            n_samples=10,
            max_iterations=5,
        )
        assert len(history) >= 1
        # Each entry should be (epsilon, certified_accuracy)
        assert len(history[0]) == 2

    def test_ibp_method(self):
        """Should work with the IBP method."""
        model = _make_ibp_model(input_dim=64, num_classes=3)
        x = torch.rand(4, 64)
        y = torch.randint(0, 3, (4,))
        critical_eps, history = find_certificate_boundary(
            model, x, y,
            method="ibp",
            max_iterations=5,
        )
        assert critical_eps >= 0.0
        assert len(history) >= 1

    def test_invalid_method_raises(self, correct_batch):
        """Invalid method should raise ValueError."""
        model, x, y = correct_batch
        with pytest.raises(ValueError):
            find_certificate_boundary(
                model, x[:4], y[:4],
                method="invalid",
                max_iterations=3,
            )

    def test_raises_on_train_mode(self, correct_batch):
        """Must raise if model is in training mode."""
        model, x, y = correct_batch
        model.train()
        try:
            with pytest.raises(ValueError):
                find_certificate_boundary(
                    model, x[:4], y[:4],
                    method="smoothing",
                    n_samples=10,
                    max_iterations=3,
                )
        finally:
            model.eval()

