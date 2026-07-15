"""Tests for non-classification adversarial attacks (Tier 3, Item 15).

Tests object_detection_attack, segmentation_attack, regression_attack,
rl_attack, recommendation_attack, and the simple model classes from
src/adv_lab/attacks/non_classification.py.
"""

from __future__ import annotations

import pytest
import torch

from adv_lab.attacks.non_classification import (
    SimpleDetector,
    SimplePolicy,
    SimpleRecommender,
    SimpleRegressor,
    SimpleSegmenter,
    object_detection_attack,
    recommendation_attack,
    regression_attack,
    rl_attack,
    segmentation_attack,
)


# ---------------------------------------------------------------------------
# SimpleDetector tests
# ---------------------------------------------------------------------------


class TestSimpleDetector:
    """Tests for the SimpleDetector model."""

    def test_output_shape(self):
        """Output should be (N, num_boxes, 5 + num_classes)."""
        det = SimpleDetector(num_classes=3, num_boxes=4, input_channels=1, input_size=8)
        det.eval()
        x = torch.rand(2, 1, 8, 8)
        out = det(x)
        assert out.shape == (2, 4, 8)  # 5 + 3 = 8

    def test_objectness_bounded(self):
        """Objectness scores should be sigmoid-bounded in [0, 1]."""
        det = SimpleDetector(num_classes=3, num_boxes=4, input_channels=1, input_size=8)
        det.eval()
        x = torch.rand(4, 1, 8, 8)
        out = det(x)
        objectness = out[:, :, 4]
        assert objectness.min().item() >= 0.0
        assert objectness.max().item() <= 1.0


# ---------------------------------------------------------------------------
# SimpleSegmenter tests
# ---------------------------------------------------------------------------


class TestSimpleSegmenter:
    """Tests for the SimpleSegmenter model."""

    def test_output_shape(self):
        """Output should be (N, num_classes, H, W)."""
        seg = SimpleSegmenter(num_classes=3, input_channels=1)
        seg.eval()
        x = torch.rand(2, 1, 8, 8)
        out = seg(x)
        assert out.shape == (2, 3, 8, 8)


# ---------------------------------------------------------------------------
# SimpleRegressor tests
# ---------------------------------------------------------------------------


class TestSimpleRegressor:
    """Tests for the SimpleRegressor model."""

    def test_output_shape(self):
        """Output should be (N, 1)."""
        reg = SimpleRegressor(input_channels=1, input_size=8)
        reg.eval()
        x = torch.rand(4, 1, 8, 8)
        out = reg(x)
        assert out.shape == (4, 1)


# ---------------------------------------------------------------------------
# SimplePolicy tests
# ---------------------------------------------------------------------------


class TestSimplePolicy:
    """Tests for the SimplePolicy model."""

    def test_output_shape(self):
        """Output should be (N, num_actions)."""
        pol = SimplePolicy(state_dim=64, num_actions=4)
        pol.eval()
        x = torch.rand(4, 64)
        out = pol(x)
        assert out.shape == (4, 4)


# ---------------------------------------------------------------------------
# SimpleRecommender tests
# ---------------------------------------------------------------------------


class TestSimpleRecommender:
    """Tests for the SimpleRecommender model."""

    def test_output_shape_single_item(self):
        """Output should be (N,) for single item per user."""
        rec = SimpleRecommender(num_users=100, num_items=50, embedding_dim=16)
        rec.eval()
        users = torch.randint(0, 100, (4,))
        items = torch.randint(0, 50, (4,))
        out = rec(users, items)
        assert out.shape == (4,)

    def test_output_shape_multiple_items(self):
        """Output should be (N, K) for multiple items per user."""
        rec = SimpleRecommender(num_users=100, num_items=50, embedding_dim=16)
        rec.eval()
        users = torch.randint(0, 100, (4,))
        items = torch.randint(0, 50, (4, 5))
        out = rec(users, items)
        assert out.shape == (4, 5)


# ---------------------------------------------------------------------------
# object_detection_attack tests
# ---------------------------------------------------------------------------


class TestObjectDetectionAttack:
    """Tests for object_detection_attack."""

    def test_output_shape(self):
        """Output must match input shape."""
        det = SimpleDetector(num_classes=3, num_boxes=4, input_channels=1, input_size=8)
        det.eval()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        x_adv = object_detection_attack(
            det, x, y, mode="disappear", epsilon=0.05, steps=5
        )
        assert x_adv.shape == x.shape

    def test_output_in_valid_range(self):
        """Output must be clamped to [0, 1]."""
        det = SimpleDetector(num_classes=3, num_boxes=4, input_channels=1, input_size=8)
        det.eval()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        x_adv = object_detection_attack(
            det, x, y, mode="disappear", epsilon=0.1, steps=5
        )
        assert x_adv.min().item() >= 0.0
        assert x_adv.max().item() <= 1.0

    def test_disappear_mode(self):
        """Disappear mode should reduce objectness scores."""
        det = SimpleDetector(num_classes=3, num_boxes=4, input_channels=1, input_size=8)
        det.eval()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        x_adv = object_detection_attack(
            det, x, y, mode="disappear", epsilon=0.1, steps=20
        )
        # Output should be different from input
        assert not torch.allclose(x_adv, x)

    def test_misclassify_mode(self):
        """Misclassify mode should produce valid outputs."""
        det = SimpleDetector(num_classes=3, num_boxes=4, input_channels=1, input_size=8)
        det.eval()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        x_adv = object_detection_attack(
            det, x, y, mode="misclassify", epsilon=0.1, steps=10
        )
        assert x_adv.shape == x.shape

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        det = SimpleDetector(num_classes=3, num_boxes=4, input_channels=1, input_size=8)
        det.eval()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        with pytest.raises(ValueError):
            object_detection_attack(det, x, y, mode="invalid", epsilon=0.05, steps=5)

    def test_raises_on_train_mode(self):
        """Must raise if model is in training mode."""
        det = SimpleDetector(num_classes=3, num_boxes=4, input_channels=1, input_size=8)
        det.train()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        with pytest.raises(ValueError):
            object_detection_attack(det, x, y, mode="disappear", epsilon=0.05, steps=5)

    def test_perturbation_respects_epsilon(self):
        """L-inf perturbation must stay within epsilon."""
        det = SimpleDetector(num_classes=3, num_boxes=4, input_channels=1, input_size=8)
        det.eval()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        eps = 0.05
        x_adv = object_detection_attack(
            det, x, y, mode="disappear", epsilon=eps, steps=10
        )
        linf = (x_adv - x).abs().max().item()
        assert linf <= eps + 1e-6


# ---------------------------------------------------------------------------
# segmentation_attack tests
# ---------------------------------------------------------------------------


class TestSegmentationAttack:
    """Tests for segmentation_attack."""

    def test_output_shape(self):
        """Output must match input shape."""
        seg = SimpleSegmenter(num_classes=3, input_channels=1)
        seg.eval()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        x_adv = segmentation_attack(seg, x, y, epsilon=0.05, steps=5)
        assert x_adv.shape == x.shape

    def test_output_in_valid_range(self):
        """Output must be clamped to [0, 1]."""
        seg = SimpleSegmenter(num_classes=3, input_channels=1)
        seg.eval()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        x_adv = segmentation_attack(seg, x, y, epsilon=0.1, steps=5)
        assert x_adv.min().item() >= 0.0
        assert x_adv.max().item() <= 1.0

    def test_perturbation_respects_epsilon(self):
        """L-inf perturbation must stay within epsilon."""
        seg = SimpleSegmenter(num_classes=3, input_channels=1)
        seg.eval()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        eps = 0.05
        x_adv = segmentation_attack(seg, x, y, epsilon=eps, steps=10)
        linf = (x_adv - x).abs().max().item()
        assert linf <= eps + 1e-6

    def test_raises_on_train_mode(self):
        """Must raise if model is in training mode."""
        seg = SimpleSegmenter(num_classes=3, input_channels=1)
        seg.train()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        with pytest.raises(ValueError):
            segmentation_attack(seg, x, y, epsilon=0.05, steps=5)


# ---------------------------------------------------------------------------
# regression_attack tests
# ---------------------------------------------------------------------------


class TestRegressionAttack:
    """Tests for regression_attack."""

    def test_output_shape(self):
        """Output must match input shape."""
        reg = SimpleRegressor(input_channels=1, input_size=8)
        reg.eval()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        x_adv = regression_attack(reg, x, y, target_value=10.0, epsilon=0.05, steps=5)
        assert x_adv.shape == x.shape

    def test_output_in_valid_range(self):
        """Output must be clamped to [0, 1]."""
        reg = SimpleRegressor(input_channels=1, input_size=8)
        reg.eval()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        x_adv = regression_attack(reg, x, y, target_value=10.0, epsilon=0.1, steps=5)
        assert x_adv.min().item() >= 0.0
        assert x_adv.max().item() <= 1.0

    def test_perturbation_respects_epsilon(self):
        """L-inf perturbation must stay within epsilon."""
        reg = SimpleRegressor(input_channels=1, input_size=8)
        reg.eval()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        eps = 0.05
        x_adv = regression_attack(reg, x, y, target_value=5.0, epsilon=eps, steps=10)
        linf = (x_adv - x).abs().max().item()
        assert linf <= eps + 1e-6

    def test_raises_on_train_mode(self):
        """Must raise if model is in training mode."""
        reg = SimpleRegressor(input_channels=1, input_size=8)
        reg.train()
        x = torch.rand(4, 1, 8, 8)
        y = torch.zeros(4, dtype=torch.long)
        with pytest.raises(ValueError):
            regression_attack(reg, x, y, target_value=10.0, epsilon=0.05, steps=5)


# ---------------------------------------------------------------------------
# rl_attack tests
# ---------------------------------------------------------------------------


class TestRLAttack:
    """Tests for rl_attack."""

    def test_output_shape(self):
        """Output must match input shape."""
        pol = SimplePolicy(state_dim=64, num_actions=4)
        pol.eval()
        x = torch.rand(4, 64)
        y = torch.zeros(4, dtype=torch.long)
        x_adv = rl_attack(pol, x, y, epsilon=0.05, steps=5)
        assert x_adv.shape == x.shape

    def test_output_in_valid_range(self):
        """Output must be clamped to [0, 1]."""
        pol = SimplePolicy(state_dim=64, num_actions=4)
        pol.eval()
        x = torch.rand(4, 64)
        y = torch.zeros(4, dtype=torch.long)
        x_adv = rl_attack(pol, x, y, epsilon=0.1, steps=5)
        assert x_adv.min().item() >= 0.0
        assert x_adv.max().item() <= 1.0

    def test_perturbation_respects_epsilon(self):
        """L-inf perturbation must stay within epsilon."""
        pol = SimplePolicy(state_dim=64, num_actions=4)
        pol.eval()
        x = torch.rand(4, 64)
        y = torch.zeros(4, dtype=torch.long)
        eps = 0.05
        x_adv = rl_attack(pol, x, y, epsilon=eps, steps=10)
        linf = (x_adv - x).abs().max().item()
        assert linf <= eps + 1e-6

    def test_raises_on_train_mode(self):
        """Must raise if model is in training mode."""
        pol = SimplePolicy(state_dim=64, num_actions=4)
        pol.train()
        x = torch.rand(4, 64)
        y = torch.zeros(4, dtype=torch.long)
        with pytest.raises(ValueError):
            rl_attack(pol, x, y, epsilon=0.05, steps=5)

    def test_specific_optimal_action(self):
        """Specifying optimal_action_idx should produce valid output."""
        pol = SimplePolicy(state_dim=64, num_actions=4)
        pol.eval()
        x = torch.rand(4, 64)
        y = torch.zeros(4, dtype=torch.long)
        x_adv = rl_attack(pol, x, y, epsilon=0.05, steps=5, optimal_action_idx=2)
        assert x_adv.shape == x.shape


# ---------------------------------------------------------------------------
# recommendation_attack tests
# ---------------------------------------------------------------------------


class TestRecommendationAttack:
    """Tests for recommendation_attack."""

    def test_output_shape(self):
        """Output must match input shape."""
        # Use a simple linear model as proxy for recommendation scoring
        scorer = torch.nn.Sequential(
            torch.nn.Linear(16, 50),
        )
        scorer.eval()
        x = torch.rand(4, 16)
        y = torch.zeros(4, dtype=torch.long)
        x_adv = recommendation_attack(
            scorer, x, y, target_item_idx=5, num_items=50, epsilon=0.1, steps=5
        )
        assert x_adv.shape == x.shape

    def test_output_in_valid_range(self):
        """Output must be clamped to [0, 1]."""
        scorer = torch.nn.Sequential(
            torch.nn.Linear(16, 50),
        )
        scorer.eval()
        x = torch.rand(4, 16)
        y = torch.zeros(4, dtype=torch.long)
        x_adv = recommendation_attack(
            scorer, x, y, target_item_idx=3, num_items=50, epsilon=0.1, steps=5
        )
        assert x_adv.min().item() >= 0.0
        assert x_adv.max().item() <= 1.0

    def test_perturbation_respects_epsilon(self):
        """L-inf perturbation must stay within epsilon."""
        scorer = torch.nn.Sequential(
            torch.nn.Linear(16, 50),
        )
        scorer.eval()
        x = torch.rand(4, 16)
        y = torch.zeros(4, dtype=torch.long)
        eps = 0.1
        x_adv = recommendation_attack(
            scorer, x, y, target_item_idx=0, num_items=50, epsilon=eps, steps=10
        )
        linf = (x_adv - x).abs().max().item()
        assert linf <= eps + 1e-6

    def test_raises_on_train_mode(self):
        """Must raise if model is in training mode."""
        scorer = torch.nn.Sequential(
            torch.nn.Linear(16, 50),
        )
        scorer.train()
        x = torch.rand(4, 16)
        y = torch.zeros(4, dtype=torch.long)
        with pytest.raises(ValueError):
            recommendation_attack(
                scorer, x, y, target_item_idx=0, num_items=50, epsilon=0.1, steps=5
            )
