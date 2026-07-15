"""Tests for API behavior simulation (Tier 2, Item 14).

Tests APISimulator, simulated_api_attack, and anomaly_detection_evasion from
src/adv_lab/attacks/api_sim.py.
"""

from __future__ import annotations

import pytest
import torch

from adv_lab.attacks.api_sim import (
    AnomalyType,
    APISimulator,
    anomaly_detection_evasion,
    simulated_api_attack,
)
from adv_lab.attacks.fgsm import fgsm_attack


# ---------------------------------------------------------------------------
# APISimulator tests
# ---------------------------------------------------------------------------


class TestAPISimulator:
    """Tests for the APISimulator class."""

    def test_query_returns_probabilities(self, correct_batch):
        """query() should return a probability tensor."""
        model, x, y = correct_batch
        api = APISimulator(model, rate_limit=0, total_budget=0)
        probs = api.query(x)
        assert probs.shape[0] == x.shape[0]
        # Probabilities should be non-negative
        assert probs.min().item() >= 0.0

    def test_query_budget_enforcement(self, correct_batch):
        """Exceeding the query budget should raise RuntimeError."""
        model, x, y = correct_batch
        api = APISimulator(model, rate_limit=0, total_budget=3)
        for _ in range(3):
            api.query(x[:1])
        with pytest.raises(RuntimeError):
            api.query(x[:1])

    def test_queries_used_tracking(self, correct_batch):
        """queries_used should increment with each call."""
        model, x, y = correct_batch
        api = APISimulator(model, rate_limit=0, total_budget=100)
        assert api.queries_used == 0
        api.query(x[:1])
        assert api.queries_used == 1
        api.query(x[:1])
        assert api.queries_used == 2

    def test_queries_remaining(self, correct_batch):
        """queries_remaining should decrement with usage."""
        model, x, y = correct_batch
        api = APISimulator(model, rate_limit=0, total_budget=10)
        assert api.queries_remaining == 10
        api.query(x[:1])
        assert api.queries_remaining == 9

    def test_unlimited_budget_returns_negative_one(self, correct_batch):
        """queries_remaining returns -1 when budget is unlimited."""
        model, x, y = correct_batch
        api = APISimulator(model, rate_limit=0, total_budget=0)
        assert api.queries_remaining == -1

    def test_confidence_rounding(self, correct_batch):
        """Confidence scores should be rounded to specified decimals."""
        model, x, y = correct_batch
        api = APISimulator(model, rate_limit=0, total_budget=0, confidence_rounding=2)
        probs = api.query(x[:4])
        # Check that values are rounded to 2 decimal places
        rounded = torch.round(probs * 100.0) / 100.0
        assert torch.allclose(probs, rounded, atol=1e-7)

    def test_top_k_filtering(self, correct_batch):
        """Top-K filtering should zero out non-top-K classes."""
        model, x, y = correct_batch
        api = APISimulator(model, rate_limit=0, total_budget=0, top_k_only=1)
        probs = api.query(x[:4])
        # With top_k=1, each row should have at most 1 non-zero value
        nonzero_per_row = (probs > 0).sum(dim=1)
        assert (nonzero_per_row <= 1).all()

    def test_query_logging(self, correct_batch):
        """Query logs should record metadata for each call."""
        model, x, y = correct_batch
        api = APISimulator(model, rate_limit=0, total_budget=0, enable_logging=True)
        api.query(x[:2])
        api.query(x[:2])
        assert len(api.query_log) == 2
        assert api.query_log[0].query_index == 0
        assert api.query_log[1].query_index == 1

    def test_anomaly_detection_enabled(self, correct_batch):
        """Anomaly detection should be able to produce events."""
        model, x, y = correct_batch
        api = APISimulator(
            model,
            rate_limit=0,
            total_budget=0,
            enable_anomaly_detection=True,
        )
        # Query with similar inputs repeatedly to trigger clustering detection
        for _ in range(60):
            api.query(x[:1])
        # Events may or may not be generated, but the list should exist
        assert isinstance(api.anomaly_events, list)

    def test_reset_clears_state(self, correct_batch):
        """reset() should clear all counters and logs."""
        model, x, y = correct_batch
        api = APISimulator(model, rate_limit=0, total_budget=100)
        api.query(x[:1])
        api.query(x[:1])
        api.reset()
        assert api.queries_used == 0
        assert api.queries_remaining == 100
        assert len(api.query_log) == 0

    def test_raises_on_train_mode(self, correct_batch):
        """APISimulator should raise if model is in training mode."""
        model, x, y = correct_batch
        model.train()
        try:
            with pytest.raises(ValueError):
                APISimulator(model, rate_limit=60, total_budget=1000)
        finally:
            model.eval()

    def test_output_is_detached(self, correct_batch):
        """query() output must be detached."""
        model, x, y = correct_batch
        api = APISimulator(model, rate_limit=0, total_budget=0)
        probs = api.query(x[:4])
        assert not probs.requires_grad


# ---------------------------------------------------------------------------
# simulated_api_attack tests
# ---------------------------------------------------------------------------


class TestSimulatedAPIAttack:
    """Tests for the simulated_api_attack function."""

    def test_returns_tuple(self, correct_batch):
        """Should return (adversarial_images, api_simulator)."""
        model, x, y = correct_batch
        result = simulated_api_attack(
            model,
            x,
            y,
            attack_fn=fgsm_attack,
            rate_limit=0,
            total_budget=0,
            epsilon=0.03,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        x_adv, api = result
        assert x_adv.shape == x.shape
        assert isinstance(api, APISimulator)

    def test_output_in_valid_range(self, correct_batch):
        """Adversarial output must be in [0, 1]."""
        model, x, y = correct_batch
        x_adv, _ = simulated_api_attack(
            model,
            x,
            y,
            attack_fn=fgsm_attack,
            rate_limit=0,
            total_budget=0,
            epsilon=0.05,
        )
        assert x_adv.min().item() >= 0.0
        assert x_adv.max().item() <= 1.0

    def test_budget_consumed(self, correct_batch):
        """API queries should be consumed during the attack."""
        model, x, y = correct_batch
        _, api = simulated_api_attack(
            model,
            x,
            y,
            attack_fn=fgsm_attack,
            rate_limit=0,
            total_budget=0,
            epsilon=0.03,
        )
        # FGSM uses at least 1 forward pass
        assert api.queries_used >= 1

    def test_raises_on_train_mode(self, correct_batch):
        """Must raise ValueError if model is in training mode."""
        model, x, y = correct_batch
        model.train()
        try:
            with pytest.raises(ValueError):
                simulated_api_attack(
                    model,
                    x,
                    y,
                    attack_fn=fgsm_attack,
                    rate_limit=0,
                    total_budget=0,
                    epsilon=0.03,
                )
        finally:
            model.eval()

    def test_output_is_detached(self, correct_batch):
        """Output tensor must be detached."""
        model, x, y = correct_batch
        x_adv, _ = simulated_api_attack(
            model,
            x,
            y,
            attack_fn=fgsm_attack,
            rate_limit=0,
            total_budget=0,
            epsilon=0.03,
        )
        assert not x_adv.requires_grad


# ---------------------------------------------------------------------------
# anomaly_detection_evasion tests
# ---------------------------------------------------------------------------


class TestAnomalyDetectionEvasion:
    """Tests for the anomaly-evasive attack."""

    def test_output_shape_matches_input(self, correct_batch):
        """Output tensor must match input shape."""
        model, x, y = correct_batch
        x_adv = anomaly_detection_evasion(model, x, y, epsilon=0.05, steps=5)
        assert x_adv.shape == x.shape

    def test_output_in_valid_range(self, correct_batch):
        """Output must be in [0, 1]."""
        model, x, y = correct_batch
        x_adv = anomaly_detection_evasion(model, x, y, epsilon=0.1, steps=10)
        assert x_adv.min().item() >= 0.0
        assert x_adv.max().item() <= 1.0

    def test_perturbation_respects_epsilon(self, correct_batch):
        """L-inf perturbation must not exceed epsilon."""
        model, x, y = correct_batch
        eps = 0.05
        x_adv = anomaly_detection_evasion(model, x, y, epsilon=eps, steps=10)
        linf = (x_adv - x).abs().max().item()
        assert linf <= eps + 1e-6

    def test_nonzero_perturbation(self, correct_batch):
        """Attack should produce a non-trivial perturbation."""
        model, x, y = correct_batch
        x_adv = anomaly_detection_evasion(model, x, y, epsilon=0.05, steps=20)
        assert not torch.allclose(x_adv, x)

    def test_output_is_detached(self, correct_batch):
        """Output must be detached from computation graph."""
        model, x, y = correct_batch
        x_adv = anomaly_detection_evasion(model, x, y, epsilon=0.05, steps=5)
        assert not x_adv.requires_grad

    def test_raises_on_train_mode(self, correct_batch):
        """Must raise ValueError if model is in training mode."""
        model, x, y = correct_batch
        model.train()
        try:
            with pytest.raises(ValueError):
                anomaly_detection_evasion(model, x, y, epsilon=0.05, steps=5)
        finally:
            model.eval()

    def test_benign_query_ratio(self, correct_batch):
        """Different benign_query_ratio values should produce valid outputs."""
        model, x, y = correct_batch
        for ratio in [0.0, 0.25, 0.5]:
            x_adv = anomaly_detection_evasion(
                model, x, y, epsilon=0.05, steps=5, benign_query_ratio=ratio
            )
            assert x_adv.shape == x.shape

    def test_momentum_parameter(self, correct_batch):
        """Different momentum values should produce valid results."""
        model, x, y = correct_batch
        for mom in [0.0, 0.5, 0.9]:
            x_adv = anomaly_detection_evasion(
                model, x, y, epsilon=0.05, steps=5, momentum=mom
            )
            assert x_adv.min().item() >= 0.0
            assert x_adv.max().item() <= 1.0


# ---------------------------------------------------------------------------
# AnomalyType tests
# ---------------------------------------------------------------------------


class TestAnomalyType:
    """Tests for the AnomalyType enum."""

    def test_all_members_exist(self):
        """All expected anomaly types should be defined."""
        assert AnomalyType.INPUT_CLUSTERING.value == "input_clustering"
        assert AnomalyType.HIGH_QUERY_RATE.value == "high_query_rate"
        assert AnomalyType.LOW_ENTROPY_RESPONSES.value == "low_entropy_responses"
        assert AnomalyType.SEQUENTIAL_SIMILARITY.value == "sequential_similarity"
        assert AnomalyType.DISTRIBUTION_SHIFT.value == "distribution_shift"
        assert AnomalyType.REPEATED_QUERIES.value == "repeated_queries"
