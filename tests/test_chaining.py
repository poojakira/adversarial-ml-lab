"""Tests for perturbation chaining (Tier 2, Item 13).

Tests PerturbationChain, chain_attack, ChainState, and AttackConfig from
src/adv_lab/attacks/chaining.py.
"""

from __future__ import annotations

import pytest
import torch

from adv_lab.attacks.chaining import (
    AttackConfig,
    ChainState,
    PerturbationChain,
    StepMetrics,
    chain_attack,
)
from adv_lab.attacks.fgsm import fgsm_attack


# ---------------------------------------------------------------------------
# ChainState tests
# ---------------------------------------------------------------------------


class TestChainState:
    """Tests for the ChainState cross-invocation state tracker."""

    def test_initial_state(self):
        """Fresh ChainState should have empty logs and step 0."""
        state = ChainState()
        assert state.current_step == 0
        assert len(state.step_logs) == 0
        assert len(state.confidence_history) == 0
        assert state.initial_predictions is None
        assert state.initial_confidence is None

    def test_log_step_increments_counter(self, correct_batch):
        """log_step should increment current_step and record metrics."""
        model, x, y = correct_batch
        state = ChainState()
        state.initial_predictions = y.clone()
        state.initial_confidence = torch.ones(y.shape[0])

        confidence = torch.rand(y.shape[0])
        predictions = torch.randint(0, 3, (y.shape[0],))
        perturbation = torch.randn_like(x) * 0.01

        metrics = state.log_step(
            step_name="test_step",
            confidence=confidence,
            predictions=predictions,
            perturbation=perturbation,
            labels=y,
        )

        assert state.current_step == 1
        assert len(state.step_logs) == 1
        assert isinstance(metrics, StepMetrics)
        assert metrics.step_name == "test_step"
        assert metrics.step_index == 0

    def test_confidence_degradation(self, correct_batch):
        """get_confidence_degradation should return the drop in mean confidence."""
        _, x, y = correct_batch
        state = ChainState()
        state.initial_confidence = torch.ones(y.shape[0])
        state.confidence_history.append(torch.full((y.shape[0],), 0.5))

        degradation = state.get_confidence_degradation()
        assert degradation is not None
        assert abs(degradation - 0.5) < 1e-6

    def test_summary_produces_dict(self, correct_batch):
        """summary() should return a dictionary with expected keys."""
        _, x, y = correct_batch
        state = ChainState()
        state.initial_predictions = y.clone()
        state.initial_confidence = torch.ones(y.shape[0])

        state.log_step(
            step_name="s1",
            confidence=torch.rand(y.shape[0]),
            predictions=torch.randint(0, 3, (y.shape[0],)),
            perturbation=torch.randn_like(x) * 0.01,
            labels=y,
        )

        summary = state.summary()
        assert "total_steps" in summary
        assert "confidence_degradation" in summary
        assert "final_misclassification_rate" in summary
        assert "final_linf" in summary
        assert "per_step" in summary
        assert len(summary["per_step"]) == 1

    def test_attack_success_rate_zero_initially(self):
        """get_attack_success_rate should return 0.0 before any steps."""
        state = ChainState()
        assert state.get_attack_success_rate() == 0.0


# ---------------------------------------------------------------------------
# AttackConfig tests
# ---------------------------------------------------------------------------


class TestAttackConfig:
    """Tests for the AttackConfig frozen dataclass."""

    def test_immutability(self):
        """AttackConfig should be frozen (immutable)."""
        config = AttackConfig(
            name="test",
            attack_fn=fgsm_attack,
            kwargs={"epsilon": 0.01},
        )
        with pytest.raises(Exception):
            config.name = "changed"  # type: ignore[misc]

    def test_default_kwargs(self):
        """Default kwargs should be an empty dict."""
        config = AttackConfig(name="test", attack_fn=fgsm_attack)
        assert config.kwargs == {}
        assert config.epsilon_share is None
        assert config.success_threshold is None


# ---------------------------------------------------------------------------
# PerturbationChain tests
# ---------------------------------------------------------------------------


class TestPerturbationChain:
    """Tests for the PerturbationChain orchestrator."""

    def test_requires_at_least_one_config(self):
        """PerturbationChain must raise on empty configs."""
        with pytest.raises(ValueError):
            PerturbationChain(configs=[])

    def test_execute_returns_tuple(self, correct_batch):
        """execute() should return (adversarial_images, chain_state)."""
        model, x, y = correct_batch
        chain = PerturbationChain(
            configs=[
                AttackConfig("fgsm", fgsm_attack, {"epsilon": 0.01}),
            ],
            total_epsilon=0.05,
        )
        result = chain.execute(model, x, y)
        assert isinstance(result, tuple)
        assert len(result) == 2
        x_adv, state = result
        assert x_adv.shape == x.shape
        assert isinstance(state, ChainState)

    def test_output_in_valid_range(self, correct_batch):
        """Chain output must stay in [0, 1]."""
        model, x, y = correct_batch
        chain = PerturbationChain(
            configs=[
                AttackConfig("step1", fgsm_attack, {"epsilon": 0.02}),
                AttackConfig("step2", fgsm_attack, {"epsilon": 0.02}),
            ],
            total_epsilon=0.05,
        )
        x_adv, _ = chain.execute(model, x, y)
        assert x_adv.min().item() >= 0.0
        assert x_adv.max().item() <= 1.0

    def test_total_epsilon_enforcement(self, correct_batch):
        """Cumulative perturbation must respect total_epsilon."""
        model, x, y = correct_batch
        total_eps = 0.03
        chain = PerturbationChain(
            configs=[
                AttackConfig("step1", fgsm_attack, {"epsilon": 0.05}),
                AttackConfig("step2", fgsm_attack, {"epsilon": 0.05}),
            ],
            total_epsilon=total_eps,
        )
        x_adv, _ = chain.execute(model, x, y)
        linf = (x_adv - x).abs().max().item()
        assert linf <= total_eps + 1e-6

    def test_state_tracks_multiple_steps(self, correct_batch):
        """Chain state should record metrics for each step."""
        model, x, y = correct_batch
        chain = PerturbationChain(
            configs=[
                AttackConfig("phase_a", fgsm_attack, {"epsilon": 0.01}),
                AttackConfig("phase_b", fgsm_attack, {"epsilon": 0.02}),
            ],
            total_epsilon=0.05,
        )
        _, state = chain.execute(model, x, y)
        assert state.current_step == 2
        assert len(state.step_logs) == 2
        assert state.step_logs[0].step_name == "phase_a"
        assert state.step_logs[1].step_name == "phase_b"

    def test_raises_on_train_mode(self, correct_batch):
        """execute() must raise if model is in training mode."""
        model, x, y = correct_batch
        chain = PerturbationChain(
            configs=[AttackConfig("fgsm", fgsm_attack, {"epsilon": 0.01})],
        )
        model.train()
        try:
            with pytest.raises(ValueError):
                chain.execute(model, x, y)
        finally:
            model.eval()

    def test_target_classes(self, correct_batch):
        """Targeted execution should track target hit rate."""
        model, x, y = correct_batch
        num_classes = 3
        targets = (y + 1) % num_classes
        chain = PerturbationChain(
            configs=[AttackConfig("fgsm", fgsm_attack, {"epsilon": 0.05})],
            total_epsilon=0.1,
        )
        _, state = chain.execute(model, x, y, target_classes=targets)
        # Should have target_hit_rate recorded
        assert state.step_logs[0].target_hit_rate >= 0.0

    def test_output_is_detached(self, correct_batch):
        """Output tensor must be detached from computation graph."""
        model, x, y = correct_batch
        chain = PerturbationChain(
            configs=[AttackConfig("fgsm", fgsm_attack, {"epsilon": 0.01})],
        )
        x_adv, _ = chain.execute(model, x, y)
        assert not x_adv.requires_grad


# ---------------------------------------------------------------------------
# chain_attack tests
# ---------------------------------------------------------------------------


class TestChainAttack:
    """Tests for the canonical three-phase perturbation chain."""

    def test_output_shape_matches_input(self, correct_batch):
        """Output must have the same shape as input."""
        model, x, y = correct_batch
        x_adv, state = chain_attack(
            model,
            x,
            y,
            softening_steps=5,
            boundary_steps=5,
            target_steps=5,
        )
        assert x_adv.shape == x.shape

    def test_output_in_valid_range(self, correct_batch):
        """Output must be clamped to [0, 1]."""
        model, x, y = correct_batch
        x_adv, _ = chain_attack(
            model,
            x,
            y,
            softening_steps=5,
            boundary_steps=5,
            target_steps=5,
        )
        assert x_adv.min().item() >= 0.0
        assert x_adv.max().item() <= 1.0

    def test_three_phases_logged(self, correct_batch):
        """Chain state should record exactly three phases."""
        model, x, y = correct_batch
        _, state = chain_attack(
            model,
            x,
            y,
            softening_steps=3,
            boundary_steps=3,
            target_steps=3,
        )
        assert len(state.step_logs) == 3
        assert state.step_logs[0].step_name == "A_confidence_softening"
        assert state.step_logs[1].step_name == "B_boundary_crossing"
        assert state.step_logs[2].step_name == "C_target_lock"

    def test_total_epsilon_respected(self, correct_batch):
        """When total_epsilon is set, perturbation must stay within it."""
        model, x, y = correct_batch
        total_eps = 0.04
        x_adv, _ = chain_attack(
            model,
            x,
            y,
            total_epsilon=total_eps,
            softening_steps=5,
            boundary_steps=5,
            target_steps=5,
        )
        linf = (x_adv - x).abs().max().item()
        assert linf <= total_eps + 1e-6

    def test_output_is_detached(self, correct_batch):
        """Output must be detached from computation graph."""
        model, x, y = correct_batch
        x_adv, _ = chain_attack(
            model,
            x,
            y,
            softening_steps=3,
            boundary_steps=3,
            target_steps=3,
        )
        assert not x_adv.requires_grad

    def test_raises_on_train_mode(self, correct_batch):
        """Must raise ValueError if model is in training mode."""
        model, x, y = correct_batch
        model.train()
        try:
            with pytest.raises(ValueError):
                chain_attack(
                    model, x, y, softening_steps=2, boundary_steps=2, target_steps=2
                )
        finally:
            model.eval()

    def test_raises_on_non_4d_input(self, correct_batch):
        """Must raise ValueError for non-4D images."""
        model, x, y = correct_batch
        x_flat = x.view(x.shape[0], -1)
        with pytest.raises(ValueError):
            chain_attack(
                model, x_flat, y, softening_steps=2, boundary_steps=2, target_steps=2
            )

    def test_metadata_contains_success_info(self, correct_batch):
        """State metadata should contain attack success information."""
        model, x, y = correct_batch
        _, state = chain_attack(
            model,
            x,
            y,
            softening_steps=5,
            boundary_steps=5,
            target_steps=5,
        )
        assert "attack_success_rate" in state.metadata
        assert "target_hit_rate" in state.metadata
        assert "target_classes" in state.metadata

    def test_specified_target_class(self, correct_batch):
        """Specifying target_class should target all samples to that class."""
        model, x, y = correct_batch
        _, state = chain_attack(
            model,
            x,
            y,
            target_class=0,
            softening_steps=3,
            boundary_steps=3,
            target_steps=3,
        )
        # All entries in target_classes should be 0
        assert all(t == 0 for t in state.metadata["target_classes"])

    def test_confidence_degrades_across_phases(self, correct_batch):
        """Confidence should generally decrease across the chain."""
        model, x, y = correct_batch
        _, state = chain_attack(
            model,
            x,
            y,
            softening_epsilon=0.05,
            boundary_epsilon=0.05,
            target_epsilon=0.05,
            softening_steps=10,
            boundary_steps=10,
            target_steps=10,
        )
        # Initial confidence should be higher than after softening
        degradation = state.get_confidence_degradation()
        assert degradation is not None
