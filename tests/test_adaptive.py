"""Tests for defense-aware adaptive attacks."""

from __future__ import annotations

import torch
import torch.nn as nn

from adv_lab.attacks.adaptive import (
    BPDA,
    EoT,
    GradientMaskingDetector,
    adaptive_attack,
    _random_search_with_momentum,
)


class _ConstLossModel(nn.Module):
    """Model that produces constant logits, simulating gradient masking."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return constant logits regardless of input -> zero gradients
        batch_size = x.shape[0]
        return torch.zeros(batch_size, 3, device=x.device)


def test_masking_detection_flat_loss():
    """GradientMaskingDetector triggers on constant (flat) loss."""
    detector = GradientMaskingDetector(expected_steps=100, plateau_window=5)
    # Feed constant loss values -> should trigger within 20% of steps
    for _ in range(10):
        detected = detector.record(1.5)
        if detected:
            break
    assert detector.detected
    assert detector.detection_step is not None
    assert detector.detection_step <= 20  # within 20% of expected_steps


def test_masking_detection_improving_loss():
    """GradientMaskingDetector does NOT trigger on improving loss."""
    detector = GradientMaskingDetector(expected_steps=100, plateau_window=5)
    for i in range(20):
        loss_val = 2.0 - 0.05 * i  # steadily decreasing
        detector.record(loss_val)
    assert not detector.detected


def test_bpda_produces_gradients():
    """BPDA enables gradient flow through a non-differentiable layer."""

    # Non-differentiable defense: threshold
    def hard_threshold(x: torch.Tensor) -> torch.Tensor:
        return (x > 0.5).float()

    # Differentiable approximation: sigmoid
    def soft_threshold(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(10 * (x - 0.5))

    bpda = BPDA(hard_threshold, soft_threshold)

    x = torch.rand(4, 1, 8, 8, requires_grad=True)
    out = bpda(x)

    # Should be able to compute gradients through BPDA
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0  # non-zero gradients


def test_eot_averages_over_transforms():
    """EoT computes average logits over multiple transforms."""

    model = nn.Linear(64, 3)
    model.eval()

    def random_noise_transform(x: torch.Tensor) -> torch.Tensor:
        return x + 0.01 * torch.randn_like(x)

    eot = EoT(model, random_noise_transform, n_samples=10)
    x = torch.rand(2, 64)
    logits = eot(x)
    assert logits.shape == (2, 3)


def test_adaptive_attack_strategy_switch_logged():
    """adaptive_attack logs strategy switches when masking is detected."""
    model = _ConstLossModel()
    model.eval()
    x = torch.rand(4, 1, 8, 8)
    y = torch.tensor([0, 1, 2, 0])

    x_adv, log = adaptive_attack(
        model, x, y, epsilon=0.03, steps=40, random_search_steps=10
    )

    # Should have detected masking and switched strategy
    event_types = [e["event_type"] for e in log.events]
    assert "masking_detected" in event_types
    assert "switch_strategy" in event_types
    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0


def test_adaptive_attack_with_bpda():
    """adaptive_attack uses BPDA when defense is provided and masking detected."""
    model = _ConstLossModel()
    model.eval()
    x = torch.rand(4, 1, 8, 8)
    y = torch.tensor([0, 1, 2, 0])

    def defense(x: torch.Tensor) -> torch.Tensor:
        return (x > 0.5).float()

    def defense_approx(x: torch.Tensor) -> torch.Tensor:
        return x  # identity approximation

    x_adv, log = adaptive_attack(
        model,
        x,
        y,
        epsilon=0.03,
        steps=40,
        defense=defense,
        defense_approx=defense_approx,
    )

    event_types = [e["event_type"] for e in log.events]
    assert "switch_strategy" in event_types
    # Verify BPDA was selected
    switch_events = [e for e in log.events if e["event_type"] == "switch_strategy"]
    assert any(e.get("to_strategy") == "bpda" for e in switch_events)


def test_random_search_with_momentum(correct_batch):
    """Random search produces valid adversarial examples."""
    model, x, y = correct_batch
    x_adv = _random_search_with_momentum(
        model, x, y, epsilon=0.1, steps=20, n_samples=5
    )
    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0
    # L-inf constraint
    linf = (x_adv - x).abs().max().item()
    assert linf <= 0.1 + 1e-5
