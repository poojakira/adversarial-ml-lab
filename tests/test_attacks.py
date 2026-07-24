"""Tests for adversarial attacks and evaluation harness."""

import torch
import torch.nn as nn

from adv_lab.attacks.cw import generate as cw_generate
from adv_lab.attacks.fgsm import generate as fgsm_generate
from adv_lab.attacks.pgd import generate as pgd_generate
from adv_lab.eval.harness import evaluate_robustness, RobustnessGate, run_evaluation


class DummyModel(nn.Module):
    """Simple CNN for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_fgsm_attack() -> None:
    """Test FGSM attack generation."""
    model = DummyModel()
    model.eval()
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 0, 1])

    adv_x = fgsm_generate(model, x, y, epsilon=0.1)
    assert adv_x.shape == x.shape
    assert not torch.allclose(adv_x, x)
    assert adv_x.max() <= 1.0 and adv_x.min() >= 0.0


def test_pgd_attack() -> None:
    """Test PGD attack generation."""
    model = DummyModel()
    model.eval()
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 0, 1])

    adv_x = pgd_generate(model, x, y, epsilon=0.1, alpha=0.01, steps=5)
    assert adv_x.shape == x.shape
    assert not torch.allclose(adv_x, x)
    assert adv_x.max() <= 1.0 and adv_x.min() >= 0.0


def test_cw_attack() -> None:
    """Test C&W L2 attack generation."""
    model = DummyModel()
    model.eval()
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    adv_x = cw_generate(model, x, y, confidence=0.0, steps=5, learning_rate=0.01)
    assert adv_x.shape == x.shape
    assert not torch.allclose(adv_x, x)
    assert adv_x.max() <= 1.0 and adv_x.min() >= 0.0


def test_eval_harness_clean() -> None:
    """Test evaluation harness on clean data."""
    model = DummyModel()
    model.eval()
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 0, 1])
    dataloader = [(x, y)]

    acc = evaluate_robustness(model, dataloader, attack="clean")
    assert 0.0 <= acc <= 1.0


def test_eval_harness_fgsm() -> None:
    """Test evaluation harness with FGSM."""
    model = DummyModel()
    model.eval()
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 0, 1])
    dataloader = [(x, y)]

    acc = evaluate_robustness(model, dataloader, attack="fgsm", eps=0.1)
    assert 0.0 <= acc <= 1.0


def test_eval_harness_pgd() -> None:
    """Test evaluation harness with PGD."""
    model = DummyModel()
    model.eval()
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 0, 1])
    dataloader = [(x, y)]

    acc = evaluate_robustness(model, dataloader, attack="pgd", eps=0.1, steps=5)
    assert 0.0 <= acc <= 1.0


def test_robustness_gate_pass() -> None:
    """Test RobustnessGate passes when thresholds met."""
    from adv_lab.eval.harness import EvaluationMetrics

    gate = RobustnessGate(clean_threshold=0.5, fgsm_threshold=0.3, pgd_threshold=0.2)
    metrics = EvaluationMetrics(clean_accuracy=0.9, fgsm_accuracy=0.8, pgd_accuracy=0.7)
    assert gate.check(metrics) is True


def test_robustness_gate_fail() -> None:
    """Test RobustnessGate fails when thresholds not met."""
    from adv_lab.eval.harness import EvaluationMetrics

    gate = RobustnessGate(clean_threshold=0.95, fgsm_threshold=0.5, pgd_threshold=0.3)
    metrics = EvaluationMetrics(clean_accuracy=0.9, fgsm_accuracy=0.8, pgd_accuracy=0.7)
    assert gate.check(metrics) is False


def test_model_training_mode_raises() -> None:
    """Test that training mode raises RuntimeError."""
    model = DummyModel()
    model.train()  # Training mode
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 0, 1])
    dataloader = [(x, y)]

    try:
        evaluate_robustness(model, dataloader, attack="clean")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "eval() mode" in str(e)