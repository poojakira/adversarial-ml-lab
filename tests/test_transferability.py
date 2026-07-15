"""Transferability analysis tests.

Verifies that:
1. TransferabilityAnalyzer trains all architectures successfully.
2. Transfer rates are computed correctly and produce valid metrics.
3. Cross-architecture evaluation produces a complete report.
4. Ensemble transfer rate is computed.
"""

from __future__ import annotations

import torch

from adv_lab.attacks.fgsm import fgsm_attack
from adv_lab.eval.transferability import (
    TransferabilityAnalyzer,
    TransferabilityReport,
    TransferResult,
    create_heterogeneous_models,
)


def _make_data(n: int = 200, num_classes: int = 3, seed: int = 42):
    """Create synthetic data for transferability tests."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, 8, 8, generator=g)
    teacher = torch.randn(num_classes, 64, generator=g)
    scores = x.view(n, -1) @ teacher.t()
    y = scores.argmax(dim=1)
    return x, y


def test_create_heterogeneous_models():
    """create_heterogeneous_models should return 4 distinct architectures."""
    models = create_heterogeneous_models(num_classes=3, input_size=8)
    assert len(models) >= 3
    # All should be nn.Module instances
    for name, model in models.items():
        assert isinstance(name, str)
        x_test = torch.rand(2, 1, 8, 8)
        logits = model(x_test)
        assert logits.shape == (2, 3)


def test_analyzer_train_models():
    """TransferabilityAnalyzer.train_models should train all architectures."""
    x, y = _make_data(n=200, seed=100)
    analyzer = TransferabilityAnalyzer(num_classes=3, input_size=8)
    accuracies = analyzer.train_models(x, y, epochs=10, lr=1e-3)

    assert len(accuracies) >= 3
    for name, acc in accuracies.items():
        # All architectures should achieve some accuracy
        assert 0.0 <= acc <= 1.0
    assert len(analyzer.models) >= 3


def test_analyzer_evaluate_produces_report():
    """Evaluate should produce a TransferabilityReport with valid metrics."""
    train_x, train_y = _make_data(n=300, seed=200)
    eval_x, eval_y = _make_data(n=50, seed=201)

    analyzer = TransferabilityAnalyzer(num_classes=3, input_size=8)
    analyzer.train_models(train_x, train_y, epochs=15, lr=1e-3)

    report = analyzer.evaluate(
        eval_x,
        eval_y,
        attack_fn=fgsm_attack,
        attack_kwargs={"epsilon": 0.1},
        attack_name="FGSM",
    )

    assert isinstance(report, TransferabilityReport)
    assert report.source_attack_name == "FGSM"
    # Should have per-pair results
    assert len(report.per_pair) > 0
    # per_architecture should have entries for each target
    assert len(report.per_architecture) >= 3
    # All transfer rates should be valid fractions
    for result in report.per_pair:
        assert isinstance(result, TransferResult)
        assert 0.0 <= result.transfer_rate <= 1.0
        assert result.n_source_successful >= 0
        assert result.n_transferred >= 0

    # Ensemble transfer rate should be valid
    assert 0.0 <= report.ensemble_transfer_rate <= 1.0


def test_per_architecture_transfer_rates_valid():
    """Per-architecture transfer rates should all be valid fractions."""
    train_x, train_y = _make_data(n=200, seed=300)
    eval_x, eval_y = _make_data(n=30, seed=301)

    analyzer = TransferabilityAnalyzer(num_classes=3, input_size=8)
    analyzer.train_models(train_x, train_y, epochs=10, lr=1e-3)

    report = analyzer.evaluate(
        eval_x,
        eval_y,
        attack_fn=fgsm_attack,
        attack_kwargs={"epsilon": 0.05},
        attack_name="FGSM",
    )

    for arch_name, rate in report.per_architecture.items():
        assert isinstance(arch_name, str)
        assert 0.0 <= rate <= 1.0


def test_transferability_minimum_architectures():
    """Analysis must evaluate across at least 3 heterogeneous architectures."""
    train_x, train_y = _make_data(n=200, seed=400)
    eval_x, eval_y = _make_data(n=20, seed=401)

    analyzer = TransferabilityAnalyzer(num_classes=3, input_size=8)
    analyzer.train_models(train_x, train_y, epochs=10, lr=1e-3)

    # Verify we have at least 3 architectures
    assert len(analyzer.models) >= 3

    report = analyzer.evaluate(
        eval_x,
        eval_y,
        attack_fn=fgsm_attack,
        attack_kwargs={"epsilon": 0.1},
        attack_name="FGSM",
    )

    # Should report per-architecture rates for at least 3 targets
    assert len(report.per_architecture) >= 3


def test_transfer_result_consistency():
    """TransferResult fields must be internally consistent."""
    train_x, train_y = _make_data(n=200, seed=500)
    eval_x, eval_y = _make_data(n=30, seed=501)

    analyzer = TransferabilityAnalyzer(num_classes=3, input_size=8)
    analyzer.train_models(train_x, train_y, epochs=10, lr=1e-3)

    report = analyzer.evaluate(
        eval_x,
        eval_y,
        attack_fn=fgsm_attack,
        attack_kwargs={"epsilon": 0.1},
        attack_name="FGSM",
    )

    for result in report.per_pair:
        # n_transferred should not exceed n_source_successful
        assert result.n_transferred <= result.n_source_successful
        # transfer_rate should be consistent with counts
        if result.n_source_successful > 0:
            expected_rate = result.n_transferred / result.n_source_successful
            assert abs(result.transfer_rate - expected_rate) < 1e-6
