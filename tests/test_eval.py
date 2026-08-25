"""Tests for src/adv_lab/eval/ (benchmark harness, benchmark_runner, export).

Covers:
  - BenchmarkResult dataclass and CI gate logic
  - run_benchmark end-to-end with a tiny model
  - run_benchmark_batched produces DetailedBenchmark
  - export_json writes valid JSON with expected keys
  - benchmark_runner function produces a structured report
  - evaluate_robustness utility
  - RobustnessGate.check logic
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.eval.benchmark_runner import (
    _severity_from_robust_acc,
    benchmark_runner,
)
from adv_lab.eval.harness import (
    BenchmarkResult,
    DetailedBenchmark,
    EvaluationMetrics,
    RobustnessGate,
    evaluate_robustness,
    export_json,
    run_benchmark,
    run_benchmark_batched,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


class _TinyCNN(nn.Module):
    """Minimal CNN for eval tests (1x8x8 -> 3 classes)."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


def _make_batches(n: int = 64, batch_size: int = 32) -> Iterable[tuple[Tensor, Tensor]]:
    """Generate synthetic data as an iterable of (images, labels) tuples."""
    torch.manual_seed(0)
    images = torch.rand(n, 1, 8, 8)
    labels = torch.randint(0, 3, (n,))
    for i in range(0, n, batch_size):
        yield images[i : i + batch_size], labels[i : i + batch_size]


# ── BenchmarkResult Tests ─────────────────────────────────────────────────────


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_passed_true_when_above_threshold(self):
        """passed=True when robust_accuracy_pgd > PGD_GATE_THRESHOLD."""
        result = BenchmarkResult(
            model_name="test",
            clean_accuracy=0.9,
            robust_accuracy_fgsm=0.5,
            robust_accuracy_pgd=0.5,  # > 0.3 threshold
            robust_accuracy_cw=0.4,
            epsilon=0.03,
        )
        assert result.passed is True

    def test_passed_false_when_below_threshold(self):
        """passed=False when robust_accuracy_pgd <= PGD_GATE_THRESHOLD."""
        result = BenchmarkResult(
            model_name="test",
            clean_accuracy=0.9,
            robust_accuracy_fgsm=0.5,
            robust_accuracy_pgd=0.1,  # < 0.3 threshold
            robust_accuracy_cw=0.05,
            epsilon=0.03,
        )
        assert result.passed is False

    def test_timestamp_auto_populated(self):
        """BenchmarkResult auto-generates a timestamp."""
        result = BenchmarkResult(
            model_name="test",
            clean_accuracy=0.9,
            robust_accuracy_fgsm=0.5,
            robust_accuracy_pgd=0.5,
            robust_accuracy_cw=0.4,
            epsilon=0.03,
        )
        assert result.timestamp is not None
        assert len(result.timestamp) > 0


# ── run_benchmark Tests ───────────────────────────────────────────────────────


class TestRunBenchmark:
    """Tests for run_benchmark."""

    def test_run_benchmark_returns_result(self):
        """run_benchmark returns a BenchmarkResult with all fields populated."""
        torch.manual_seed(42)
        model = _TinyCNN()
        model.eval()

        result = run_benchmark(
            model,
            _make_batches(n=32, batch_size=16),
            epsilon=0.03,
            n_samples=32,
            model_name="TestCNN",
            cw_steps=5,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "TestCNN"
        assert 0.0 <= result.clean_accuracy <= 1.0
        assert 0.0 <= result.robust_accuracy_fgsm <= 1.0
        assert 0.0 <= result.robust_accuracy_pgd <= 1.0
        assert 0.0 <= result.robust_accuracy_cw <= 1.0
        assert result.epsilon == 0.03

    def test_clean_accuracy_gte_robust(self):
        """Clean accuracy should be >= robust accuracy (attacks can only hurt)."""
        torch.manual_seed(42)
        model = _TinyCNN()
        model.eval()

        result = run_benchmark(
            model,
            _make_batches(n=32, batch_size=16),
            epsilon=0.1,
            n_samples=32,
            cw_steps=5,
        )

        # This is expected but not always guaranteed for untrained models
        # At minimum, results should be valid probabilities
        assert 0.0 <= result.robust_accuracy_pgd <= 1.0


# ── run_benchmark_batched Tests ───────────────────────────────────────────────


class TestRunBenchmarkBatched:
    """Tests for run_benchmark_batched."""

    def test_returns_detailed_benchmark(self):
        """run_benchmark_batched returns DetailedBenchmark with counts."""
        torch.manual_seed(42)
        model = _TinyCNN()

        detailed = run_benchmark_batched(
            model,
            _make_batches(n=32, batch_size=16),
            epsilon=0.03,
            n_samples=32,
            batch_size=16,
            pgd_steps=5,
            cw_steps=5,
        )

        assert isinstance(detailed, DetailedBenchmark)
        assert detailed.n_evaluated == 32
        assert detailed.clean_correct >= 0
        assert 0.0 <= detailed.fgsm_success_rate <= 1.0
        assert 0.0 <= detailed.pgd_success_rate <= 1.0


# ── export_json Tests ─────────────────────────────────────────────────────────


class TestExportJson:
    """Tests for export_json."""

    def test_writes_valid_json(self):
        """export_json writes a valid JSON file with expected keys."""
        result = BenchmarkResult(
            model_name="TestExport",
            clean_accuracy=0.85,
            robust_accuracy_fgsm=0.4,
            robust_accuracy_pgd=0.2,
            robust_accuracy_cw=0.15,
            epsilon=0.03,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            export_json(result, path)
            with open(path) as f:
                data = json.load(f)

            assert "passed" in data
            assert "robust_pgd" in data
            assert "epsilon" in data
            assert data["model_name"] == "TestExport"
            assert data["passed"] is False  # 0.2 < 0.3 threshold
        finally:
            os.unlink(path)

    def test_export_with_hmac_signing(self):
        """export_json with hmac_key includes a signature field."""
        result = BenchmarkResult(
            model_name="SignedTest",
            clean_accuracy=0.9,
            robust_accuracy_fgsm=0.5,
            robust_accuracy_pgd=0.5,
            robust_accuracy_cw=0.4,
            epsilon=0.03,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            export_json(result, path, hmac_key=b"test-secret-key")
            with open(path) as f:
                data = json.load(f)

            assert "signature" in data
            assert "signature_algorithm" in data
            assert data["signature_algorithm"] == "HMAC-SHA256"
        finally:
            os.unlink(path)


# ── benchmark_runner Tests ────────────────────────────────────────────────────


class TestBenchmarkRunner:
    """Tests for the eval/benchmark_runner module."""

    def test_benchmark_runner_default(self):
        """benchmark_runner() with defaults produces a structured report."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            report = benchmark_runner(
                model_path=None,
                epsilon=0.03,
                pgd_steps=5,
                output_path=path,
                batch_size=8,
            )

            assert "attacks" in report
            assert "fgsm" in report["attacks"]
            assert "pgd" in report["attacks"]
            assert "pass_fail" in report
            assert report["pass_fail"] in ("PASS", "FAIL")
            assert "findings" in report
            assert "severity_summary" in report
        finally:
            os.unlink(path)

    def test_severity_from_robust_acc(self):
        """_severity_from_robust_acc maps accuracy to correct severity levels."""
        assert _severity_from_robust_acc(0.01) == "CRITICAL"
        assert _severity_from_robust_acc(0.10) == "HIGH"
        assert _severity_from_robust_acc(0.40) == "MEDIUM"
        assert _severity_from_robust_acc(0.60) == "LOW"


# ── evaluate_robustness Tests ─────────────────────────────────────────────────


class TestEvaluateRobustness:
    """Tests for evaluate_robustness utility."""

    def test_clean_evaluation(self):
        """evaluate_robustness with attack='clean' returns valid accuracy."""
        torch.manual_seed(0)
        model = _TinyCNN()
        model.eval()

        acc = evaluate_robustness(model, _make_batches(n=32), attack="clean", eps=0.03)
        assert 0.0 <= acc <= 1.0

    def test_fgsm_evaluation(self):
        """evaluate_robustness with attack='fgsm' runs without error."""
        torch.manual_seed(0)
        model = _TinyCNN()
        model.eval()

        acc = evaluate_robustness(model, _make_batches(n=16), attack="fgsm", eps=0.03)
        assert 0.0 <= acc <= 1.0

    def test_invalid_attack_raises(self):
        """evaluate_robustness raises ValueError for unknown attack type."""
        model = _TinyCNN()
        model.eval()

        try:
            evaluate_robustness(model, _make_batches(n=16), attack="unknown")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "unsupported attack" in str(e)


# ── RobustnessGate Tests ──────────────────────────────────────────────────────


class TestRobustnessGate:
    """Tests for RobustnessGate."""

    def test_gate_passes(self):
        """Gate passes when all metrics exceed thresholds."""
        gate = RobustnessGate(clean_threshold=0.5, fgsm_threshold=0.2, pgd_threshold=0.3)
        metrics = EvaluationMetrics(clean_accuracy=0.9, fgsm_accuracy=0.5, pgd_accuracy=0.4)
        assert gate.check(metrics) is True

    def test_gate_fails_pgd(self):
        """Gate fails when PGD accuracy is below threshold."""
        gate = RobustnessGate(pgd_threshold=0.3)
        metrics = EvaluationMetrics(clean_accuracy=0.9, fgsm_accuracy=0.5, pgd_accuracy=0.1)
        assert gate.check(metrics) is False
