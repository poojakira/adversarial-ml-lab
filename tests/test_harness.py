"""Harness tests (2)."""

from __future__ import annotations

import json

from adv_lab.eval.harness import BenchmarkResult, export_json, run_benchmark


def _one_batch(x, y):
    yield x, y


def test_benchmark_result_json_valid(correct_batch, tmp_path):
    model, x, y = correct_batch
    result = run_benchmark(
        model,
        _one_batch(x, y),
        epsilon=0.03,
        n_samples=x.shape[0],
        cw_steps=50,
    )
    out = tmp_path / "report.json"
    export_json(result, str(out))

    payload = json.loads(out.read_text())
    # CI consumes these keys directly.
    assert "passed" in payload and isinstance(payload["passed"], bool)
    assert "robust_pgd" in payload
    assert 0.0 <= payload["robust_pgd"] <= 1.0
    assert "clean_accuracy" in payload


def test_benchmark_passed_flag_logic():
    # Above threshold (0.3) -> pass.
    passing = BenchmarkResult(
        model_name="m",
        clean_accuracy=0.9,
        robust_accuracy_fgsm=0.6,
        robust_accuracy_pgd=0.41,
        robust_accuracy_cw=0.2,
        epsilon=0.03,
    )
    assert passing.passed is True

    # At/below threshold -> fail.
    failing = BenchmarkResult(
        model_name="m",
        clean_accuracy=0.9,
        robust_accuracy_fgsm=0.4,
        robust_accuracy_pgd=0.1,
        robust_accuracy_cw=0.05,
        epsilon=0.03,
    )
    assert failing.passed is False
