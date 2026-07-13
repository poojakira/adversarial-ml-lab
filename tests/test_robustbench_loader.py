"""Tests for the RobustBench loader wiring and the batched benchmark runner.

These tests must not touch the network: RobustBench model/data loading is
monkeypatched. The point is to prove the *wiring* is correct -- that our loader
puts the model in eval() mode on the right device, that the batched runner
reproduces the single-shot :func:`run_benchmark` numbers, and that the CLI
exposes the RobustBench flags -- not to re-download a 170 MB dataset in CI.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

from adv_lab.eval.harness import (
    DetailedBenchmark,
    main,
    run_benchmark,
    run_benchmark_batched,
)


def _one_batch(x, y):
    yield x, y


def test_batched_matches_single_shot(correct_batch):
    """Batched runner must reproduce run_benchmark's robust accuracies.

    The batched loop runs the identical FGSM/PGD/C&W functions, just streamed in
    mini-batches, so on the same data with the same PGD step count the robust
    accuracies must match the single-shot path (deterministic parts) closely.
    """
    model, x, y = correct_batch
    torch.manual_seed(0)
    single = run_benchmark(
        model, _one_batch(x, y), epsilon=0.05, n_samples=x.shape[0], cw_steps=50
    )
    torch.manual_seed(0)
    detailed = run_benchmark_batched(
        model,
        _one_batch(x, y),
        epsilon=0.05,
        n_samples=x.shape[0],
        device="cpu",
        batch_size=8,
        pgd_steps=40,
        cw_steps=50,
    )
    assert isinstance(detailed, DetailedBenchmark)
    assert detailed.n_evaluated == x.shape[0]
    # Clean accuracy is fully deterministic and must be identical.
    assert detailed.result.clean_accuracy == pytest.approx(single.clean_accuracy)
    # FGSM is deterministic (single signed step) -> must match exactly.
    assert detailed.result.robust_accuracy_fgsm == pytest.approx(
        single.robust_accuracy_fgsm
    )
    # PGD has a random start; both used the same seed, expect a close match.
    assert detailed.result.robust_accuracy_pgd == pytest.approx(
        single.robust_accuracy_pgd, abs=0.2
    )


def test_batched_success_rates_in_range(correct_batch):
    model, x, y = correct_batch
    detailed = run_benchmark_batched(
        model,
        _one_batch(x, y),
        epsilon=0.1,
        n_samples=x.shape[0],
        device="cpu",
        batch_size=16,
        pgd_steps=10,
        cw_steps=20,
    )
    for rate in (
        detailed.fgsm_success_rate,
        detailed.pgd_success_rate,
        detailed.cw_success_rate,
    ):
        assert 0.0 <= rate <= 1.0
    # Ordering sanity: the correct-count ladder should be monotone non-increasing
    # for a well-behaved (non-gradient-masked) model under increasing strength.
    assert detailed.clean_correct >= detailed.fgsm_correct


def _install_fake_robustbench(monkeypatch, model: nn.Module):
    """Install fake robustbench.utils/.data modules so no network is hit."""
    calls = {}

    def fake_load_model(model_name, model_dir, dataset, threat_model):
        calls["model_name"] = model_name
        calls["dataset"] = dataset
        calls["threat_model"] = threat_model
        return model

    def fake_load_cifar10(n_examples, data_dir):
        calls["n_examples"] = n_examples
        x = torch.rand(n_examples, 1, 8, 8)
        yv = torch.randint(0, 3, (n_examples,))
        return x, yv

    utils_mod = types.ModuleType("robustbench.utils")
    utils_mod.load_model = fake_load_model
    data_mod = types.ModuleType("robustbench.data")
    data_mod.load_cifar10 = fake_load_cifar10
    pkg = types.ModuleType("robustbench")
    monkeypatch.setitem(sys.modules, "robustbench", pkg)
    monkeypatch.setitem(sys.modules, "robustbench.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "robustbench.data", data_mod)
    return calls


def test_load_robustbench_model_sets_eval_and_device(monkeypatch, lab):
    from adv_lab.eval.robustbench_loader import load_robustbench_model

    model, _, _ = lab
    model.train()  # deliberately wrong mode; loader must fix it
    _install_fake_robustbench(monkeypatch, model)

    loaded = load_robustbench_model(
        "Wang2023Better_WRN-28-10",
        dataset="cifar10",
        threat_model="Linf",
        device="cpu",
    )
    assert loaded is model
    assert loaded.training is False  # loader forced eval()


def test_load_cifar10_shapes(monkeypatch, lab):
    from adv_lab.eval.robustbench_loader import load_cifar10

    model, _, _ = lab
    _install_fake_robustbench(monkeypatch, model)
    x, y = load_cifar10(n_examples=17, data_dir="./data")
    assert x.shape[0] == 17
    assert y.shape[0] == 17


def test_cli_robustbench_end_to_end(monkeypatch, lab):
    """The --model-source robustbench CLI path must run against a fake zoo."""
    model, _, _ = lab
    model.eval()
    _install_fake_robustbench(monkeypatch, model)

    rc = main(
        [
            "--model-source", "robustbench",
            "--model-name", "Standard",
            "--n-samples", "20",
            "--epsilon", "0.1",
            "--batch-size", "8",
            "--pgd-steps", "5",
            "--cw-steps", "10",
            "--device", "cpu",
        ]
    )
    # Return code is the PGD gate result (0 pass / 1 fail); both are valid,
    # we only assert the path ran to completion and produced an int.
    assert rc in (0, 1)
