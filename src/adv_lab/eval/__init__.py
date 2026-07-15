"""Lazy exports for evaluation helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "BenchmarkResult": "adv_lab.eval.harness",
    "DetailedBenchmark": "adv_lab.eval.harness",
    "export_json": "adv_lab.eval.harness",
    "run_benchmark": "adv_lab.eval.harness",
    "run_benchmark_batched": "adv_lab.eval.harness",
    "load_robustbench_model": "adv_lab.eval.robustbench_loader",
    "load_cifar10": "adv_lab.eval.robustbench_loader",
    "iter_batches": "adv_lab.eval.robustbench_loader",
    "CIFAR10_LINF_EPSILON": "adv_lab.eval.robustbench_loader",
    "TransferabilityAnalyzer": "adv_lab.eval.transferability",
    "TransferabilityReport": "adv_lab.eval.transferability",
    "TransferResult": "adv_lab.eval.transferability",
    "create_heterogeneous_models": "adv_lab.eval.transferability",
    "RandomizedSmoothing": "adv_lab.eval.certified",
    "SmoothingResult": "adv_lab.eval.certified",
    "LipschitzNetwork": "adv_lab.eval.certified",
    "lipschitz_eval": "adv_lab.eval.certified",
    "ibp_eval": "adv_lab.eval.certified",
    "IBPBounds": "adv_lab.eval.certified",
    "find_certificate_boundary": "adv_lab.eval.certified",
    "sign_report": "adv_lab.eval.ci_signing",
    "verify_report": "adv_lab.eval.ci_signing",
    "log_input_hashes": "adv_lab.eval.ci_signing",
    "detect_replay": "adv_lab.eval.ci_signing",
    "derive_key": "adv_lab.eval.ci_signing",
    "create_signed_manifest": "adv_lab.eval.ci_signing",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(name)
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
