"""Evaluation: the CI-gateable robustness benchmark harness, transferability analysis, certified defense evaluation, and CI signing."""

from adv_lab.eval.harness import (
    BenchmarkResult,
    DetailedBenchmark,
    export_json,
    run_benchmark,
    run_benchmark_batched,
)
from adv_lab.eval.robustbench_loader import (
    CIFAR10_LINF_EPSILON,
    iter_batches,
    load_cifar10,
    load_robustbench_model,
)
from adv_lab.eval.transferability import (
    TransferabilityAnalyzer,
    TransferabilityReport,
    TransferResult,
    create_heterogeneous_models,
)
from adv_lab.eval.certified import (
    IBPBounds,
    LipschitzNetwork,
    RandomizedSmoothing,
    SmoothingResult,
    find_certificate_boundary,
    ibp_eval,
    lipschitz_eval,
)
from adv_lab.eval.ci_signing import (
    create_signed_manifest,
    derive_key,
    detect_replay,
    log_input_hashes,
    sign_report,
    verify_report,
)

__all__ = [
    "BenchmarkResult",
    "run_benchmark",
    "run_benchmark_batched",
    "DetailedBenchmark",
    "export_json",
    # robustbench_loader module
    "load_robustbench_model",
    "load_cifar10",
    "iter_batches",
    "CIFAR10_LINF_EPSILON",
    "TransferabilityAnalyzer",
    "TransferabilityReport",
    "TransferResult",
    "create_heterogeneous_models",
    # certified module
    "RandomizedSmoothing",
    "SmoothingResult",
    "LipschitzNetwork",
    "lipschitz_eval",
    "ibp_eval",
    "IBPBounds",
    "find_certificate_boundary",
    # ci_signing module
    "sign_report",
    "verify_report",
    "log_input_hashes",
    "detect_replay",
    "derive_key",
    "create_signed_manifest",
]
