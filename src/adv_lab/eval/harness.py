"""Benchmark harness with CI-gateable JSON output.

The point of this module: turn "is my classifier robust?" from a vibe into a
number that a CI job can pass or fail on. It runs the full attack ladder
(FGSM < PGD < C&W) against a fixed sample of the data, records robust accuracy
under each, and stamps a ``passed`` flag based on a PGD threshold.

    passed = robust_accuracy_pgd > 0.3

PGD is chosen as the gate (not FGSM, not C&W) deliberately: FGSM is too weak
and over-reports robustness (gradient-masking risk), while C&W is a slow
optimization better suited to offline audits than a per-commit gate. PGD is the
honest, reproducible middle -- strong enough to be meaningful, fast enough for CI.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.cw import cw_l2_attack
from adv_lab.attacks.fgsm import fgsm_attack
from adv_lab.attacks.pgd import pgd_attack

# CI gate: a model must keep more than this fraction correct under PGD to pass.
PGD_GATE_THRESHOLD = 0.3


@dataclass
class BenchmarkResult:
    """A single robustness benchmark run.

    ``passed`` is derived from :data:`PGD_GATE_THRESHOLD` and is what a CI job
    keys off of.
    """

    model_name: str
    clean_accuracy: float
    robust_accuracy_fgsm: float
    robust_accuracy_pgd: float
    robust_accuracy_cw: float
    epsilon: float
    timestamp: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    passed: bool = False

    def __post_init__(self) -> None:
        # Keep passed consistent with the PGD robust accuracy at all times.
        self.passed = self.robust_accuracy_pgd > PGD_GATE_THRESHOLD


def _take_samples(
    dataloader: Iterable[tuple[Tensor, Tensor]], n_samples: int
) -> tuple[Tensor, Tensor]:
    """Collect up to ``n_samples`` examples from a dataloader into one batch."""
    xs: list[Tensor] = []
    ys: list[Tensor] = []
    collected = 0
    for images, labels in dataloader:
        remaining = n_samples - collected
        if remaining <= 0:
            break
        xs.append(images[:remaining])
        ys.append(labels[:remaining])
        collected += min(remaining, images.shape[0])
    if not xs:
        raise ValueError("dataloader yielded no samples")
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _accuracy(model: nn.Module, x: Tensor, y: Tensor) -> float:
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    return float((pred == y).float().mean().item())


def run_benchmark(
    model: nn.Module,
    dataloader: Iterable[tuple[Tensor, Tensor]],
    epsilon: float = 0.03,
    n_samples: int = 500,
    model_name: str | None = None,
    cw_steps: int = 200,
    cw_c: float = 1.0,
) -> BenchmarkResult:
    """Run the FGSM/PGD/C&W ladder and return a :class:`BenchmarkResult`.

    Robust accuracy for each attack is the fraction of the sampled set that the
    model still classifies correctly after the perturbation. By construction of
    the attack strengths we expect::

        clean >= robust_fgsm >= robust_pgd >= robust_cw

    Args:
        model: classifier; forced into ``eval()`` for the duration.
        dataloader: yields ``(images, labels)`` batches in ``[0, 1]``.
        epsilon: L-inf budget for FGSM and PGD.
        n_samples: number of examples to evaluate on.
        model_name: label for the report; defaults to the model's class name.
        cw_steps: iterations for the (slower) C&W attack in the benchmark.
        cw_c: C&W trade-off constant used in the benchmark.

    Returns:
        A populated :class:`BenchmarkResult` with ``passed`` set by the gate.
    """
    was_training = model.training
    model.eval()
    try:
        x, y = _take_samples(dataloader, n_samples)
        name = model_name or model.__class__.__name__

        clean_acc = _accuracy(model, x, y)

        x_fgsm = fgsm_attack(model, x, y, epsilon=epsilon)
        robust_fgsm = _accuracy(model, x_fgsm, y)

        x_pgd = pgd_attack(model, x, y, epsilon=epsilon, alpha=epsilon / 4.0, steps=40)
        robust_pgd = _accuracy(model, x_pgd, y)

        x_cw = cw_l2_attack(model, x, y, c=cw_c, steps=cw_steps)
        robust_cw = _accuracy(model, x_cw, y)
    finally:
        if was_training:
            model.train()

    return BenchmarkResult(
        model_name=name,
        clean_accuracy=clean_acc,
        robust_accuracy_fgsm=robust_fgsm,
        robust_accuracy_pgd=robust_pgd,
        robust_accuracy_cw=robust_cw,
        epsilon=epsilon,
    )


@dataclass
class DetailedBenchmark:
    """A benchmark run enriched with per-attack success rates and raw counts.

    :attr:`result` is the canonical :class:`BenchmarkResult` a CI gate keys off
    of; the remaining fields expose what that summary hides -- how many of the
    *originally-correct* predictions each attack actually flipped (the honest
    definition of "attack success"), plus the raw counts behind every rate so a
    reviewer can recompute the numbers.
    """

    result: BenchmarkResult
    n_evaluated: int
    clean_correct: int
    fgsm_correct: int
    pgd_correct: int
    cw_correct: int
    fgsm_success_rate: float
    pgd_success_rate: float
    cw_success_rate: float
    batch_size: int
    device: str
    pgd_steps: int


def run_benchmark_batched(
    model: nn.Module,
    dataloader: Iterable[tuple[Tensor, Tensor]],
    epsilon: float = 0.03,
    n_samples: int = 1000,
    model_name: str | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 100,
    pgd_steps: int = 40,
    cw_steps: int = 100,
    cw_c: float = 1.0,
) -> DetailedBenchmark:
    """Memory-safe, device-aware sibling of :func:`run_benchmark`.

    :func:`run_benchmark` collects the whole evaluation set into a single batch
    and attacks it in one shot. That is fine for the tiny synthetic demo, but it
    OOMs the moment you point it at a real model like WRN-28-10 on a 1000-image
    CIFAR-10 subset with an 8 GB GPU. This variant streams the data through the
    model in mini-batches, running the identical FGSM/PGD/C&W attack functions
    per batch and accumulating counts, so the numbers are directly comparable to
    the RobustBench leaderboard without blowing the VRAM budget.

    "Robust accuracy" is the fraction of the whole evaluated set still classified
    correctly after the attack. "Attack success rate" is the fraction of the
    *originally-correct* predictions the attack managed to flip -- fooling an
    already-wrong prediction is not counted as a success.

    Args:
        model: classifier; forced into ``eval()`` for the duration.
        dataloader: yields ``(images, labels)`` batches in ``[0, 1]``.
        epsilon: L-inf budget for FGSM and PGD (RobustBench CIFAR-10 uses 8/255).
        n_samples: number of examples to evaluate on.
        model_name: label for the report; defaults to the model's class name.
        device: device to run on; defaults to CUDA when available, else CPU.
        batch_size: mini-batch size for streaming (keep small to fit VRAM).
        pgd_steps: PGD iterations (RobustBench-style evaluations use >=40).
        cw_steps: C&W Adam iterations (kept modest so the ladder stays tractable
            on n>=1000 real images).
        cw_c: C&W trade-off constant.

    Returns:
        A :class:`DetailedBenchmark` wrapping a populated :class:`BenchmarkResult`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    was_training = model.training
    model.eval()
    model.to(device)
    try:
        x, y = _take_samples(dataloader, n_samples)
        name = model_name or model.__class__.__name__
        n_eval = x.shape[0]

        clean_correct = 0
        fgsm_correct = 0
        pgd_correct = 0
        cw_correct = 0
        # Attack success is measured only on originally-correct predictions.
        clean_correct_for_success = 0
        fgsm_flipped = 0
        pgd_flipped = 0
        cw_flipped = 0

        for i in range(0, n_eval, batch_size):
            xb = x[i : i + batch_size].to(device)
            yb = y[i : i + batch_size].to(device)

            with torch.no_grad():
                clean_pred = model(xb).argmax(dim=1)
            correct_mask = clean_pred == yb
            clean_correct += int(correct_mask.sum().item())
            clean_correct_for_success += int(correct_mask.sum().item())

            xb_fgsm = fgsm_attack(model, xb, yb, epsilon=epsilon)
            with torch.no_grad():
                fgsm_pred = model(xb_fgsm).argmax(dim=1)
            fgsm_correct += int((fgsm_pred == yb).sum().item())
            fgsm_flipped += int((correct_mask & (fgsm_pred != yb)).sum().item())

            xb_pgd = pgd_attack(
                model, xb, yb, epsilon=epsilon, alpha=epsilon / 4.0, steps=pgd_steps
            )
            with torch.no_grad():
                pgd_pred = model(xb_pgd).argmax(dim=1)
            pgd_correct += int((pgd_pred == yb).sum().item())
            pgd_flipped += int((correct_mask & (pgd_pred != yb)).sum().item())

            xb_cw = cw_l2_attack(model, xb, yb, c=cw_c, steps=cw_steps)
            with torch.no_grad():
                cw_pred = model(xb_cw).argmax(dim=1)
            cw_correct += int((cw_pred == yb).sum().item())
            cw_flipped += int((correct_mask & (cw_pred != yb)).sum().item())
    finally:
        if was_training:
            model.train()

    clean_acc = clean_correct / n_eval if n_eval else 0.0
    result = BenchmarkResult(
        model_name=name,
        clean_accuracy=clean_acc,
        robust_accuracy_fgsm=fgsm_correct / n_eval if n_eval else 0.0,
        robust_accuracy_pgd=pgd_correct / n_eval if n_eval else 0.0,
        robust_accuracy_cw=cw_correct / n_eval if n_eval else 0.0,
        epsilon=epsilon,
    )

    denom = clean_correct_for_success or 1
    return DetailedBenchmark(
        result=result,
        n_evaluated=n_eval,
        clean_correct=clean_correct,
        fgsm_correct=fgsm_correct,
        pgd_correct=pgd_correct,
        cw_correct=cw_correct,
        fgsm_success_rate=fgsm_flipped / denom,
        pgd_success_rate=pgd_flipped / denom,
        cw_success_rate=cw_flipped / denom,
        batch_size=batch_size,
        device=str(device),
        pgd_steps=pgd_steps,
    )


def export_json(
    result: BenchmarkResult,
    path: str,
    hmac_key: bytes | None = None,
) -> None:
    """Write a CI-consumable JSON report, optionally HMAC-signed.

    The payload includes short keys a CI job can grep (``passed``,
    ``robust_pgd``) alongside the full record for humans.

    If ``hmac_key`` is provided, the output JSON includes an HMAC-SHA256
    signature over the canonical payload, making the report tamper-evident.
    An unsigned CI gate is a broken CI gate -- use signing in production.

    Args:
        result: The benchmark result to export.
        path: File path for the JSON output.
        hmac_key: Optional HMAC signing key. If None (default), the report
            is written unsigned for backward compatibility. If provided,
            a ``signature`` field is added using HMAC-SHA256.
    """
    import hashlib as _hashlib
    import hmac as _hmac
    import os

    payload = {
        "passed": result.passed,
        "robust_pgd": result.robust_accuracy_pgd,
        "robust_fgsm": result.robust_accuracy_fgsm,
        "robust_cw": result.robust_accuracy_cw,
        "clean_accuracy": result.clean_accuracy,
        "epsilon": result.epsilon,
        "gate_threshold": PGD_GATE_THRESHOLD,
        "model_name": result.model_name,
        "timestamp": result.timestamp,
        "detail": asdict(result),
    }

    if hmac_key is not None:
        # Canonicalize payload for signing
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        signature = _hmac.new(
            hmac_key, canonical.encode("utf-8"), _hashlib.sha256
        ).hexdigest()
        payload["signature"] = signature
        payload["signature_algorithm"] = "HMAC-SHA256"

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


# --------------------------------------------------------------------------- #
# CLI: a self-contained demo so `adv-eval` / `py -m adv_lab.eval.harness` runs
# end-to-end with no external dataset download. It trains a small CNN on a
# synthetic-but-learnable image task, then benchmarks it. An undefended model
# typically FAILS the PGD gate -- which is exactly the point.
# --------------------------------------------------------------------------- #


class _SmallCNN(nn.Module):
    """Tiny CNN for the CLI demo (1x8x8 inputs, configurable #classes)."""

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
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


def _make_synthetic_dataset(
    n: int, num_classes: int, seed: int
) -> tuple[Tensor, Tensor]:
    """Generate a learnable image classification task in ``[0, 1]``.

    Labels come from a fixed random linear teacher over the pixels, so a small
    CNN can reach high clean accuracy quickly and the robustness drop is stark.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, 8, 8, generator=g)
    teacher = torch.randn(num_classes, 64, generator=g)
    scores = x.view(n, -1) @ teacher.t()
    y = scores.argmax(dim=1)
    return x, y


def _iter_batches(
    x: Tensor, y: Tensor, batch_size: int
) -> Iterable[tuple[Tensor, Tensor]]:
    for i in range(0, x.shape[0], batch_size):
        yield x[i : i + batch_size], y[i : i + batch_size]


def _train_demo_model(
    model: nn.Module, x: Tensor, y: Tensor, epochs: int, lr: float
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in _iter_batches(x, y, 128):
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()


def _run_robustbench(args) -> int:
    """Evaluate a pretrained RobustBench model on real CIFAR-10.

    This is the leaderboard-comparable path: load a zoo model + the fixed
    RobustBench CIFAR-10 subset, run the FGSM/PGD/C&W ladder in a VRAM-safe
    batched loop, print a summary, and (optionally) write a JSON report that
    also carries per-attack success rates and provenance.
    """
    from adv_lab.eval.robustbench_loader import (
        load_cifar10 as _load_cifar10,
        load_robustbench_model as _load_rb_model,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"[robustbench] loading model '{args.model_name}' "
        f"({args.dataset}/{args.threat_model}) on {device} ..."
    )
    model = _load_rb_model(
        model_name=args.model_name,
        dataset=args.dataset,
        threat_model=args.threat_model,
        model_dir=args.model_dir,
        device=device,
    )

    print(f"[robustbench] loading {args.n_samples} CIFAR-10 test images ...")
    x, y = _load_cifar10(n_examples=args.n_samples, data_dir=args.data_dir)

    detailed = run_benchmark_batched(
        model,
        _iter_batches(x, y, args.batch_size),
        epsilon=args.epsilon,
        n_samples=args.n_samples,
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
        pgd_steps=args.pgd_steps,
        cw_steps=args.cw_steps,
    )
    result = detailed.result

    print("=" * 64)
    print(f"model            : {result.model_name}")
    print(f"dataset          : {args.dataset} / {args.threat_model}")
    print(f"n evaluated      : {detailed.n_evaluated}")
    print(f"device           : {detailed.device}")
    print(f"epsilon          : {result.epsilon:.5f}  (~{result.epsilon * 255:.1f}/255)")
    print(f"clean accuracy   : {result.clean_accuracy:.4f}")
    print(
        f"robust  (FGSM)   : {result.robust_accuracy_fgsm:.4f}"
        f"   success={detailed.fgsm_success_rate:.4f}"
    )
    print(
        f"robust  (PGD-{detailed.pgd_steps}) : {result.robust_accuracy_pgd:.4f}"
        f"   success={detailed.pgd_success_rate:.4f}"
    )
    print(
        f"robust  (C&W)    : {result.robust_accuracy_cw:.4f}"
        f"   success={detailed.cw_success_rate:.4f}"
    )
    print(f"PGD gate (>{PGD_GATE_THRESHOLD}) : {'PASS' if result.passed else 'FAIL'}")
    print("=" * 64)

    if args.output:
        export_json(result, args.output)
        # Augment the report with the detailed fields RobustBench comparisons need.
        with open(args.output, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["model_source"] = "robustbench"
        payload["dataset"] = args.dataset
        payload["threat_model"] = args.threat_model
        payload["n_evaluated"] = detailed.n_evaluated
        payload["device"] = detailed.device
        payload["pgd_steps"] = detailed.pgd_steps
        payload["cw_steps"] = args.cw_steps
        payload["success_rate"] = {
            "fgsm": detailed.fgsm_success_rate,
            "pgd": detailed.pgd_success_rate,
            "cw": detailed.cw_success_rate,
        }
        payload["correct_counts"] = {
            "clean": detailed.clean_correct,
            "fgsm": detailed.fgsm_correct,
            "pgd": detailed.pgd_correct,
            "cw": detailed.cw_correct,
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"wrote JSON report -> {args.output}")

    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``adv-eval`` / ``py -m adv_lab.eval.harness``)."""
    parser = argparse.ArgumentParser(
        prog="adv-eval",
        description="Stress-test a classifier with FGSM/PGD/C&W and emit a "
        "CI-gateable JSON robustness report.",
    )
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cw-steps", type=int, default=200)
    parser.add_argument(
        "--model-source",
        type=str,
        default="synthetic",
        choices=["synthetic", "robustbench"],
        help="Where to get the target model. 'synthetic' trains the built-in "
        "demo CNN; 'robustbench' loads a pretrained model from the RobustBench "
        "zoo and evaluates on real CIFAR-10 for leaderboard-comparable numbers.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Standard",
        help="RobustBench zoo id (e.g. 'Standard', 'Wang2023Better_WRN-28-10'). "
        "Only used when --model-source robustbench.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="RobustBench dataset id (cifar10 supported here).",
    )
    parser.add_argument(
        "--threat-model",
        type=str,
        default="Linf",
        help="RobustBench threat model (Linf, L2, corruptions).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Cache dir for CIFAR-10 (RobustBench download).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="Cache dir for RobustBench model checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. 'cuda', 'cpu'). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Mini-batch size for the batched RobustBench evaluation (keep "
        "small to fit VRAM on WRN-28-10).",
    )
    parser.add_argument(
        "--pgd-steps",
        type=int,
        default=40,
        help="PGD iterations for the RobustBench evaluation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the JSON report (e.g. results/report.json).",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    if args.model_source == "robustbench":
        return _run_robustbench(args)

    # Build a slightly larger pool so we can train and still hold out n_samples.
    pool = max(args.n_samples * 4, 2000)
    x, y = _make_synthetic_dataset(pool, args.num_classes, args.seed)
    x_train, y_train = x[args.n_samples :], y[args.n_samples :]
    x_eval, y_eval = x[: args.n_samples], y[: args.n_samples]

    model = _SmallCNN(num_classes=args.num_classes)
    _train_demo_model(model, x_train, y_train, epochs=args.epochs, lr=args.lr)

    result = run_benchmark(
        model,
        _iter_batches(x_eval, y_eval, 256),
        epsilon=args.epsilon,
        n_samples=args.n_samples,
        model_name="SmallCNN(undefended-demo)",
        cw_steps=args.cw_steps,
    )

    print("=" * 60)
    print(f"model          : {result.model_name}")
    print(f"clean accuracy : {result.clean_accuracy:.3f}")
    print(f"robust  (FGSM) : {result.robust_accuracy_fgsm:.3f}")
    print(f"robust  (PGD)  : {result.robust_accuracy_pgd:.3f}")
    print(f"robust  (C&W)  : {result.robust_accuracy_cw:.3f}")
    print(f"epsilon        : {result.epsilon}")
    print(f"PGD gate (>{PGD_GATE_THRESHOLD}) : {'PASS' if result.passed else 'FAIL'}")
    print("=" * 60)

    if args.output:
        export_json(result, args.output)
        print(f"wrote JSON report -> {args.output}")

    # Non-zero exit on gate failure so CI can key off the process exit code too.
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
