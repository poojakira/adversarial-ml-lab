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


def export_json(result: BenchmarkResult, path: str) -> None:
    """Write a CI-consumable JSON report.

    The payload includes short keys a CI job can grep (``passed``,
    ``robust_pgd``) alongside the full record for humans.
    """
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
    import os

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
        "--output",
        type=str,
        default=None,
        help="Path to write the JSON report (e.g. results/report.json).",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

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
