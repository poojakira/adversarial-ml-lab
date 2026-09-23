"""
Adversarial robustness benchmark runner.

One command, one structured JSON report. Designed for developer self-service
and security design review documentation.

Usage:
    python -m adv_lab.eval.benchmark_runner --epsilon 0.03
    python -m adv_lab.eval.benchmark_runner --model-path ./my_model.pt --output report.json

MITRE ATLAS: AML.T0029  --  Discover ML Model Ontology (benchmarking model robustness)
NIST AI RMF: MANAGE 2.4  --  Measure and manage AI risks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Severity threshold for CI gate ───────────────────────────────────────────
PGD_ROBUST_ACC_GATE = 0.30  # CI fails if PGD robust accuracy < 30%


# ── Minimal dummy CNN for default benchmarking ───────────────────────────────


class _DummyCNN(nn.Module):
    """Minimal CNN used when no model path is provided."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(
    model_path: str | None,
    *,
    model_format: str = "state-dict",
) -> tuple[nn.Module, str]:
    """Load a model from path, or return a dummy CNN if no path is given."""
    if model_path is None:
        model = _DummyCNN()
        model_id = "dummy_cnn"
        logger.info("No model path provided  --  benchmarking dummy CNN.")
    else:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if model_format == "torchscript":
            try:
                model = torch.jit.load(model_path, map_location="cpu")
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"failed to load TorchScript model '{model_path}': {exc}") from exc
        elif model_format == "state-dict":
            # Load state dict only -- never deserialize arbitrary Python objects.
            try:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
                raise ValueError(
                    f"failed to load checkpoint '{model_path}' as a weights-only "
                    f"state dict: {exc}. The file must be a torch.save() of a "
                    "plain state_dict (no pickled objects)."
                ) from exc
            model = _DummyCNN()
            try:
                model.load_state_dict(state_dict)
            except (RuntimeError, TypeError) as exc:
                raise ValueError(
                    f"checkpoint '{model_path}' does not match the expected "
                    f"_DummyCNN architecture: {exc}. Use --model-format torchscript "
                    "for an arbitrary trusted model architecture."
                ) from exc
        else:
            raise ValueError("model_format must be 'state-dict' or 'torchscript'")
        model_id = Path(model_path).name
        logger.info("Loaded %s model from %s", model_format, model_path)
    model.eval()
    return model, model_id


def _make_test_batch(
    batch_size: int = 32,
    channels: int = 1,
    height: int = 28,
    width: int = 28,
    num_classes: int = 10,
    *,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a deterministic synthetic batch for smoke/demo benchmarking."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    images = torch.rand(batch_size, channels, height, width, generator=generator)
    labels = torch.randint(0, num_classes, (batch_size,), generator=generator)
    return images, labels


def _load_evaluation_batch(dataset_path: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load an explicit NPZ evaluation batch without pickle deserialization."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            if "images" not in data or "labels" not in data:
                raise ValueError("evaluation NPZ must contain 'images' and 'labels' arrays")
            images_np = np.asarray(data["images"])
            labels_np = np.asarray(data["labels"])
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to load evaluation dataset '{dataset_path}': {exc}") from exc

    if images_np.ndim != 4:
        raise ValueError("evaluation images must be a 4D NCHW array")
    if labels_np.ndim != 1:
        raise ValueError("evaluation labels must be a 1D array")
    if not np.issubdtype(images_np.dtype, np.number):
        raise ValueError("evaluation images must use a numeric dtype")
    if not np.issubdtype(labels_np.dtype, np.integer):
        raise ValueError("evaluation labels must use an integer dtype")
    if not np.isfinite(images_np).all():
        raise ValueError("evaluation images contain NaN or infinity")
    if images_np.size and (float(images_np.min()) < 0.0 or float(images_np.max()) > 1.0):
        raise ValueError("evaluation images must be normalized to the [0, 1] range")
    if labels_np.size and int(labels_np.min()) < 0:
        raise ValueError("evaluation labels must be non-negative class indices")
    if len(images_np) != len(labels_np):
        raise ValueError("evaluation image/label counts do not match")
    if len(images_np) < batch_size:
        raise ValueError(
            "evaluation dataset contains "
            f"{len(images_np)} samples, fewer than batch_size={batch_size}"
        )

    images = torch.from_numpy(images_np[:batch_size]).to(dtype=torch.float32)
    labels = torch.from_numpy(labels_np[:batch_size]).to(dtype=torch.long)
    return images, labels


def _validate_model_batch_contract(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Fail before attacks when the deployed model and evidence batch are incompatible."""
    try:
        with torch.no_grad():
            logits = model(images[: min(2, len(images))])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"model cannot evaluate the supplied dataset shape: {exc}") from exc

    if logits.ndim != 2 or logits.shape[0] != min(2, len(images)):
        raise ValueError(
            "model output must be a 2D [batch, classes] logits tensor for this evaluator"
        )
    if logits.shape[1] < 2:
        raise ValueError("model must expose at least two output classes")
    if not torch.isfinite(logits).all():
        raise ValueError("model produced NaN or infinity on the supplied evaluation data")
    if labels.numel() and int(labels.max().item()) >= logits.shape[1]:
        raise ValueError(
            f"evaluation label {int(labels.max().item())} exceeds model class range "
            f"0..{logits.shape[1] - 1}"
        )


# ── Attack imports (canonical implementations from adv_lab.attacks) ───────────

from adv_lab.attacks.fgsm import fgsm_attack
from adv_lab.attacks.pgd import pgd_attack


def _fgsm(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """FGSM via canonical implementation.

    MITRE ATLAS: AML.T0043  --  Craft Adversarial Data
    """
    return fgsm_attack(model, images, labels, epsilon=epsilon)


def _pgd(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int = 40,
    step_size: float | None = None,
) -> torch.Tensor:
    """PGD via canonical implementation.

    MITRE ATLAS: AML.T0015  --  Evade ML Model
    """
    if step_size is None:
        step_size = epsilon / (steps**0.5)
    return pgd_attack(
        model, images, labels, epsilon=epsilon, alpha=step_size, steps=steps, random_start=False
    )


def _predictions(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return class predictions for a batch under eval mode."""
    with torch.no_grad():
        return model(images).argmax(dim=1)


def _robust_accuracy(
    model: nn.Module,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Fraction of all evaluated examples still classified correctly after attack."""
    preds = _predictions(model, adv_images)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def _attack_success_rate(
    model: nn.Module,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    clean_correct: torch.Tensor,
) -> tuple[float | None, int]:
    """Attack success only among examples classified correctly before attack."""
    denominator = int(clean_correct.sum().item())
    if denominator == 0:
        return None, 0
    adv_preds = _predictions(model, adv_images)
    successes = (clean_correct & (adv_preds != labels)).sum().item()
    return float(successes / denominator), denominator


# ── Report construction ───────────────────────────────────────────────────────


def _severity_from_robust_acc(robust_acc: float, gate: float) -> str:
    """Map measured PGD accuracy to severity relative to the configured gate."""
    if robust_acc < min(0.05, gate):
        return "CRITICAL"
    if robust_acc < gate:
        return "HIGH"
    return "LOW"


def benchmark_runner(
    model_path: str | None = None,
    epsilon: float = 0.03,
    pgd_steps: int = 40,
    output_path: str = "benchmark_report.json",
    batch_size: int = 32,
    *,
    dataset_path: str | None = None,
    model_format: str = "state-dict",
    production: bool = False,
    min_pgd_robust_accuracy: float = PGD_ROBUST_ACC_GATE,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run the full adversarial robustness benchmark and return a structured report.

    Runs FGSM, PGD (L-inf), and a C&W proxy attack against the specified model.
    Produces a failing result when measured PGD robust accuracy is below the
    configured deployment threshold, enabling use as an explicit CI admission gate.

    MITRE ATLAS:
        AML.T0029  --  Discover ML Model Ontology
        AML.T0043  --  Craft Adversarial Data
        AML.T0015  --  Evade ML Model

    NIST AI RMF: MANAGE 2.4

    Args:
        model_path: Path to a PyTorch state dict (.pt). None = benchmark dummy CNN.
        epsilon: L-inf perturbation budget. Typical: 0.03 (MNIST-scale), 8/255 (CIFAR-scale).
        pgd_steps: Number of PGD iterations. 40 is the standard evaluation setting.
        output_path: File path to write the JSON report.
        batch_size: Number of test samples per attack run.

    Returns:
        dict: Structured benchmark report with findings, severity summary, pass/fail.

    Raises:
        FileNotFoundError: If model_path is provided but file does not exist.
    """
    if epsilon < 0.0 or epsilon != epsilon or epsilon == float("inf"):
        raise ValueError(f"epsilon must be a finite, non-negative float, got {epsilon}")
    if pgd_steps < 1:
        raise ValueError(f"pgd_steps must be >= 1, got {pgd_steps}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if production and not model_path:
        raise ValueError("production mode requires --model-path")
    if production and not dataset_path:
        raise ValueError("production mode requires --dataset-path")
    if production and model_format != "torchscript":
        raise ValueError("production mode requires --model-format torchscript")
    if not 0.0 <= min_pgd_robust_accuracy <= 1.0:
        raise ValueError("min_pgd_robust_accuracy must be in [0, 1]")

    model, model_id = _load_model(model_path, model_format=model_format)
    if dataset_path:
        images, labels = _load_evaluation_batch(dataset_path, batch_size)
        dataset_source = str(Path(dataset_path).resolve())
        dataset_sha256 = _sha256_file(dataset_path)
    else:
        images, labels = _make_test_batch(batch_size=batch_size, seed=seed)
        dataset_source = "synthetic_seeded_smoke_batch"
        dataset_sha256 = None

    # Ensure model is in eval mode before running attacks.
    # Attacks measured during training mode produce incorrect results because
    # batch normalization and dropout behave differently.
    model.eval()
    _validate_model_batch_contract(model, images, labels)

    clean_preds = _predictions(model, images)
    clean_correct = clean_preds == labels
    clean_correct_count = int(clean_correct.sum().item())
    clean_accuracy = float(clean_correct_count / len(labels))

    logger.info("Running FGSM attack (epsilon=%.3f)...", epsilon)
    # MITRE ATLAS: AML.T0043  --  Craft Adversarial Data
    adv_fgsm = _fgsm(model, images, labels, epsilon=epsilon)
    fgsm_robust_acc = _robust_accuracy(model, adv_fgsm, labels)
    fgsm_asr, fgsm_asr_n = _attack_success_rate(model, adv_fgsm, labels, clean_correct)

    logger.info("Running PGD attack (epsilon=%.3f, steps=%d)...", epsilon, pgd_steps)
    # MITRE ATLAS: AML.T0015  --  Evade ML Model
    adv_pgd = _pgd(model, images, labels, epsilon=epsilon, steps=pgd_steps)
    pgd_robust_acc = _robust_accuracy(model, adv_pgd, labels)
    pgd_asr, pgd_asr_n = _attack_success_rate(model, adv_pgd, labels, clean_correct)

    # C&W proxy: use PGD with more steps as a computationally feasible proxy
    # Full C&W optimization (Carlini & Wagner 2017) is available via cw_l2_attack
    # but is slow for batch evaluation. PGD-100 is a practical proxy.
    logger.info("Running C&W proxy attack (PGD-100)...")
    # MITRE ATLAS: AML.T0043  --  Craft Adversarial Data
    adv_cw_proxy = _pgd(model, images, labels, epsilon=epsilon, steps=100)
    cw_robust_acc = _robust_accuracy(model, adv_cw_proxy, labels)
    cw_asr, cw_asr_n = _attack_success_rate(model, adv_cw_proxy, labels, clean_correct)

    # Findings
    pgd_severity = _severity_from_robust_acc(pgd_robust_acc, min_pgd_robust_accuracy)
    findings: list[dict[str, Any]] = []

    if pgd_robust_acc < min_pgd_robust_accuracy:
        findings.append(
            {
                "severity": pgd_severity,
                "attack": "pgd",
                "atlas_technique": "AML.T0015",
                "message": (
                    f"PGD robust accuracy {pgd_robust_acc:.1%} is below the {min_pgd_robust_accuracy:.0%} "  # noqa: E501
                    "configured deployment gate for this evaluation. Review the model, data, "
                    "threat model, and policy before promotion."
                ),
            }
        )

    if fgsm_robust_acc > pgd_robust_acc + 0.20:
        findings.append(
            {
                "severity": "MEDIUM",
                "attack": "gradient_masking_indicator",
                "atlas_technique": "AML.T0015",
                "message": (
                    f"FGSM robust acc ({fgsm_robust_acc:.1%}) significantly exceeds "
                    f"PGD robust acc ({pgd_robust_acc:.1%}). "
                    "This is a gradient masking indicator  --  apparent FGSM robustness is likely false."  # noqa: E501
                ),
            }
        )

    severity_summary = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for f in findings:
        sev = f.get("severity", "LOW")
        if sev in severity_summary:
            severity_summary[sev] += 1

    pass_fail = "FAIL" if (severity_summary["CRITICAL"] + severity_summary["HIGH"] > 0) else "PASS"

    remediation_hints: list[str] = []
    if pgd_robust_acc < min_pgd_robust_accuracy:
        remediation_hints.append(
            f"Evaluate adversarial training (for example PGD-based training at eps={epsilon:.3f}) "
            "and re-run this same gate; do not assume a literature result transfers to this model."
        )
        remediation_hints.append(
            "Consider randomized smoothing for certified robustness guarantees."
        )

    defense_roi = [
        {
            "defense": "no_defense",
            "robust_accuracy_eps_provided": round(pgd_robust_acc, 4),
            "training_overhead": "none",
            "recommendation": "Fails configured robustness gate"
            if pgd_robust_acc < min_pgd_robust_accuracy
            else "Meets configured robustness gate",
        },
        {
            "defense": "madry_pgd7_adversarial_training",
            "robust_accuracy_eps8_255_literature": 0.45,
            "training_overhead": "+3x training time",
            "recommendation": "Candidate defense to evaluate under the same deployment benchmark",
            "note": "Literature estimate (Madry et al. 2018); run on your model to verify",
        },
        {
            "defense": "randomized_smoothing",
            "robust_accuracy": "certified L2 radius guarantee",
            "training_overhead": "+2x inference time",
            "recommendation": (
                "Evaluate when an L2 certification objective is part of the threat model"
            ),
            "note": "Cohen et al. 2019; provides provable L2 bounds, not L-inf",
        },
    ]

    report: dict[str, Any] = {
        "tool": "adversarial-ml-lab",
        "version": "1.0.0",
        "scan_date": date.today().isoformat(),
        "scan_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "model_id": model_id,
        "model_format": model_format,
        "model_sha256": _sha256_file(model_path) if model_path else None,
        "dataset_source": dataset_source,
        "dataset_sha256": dataset_sha256,
        "synthetic_seed": None if dataset_path else seed,
        "production_mode": production,
        "min_pgd_robust_accuracy": min_pgd_robust_accuracy,
        "epsilon": epsilon,
        "pgd_steps": pgd_steps,
        "batch_size": batch_size,
        "clean_accuracy": round(clean_accuracy, 4),
        "clean_correct": clean_correct_count,
        "attacks": {
            "fgsm": {
                "robust_accuracy": round(fgsm_robust_acc, 4),
                "attack_success_rate": None if fgsm_asr is None else round(fgsm_asr, 4),
                "attack_success_denominator": fgsm_asr_n,
                "atlas_technique": "AML.T0043",
                "note": "Single-step; use PGD for honest evaluation",
            },
            "pgd": {
                "robust_accuracy": round(pgd_robust_acc, 4),
                "attack_success_rate": None if pgd_asr is None else round(pgd_asr, 4),
                "attack_success_denominator": pgd_asr_n,
                "steps": pgd_steps,
                "atlas_technique": "AML.T0015",
                "note": "Standard honest white-box evaluation metric",
            },
            "cw_l2_proxy": {
                "robust_accuracy": round(cw_robust_acc, 4),
                "attack_success_rate": None if cw_asr is None else round(cw_asr, 4),
                "attack_success_denominator": cw_asr_n,
                "note": "PGD-100 proxy; full C&W optimization available via cw_l2_attack()",
                "atlas_technique": "AML.T0043",
            },
        },
        "defense_roi": defense_roi,
        "pass_fail": pass_fail,
        "findings": findings,
        "severity_summary": severity_summary,
        "remediation_hints": remediation_hints,
        "mitre_atlas_techniques": ["AML.T0015", "AML.T0043", "AML.T0029"],
        "nist_ai_rmf_function": "MANAGE 2.4",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Report written to %s  --  result: %s", output_path, pass_fail)
    return report


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
    )

    parser = argparse.ArgumentParser(
        description="Adversarial robustness benchmark  --  one command, one JSON report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deterministic smoke/demo only (not deployment evidence)
  python -m adv_lab.eval.benchmark_runner --epsilon 0.03 --seed 42

  # Production admission gate: explicit deployed model + representative data
  python -m adv_lab.eval.benchmark_runner --production \
    --model-path ./model.ts --model-format torchscript \
    --dataset-path ./evaluation.npz --output report.json
        """,
    )
    parser.add_argument("--model-path", default=None, help="Path to model artifact")
    parser.add_argument(
        "--model-format",
        choices=("state-dict", "torchscript"),
        default="state-dict",
        help="Model artifact format. Use torchscript for arbitrary deployed architectures.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="NPZ containing images (NCHW) and labels arrays. Required in production mode.",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help=(
            "Fail closed unless an explicit TorchScript model and evaluation "
            "dataset are supplied."
        ),
    )
    parser.add_argument(
        "--min-pgd-robust-accuracy",
        type=float,
        default=PGD_ROBUST_ACC_GATE,
        help="Deployment-owned minimum PGD robust accuracy gate (default: 0.30).",
    )
    parser.add_argument("--epsilon", type=float, default=0.03, help="L-inf epsilon (default: 0.03)")
    parser.add_argument("--pgd-steps", type=int, default=40, help="PGD iterations (default: 40)")
    parser.add_argument("--output", default="benchmark_report.json", help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=32, help="Test batch size")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for synthetic smoke/demo data only (ignored when --dataset-path is supplied).",
    )
    args = parser.parse_args()

    report = benchmark_runner(
        model_path=args.model_path,
        epsilon=args.epsilon,
        pgd_steps=args.pgd_steps,
        output_path=args.output,
        batch_size=args.batch_size,
        dataset_path=args.dataset_path,
        model_format=args.model_format,
        production=args.production,
        min_pgd_robust_accuracy=args.min_pgd_robust_accuracy,
        seed=args.seed,
    )

    print(json.dumps(report, indent=2))

    has_blocking = report["severity_summary"]["CRITICAL"] + report["severity_summary"]["HIGH"] > 0
    sys.exit(1 if has_blocking else 0)


if __name__ == "__main__":
    _main()
