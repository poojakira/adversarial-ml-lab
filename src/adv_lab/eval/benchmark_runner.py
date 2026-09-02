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
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

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


def _load_model(model_path: str | None) -> tuple[nn.Module, str]:
    """Load a model from path, or return a dummy CNN if no path given."""
    if model_path is None:
        model = _DummyCNN()
        model_id = "dummy_cnn"
        logger.info("No model path provided  --  benchmarking dummy CNN.")
    else:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # Load state dict only  --  never use pickle.load() on untrusted files.
        # weights_only=True prevents arbitrary code execution via pickle.
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
            raise ValueError(
                f"failed to load checkpoint '{model_path}' as a weights-only state dict: {exc}. "
                "The file must be a torch.save() of a plain state_dict (no pickled objects)."
            ) from exc
        model = _DummyCNN()
        try:
            model.load_state_dict(state_dict)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                f"checkpoint '{model_path}' does not match the expected _DummyCNN architecture: "
                f"{exc}. This runner benchmarks a fixed dummy CNN; supply a matching state_dict."
            ) from exc
        model_id = Path(model_path).name
        logger.info("Loaded model from %s", model_path)
    model.eval()
    return model, model_id


def _make_test_batch(
    batch_size: int = 32,
    channels: int = 1,
    height: int = 28,
    width: int = 28,
    num_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic test batch for benchmarking."""
    images = torch.rand(batch_size, channels, height, width)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels


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


def _robust_accuracy(
    model: nn.Module,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Fraction of adversarial examples that the model still classifies correctly."""
    with torch.no_grad():
        preds = model(adv_images).argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


# ── Report construction ───────────────────────────────────────────────────────


def _severity_from_robust_acc(robust_acc: float) -> str:
    """Map PGD robust accuracy to a severity level for the CI gate."""
    if robust_acc < 0.05:
        return "CRITICAL"
    elif robust_acc < PGD_ROBUST_ACC_GATE:
        return "HIGH"
    elif robust_acc < 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def benchmark_runner(
    model_path: str | None = None,
    epsilon: float = 0.03,
    pgd_steps: int = 40,
    output_path: str = "benchmark_report.json",
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Run the full adversarial robustness benchmark and return a structured report.

    Runs FGSM, PGD (L-inf), and a C&W proxy attack against the specified model.
    Exits with code 1 if any HIGH or CRITICAL severity finding is present
    (i.e., PGD robust accuracy < 30%)  --  enabling use as a CI gate.

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

    model, model_id = _load_model(model_path)
    images, labels = _make_test_batch(batch_size=batch_size)

    # Ensure model is in eval mode before running attacks.
    # Attacks measured during training mode produce incorrect results because
    # batch normalization and dropout behave differently.
    model.eval()

    logger.info("Running FGSM attack (epsilon=%.3f)...", epsilon)
    # MITRE ATLAS: AML.T0043  --  Craft Adversarial Data
    adv_fgsm = _fgsm(model, images, labels, epsilon=epsilon)
    fgsm_robust_acc = _robust_accuracy(model, adv_fgsm, labels)

    logger.info("Running PGD attack (epsilon=%.3f, steps=%d)...", epsilon, pgd_steps)
    # MITRE ATLAS: AML.T0015  --  Evade ML Model
    adv_pgd = _pgd(model, images, labels, epsilon=epsilon, steps=pgd_steps)
    pgd_robust_acc = _robust_accuracy(model, adv_pgd, labels)

    # C&W proxy: use PGD with more steps as a computationally feasible proxy
    # Full C&W optimization (Carlini & Wagner 2017) is available via cw_l2_attack
    # but is slow for batch evaluation. PGD-100 is a practical proxy.
    logger.info("Running C&W proxy attack (PGD-100)...")
    # MITRE ATLAS: AML.T0043  --  Craft Adversarial Data
    adv_cw_proxy = _pgd(model, images, labels, epsilon=epsilon, steps=100)
    cw_robust_acc = _robust_accuracy(model, adv_cw_proxy, labels)

    # Findings
    pgd_severity = _severity_from_robust_acc(pgd_robust_acc)
    findings: list[dict[str, Any]] = []

    if pgd_robust_acc < PGD_ROBUST_ACC_GATE:
        findings.append(
            {
                "severity": pgd_severity,
                "attack": "pgd",
                "atlas_technique": "AML.T0015",
                "message": (
                    f"PGD robust accuracy {pgd_robust_acc:.1%} is below the {PGD_ROBUST_ACC_GATE:.0%} "  # noqa: E501
                    f"CI gate threshold. This model is not safe for adversarial environments."
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
    if pgd_robust_acc < PGD_ROBUST_ACC_GATE:
        remediation_hints.append(
            f"Apply Madry adversarial training (PGD-7, eps={epsilon:.3f})  --  "
            "expected to improve robust accuracy to ~40-50%."
        )
        remediation_hints.append(
            "Consider randomized smoothing for certified robustness guarantees."
        )

    defense_roi = [
        {
            "defense": "no_defense",
            "robust_accuracy_eps_provided": round(pgd_robust_acc, 4),
            "training_overhead": "none",
            "recommendation": "Not safe for adversarial environments"
            if pgd_robust_acc < PGD_ROBUST_ACC_GATE
            else "Acceptable for low-risk deployments",
        },
        {
            "defense": "madry_pgd7_adversarial_training",
            "robust_accuracy_eps8_255_literature": 0.45,
            "training_overhead": "+3x training time",
            "recommendation": "Recommended for high-risk deployments (fraud detection, biometric auth)",  # noqa: E501
            "note": "Literature estimate (Madry et al. 2018); run on your model to verify",
        },
        {
            "defense": "randomized_smoothing",
            "robust_accuracy": "certified L2 radius guarantee",
            "training_overhead": "+2x inference time",
            "recommendation": "Use when certified robustness is required",
            "note": "Cohen et al. 2019; provides provable L2 bounds, not L-inf",
        },
    ]

    report: dict[str, Any] = {
        "tool": "adversarial-ml-lab",
        "version": "1.0.0",
        "scan_date": date.today().isoformat(),
        "scan_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "model_id": model_id,
        "epsilon": epsilon,
        "pgd_steps": pgd_steps,
        "batch_size": batch_size,
        "attacks": {
            "fgsm": {
                "robust_accuracy": round(fgsm_robust_acc, 4),
                "attack_success_rate": round(1.0 - fgsm_robust_acc, 4),
                "atlas_technique": "AML.T0043",
                "note": "Single-step; use PGD for honest evaluation",
            },
            "pgd": {
                "robust_accuracy": round(pgd_robust_acc, 4),
                "attack_success_rate": round(1.0 - pgd_robust_acc, 4),
                "steps": pgd_steps,
                "atlas_technique": "AML.T0015",
                "note": "Standard honest white-box evaluation metric",
            },
            "cw_l2_proxy": {
                "robust_accuracy": round(cw_robust_acc, 4),
                "attack_success_rate": round(1.0 - cw_robust_acc, 4),
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
  # Benchmark dummy model (default)
  python -m adv_lab.eval.benchmark_runner --epsilon 0.03

  # Benchmark your model, write to file
  python -m adv_lab.eval.benchmark_runner --model-path ./model.pt --output report.json

  # Use as CI gate (exits 1 if robust_acc < 30%)
  python -m adv_lab.eval.benchmark_runner && echo PASS || echo FAIL
        """,
    )
    parser.add_argument("--model-path", default=None, help="Path to PyTorch state dict (.pt)")
    parser.add_argument("--epsilon", type=float, default=0.03, help="L-inf epsilon (default: 0.03)")
    parser.add_argument("--pgd-steps", type=int, default=40, help="PGD iterations (default: 40)")
    parser.add_argument("--output", default="benchmark_report.json", help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=32, help="Test batch size")
    args = parser.parse_args()

    report = benchmark_runner(
        model_path=args.model_path,
        epsilon=args.epsilon,
        pgd_steps=args.pgd_steps,
        output_path=args.output,
        batch_size=args.batch_size,
    )

    print(json.dumps(report, indent=2))

    has_blocking = report["severity_summary"]["CRITICAL"] + report["severity_summary"]["HIGH"] > 0
    sys.exit(1 if has_blocking else 0)


if __name__ == "__main__":
    _main()
