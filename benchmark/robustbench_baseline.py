#!/usr/bin/env python3
"""Real adversarial robustness benchmark: ResNet-18 (ImageNet-pretrained) vs CIFAR-10.

This script runs REAL attacks (FGSM, PGD-20) from this repo against a REAL
pretrained model (torchvision ResNet-18) on REAL data (CIFAR-10 test set).

The ImageNet-pretrained ResNet-18 is not trained on CIFAR-10 classes, so clean
accuracy will be low. The point is to demonstrate real attack execution against
real model weights — not to claim state-of-the-art robustness numbers.

Requirements:
    pip install torch torchvision

Usage:
    python benchmark/robustbench_baseline.py
"""
# MITRE ATLAS: AML.T0043 - Craft Adversarial Data | AML.T0015 - Evade ML Model

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Dependency check ---
try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.models as models
    import torchvision.transforms as transforms
except ImportError:
    print(
        "ERROR: This benchmark requires PyTorch and torchvision.\n"
        "Install them with:\n\n"
        "    pip install torch torchvision\n\n"
        "Then re-run this script.",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Ensure repo root is importable ---
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from adv_lab.attacks.fgsm import fgsm_attack  # noqa: E402
from adv_lab.attacks.pgd import pgd_attack  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_IMAGES = 100  # First N images from CIFAR-10 test set
EPS = 8 / 255  # Standard L-inf perturbation budget (RobustBench standard)
PGD_ALPHA = 2 / 255  # PGD step size (standard: eps/4)
PGD_STEPS = 20  # PGD iterations (PGD-20 is the standard benchmark)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Model wrapper: adapt ImageNet ResNet-18 to accept 32x32 CIFAR-10 inputs
# ---------------------------------------------------------------------------
class ResNet18CIFAR10Adapter(nn.Module):
    """Wraps a pretrained ImageNet ResNet-18 to handle CIFAR-10 inputs.

    CIFAR-10 images are 32x32x3. ImageNet ResNet-18 expects 224x224x3.
    We upsample the input to 224x224 so the pretrained conv filters work.
    We do NOT retrain or fine-tune — we use the raw ImageNet weights.

    For label mapping: ImageNet has 1000 classes, CIFAR-10 has 10. We take
    the model's top-1 prediction from the 1000 ImageNet classes and map it
    to one of 10 buckets (using argmax % 10) so we can compute a "pseudo-accuracy"
    for attack success measurement. The absolute accuracy is meaningless for
    CIFAR-10 semantics, but attack success (flipping the model's prediction)
    is a valid robustness measurement regardless of label semantics.
    """

    def __init__(self) -> None:
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.eval()
        self.upsample = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample 32x32 -> 224x224
        x = self.upsample(x)
        # ImageNet normalization (applied after upsampling for correct scale)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        logits_1000 = self.resnet(x)
        # Reduce 1000 ImageNet logits to 10 pseudo-classes by summing groups of 100
        # This gives a consistent 10-class output for attack evaluation
        logits_10 = logits_1000.view(x.size(0), 10, 100).sum(dim=2)
        return logits_10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_cifar10_batch(num_images: int = NUM_IMAGES) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the first `num_images` from CIFAR-10 test set as normalized tensors."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts to [0, 1] float tensor
        ]
    )

    # Download CIFAR-10 to a cache directory within the repo
    cache_dir = REPO_ROOT / ".data"
    dataset = torchvision.datasets.CIFAR10(
        root=str(cache_dir),
        train=False,
        download=True,
        transform=transform,
    )

    images = []
    labels = []
    for i in range(min(num_images, len(dataset))):
        img, lbl = dataset[i]
        images.append(img)
        labels.append(lbl)

    images_tensor = torch.stack(images)  # (N, 3, 32, 32) in [0, 1]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return images_tensor, labels_tensor


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_clean(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> dict:
    """Measure clean (no attack) accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
    }


def evaluate_fgsm(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = EPS,
) -> dict:
    """Run FGSM and measure attack success rate."""
    model.eval()

    # Get clean predictions first
    with torch.no_grad():
        clean_preds = model(images).argmax(dim=1)
    correct_mask = clean_preds == labels

    # Generate adversarial examples
    t0 = time.perf_counter()
    adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)
    elapsed = time.perf_counter() - t0

    # Evaluate adversarial predictions
    with torch.no_grad():
        adv_preds = model(adv_images).argmax(dim=1)

    # Attack success = was correct, now wrong
    n_correct_clean = correct_mask.sum().item()
    n_fooled = (correct_mask & (adv_preds != labels)).sum().item()
    success_rate = n_fooled / n_correct_clean if n_correct_clean > 0 else 0.0

    # Adversarial accuracy = fraction still correct after attack
    adv_correct = (adv_preds == labels).sum().item()

    return {
        "attack": "FGSM",
        "epsilon": epsilon,
        "clean_correct": int(n_correct_clean),
        "fooled": int(n_fooled),
        "success_rate": success_rate,
        "adversarial_accuracy": adv_correct / labels.size(0),
        "elapsed_seconds": round(elapsed, 3),
    }


def evaluate_pgd(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = EPS,
    alpha: float = PGD_ALPHA,
    steps: int = PGD_STEPS,
) -> dict:
    """Run PGD-20 and measure attack success rate."""
    model.eval()

    # Get clean predictions first
    with torch.no_grad():
        clean_preds = model(images).argmax(dim=1)
    correct_mask = clean_preds == labels

    # Generate adversarial examples
    t0 = time.perf_counter()
    adv_images = pgd_attack(
        model, images, labels, epsilon=epsilon, alpha=alpha, steps=steps, random_start=True
    )
    elapsed = time.perf_counter() - t0

    # Evaluate adversarial predictions
    with torch.no_grad():
        adv_preds = model(adv_images).argmax(dim=1)

    # Attack success = was correct, now wrong
    n_correct_clean = correct_mask.sum().item()
    n_fooled = (correct_mask & (adv_preds != labels)).sum().item()
    success_rate = n_fooled / n_correct_clean if n_correct_clean > 0 else 0.0

    # Adversarial accuracy = fraction still correct after attack
    adv_correct = (adv_preds == labels).sum().item()

    return {
        "attack": "PGD-20",
        "epsilon": epsilon,
        "alpha": alpha,
        "steps": steps,
        "random_start": True,
        "clean_correct": int(n_correct_clean),
        "fooled": int(n_fooled),
        "success_rate": success_rate,
        "adversarial_accuracy": adv_correct / labels.size(0),
        "elapsed_seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("REAL Adversarial Robustness Benchmark")
    print("Model: torchvision.resnet18 (pretrained=True, ImageNet weights)")
    print("Data:  CIFAR-10 test set (first 100 images)")
    print(f"Device: {DEVICE}")
    print(f"L-inf budget: eps={EPS:.5f} ({EPS*255:.0f}/255)")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading pretrained ResNet-18...")
    model = ResNet18CIFAR10Adapter().to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"       Model loaded. Parameters: {n_params:,}")

    # Load data
    print("\n[2/4] Loading CIFAR-10 test data...")
    images, labels = load_cifar10_batch(NUM_IMAGES)
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    print(f"       Loaded {images.size(0)} images, shape={tuple(images.shape)}")

    # Clean evaluation
    print("\n[3/4] Evaluating clean accuracy...")
    clean_results = evaluate_clean(model, images, labels)
    print(
        f"       Clean accuracy: {clean_results['correct']}/{clean_results['total']}"
        f" = {clean_results['accuracy']:.1%}"
    )
    print("       (Low accuracy expected: ImageNet model on CIFAR-10 classes)")

    # Attack evaluations
    print("\n[4/4] Running adversarial attacks...")
    print(f"\n  FGSM (eps={EPS*255:.0f}/255)...")
    fgsm_results = evaluate_fgsm(model, images, labels, epsilon=EPS)
    print(
        f"       Fooled: {fgsm_results['fooled']}/{fgsm_results['clean_correct']}"
        f" = {fgsm_results['success_rate']:.1%} success rate"
    )
    print(f"       Adversarial accuracy: {fgsm_results['adversarial_accuracy']:.1%}")
    print(f"       Time: {fgsm_results['elapsed_seconds']:.3f}s")

    print(
        f"\n  PGD-20 (eps={EPS*255:.0f}/255, alpha={PGD_ALPHA*255:.0f}/255, steps={PGD_STEPS})..."
    )
    pgd_results = evaluate_pgd(model, images, labels, epsilon=EPS, alpha=PGD_ALPHA, steps=PGD_STEPS)
    print(
        f"       Fooled: {pgd_results['fooled']}/{pgd_results['clean_correct']}"
        f" = {pgd_results['success_rate']:.1%} success rate"
    )
    print(f"       Adversarial accuracy: {pgd_results['adversarial_accuracy']:.1%}")
    print(f"       Time: {pgd_results['elapsed_seconds']:.3f}s")

    # Compose results
    results = {
        "benchmark": "Real Adversarial Robustness Evaluation",
        "model": "torchvision.resnet18 (pretrained=True, adapted to CIFAR-10 input size)",
        "model_source": "torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)",
        "model_parameters": n_params,
        "dataset": "CIFAR-10 test set (first 100 images)",
        "num_images": NUM_IMAGES,
        "device": DEVICE,
        "attack_config": {
            "linf_budget_eps": EPS,
            "linf_budget_eps_int": f"{int(EPS * 255)}/255",
            "fgsm": {"epsilon": EPS},
            "pgd": {
                "epsilon": EPS,
                "alpha": PGD_ALPHA,
                "steps": PGD_STEPS,
                "random_start": True,
            },
        },
        "results": {
            "clean": clean_results,
            "fgsm": fgsm_results,
            "pgd_20": pgd_results,
        },
        "methodology_notes": [
            "All measurements are REAL — run on real pretrained weights with real data.",
            "No synthetic/projected numbers. Every value comes from actual forward/backward passes.",
            "ImageNet-pretrained model evaluated on CIFAR-10 via input upsampling + logit grouping.",
            "Clean accuracy is expected to be low due to domain mismatch (ImageNet vs CIFAR-10).",
            "Attack success rates measure the fraction of correctly-classified images that are fooled.",
            "PGD-20 is strictly stronger than FGSM; if FGSM > PGD success, suspect gradient masking.",
        ],
        "attack_implementations": {
            "fgsm": "adv_lab.attacks.fgsm.fgsm_attack",
            "pgd": "adv_lab.attacks.pgd.pgd_attack",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }

    # Write results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "resnet18_cifar10_robustbench.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results written to: {output_path}")
    print(f"{'=' * 70}")

    # Summary
    print("\n  SUMMARY:")
    print(f"    Clean accuracy:       {clean_results['accuracy']:.1%}")
    print(f"    FGSM success rate:    {fgsm_results['success_rate']:.1%}")
    print(f"    PGD-20 success rate:  {pgd_results['success_rate']:.1%}")
    print(f"    FGSM adv accuracy:    {fgsm_results['adversarial_accuracy']:.1%}")
    print(f"    PGD-20 adv accuracy:  {pgd_results['adversarial_accuracy']:.1%}")


if __name__ == "__main__":
    main()
