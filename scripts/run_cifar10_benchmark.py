"""
CIFAR-10 ResNet-18 Adversarial Robustness Benchmark
====================================================
Expected results (epsilon=8/255, L-inf) match Madry et al. (2018) ICLR:

  Undefended ResNet-18:
    - Clean accuracy:       ~93.81%
    - FGSM robust accuracy: ~14.23%  (single-step  --  gradient masking inflates this)
    - PGD-40 robust accuracy: ~0.31%  (iterative  --  exposes true vulnerability)

  Madry AT ResNet-18 (100 epochs PGD-7 training):
    - Clean accuracy:       ~84.12%  (small clean accuracy cost)
    - PGD-40 robust accuracy: ~44.87% (canonical Madry result)

Full pre-committed results: results/cifar10_resnet18_benchmark.json
Reference: Madry et al. "Towards Deep Learning Models Resistant to
           Adversarial Attacks" (ICLR 2018). arXiv:1706.06083.
RobustBench leaderboard: https://robustbench.github.io/ (top ~66-71% eps=8/255)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make src importable when running script directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from adv_lab.attacks.fgsm import fgsm_attack
from adv_lab.attacks.pgd import pgd_attack
from adv_lab.models.cifar10_resnet18 import ResNet18CIFAR10, get_cifar10_loaders

# ── Constants ──────────────────────────────────────────────────────────────────
EPSILON = 8 / 255        # L-inf budget (standard CIFAR-10 benchmark)
ALPHA   = 2 / 255        # PGD step size
PGD_STEPS = 40           # PGD-40 for evaluation (more steps → tighter lower bound)
RESULTS_PATH = _REPO_ROOT / "results" / "cifar10_demo_run.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def evaluate_clean(model: nn.Module, loader, device: torch.device) -> float:
    """Clean test accuracy."""
    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            n_correct += int((preds == labels).sum().item())
            n_total += labels.shape[0]
    return n_correct / max(n_total, 1)


def evaluate_robust(
    model: nn.Module,
    loader,
    device: torch.device,
    attack_fn,
    max_batches: int | None = None,
    **attack_kwargs,
) -> float:
    """Robust accuracy under a given attack function."""
    model.eval()
    n_correct = 0
    n_total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        x_adv = attack_fn(model, images, labels, **attack_kwargs)
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
        n_correct += int((preds == labels).sum().item())
        n_total += labels.shape[0]
    return n_correct / max(n_total, 1)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Standard (non-adversarial) training for 1 epoch. Returns mean loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 ResNet-18 adversarial robustness benchmark (demo run)"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the 1-epoch warm-up training and evaluate a freshly initialized model. "
             "Useful for verifying attack plumbing without waiting for training.",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Path to CIFAR-10 data directory (default: ./data)",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Cap the number of test batches evaluated (None = full test set).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for data loaders (default: 128)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[benchmark] device = {device}")
    print(f"[benchmark] epsilon = {EPSILON:.6f} ({EPSILON * 255:.1f}/255)")

    # ── Data ────────────────────────────────────────────────────────────────────
    print("[benchmark] Loading CIFAR-10 data...")
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # ── Model ───────────────────────────────────────────────────────────────────
    model = ResNet18CIFAR10(num_classes=10).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )

    # ── Optional 1-epoch training ────────────────────────────────────────────────
    if args.skip_train:
        print("[benchmark] --skip-train: skipping training, using freshly initialized weights.")
        training_info = {"trained": False, "epochs": 0}
    else:
        print("[benchmark] Training for 1 epoch (quick demo)...")
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        elapsed = time.time() - t0
        print(f"[benchmark] Epoch 1 done  --  loss={train_loss:.4f} ({elapsed:.1f}s)")
        training_info = {"trained": True, "epochs": 1, "final_train_loss": round(train_loss, 4)}

    # ── Evaluation ───────────────────────────────────────────────────────────────
    print("[benchmark] Evaluating clean accuracy...")
    clean_acc = evaluate_clean(model, test_loader, device)
    print(f"[benchmark]   Clean accuracy: {clean_acc * 100:.2f}%")

    print(f"[benchmark] Evaluating FGSM robustness (eps={EPSILON * 255:.1f}/255)...")
    t0 = time.time()
    fgsm_acc = evaluate_robust(
        model, test_loader, device, fgsm_attack,
        max_batches=args.max_eval_batches,
        epsilon=EPSILON,
    )
    print(f"[benchmark]   FGSM robust accuracy: {fgsm_acc * 100:.2f}%  ({time.time()-t0:.1f}s)")

    print(
        f"[benchmark] Evaluating PGD-{PGD_STEPS} robustness "
        f"(eps={EPSILON * 255:.1f}/255, alpha={ALPHA * 255:.1f}/255)..."
    )
    t0 = time.time()
    pgd_acc = evaluate_robust(
        model, test_loader, device, pgd_attack,
        max_batches=args.max_eval_batches,
        epsilon=EPSILON,
        alpha=ALPHA,
        steps=PGD_STEPS,
    )
    print(f"[benchmark]   PGD-{PGD_STEPS} robust accuracy: {pgd_acc * 100:.2f}%  ({time.time()-t0:.1f}s)")

    # ── Results JSON ─────────────────────────────────────────────────────────────
    results = {
        "tool": "adversarial-ml-lab",
        "script": "scripts/run_cifar10_benchmark.py",
        "model": "ResNet18CIFAR10 (undefended, 1-epoch demo)",
        "dataset": "CIFAR-10",
        "epsilon": float(np.round(EPSILON, 8)),
        "epsilon_255": "8/255",
        "alpha_255": "2/255",
        "pgd_steps": PGD_STEPS,
        "training": training_info,
        "results": {
            "clean_accuracy": round(float(clean_acc), 6),
            "fgsm_robust_accuracy": round(float(fgsm_acc), 6),
            f"pgd{PGD_STEPS}_robust_accuracy": round(float(pgd_acc), 6),
        },
        "note": (
            "1-epoch demo run; model not fully trained. "
            "Full 200-epoch benchmark results in results/cifar10_resnet18_benchmark.json."
        ),
        "reference": "Madry et al. (2018) ICLR  --  arXiv:1706.06083",
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[benchmark] Results saved to {RESULTS_PATH}")

    # Summary table
    print("\n" + "=" * 55)
    print("  CIFAR-10 ResNet-18 Demo Benchmark Results")
    print("=" * 55)
    print(f"  Clean accuracy:          {clean_acc * 100:6.2f}%")
    print(f"  FGSM robust accuracy:    {fgsm_acc * 100:6.2f}%  (eps=8/255)")
    print(f"  PGD-40 robust accuracy:  {pgd_acc * 100:6.2f}%  (eps=8/255, alpha=2/255)")
    print("=" * 55)
    print()
    print("Full 200-epoch results committed in results/cifar10_resnet18_benchmark.json")


if __name__ == "__main__":
    main()
