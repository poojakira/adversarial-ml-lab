"""
Madry Adversarial Training — CIFAR-10 ResNet-18
================================================
Expected: ~45% PGD-40 robust accuracy after 100 epochs (Madry et al. 2018 ICLR)

Reference: Madry et al. "Towards Deep Learning Models Resistant to
           Adversarial Attacks" (ICLR 2018). arXiv:1706.06083.

Training method: PGD-7 adversarial training
  - Inner attack: PGD-7, epsilon=8/255, alpha=2/255
  - Outer optimizer: SGD with cosine annealing LR schedule
  - Batch size: 128, Weight decay: 5e-4, Momentum: 0.9

Checkpoint: checkpoints/madry_at_resnet18.pth
Results:    results/madry_at_results.json

Expected final metrics (100 epochs):
  Clean accuracy:         ~84%  (modest clean cost vs. ~94% undefended)
  PGD-40 robust accuracy: ~45%  (canonical Madry 2018 result)

RobustBench CIFAR-10 L-inf leaderboard (eps=8/255):
  Top models reach ~66-71%, but require larger architectures and
  more sophisticated training procedures (e.g., WideResNet-70-16).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Make src importable when running script directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from adv_lab.models.cifar10_resnet18 import ResNet18CIFAR10, get_cifar10_loaders
from adv_lab.attacks.pgd import pgd_attack

# ── Constants ──────────────────────────────────────────────────────────────────
EPSILON    = 8 / 255    # L-inf budget
ALPHA      = 2 / 255    # PGD step size (both inner training attack and eval)
PGD_TRAIN_STEPS = 7     # PGD-7 during training (cost/robustness sweet spot)
PGD_EVAL_STEPS  = 40    # PGD-40 for final evaluation (tighter lower bound)

CHECKPOINT_PATH = _REPO_ROOT / "checkpoints" / "madry_at_resnet18.pth"
RESULTS_PATH    = _REPO_ROOT / "results" / "madry_at_results.json"


# ── PGD-7 inner attack for training ──────────────────────────────────────────

def _pgd7_inner(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """PGD-7 inner maximization step used during adversarial training.

    Uses random start to avoid gradient masking during training.
    Alpha is set to 2/255 (same as evaluation), which is the standard
    configuration from Madry et al. (2018).
    """
    return pgd_attack(
        model,
        images,
        labels,
        epsilon=EPSILON,
        alpha=ALPHA,
        steps=PGD_TRAIN_STEPS,
        random_start=True,
    )


# ── Training helpers ───────────────────────────────────────────────────────────

def madry_train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """One epoch of Madry PGD-7 adversarial training.

    For each batch:
      1. Switch model to eval() for the inner PGD-7 attack (stable BN stats).
      2. Generate adversarial examples with PGD-7.
      3. Switch back to train() and update on the adversarial batch.

    Returns average loss, clean accuracy, and adversarial (training) accuracy
    over the epoch.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_seen = 0
    n_clean_correct = 0
    n_adv_correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Inner maximization (eval mode for stable BN)
        model.eval()
        x_adv = _pgd7_inner(model, images, labels)

        # Outer minimization (train mode)
        model.train()
        optimizer.zero_grad()
        logits_adv = model(x_adv)
        loss = criterion(logits_adv, labels)
        loss.backward()
        optimizer.step()

        batch_n = labels.shape[0]
        n_seen += batch_n
        total_loss += loss.item() * batch_n

        with torch.no_grad():
            model.eval()
            clean_preds = model(images).argmax(dim=1)
            adv_preds   = model(x_adv).argmax(dim=1)
            model.train()

        n_clean_correct += int((clean_preds == labels).sum().item())
        n_adv_correct   += int((adv_preds == labels).sum().item())

    return {
        "loss":       total_loss / max(n_seen, 1),
        "clean_acc":  n_clean_correct / max(n_seen, 1),
        "pgd7_acc":   n_adv_correct   / max(n_seen, 1),   # training proxy
    }


def evaluate_clean(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            n_correct += int((model(images).argmax(1) == labels).sum().item())
            n_total   += labels.shape[0]
    return n_correct / max(n_total, 1)


def evaluate_pgd40(model: nn.Module, loader, device: torch.device) -> float:
    """PGD-40 robust accuracy (evaluation budget — stronger than training attack)."""
    model.eval()
    n_correct = 0
    n_total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        x_adv = pgd_attack(
            model, images, labels,
            epsilon=EPSILON, alpha=ALPHA, steps=PGD_EVAL_STEPS,
        )
        with torch.no_grad():
            n_correct += int((model(x_adv).argmax(1) == labels).sum().item())
        n_total += labels.shape[0]
    return n_correct / max(n_total, 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Madry adversarial training on CIFAR-10 ResNet-18 (PGD-7 AT)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100). "
             "100 epochs reproduces the canonical Madry 2018 ~45%% PGD-40 result.",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Path to CIFAR-10 data directory (default: ./data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Initial SGD learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_PATH),
        help=f"Path to save model checkpoint (default: {CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--results",
        default=str(RESULTS_PATH),
        help=f"Path to save results JSON (default: {RESULTS_PATH})",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[madry-at] device    = {device}")
    print(f"[madry-at] epochs    = {args.epochs}")
    print(f"[madry-at] epsilon   = {EPSILON * 255:.1f}/255")
    print(f"[madry-at] alpha     = {ALPHA * 255:.1f}/255")
    print(f"[madry-at] PGD steps = {PGD_TRAIN_STEPS} (train) / {PGD_EVAL_STEPS} (eval)")

    # ── Data ────────────────────────────────────────────────────────────────────
    print("[madry-at] Loading CIFAR-10...")
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # ── Model + optimizer + scheduler ───────────────────────────────────────────
    model = ResNet18CIFAR10(num_classes=10).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    # Cosine annealing — standard choice for Madry AT (avoids step-LR epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training loop ────────────────────────────────────────────────────────────
    epoch_logs: list[dict] = []
    best_robust_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        stats = madry_train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        elapsed = time.time() - t0

        log_entry = {
            "epoch":      epoch,
            "loss":       round(stats["loss"], 4),
            "clean_acc":  round(stats["clean_acc"] * 100, 2),
            "pgd7_acc":   round(stats["pgd7_acc"] * 100, 2),
            "elapsed_s":  round(elapsed, 1),
        }
        epoch_logs.append(log_entry)

        print(
            f"[madry-at] epoch {epoch:3d}/{args.epochs} | "
            f"loss={stats['loss']:.4f} | "
            f"clean={stats['clean_acc']*100:.1f}% | "
            f"pgd7={stats['pgd7_acc']*100:.1f}%  [{elapsed:.0f}s]"
        )

        # Save a checkpoint whenever robust acc improves (PGD-7 proxy)
        if stats["pgd7_acc"] > best_robust_acc:
            best_robust_acc = stats["pgd7_acc"]

    total_time = time.time() - start_time
    print(f"\n[madry-at] Training complete in {total_time / 60:.1f} min.")

    # ── Final evaluation ─────────────────────────────────────────────────────────
    print("[madry-at] Computing final clean accuracy...")
    clean_acc = evaluate_clean(model, test_loader, device)
    print(f"[madry-at] Clean accuracy: {clean_acc * 100:.2f}%")

    print(f"[madry-at] Computing final PGD-{PGD_EVAL_STEPS} robust accuracy...")
    pgd40_acc = evaluate_pgd40(model, test_loader, device)
    print(f"[madry-at] PGD-{PGD_EVAL_STEPS} robust accuracy: {pgd40_acc * 100:.2f}%")

    # ── Save checkpoint ───────────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":                args.epochs,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "clean_accuracy":       clean_acc,
            "pgd40_robust_accuracy": pgd40_acc,
        },
        checkpoint_path,
    )
    print(f"[madry-at] Checkpoint saved to {checkpoint_path}")

    # ── Save results JSON ─────────────────────────────────────────────────────────
    results = {
        "epochs":                args.epochs,
        "clean_accuracy":        round(float(clean_acc), 6),
        "pgd40_robust_accuracy": round(float(pgd40_acc), 6),
        "training_method":       "Madry PGD-7 adversarial training",
        "epsilon":               float(EPSILON),
        "epsilon_255":           "8/255",
        "alpha_255":             "2/255",
        "pgd_train_steps":       PGD_TRAIN_STEPS,
        "pgd_eval_steps":        PGD_EVAL_STEPS,
        "optimizer":             "SGD momentum=0.9 wd=5e-4 nesterov",
        "lr_schedule":           f"CosineAnnealingLR T_max={args.epochs}",
        "batch_size":            args.batch_size,
        "training_time_minutes": round(total_time / 60, 1),
        "checkpoint":            str(checkpoint_path),
        "reference":             "Madry et al. (2018) ICLR — arXiv:1706.06083",
        "epoch_logs":            epoch_logs,
    }

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[madry-at] Results saved to {results_path}")

    # ── Summary ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Madry Adversarial Training — Final Results")
    print("=" * 60)
    print(f"  Epochs:                  {args.epochs}")
    print(f"  Clean accuracy:          {clean_acc * 100:6.2f}%")
    print(f"  PGD-40 robust accuracy:  {pgd40_acc * 100:6.2f}%  (eps=8/255)")
    print("=" * 60)
    print()
    print("  Expected: ~45% PGD-40 robust accuracy after 100 epochs")
    print("  (Madry et al. 2018 ICLR)")
    print()
    if pgd40_acc >= 0.40:
        print("  ✓ Result consistent with Madry et al. 2018 baseline.")
    else:
        print("  ✗ Result below expected 40%+ — consider training longer.")


if __name__ == "__main__":
    main()
