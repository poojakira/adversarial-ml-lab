"""
REAL, reproducible small-CNN CIFAR-10 adversarial robustness benchmark.
======================================================================

This script trains a SMALL but REAL convolutional network on REAL CIFAR-10
(torchvision.datasets.CIFAR10, download=True) for a few epochs on CPU, then
runs the honest attack ladder (FGSM, PGD L-inf eps=8/255, C&W L2) against it
on the REAL CIFAR-10 test set.

WHY A SMALL CNN, NOT ResNet-18 @ 93%?
    torch here is CPU-ONLY. Training ResNet-18 to ~93% on CPU is infeasible in
    a few minutes. Instead we train a compact CNN for a handful of epochs and
    report the REAL clean accuracy we actually measure (typically ~60-70%).
    This is a SMALL-COMPUTE CPU run, NOT a state-of-the-art result. Every number
    emitted by this script comes from a run that actually executed here and is
    reproducible with the recorded seed.

The attacks in adv_lab expect inputs in [0, 1]. We therefore load CIFAR-10 with
ToTensor() only (no Normalize) so images live in [0, 1], and fold the standard
CIFAR-10 mean/std normalization INTO the model as a fixed first layer. This is
the correct way to attack in pixel space while the network sees normalized
inputs.
"""
# MITRE ATLAS: AML.T0043 - Craft Adversarial Data | AML.T0015 - Evade ML Model

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make src importable when running the script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from adv_lab.attacks.cw import cw_l2_attack
from adv_lab.attacks.fgsm import fgsm_attack
from adv_lab.attacks.pgd import pgd_attack

# ── Standard CIFAR-10 benchmark constants ───────────────────────────────────────
EPSILON = 8 / 255          # L-inf budget (canonical CIFAR-10 benchmark)
ALPHA = 2 / 255            # PGD step size
PGD_STEPS = 20             # PGD iterations for evaluation
CW_STEPS = 100             # C&W Adam iterations (kept small for CPU budget)
CW_C = 1.0                 # C&W loss trade-off constant
SEED = 42

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class SmallCNN(nn.Module):
    """A small but real CNN for CIFAR-10 with built-in input normalization.

    Normalization is a fixed (non-learned) first operation so that the attacks
    can operate on inputs in the [0, 1] pixel range while the convolutional
    stack still sees standardized inputs.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._mean) / self._std
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_loaders(data_dir: str, batch_size: int):
    """CIFAR-10 loaders in [0, 1] pixel space (ToTensor only, no Normalize)."""
    import torchvision
    import torchvision.transforms as transforms

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def train(model, loader, device, epochs, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        total, n = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        mean_loss = total / max(n, 1)
        dt = time.time() - t0
        print(f"[train] epoch {epoch + 1}/{epochs}  loss={mean_loss:.4f}  ({dt:.1f}s)")
        history.append(round(mean_loss, 4))
    return history


@torch.no_grad()
def clean_accuracy(model, loader, device, max_samples=None):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.shape[0]
        if max_samples is not None and total >= max_samples:
            break
    return correct / max(total, 1), total


def robust_accuracy(model, loader, device, attack_fn, max_samples, **kwargs):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        x_adv = attack_fn(model, images, labels, **kwargs)
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.shape[0]
        if max_samples is not None and total >= max_samples:
            break
    return correct / max(total, 1), total


def git_sha() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        # Fixed argv, resolved absolute git path, no shell, no untrusted input.
        out = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(_REPO_ROOT / "data"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--attack-samples",
        type=int,
        default=1000,
        help="Number of test samples used for the (expensive) attack ladder.",
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "results" / "cifar10_smallcnn_real.json"),
    )
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cpu")  # torch here is CPU-only.
    print(f"[bench] device={device}  seed={SEED}  eps={EPSILON:.6f} (8/255)")

    wall0 = time.time()

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    model = SmallCNN(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bench] SmallCNN parameters: {n_params:,}")

    t_train0 = time.time()
    loss_history = train(model, train_loader, device, args.epochs, args.lr)
    train_seconds = time.time() - t_train0

    print("[bench] measuring clean accuracy on full test set...")
    clean_acc, clean_n = clean_accuracy(model, test_loader, device)
    print(f"[bench]   clean accuracy = {clean_acc * 100:.2f}%  (n={clean_n})")

    print(f"[bench] FGSM (eps=8/255) on {args.attack_samples} samples...")
    t0 = time.time()
    fgsm_acc, fgsm_n = robust_accuracy(
        model, test_loader, device, fgsm_attack, args.attack_samples, epsilon=EPSILON
    )
    fgsm_secs = time.time() - t0
    print(f"[bench]   FGSM robust acc = {fgsm_acc * 100:.2f}%  (n={fgsm_n}, {fgsm_secs:.1f}s)")

    print(f"[bench] PGD-{PGD_STEPS} (eps=8/255, alpha=2/255) on {args.attack_samples} samples...")
    t0 = time.time()
    pgd_acc, pgd_n = robust_accuracy(
        model,
        test_loader,
        device,
        pgd_attack,
        args.attack_samples,
        epsilon=EPSILON,
        alpha=ALPHA,
        steps=PGD_STEPS,
    )
    pgd_secs = time.time() - t0
    print(f"[bench]   PGD-{PGD_STEPS} robust acc = {pgd_acc * 100:.2f}%  (n={pgd_n}, {pgd_secs:.1f}s)")

    print(f"[bench] C&W L2 ({CW_STEPS} steps, c={CW_C}) on {args.attack_samples} samples...")
    t0 = time.time()
    cw_acc, cw_n = robust_accuracy(
        model,
        test_loader,
        device,
        cw_l2_attack,
        args.attack_samples,
        c=CW_C,
        kappa=0.0,
        steps=CW_STEPS,
        lr=0.01,
    )
    cw_secs = time.time() - t0
    print(f"[bench]   C&W L2 robust acc = {cw_acc * 100:.2f}%  (n={cw_n}, {cw_secs:.1f}s)")

    wall_seconds = time.time() - wall0

    results = {
        "_schema": "adv-lab-benchmark-v1",
        "_status": "REAL_MEASURED - every number produced by an actual run on this machine",
        "_synthetic": False,
        "_reproducibility": (
            f"python scripts/run_real_smallcnn_benchmark.py --epochs {args.epochs} "
            f"--attack-samples {args.attack_samples}  (seed={SEED}, CPU-only torch)"
        ),
        "dataset": "CIFAR-10 (torchvision.datasets.CIFAR10, download=True)",
        "architecture": "SmallCNN (4-conv + 2-FC, built-in CIFAR-10 normalization)",
        "n_parameters": int(n_params),
        "framework": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
        "seed": SEED,
        "training": {
            "epochs": args.epochs,
            "optimizer": "Adam",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "train_loss_history": loss_history,
            "train_seconds": round(train_seconds, 1),
        },
        "epsilon": float(np.round(EPSILON, 8)),
        "epsilon_255": "8/255",
        "alpha_255": "2/255",
        "attack_eval_samples": args.attack_samples,
        "results": {
            "clean_accuracy": round(float(clean_acc), 6),
            "clean_eval_samples": clean_n,
            "fgsm": {
                "robust_accuracy": round(float(fgsm_acc), 6),
                "epsilon_255": "8/255",
                "samples": fgsm_n,
                "seconds": round(fgsm_secs, 1),
            },
            "pgd_linf": {
                "robust_accuracy": round(float(pgd_acc), 6),
                "epsilon_255": "8/255",
                "alpha_255": "2/255",
                "steps": PGD_STEPS,
                "samples": pgd_n,
                "seconds": round(pgd_secs, 1),
            },
            "cw_l2": {
                "robust_accuracy": round(float(cw_acc), 6),
                "c": CW_C,
                "kappa": 0.0,
                "steps": CW_STEPS,
                "samples": cw_n,
                "seconds": round(cw_secs, 1),
            },
        },
        "commit_sha": git_sha(),
        "wall_clock_seconds": round(wall_seconds, 1),
        "compute_budget": (
            "SMALL CPU RUN, NOT SOTA. Trained a compact CNN for a few epochs on "
            "CPU-only PyTorch. Clean accuracy is far below the ~93% ResNet-18 "
            "figure and is honest for this budget. Attack robust accuracy is "
            "measured on a subset of the test set for the expensive iterative/"
            "optimization attacks. Reproducible with the recorded seed."
        ),
        "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("  REAL SmallCNN CIFAR-10 Benchmark (small CPU budget)")
    print("=" * 60)
    print(f"  Clean accuracy:        {clean_acc * 100:6.2f}%  (full test set)")
    print(f"  FGSM  robust acc:      {fgsm_acc * 100:6.2f}%  (eps=8/255)")
    print(f"  PGD-{PGD_STEPS} robust acc:      {pgd_acc * 100:6.2f}%  (eps=8/255)")
    print(f"  C&W L2 robust acc:     {cw_acc * 100:6.2f}%")
    print("=" * 60)
    print(f"  wall clock: {wall_seconds:.1f}s   -> {out_path}")


if __name__ == "__main__":
    main()
