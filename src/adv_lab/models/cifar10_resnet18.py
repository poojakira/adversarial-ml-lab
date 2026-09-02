"""
src/adv_lab/models/cifar10_resnet18.py
──────────────────────────────────────────────────────────────────────────────
CIFAR-10 data loader and ResNet-18 model for adversarial robustness benchmarks.

# MITRE ATLAS: AML.T0043  --  Craft Adversarial Data (model used as attack target)
# MITRE ATLAS: AML.T0015  --  Evade ML Model (evaluation of evasion robustness)

Standard ResNet-18 adapted for CIFAR-10 (32×32 images, 10 classes).
The architecture follows He et al. (2016) with the first conv layer changed
from 7×7 stride-2 to 3×3 stride-1 (standard practice for small images).

Reference results on this model (epsilon=8/255, PGD-40 steps, alpha=2/255):
  - Clean accuracy:            ~93% (undefended)
  - FGSM robust accuracy:      ~10-20% (single-step, gradient masking artifacts)
  - PGD-40 robust accuracy:     ~0%  (undefended  --  collapses completely)
  - C&W robust accuracy:        ~0%  (undefended)
  - Madry AT PGD-40 robust:    ~45%  (after 100 epochs adversarial training)

These results are consistent with the published Madry et al. (2018) paper and
the RobustBench CIFAR-10 L∞ leaderboard (eps=8/255).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

# ── ResNet building blocks ─────────────────────────────────────────────────────


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet18CIFAR10(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32×32 input, 10 classes).

    Changes from ImageNet ResNet-18:
      - First conv: 7×7 stride-2 → 3×3 stride-1 (preserves spatial resolution)
      - Removes the initial MaxPool layer
      - Final FC layer outputs 10 classes

    Parameters
    ----------
    num_classes:
        Number of output classes (10 for CIFAR-10).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 64

        # CIFAR-10 adapted stem: 3×3, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Weight initialisation (He et al.)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ── CIFAR-10 data loader ───────────────────────────────────────────────────────


def get_cifar10_loaders(
    data_dir: str | Path = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[object, object]:
    """Return (train_loader, test_loader) for CIFAR-10.

    Parameters
    ----------
    data_dir:
        Directory to download/cache CIFAR-10 data.
    batch_size:
        Minibatch size for both train and test loaders.
    num_workers:
        DataLoader worker processes.

    Returns
    -------
    tuple
        (train_loader, test_loader)  --  standard PyTorch DataLoader objects.
    """
    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for CIFAR-10 loading. Install with: pip install torchvision"
        ) from exc

    # Standard CIFAR-10 normalisation (mean/std computed over training set)
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader
