"""Transferability analysis across heterogeneous model architectures.

Adversarial examples often transfer between models: an example crafted against
one architecture can fool a completely different architecture. This module
quantifies that phenomenon by evaluating adversarial examples generated on a
source model against multiple target architectures.

References:

* Papernot et al., "Transferability in Machine Learning: from Phenomena to
  Black-Box Attacks using Adversarial Samples" (arXiv 2016).
* Demontis et al., "Why Do Adversarial Attacks Transfer? Explaining
  Transferability of Evasion and Poisoning Attacks" (USENIX Security 2019).
* Tramer et al., "Ensemble Adversarial Training: Attacks and Defenses"
  (ICLR 2018). Showed that ensemble-based generation improves transferability.
* Liu et al., "Delving into Transferable Adversarial Examples and Black-box
  Attacks" (ICLR 2017). Studied cross-architecture transfer systematically.

The module provides:
* :class:`TransferabilityAnalyzer` -- evaluates adversarial examples across 3+
  heterogeneous architectures and reports per-architecture and ensemble transfer
  rates.
* Helper functions to create small architectural variants that simulate diverse
  network families (CNN, wide-CNN, deep-CNN) at the scale of the lab's 1x8x8
  synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


# --------------------------------------------------------------------------- #
# Heterogeneous architecture definitions
# --------------------------------------------------------------------------- #


class _ShallowCNN(nn.Module):
    """Shallow CNN variant (simulates a simple ConvNet baseline).

    Architecture: Conv(1->8, 3x3) -> ReLU -> Flatten -> FC -> classes.
    One conv layer, minimal depth. Analogous to a very simple LeNet variant.
    """

    def __init__(self, num_classes: int = 3, input_size: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * input_size * input_size, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


class _WideCNN(nn.Module):
    """Wide CNN variant (simulates a WideResNet-like architecture).

    Architecture: Conv(1->32, 3x3) -> ReLU -> Conv(32->64, 3x3) -> ReLU ->
    Flatten -> FC(128) -> FC(classes).
    Wide channels with fewer layers. Mimics the width-over-depth philosophy.
    """

    def __init__(self, num_classes: int = 3, input_size: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * input_size * input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


class _DeepCNN(nn.Module):
    """Deep CNN variant (simulates a deeper ResNet-like architecture).

    Architecture: Conv(1->8) -> ReLU -> Conv(8->16) -> ReLU -> Conv(16->32) ->
    ReLU -> Conv(32->32) -> ReLU -> Flatten -> FC(64) -> FC(classes).
    More layers with moderate width. Mimics depth-over-width philosophy.
    """

    def __init__(self, num_classes: int = 3, input_size: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * input_size * input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


class _MLPModel(nn.Module):
    """MLP-only model (simulates a non-convolutional architecture like ViT).

    Architecture: Flatten -> FC(256) -> ReLU -> FC(128) -> ReLU -> FC(classes).
    No convolutions at all, providing maximum architectural diversity from the
    CNN variants. Analogous to a patch-based MLP (simplified ViT without
    attention).
    """

    def __init__(self, num_classes: int = 3, input_size: int = 8) -> None:
        super().__init__()
        input_dim = 1 * input_size * input_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# TransferabilityAnalyzer
# --------------------------------------------------------------------------- #


@dataclass
class TransferResult:
    """Result of transferability evaluation for a single source-target pair.

    Attributes:
        source_name: name of the model that generated the adversarial examples.
        target_name: name of the model being evaluated.
        transfer_rate: fraction of adversarial examples that also fool the
            target model (computed only over examples that successfully fooled
            the source model).
        n_source_successful: number of examples that fooled the source model.
        n_transferred: number of those that also fooled the target.
    """

    source_name: str
    target_name: str
    transfer_rate: float
    n_source_successful: int
    n_transferred: int


@dataclass
class TransferabilityReport:
    """Complete transferability analysis report.

    Attributes:
        per_pair: per source-target pair results.
        per_architecture: average transfer rate for each target architecture
            (averaged over all sources).
        ensemble_transfer_rate: fraction of adversarial examples that fool
            ALL target architectures simultaneously.
        source_attack_name: name of the attack used to generate adversarial
            examples.
    """

    per_pair: list[TransferResult] = field(default_factory=list)
    per_architecture: dict[str, float] = field(default_factory=dict)
    ensemble_transfer_rate: float = 0.0
    source_attack_name: str = ""


class TransferabilityAnalyzer:
    """Evaluate adversarial example transferability across architectures.

    Creates and trains multiple heterogeneous architectures on the same task,
    generates adversarial examples against each, and measures how well those
    examples transfer to the other architectures.

    Usage::

        analyzer = TransferabilityAnalyzer(num_classes=3, input_size=8)
        analyzer.train_models(train_x, train_y, epochs=25)
        report = analyzer.evaluate(eval_x, eval_y, attack_fn=fgsm_attack)

    Args:
        num_classes: number of output classes.
        input_size: spatial dimension of input images (square assumed).
        architectures: optional dict mapping names to model constructors.
            If not provided, uses the 4 built-in heterogeneous architectures.
    """

    def __init__(
        self,
        num_classes: int = 3,
        input_size: int = 8,
        architectures: dict[str, Callable[[], nn.Module]] | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.input_size = input_size

        if architectures is not None:
            self._arch_factories = architectures
        else:
            self._arch_factories = {
                "ShallowCNN": lambda: _ShallowCNN(
                    num_classes=num_classes, input_size=input_size
                ),
                "WideCNN": lambda: _WideCNN(
                    num_classes=num_classes, input_size=input_size
                ),
                "DeepCNN": lambda: _DeepCNN(
                    num_classes=num_classes, input_size=input_size
                ),
                "MLP": lambda: _MLPModel(
                    num_classes=num_classes, input_size=input_size
                ),
            }

        self.models: dict[str, nn.Module] = {}

    def train_models(
        self,
        train_x: Tensor,
        train_y: Tensor,
        epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 128,
    ) -> dict[str, float]:
        """Train all architectures on the provided data.

        Args:
            train_x: training images in ``[0, 1]``, shape ``(N, 1, H, W)``.
            train_y: training labels, shape ``(N,)``.
            epochs: number of training epochs per model.
            lr: learning rate for Adam optimizer.
            batch_size: mini-batch size.

        Returns:
            Dict mapping architecture name to clean training accuracy.
        """
        accuracies: dict[str, float] = {}
        for name, factory in self._arch_factories.items():
            model = factory()
            _train_model(
                model, train_x, train_y, epochs=epochs, lr=lr, batch_size=batch_size
            )
            model.eval()
            self.models[name] = model
            # Compute training accuracy
            with torch.no_grad():
                preds = model(train_x).argmax(dim=1)
                acc = float((preds == train_y).float().mean().item())
            accuracies[name] = acc
        return accuracies

    def evaluate(
        self,
        images: Tensor,
        labels: Tensor,
        attack_fn: Callable[..., Tensor],
        attack_kwargs: dict | None = None,
        attack_name: str = "unknown",
    ) -> TransferabilityReport:
        """Evaluate transferability of adversarial examples across architectures.

        For each source model, generates adversarial examples using the provided
        attack function, then evaluates those examples against all other
        (target) models.

        Args:
            images: clean evaluation images in ``[0, 1]``, shape ``(N, 1, H, W)``.
            labels: ground-truth labels, shape ``(N,)``.
            attack_fn: attack function with signature
                ``(model, images, labels, **kwargs) -> adversarial_images``.
                Must follow the standard attack interface (model in eval mode,
                returns detached tensor in [0,1]).
            attack_kwargs: additional keyword arguments for the attack function.
            attack_name: name of the attack for the report.

        Returns:
            A :class:`TransferabilityReport` with per-pair, per-architecture,
            and ensemble transfer rates.

        Raises:
            ValueError: if fewer than 2 models have been trained.
        """
        if len(self.models) < 2:
            raise ValueError(
                "Need at least 2 trained models for transferability analysis. "
                "Call train_models() first."
            )

        if attack_kwargs is None:
            attack_kwargs = {}

        report = TransferabilityReport(source_attack_name=attack_name)
        per_target_rates: dict[str, list[float]] = {name: [] for name in self.models}

        # Track ensemble: for each example, whether ALL targets are fooled
        # across all source models
        ensemble_numerator = 0
        ensemble_denominator = 0

        for source_name, source_model in self.models.items():
            source_model.eval()
            # Generate adversarial examples against the source
            x_adv = attack_fn(source_model, images, labels, **attack_kwargs)

            # Determine which examples successfully fooled the source
            with torch.no_grad():
                source_preds_adv = source_model(x_adv).argmax(dim=1)
            source_success_mask = source_preds_adv != labels
            n_source_success = int(source_success_mask.sum().item())

            if n_source_success == 0:
                # Attack failed entirely on this source, skip
                for target_name in self.models:
                    if target_name == source_name:
                        continue
                    result = TransferResult(
                        source_name=source_name,
                        target_name=target_name,
                        transfer_rate=0.0,
                        n_source_successful=0,
                        n_transferred=0,
                    )
                    report.per_pair.append(result)
                    per_target_rates[target_name].append(0.0)
                continue

            # Get adversarial examples that fooled the source
            x_adv_success = x_adv[source_success_mask]
            y_success = labels[source_success_mask]

            # Check ensemble: all targets must be fooled
            all_targets_fooled = torch.ones(n_source_success, dtype=torch.bool)

            for target_name, target_model in self.models.items():
                if target_name == source_name:
                    continue

                target_model.eval()
                with torch.no_grad():
                    target_preds = target_model(x_adv_success).argmax(dim=1)
                n_transferred = int((target_preds != y_success).sum().item())
                transfer_rate = n_transferred / n_source_success

                result = TransferResult(
                    source_name=source_name,
                    target_name=target_name,
                    transfer_rate=transfer_rate,
                    n_source_successful=n_source_success,
                    n_transferred=n_transferred,
                )
                report.per_pair.append(result)
                per_target_rates[target_name].append(transfer_rate)

                # Update ensemble tracking
                all_targets_fooled = all_targets_fooled & (target_preds != y_success)

            ensemble_numerator += int(all_targets_fooled.sum().item())
            ensemble_denominator += n_source_success

        # Compute per-architecture average transfer rates
        for target_name, rates in per_target_rates.items():
            if rates:
                report.per_architecture[target_name] = sum(rates) / len(rates)
            else:
                report.per_architecture[target_name] = 0.0

        # Compute ensemble transfer rate
        if ensemble_denominator > 0:
            report.ensemble_transfer_rate = ensemble_numerator / ensemble_denominator
        else:
            report.ensemble_transfer_rate = 0.0

        return report


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _train_model(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> None:
    """Train a model on the provided data (internal helper)."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n = x.shape[0]

    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x[idx], y[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()


def create_heterogeneous_models(
    num_classes: int = 3,
    input_size: int = 8,
) -> dict[str, nn.Module]:
    """Create a set of 4 heterogeneous architectures for transferability testing.

    Returns untrained models. Use :meth:`TransferabilityAnalyzer.train_models`
    or the :func:`_train_model` helper to train them.

    The architectures span different design philosophies:
    - ShallowCNN: minimal depth, single conv layer (LeNet-like)
    - WideCNN: wide channels, moderate depth (WideResNet-like)
    - DeepCNN: many conv layers, moderate width (ResNet-like)
    - MLP: no convolutions, pure fully-connected (ViT/MLP-Mixer-like)

    Args:
        num_classes: number of output classes.
        input_size: spatial dimension of input (square assumed).

    Returns:
        Dict mapping architecture name to untrained model instance.
    """
    return {
        "ShallowCNN": _ShallowCNN(num_classes=num_classes, input_size=input_size),
        "WideCNN": _WideCNN(num_classes=num_classes, input_size=input_size),
        "DeepCNN": _DeepCNN(num_classes=num_classes, input_size=input_size),
        "MLP": _MLPModel(num_classes=num_classes, input_size=input_size),
    }
