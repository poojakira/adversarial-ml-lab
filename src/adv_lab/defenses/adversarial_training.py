"""Adversarial training (Madry et al., 2018).

The min-max defense: on every batch, generate worst-case perturbations with an
inner PGD attack, then train on those instead of the clean inputs::

    min_theta  E_(x,y) [ max_{||delta|| <= eps}  L(f_theta(x + delta), y) ]

Why PGD (and specifically PGD-7) for the inner maximization and not FGSM?
FGSM adversarial training is cheap but famously collapses into "catastrophic
overfitting": the model learns to defeat the single-step attack while staying
wide open to a multi-step one -- a textbook case of gradient masking. Multi-step
PGD approximates the inner max far better, and ~7 steps is the community's
accepted cost/robustness sweet spot for training (evaluation later uses many
more steps). See the README for the full argument.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.pgd import pgd_attack


def _pgd7(model: nn.Module, images: Tensor, labels: Tensor, epsilon: float) -> Tensor:
    """Default inner attack: PGD with 7 steps (standard for adv. training)."""
    return pgd_attack(
        model,
        images,
        labels,
        epsilon=epsilon,
        alpha=max(epsilon / 4.0, 1e-3),
        steps=7,
        random_start=True,
    )


class AdversarialTrainer:
    """Trains a classifier on PGD-generated adversarial examples.

    Args:
        model: the classifier to harden.
        optimizer: optimizer over ``model.parameters()``.
        attack_fn: inner attack with signature
            ``fn(model, images, labels, epsilon) -> Tensor``. Defaults to PGD-7.
        epsilon: perturbation budget handed to the inner attack.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        attack_fn: Callable[..., Tensor] | None = None,
        epsilon: float = 0.03,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.attack_fn = attack_fn if attack_fn is not None else _pgd7
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()

    def _generate(self, images: Tensor, labels: Tensor) -> Tensor:
        """Run the inner attack with the model temporarily in eval() mode.

        Attacks require eval() (stable BatchNorm/Dropout); training requires
        train(). We flip modes around the attack and always restore train().
        """
        was_training = self.model.training
        self.model.eval()
        try:
            adv = self.attack_fn(self.model, images, labels, epsilon=self.epsilon)
        finally:
            if was_training:
                self.model.train()
        return adv

    def train_epoch(self, dataloader) -> dict:
        """One epoch of adversarial training.

        Returns:
            ``{"loss": float, "clean_acc": float, "robust_acc": float}`` averaged
            over the epoch. ``robust_acc`` is accuracy on the inner-attack
            adversarial batch (a cheap running proxy, not the final benchmark).
        """
        self.model.train()
        total_loss = 0.0
        n_seen = 0
        n_clean_correct = 0
        n_robust_correct = 0

        for images, labels in dataloader:
            adv = self._generate(images, labels)

            self.model.train()
            self.optimizer.zero_grad()
            adv_logits = self.model(adv)
            loss = self.criterion(adv_logits, labels)
            loss.backward()
            self.optimizer.step()

            batch_n = labels.shape[0]
            n_seen += batch_n
            total_loss += loss.item() * batch_n

            with torch.no_grad():
                self.model.eval()
                clean_pred = self.model(images).argmax(dim=1)
                robust_pred = self.model(adv).argmax(dim=1)
                self.model.train()
            n_clean_correct += int((clean_pred == labels).sum().item())
            n_robust_correct += int((robust_pred == labels).sum().item())

        return {
            "loss": total_loss / max(n_seen, 1),
            "clean_acc": n_clean_correct / max(n_seen, 1),
            "robust_acc": n_robust_correct / max(n_seen, 1),
        }

    def evaluate_robust(self, dataloader, attack_fn: Callable[..., Tensor]) -> float:
        """Robust accuracy under a caller-supplied attack.

        Args:
            dataloader: yields ``(images, labels)`` batches.
            attack_fn: ``fn(model, images, labels, epsilon) -> Tensor``.

        Returns:
            Fraction of examples still classified correctly under attack.
        """
        self.model.eval()
        n_seen = 0
        n_correct = 0

        for images, labels in dataloader:
            adv = attack_fn(self.model, images, labels, epsilon=self.epsilon)
            with torch.no_grad():
                pred = self.model(adv).argmax(dim=1)
            n_seen += labels.shape[0]
            n_correct += int((pred == labels).sum().item())

        return n_correct / max(n_seen, 1)
