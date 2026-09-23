"""Tests for adversarial attack math properties (fast, no CIFAR-10 download).

These tests verify that FGSM and PGD produce adversarial examples that:
1. Differ from clean inputs (perturbation is non-zero for non-trivial models)
2. Stay within the epsilon L-inf ball
3. Remain clamped to valid image range [0, 1]
"""

import torch
import torch.nn as nn

from adv_lab.attacks.fgsm import fgsm_attack
from adv_lab.attacks.pgd import pgd_attack, pgd_l2


class SmallCNN(nn.Module):
    """Minimal CNN that produces non-zero gradients for testing."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _make_test_data(batch_size: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random test images and labels."""
    torch.manual_seed(42)
    images = torch.rand(batch_size, 3, 8, 8)  # Small 8x8 images for speed
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


def _make_model() -> SmallCNN:
    """Create a small model in eval mode with fixed weights."""
    torch.manual_seed(0)
    model = SmallCNN()
    model.eval()
    return model


class TestFGSMProperties:
    """Verify FGSM attack mathematical properties."""

    def test_fgsm_changes_input(self) -> None:
        """FGSM with non-zero epsilon must modify the input."""
        model = _make_model()
        images, labels = _make_test_data()
        epsilon = 8 / 255

        adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)

        # At least some pixels should differ
        assert not torch.allclose(
            adv_images, images
        ), "FGSM did not modify the input  --  attack is broken"

    def test_fgsm_linf_bound(self) -> None:
        """Perturbation must be within epsilon L-inf ball."""
        model = _make_model()
        images, labels = _make_test_data()
        epsilon = 8 / 255

        adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)
        perturbation = (adv_images - images).abs()

        assert (
            perturbation.max().item() <= epsilon + 1e-6
        ), f"FGSM perturbation {perturbation.max().item():.6f} exceeds epsilon {epsilon:.6f}"

    def test_fgsm_valid_range(self) -> None:
        """Adversarial examples must be in [0, 1]."""
        model = _make_model()
        images, labels = _make_test_data()
        epsilon = 8 / 255

        adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)

        assert adv_images.min().item() >= 0.0, "Adversarial image has values < 0"
        assert adv_images.max().item() <= 1.0, "Adversarial image has values > 1"

    def test_fgsm_zero_epsilon_no_change(self) -> None:
        """With epsilon=0, FGSM should return the input unchanged."""
        model = _make_model()
        images, labels = _make_test_data()

        adv_images = fgsm_attack(model, images, labels, epsilon=0.0)

        assert torch.allclose(adv_images, images), "FGSM with epsilon=0 should not modify the input"

    def test_fgsm_output_shape(self) -> None:
        """Output shape must match input shape."""
        model = _make_model()
        images, labels = _make_test_data()

        adv_images = fgsm_attack(model, images, labels, epsilon=8 / 255)

        assert adv_images.shape == images.shape

    def test_fgsm_output_detached(self) -> None:
        """Output must be detached from computation graph."""
        model = _make_model()
        images, labels = _make_test_data()

        adv_images = fgsm_attack(model, images, labels, epsilon=8 / 255)

        assert not adv_images.requires_grad


class TestPGDProperties:
    """Verify PGD L-inf attack mathematical properties."""

    def test_pgd_changes_input(self) -> None:
        """PGD with non-zero epsilon must modify the input."""
        model = _make_model()
        images, labels = _make_test_data()
        epsilon = 8 / 255

        adv_images = pgd_attack(model, images, labels, epsilon=epsilon, alpha=2 / 255, steps=5)

        assert not torch.allclose(
            adv_images, images
        ), "PGD did not modify the input  --  attack is broken"

    def test_pgd_linf_bound(self) -> None:
        """Perturbation must be within epsilon L-inf ball."""
        model = _make_model()
        images, labels = _make_test_data()
        epsilon = 8 / 255

        adv_images = pgd_attack(model, images, labels, epsilon=epsilon, alpha=2 / 255, steps=10)
        perturbation = (adv_images - images).abs()

        assert (
            perturbation.max().item() <= epsilon + 1e-6
        ), f"PGD perturbation {perturbation.max().item():.6f} exceeds epsilon {epsilon:.6f}"

    def test_pgd_valid_range(self) -> None:
        """Adversarial examples must be in [0, 1]."""
        model = _make_model()
        images, labels = _make_test_data()
        epsilon = 8 / 255

        adv_images = pgd_attack(model, images, labels, epsilon=epsilon, alpha=2 / 255, steps=10)

        assert adv_images.min().item() >= 0.0, "Adversarial image has values < 0"
        assert adv_images.max().item() <= 1.0, "Adversarial image has values > 1"

    def test_pgd_stronger_than_fgsm(self) -> None:
        """PGD should achieve at least as much loss increase as FGSM."""
        model = _make_model()
        images, labels = _make_test_data()
        epsilon = 8 / 255

        fgsm_adv = fgsm_attack(model, images, labels, epsilon=epsilon)
        pgd_adv = pgd_attack(
            model,
            images,
            labels,
            epsilon=epsilon,
            alpha=2 / 255,
            steps=20,
            random_start=False,
        )

        # PGD loss should be >= FGSM loss (PGD is iterative maximization)
        with torch.no_grad():
            fgsm_loss = nn.functional.cross_entropy(model(fgsm_adv), labels)
            pgd_loss = nn.functional.cross_entropy(model(pgd_adv), labels)

        assert (
            pgd_loss.item() >= fgsm_loss.item() - 1e-4
        ), f"PGD loss {pgd_loss.item():.4f} < FGSM loss {fgsm_loss.item():.4f}"

    def test_pgd_output_detached(self) -> None:
        """Output must be detached from computation graph."""
        model = _make_model()
        images, labels = _make_test_data()

        adv_images = pgd_attack(model, images, labels, epsilon=8 / 255, alpha=2 / 255, steps=3)

        assert not adv_images.requires_grad


class TestPGDL2Properties:
    """Verify PGD L2 attack mathematical properties."""

    def test_pgd_l2_changes_input(self) -> None:
        """PGD-L2 with non-zero epsilon must modify the input."""
        model = _make_model()
        images, labels = _make_test_data()
        epsilon = 0.5

        adv_images = pgd_l2(model, images, labels, epsilon=epsilon, alpha=0.1, steps=5)

        assert not torch.allclose(
            adv_images, images
        ), "PGD-L2 did not modify the input  --  attack is broken"

    def test_pgd_l2_norm_bound(self) -> None:
        """Perturbation L2 norm must be within epsilon for each sample."""
        model = _make_model()
        images, labels = _make_test_data()
        epsilon = 0.5

        adv_images = pgd_l2(model, images, labels, epsilon=epsilon, alpha=0.1, steps=10)

        batch_size = images.size(0)
        delta = (adv_images - images).view(batch_size, -1)
        l2_norms = delta.norm(p=2, dim=1)

        assert (
            l2_norms <= epsilon + 1e-4
        ).all(), f"PGD-L2 perturbation exceeds epsilon: max norm = {l2_norms.max().item():.4f}"

    def test_pgd_l2_valid_range(self) -> None:
        """Adversarial examples must be in [0, 1]."""
        model = _make_model()
        images, labels = _make_test_data()

        adv_images = pgd_l2(model, images, labels, epsilon=0.5, alpha=0.1, steps=10)

        assert adv_images.min().item() >= 0.0
        assert adv_images.max().item() <= 1.0
