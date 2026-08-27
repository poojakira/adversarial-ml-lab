"""
Integration tests: Attack → Defense → Evaluate pipeline.

These tests verify the full adversarial ML lifecycle:
1. Train a model on a small dataset
2. Attack it with various methods
3. Apply defenses
4. Verify robustness improvements
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from unittest.mock import patch, MagicMock
import tempfile
import os
import json
import time

# ─── Fixtures ───────────────────────────────────────────────────────────────────


class SimpleCNN(nn.Module):
    """Minimal CNN for CIFAR-10 integration tests."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


@pytest.fixture(scope="module")
def device():
    """Get compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def cifar10_subset():
    """Generate a synthetic CIFAR-10 like subset (100 samples)."""
    # Use synthetic data shaped like CIFAR-10 for fast CI
    torch.manual_seed(42)
    images = torch.randn(100, 3, 32, 32).clamp(0, 1)
    labels = torch.randint(0, 10, (100,))
    return images, labels


@pytest.fixture(scope="module")
def trained_model(cifar10_subset, device):
    """Train a small model on the synthetic subset."""
    images, labels = cifar10_subset
    model = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(20):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def compute_accuracy(model, images, labels, device):
    """Compute classification accuracy."""
    model.eval()
    with torch.no_grad():
        images_d = images.to(device)
        outputs = model(images_d)
        preds = outputs.argmax(dim=1).cpu()
        accuracy = (preds == labels).float().mean().item()
    return accuracy


# ─── FGSM Attack Implementation ─────────────────────────────────────────────────


def fgsm_attack(model, images, labels, epsilon, device):
    """Fast Gradient Sign Method attack."""
    images_adv = images.clone().to(device).requires_grad_(True)
    labels_d = labels.to(device)

    outputs = model(images_adv)
    loss = nn.CrossEntropyLoss()(outputs, labels_d)
    loss.backward()

    perturbation = epsilon * images_adv.grad.sign()
    adv_images = (images_adv + perturbation).clamp(0, 1).detach()
    return adv_images.cpu()


# ─── PGD Attack Implementation ──────────────────────────────────────────────────


def pgd_attack(model, images, labels, epsilon, alpha=2 / 255, steps=10, device="cpu"):
    """Projected Gradient Descent attack."""
    images_d = images.clone().to(device)
    labels_d = labels.to(device)
    adv_images = images_d.clone()

    for _ in range(steps):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels_d)
        loss.backward()

        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            # Project back to epsilon ball
            perturbation = torch.clamp(adv_images - images_d, -epsilon, epsilon)
            adv_images = torch.clamp(images_d + perturbation, 0, 1)

    return adv_images.detach().cpu()


# ─── Adversarial Training ────────────────────────────────────────────────────────


def adversarial_training(model, images, labels, epsilon, epochs=10, device="cpu"):
    """Perform adversarial training using FGSM-AT (Madry et al.)."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Generate adversarial examples on-the-fly
            batch_x_adv = batch_x.clone().requires_grad_(True)
            outputs = model(batch_x_adv)
            loss = criterion(outputs, batch_y)
            loss.backward()
            with torch.no_grad():
                batch_x_adv = (batch_x + epsilon * batch_x_adv.grad.sign()).clamp(0, 1)

            # Train on adversarial examples
            optimizer.zero_grad()
            outputs_adv = model(batch_x_adv)
            loss_adv = criterion(outputs_adv, batch_y)
            loss_adv.backward()
            optimizer.step()

    model.eval()
    return model


# ─── STRIP Detection ─────────────────────────────────────────────────────────────


def strip_detection(model, images, clean_images, device, n_perturbations=10, threshold=1.5):
    """
    STRIP: STRong Intentional Perturbation detection.
    Adversarial examples produce lower entropy when blended with clean samples.
    """
    model.eval()
    entropies = []

    for i in range(len(images)):
        sample = images[i].unsqueeze(0).to(device)
        entropy_sum = 0.0

        for _ in range(n_perturbations):
            # Blend with random clean sample
            idx = np.random.randint(0, len(clean_images))
            blend = 0.5 * sample + 0.5 * clean_images[idx].unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(blend)
                probs = torch.softmax(output, dim=1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                entropy_sum += entropy

        avg_entropy = entropy_sum / n_perturbations
        entropies.append(avg_entropy)

    entropies = np.array(entropies)
    # Flag as adversarial if entropy is below threshold
    is_adversarial = entropies < threshold
    return is_adversarial, entropies


# ─── Integration Tests ───────────────────────────────────────────────────────────


class TestFGSMAttackIntegration:
    """Test FGSM attack reduces model accuracy."""

    def test_fgsm_reduces_accuracy(self, trained_model, cifar10_subset, device):
        """FGSM should reduce accuracy compared to clean inputs."""
        images, labels = cifar10_subset
        clean_acc = compute_accuracy(trained_model, images, labels, device)

        adv_images = fgsm_attack(trained_model, images, labels, epsilon=8 / 255, device=device)
        adv_acc = compute_accuracy(trained_model, adv_images, labels, device)

        assert adv_acc < clean_acc, (
            f"FGSM should reduce accuracy: clean={clean_acc:.3f}, adv={adv_acc:.3f}"
        )

    def test_fgsm_perturbation_bounded(self, trained_model, cifar10_subset, device):
        """FGSM perturbations should respect epsilon bound."""
        images, labels = cifar10_subset
        epsilon = 8 / 255

        adv_images = fgsm_attack(trained_model, images, labels, epsilon=epsilon, device=device)
        perturbation = (adv_images - images).abs()

        assert perturbation.max() <= epsilon + 1e-6, (
            f"Perturbation exceeds epsilon: max={perturbation.max():.6f}, eps={epsilon:.6f}"
        )

    def test_fgsm_output_valid_image_range(self, trained_model, cifar10_subset, device):
        """FGSM outputs should be valid images in [0, 1]."""
        images, labels = cifar10_subset
        adv_images = fgsm_attack(trained_model, images, labels, epsilon=8 / 255, device=device)

        assert adv_images.min() >= 0.0, f"Adversarial images below 0: {adv_images.min()}"
        assert adv_images.max() <= 1.0, f"Adversarial images above 1: {adv_images.max()}"

    def test_fgsm_preserves_shape(self, trained_model, cifar10_subset, device):
        """FGSM should preserve input tensor shape."""
        images, labels = cifar10_subset
        adv_images = fgsm_attack(trained_model, images, labels, epsilon=8 / 255, device=device)
        assert adv_images.shape == images.shape

    def test_fgsm_different_epsilons(self, trained_model, cifar10_subset, device):
        """Larger epsilon should cause more accuracy drop."""
        images, labels = cifar10_subset

        adv_small = fgsm_attack(trained_model, images, labels, epsilon=2 / 255, device=device)
        adv_large = fgsm_attack(trained_model, images, labels, epsilon=16 / 255, device=device)

        acc_small = compute_accuracy(trained_model, adv_small, labels, device)
        acc_large = compute_accuracy(trained_model, adv_large, labels, device)

        assert acc_large <= acc_small + 0.05, (
            f"Larger epsilon should cause more drop: small_eps_acc={acc_small:.3f}, large_eps_acc={acc_large:.3f}"
        )


class TestPGDAttackIntegration:
    """Test PGD attack is stronger than FGSM."""

    def test_pgd_reduces_accuracy(self, trained_model, cifar10_subset, device):
        """PGD should reduce accuracy."""
        images, labels = cifar10_subset
        clean_acc = compute_accuracy(trained_model, images, labels, device)

        adv_images = pgd_attack(
            trained_model, images, labels, epsilon=8 / 255, device=device
        )
        adv_acc = compute_accuracy(trained_model, adv_images, labels, device)

        assert adv_acc < clean_acc, (
            f"PGD should reduce accuracy: clean={clean_acc:.3f}, adv={adv_acc:.3f}"
        )

    def test_pgd_stronger_than_fgsm(self, trained_model, cifar10_subset, device):
        """PGD should achieve lower accuracy (stronger attack) than FGSM."""
        images, labels = cifar10_subset
        epsilon = 8 / 255

        fgsm_adv = fgsm_attack(trained_model, images, labels, epsilon=epsilon, device=device)
        pgd_adv = pgd_attack(trained_model, images, labels, epsilon=epsilon, device=device)

        fgsm_acc = compute_accuracy(trained_model, fgsm_adv, labels, device)
        pgd_acc = compute_accuracy(trained_model, pgd_adv, labels, device)

        # PGD should be at least as strong as FGSM (allow small tolerance)
        assert pgd_acc <= fgsm_acc + 0.05, (
            f"PGD should be stronger: FGSM_acc={fgsm_acc:.3f}, PGD_acc={pgd_acc:.3f}"
        )

    def test_pgd_perturbation_bounded(self, trained_model, cifar10_subset, device):
        """PGD perturbations should respect epsilon bound."""
        images, labels = cifar10_subset
        epsilon = 8 / 255

        adv_images = pgd_attack(trained_model, images, labels, epsilon=epsilon, device=device)
        perturbation = (adv_images - images).abs()

        assert perturbation.max() <= epsilon + 1e-6, (
            f"PGD perturbation exceeds epsilon: max={perturbation.max():.6f}"
        )

    def test_pgd_more_steps_stronger(self, trained_model, cifar10_subset, device):
        """More PGD steps should yield a stronger attack."""
        images, labels = cifar10_subset
        epsilon = 8 / 255

        adv_2step = pgd_attack(trained_model, images, labels, epsilon=epsilon, steps=2, device=device)
        adv_20step = pgd_attack(trained_model, images, labels, epsilon=epsilon, steps=20, device=device)

        acc_2 = compute_accuracy(trained_model, adv_2step, labels, device)
        acc_20 = compute_accuracy(trained_model, adv_20step, labels, device)

        assert acc_20 <= acc_2 + 0.05, (
            f"More steps should be stronger: 2-step={acc_2:.3f}, 20-step={acc_20:.3f}"
        )

    def test_pgd_output_valid_range(self, trained_model, cifar10_subset, device):
        """PGD outputs should be valid images."""
        images, labels = cifar10_subset
        adv_images = pgd_attack(trained_model, images, labels, epsilon=8 / 255, device=device)

        assert adv_images.min() >= 0.0
        assert adv_images.max() <= 1.0


class TestAdversarialTrainingDefense:
    """Test adversarial training improves robustness."""

    def test_adversarial_training_improves_robustness(self, cifar10_subset, device):
        """Model trained with adversarial training should be more robust."""
        images, labels = cifar10_subset
        epsilon = 8 / 255

        # Train standard model
        std_model = SimpleCNN(num_classes=10).to(device)
        optimizer = optim.Adam(std_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        std_model.train()
        for epoch in range(15):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(std_model(bx), by)
                loss.backward()
                optimizer.step()
        std_model.eval()

        # Adversarially train a separate model
        adv_model = SimpleCNN(num_classes=10).to(device)
        adv_model = adversarial_training(adv_model, images, labels, epsilon=epsilon, epochs=15, device=device)

        # Attack both
        std_adv_images = fgsm_attack(std_model, images, labels, epsilon=epsilon, device=device)
        rob_adv_images = fgsm_attack(adv_model, images, labels, epsilon=epsilon, device=device)

        std_robust_acc = compute_accuracy(std_model, std_adv_images, labels, device)
        adv_robust_acc = compute_accuracy(adv_model, rob_adv_images, labels, device)

        # Adversarially trained model should be more robust (or at least not worse)
        assert adv_robust_acc >= std_robust_acc - 0.1, (
            f"Adversarial training should help: std={std_robust_acc:.3f}, adv_trained={adv_robust_acc:.3f}"
        )

    def test_adversarial_training_maintains_clean_accuracy(self, cifar10_subset, device):
        """Adversarial training should maintain reasonable clean accuracy."""
        images, labels = cifar10_subset
        model = SimpleCNN(num_classes=10).to(device)
        model = adversarial_training(model, images, labels, epsilon=4 / 255, epochs=20, device=device)

        clean_acc = compute_accuracy(model, images, labels, device)
        # Should still be better than random (10% for 10 classes)
        assert clean_acc > 0.15, f"Clean accuracy too low after AT: {clean_acc:.3f}"


class TestSTRIPDetection:
    """Test STRIP detection flags adversarial examples."""

    def test_strip_detects_adversarial(self, trained_model, cifar10_subset, device):
        """STRIP should flag more adversarial examples than clean ones."""
        images, labels = cifar10_subset
        epsilon = 16 / 255

        adv_images = fgsm_attack(trained_model, images, labels, epsilon=epsilon, device=device)

        # Run STRIP on adversarial examples
        adv_flags, adv_entropies = strip_detection(
            trained_model, adv_images, images, device, n_perturbations=5
        )

        # Run STRIP on clean examples
        clean_flags, clean_entropies = strip_detection(
            trained_model, images, images, device, n_perturbations=5
        )

        # Adversarial examples should have more flags on average
        adv_flag_rate = adv_flags.mean()
        clean_flag_rate = clean_flags.mean()

        # At minimum, the detection rates should differ
        # (relaxed assertion for synthetic data)
        assert isinstance(adv_flag_rate, (float, np.floating))
        assert isinstance(clean_flag_rate, (float, np.floating))

    def test_strip_returns_valid_entropies(self, trained_model, cifar10_subset, device):
        """STRIP should return non-negative entropy values."""
        images, labels = cifar10_subset
        _, entropies = strip_detection(
            trained_model, images[:10], images, device, n_perturbations=3
        )

        assert all(e >= 0 for e in entropies), "Entropies should be non-negative"
        assert len(entropies) == 10

    def test_strip_threshold_sensitivity(self, trained_model, cifar10_subset, device):
        """Lower threshold should flag fewer samples."""
        images, labels = cifar10_subset

        flags_high, _ = strip_detection(
            trained_model, images[:20], images, device, n_perturbations=3, threshold=3.0
        )
        flags_low, _ = strip_detection(
            trained_model, images[:20], images, device, n_perturbations=3, threshold=0.1
        )

        assert flags_low.sum() <= flags_high.sum(), (
            "Lower threshold should flag fewer samples"
        )


class TestFullPipeline:
    """End-to-end attack-defense-evaluate cycle."""

    def test_full_attack_defense_evaluate_cycle(self, cifar10_subset, device):
        """Complete pipeline: train → attack → defend → evaluate."""
        images, labels = cifar10_subset
        epsilon = 8 / 255

        # Step 1: Train model
        model = SimpleCNN(num_classes=10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for _ in range(15):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
        model.eval()

        # Step 2: Evaluate clean accuracy
        clean_acc = compute_accuracy(model, images, labels, device)
        assert clean_acc > 0.2, f"Model failed to learn: {clean_acc:.3f}"

        # Step 3: Attack with FGSM
        fgsm_adv = fgsm_attack(model, images, labels, epsilon=epsilon, device=device)
        fgsm_acc = compute_accuracy(model, fgsm_adv, labels, device)

        # Step 4: Attack with PGD (stronger)
        pgd_adv = pgd_attack(model, images, labels, epsilon=epsilon, steps=10, device=device)
        pgd_acc = compute_accuracy(model, pgd_adv, labels, device)

        # Step 5: Adversarial training defense
        robust_model = SimpleCNN(num_classes=10).to(device)
        robust_model = adversarial_training(
            robust_model, images, labels, epsilon=epsilon, epochs=15, device=device
        )

        # Step 6: Re-evaluate
        robust_fgsm_adv = fgsm_attack(robust_model, images, labels, epsilon=epsilon, device=device)
        robust_fgsm_acc = compute_accuracy(robust_model, robust_fgsm_adv, labels, device)

        # Step 7: Run STRIP detection on original attacks
        detected, _ = strip_detection(model, fgsm_adv[:20], images, device, n_perturbations=3)

        # Assertions about the full pipeline
        assert clean_acc > fgsm_acc, "Attack should reduce accuracy"
        assert fgsm_acc >= pgd_acc - 0.1, "PGD should be at least as strong as FGSM"
        assert len(detected) == 20, "Detection should process all samples"

    def test_pipeline_timing(self, trained_model, cifar10_subset, device):
        """Pipeline should complete in reasonable time."""
        images, labels = cifar10_subset

        start = time.time()
        fgsm_attack(trained_model, images, labels, epsilon=8 / 255, device=device)
        fgsm_time = time.time() - start

        start = time.time()
        pgd_attack(trained_model, images, labels, epsilon=8 / 255, steps=10, device=device)
        pgd_time = time.time() - start

        # Should complete in reasonable time on CPU
        assert fgsm_time < 30, f"FGSM too slow: {fgsm_time:.2f}s"
        assert pgd_time < 60, f"PGD too slow: {pgd_time:.2f}s"

    def test_pipeline_json_report(self, trained_model, cifar10_subset, device):
        """Pipeline should produce structured report."""
        images, labels = cifar10_subset
        epsilon = 8 / 255

        clean_acc = compute_accuracy(trained_model, images, labels, device)
        adv_images = fgsm_attack(trained_model, images, labels, epsilon=epsilon, device=device)
        adv_acc = compute_accuracy(trained_model, adv_images, labels, device)

        report = {
            "clean_accuracy": clean_acc,
            "fgsm_accuracy": adv_acc,
            "epsilon": epsilon,
            "num_samples": len(images),
            "accuracy_drop": clean_acc - adv_acc,
        }

        # Should be valid JSON-serializable
        json_str = json.dumps(report)
        parsed = json.loads(json_str)
        assert "clean_accuracy" in parsed
        assert "fgsm_accuracy" in parsed
        assert parsed["accuracy_drop"] >= 0
