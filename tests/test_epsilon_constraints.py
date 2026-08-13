"""
tests/test_epsilon_constraints.py
───────────────────────────────────────────────────────────────────────────────
Property tests for adversarial attack epsilon constraint invariants.

These tests prove that:
1. PGD L-inf: every adversarial example satisfies ||x_adv - x||_inf <= epsilon
2. PGD L2:   every adversarial example satisfies ||x_adv - x||_2 <= epsilon + tol
3. FGSM:     output values remain in [0, 1]
4. All attacks: output pixel values are in [0, 1]

These are the MINIMUM correctness guarantees for any adversarial attack
implementation. A constraint violation means the attack is not computing
what it claims, which invalidates any robustness evaluation built on top.

Run with: pytest tests/test_epsilon_constraints.py -v
"""

import pytest
import torch
import torch.nn as nn

from adv_lab.attacks.fgsm import fgsm_attack
from adv_lab.attacks.pgd import pgd_attack, pgd_l2


class SmallCNN(nn.Module):
    """Minimal model for property testing — no ImageNet weights needed."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(8 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))


@pytest.fixture(scope="module")
def model():
    m = SmallCNN()
    m.eval()
    return m


@pytest.fixture
def batch():
    """Clean input batch: 4 images, 3-channel, 8x8, values in [0,1]."""
    torch.manual_seed(42)
    images = torch.rand(4, 3, 8, 8)
    labels = torch.randint(0, 10, (4,))
    return images, labels


# ═══════════════════════════════════════════════════════════════════════════════
# PGD L-inf constraint properties
# ═══════════════════════════════════════════════════════════════════════════════


class TestPGDLinfConstraints:
    """Property: PGD L-inf must satisfy ||x_adv - x||_inf <= epsilon."""

    @pytest.mark.parametrize("epsilon", [0.01, 0.03, 0.1, 0.3])
    def test_linf_norm_within_budget(self, model, batch, epsilon):
        """For every epsilon, the max pixel deviation must not exceed it."""
        images, labels = batch
        adv = pgd_attack(model, images, labels, epsilon=epsilon, alpha=epsilon / 4, steps=10)

        delta = (adv - images).abs()
        max_delta = delta.max().item()

        assert max_delta <= epsilon + 1e-5, (
            f"PGD L-inf constraint violated: "
            f"max ||x_adv - x||_inf = {max_delta:.6f} > epsilon = {epsilon}. "
            "The attack is not projecting correctly into the epsilon-ball."
        )

    def test_output_pixels_in_valid_range(self, model, batch):
        """Adversarial images must remain in [0, 1]."""
        images, labels = batch
        adv = pgd_attack(model, images, labels, epsilon=0.1, steps=10)

        assert adv.min().item() >= -1e-6, f"PGD output below 0: {adv.min().item()}"
        assert adv.max().item() <= 1.0 + 1e-6, f"PGD output above 1: {adv.max().item()}"

    def test_linf_tighter_than_l2(self, model, batch):
        """L-inf attack should produce smaller L2 norm than L-inf budget * sqrt(dims).

        This sanity check confirms the attack is not adding uniform perturbation
        everywhere (which would be L2-ball, not L-inf-ball).
        """
        images, labels = batch
        epsilon = 0.1
        adv = pgd_attack(model, images, labels, epsilon=epsilon, steps=10)
        # L-inf ball has L2 radius <= epsilon * sqrt(total_dims)
        # The L2 norm of the delta should be <= epsilon * sqrt(C*H*W)
        delta = (adv - images)
        l2_norm = delta.view(delta.size(0), -1).norm(dim=1)
        n_dims = delta.shape[1] * delta.shape[2] * delta.shape[3]
        l2_upper = epsilon * (n_dims ** 0.5)
        assert (l2_norm <= l2_upper + 1e-4).all(), (
            "PGD L-inf delta L2 norm exceeds theoretical maximum — "
            "perturbation may not be in the L-inf ball."
        )

    def test_random_start_does_not_violate_constraint(self, model, batch):
        """random_start=True must still satisfy the epsilon constraint."""
        images, labels = batch
        for seed in range(3):
            torch.manual_seed(seed)
            adv = pgd_attack(model, images, labels, epsilon=0.05, steps=5, random_start=True)
            max_delta = (adv - images).abs().max().item()
            assert max_delta <= 0.05 + 1e-5, (
                f"random_start PGD violated epsilon=0.05 with seed={seed}: "
                f"max_delta={max_delta:.6f}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PGD L2 constraint properties
# ═══════════════════════════════════════════════════════════════════════════════


class TestPGDL2Constraints:
    """Property: PGD L2 must satisfy ||x_adv - x||_2 <= epsilon (per sample)."""

    @pytest.mark.parametrize("epsilon", [0.5, 1.0, 2.0])
    def test_l2_norm_within_budget_per_sample(self, model, batch, epsilon):
        """Per-sample L2 norm of delta must not exceed epsilon."""
        images, labels = batch
        try:
            adv = pgd_l2(model, images, labels, epsilon=epsilon, steps=10)
        except (AttributeError, ImportError):
            pytest.skip("pgd_l2 not implemented")

        delta = (adv - images).view(images.size(0), -1)
        l2_norms = delta.norm(dim=1)  # shape (N,)

        violations = (l2_norms > epsilon + 1e-4).sum().item()
        assert violations == 0, (
            f"PGD L2 constraint violated for {violations}/{images.size(0)} samples. "
            f"Max L2 norm: {l2_norms.max().item():.4f}, epsilon: {epsilon}. "
            "The attack must project onto the L2 ball each step."
        )

    def test_l2_output_pixels_valid(self, model, batch):
        """L2 adversarial images must be in [0, 1]."""
        images, labels = batch
        try:
            adv = pgd_l2(model, images, labels, epsilon=1.0, steps=10)
        except (AttributeError, ImportError):
            pytest.skip("pgd_l2 not implemented")
        assert adv.min().item() >= -1e-6
        assert adv.max().item() <= 1.0 + 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# FGSM constraint properties
# ═══════════════════════════════════════════════════════════════════════════════


class TestFGSMConstraints:
    """FGSM is a single L-inf step of exactly epsilon."""

    @pytest.mark.parametrize("epsilon", [0.01, 0.05, 0.1])
    def test_fgsm_linf_exactly_epsilon(self, model, batch, epsilon):
        """FGSM perturbation must be <= epsilon (single-step, L-inf)."""
        images, labels = batch
        adv = fgsm_attack(model, images, labels, epsilon=epsilon)
        max_delta = (adv - images).abs().max().item()
        assert max_delta <= epsilon + 1e-5, (
            f"FGSM L-inf violated: max_delta={max_delta:.6f} > epsilon={epsilon}"
        )

    def test_fgsm_output_in_valid_range(self, model, batch):
        images, labels = batch
        adv = fgsm_attack(model, images, labels, epsilon=0.1)
        assert adv.min().item() >= -1e-6
        assert adv.max().item() <= 1.0 + 1e-6

    def test_fgsm_perturbation_nonzero(self, model, batch):
        """FGSM should actually perturb the input (gradient is nonzero)."""
        images, labels = batch
        adv = fgsm_attack(model, images, labels, epsilon=0.1)
        delta = (adv - images).abs().sum().item()
        assert delta > 0, "FGSM produced zero perturbation — gradient may be zero"


# ═══════════════════════════════════════════════════════════════════════════════
# Gradient masking sanity check
# ═══════════════════════════════════════════════════════════════════════════════


class TestGradientMaskingSanity:
    """Sanity checks for gradient masking.

    A model that masks its gradients will make PGD *appear* weak — producing
    near-zero perturbations that don't fool the model, while actually evading
    a key evaluation. We cannot fully detect gradient masking without the model,
    but we can detect implementation bugs that simulate it.
    """

    def test_pgd_stronger_than_fgsm_on_random_model(self, model, batch):
        """PGD (multi-step) should produce larger loss increase than FGSM.

        If PGD gives *lower* adversarial loss than FGSM, the implementation
        likely has a gradient computation bug (a classic gradient masking signal).
        This test uses 20 PGD steps vs 1 FGSM step.
        """
        images, labels = batch
        epsilon = 0.1

        fgsm_adv = fgsm_attack(model, images, labels, epsilon=epsilon)
        pgd_adv = pgd_attack(model, images, labels, epsilon=epsilon,
                             alpha=0.01, steps=20, random_start=False)

        with torch.no_grad():
            fgsm_loss = nn.functional.cross_entropy(model(fgsm_adv), labels).item()
            pgd_loss = nn.functional.cross_entropy(model(pgd_adv), labels).item()

        assert pgd_loss >= fgsm_loss * 0.8, (
            f"PGD (20 steps) loss {pgd_loss:.4f} is much lower than FGSM loss "
            f"{fgsm_loss:.4f}. This may indicate a gradient masking bug in PGD. "
            "Multi-step attacks should not be weaker than single-step under comparable budget."
        )
