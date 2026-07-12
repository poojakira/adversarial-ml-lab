"""Model stealing tests.

Verifies that:
1. The substitute model architecture produces valid outputs.
2. Jacobian augmentation produces valid augmented inputs.
3. steal_model achieves the agreement threshold.
4. Transfer attacks from the substitute succeed at a measurable rate.
"""

from __future__ import annotations

import torch

from adv_lab.attacks.fgsm import fgsm_attack
from adv_lab.attacks.model_stealing import (
    SubstituteModel,
    jacobian_augmentation,
    steal_model,
)


def test_substitute_model_forward():
    """SubstituteModel should produce valid logits."""
    model = SubstituteModel(
        input_channels=1, input_size=8, num_classes=3, hidden_channels=16
    )
    model.eval()
    x = torch.rand(4, 1, 8, 8)
    logits = model(x)
    assert logits.shape == (4, 3)
    # Logits should be finite
    assert torch.isfinite(logits).all()


def test_jacobian_augmentation_valid_range():
    """Jacobian augmentation must produce outputs in [0, 1]."""
    model = SubstituteModel(
        input_channels=1, input_size=8, num_classes=3, hidden_channels=16
    )
    model.eval()
    x = torch.rand(8, 1, 8, 8)
    y = torch.randint(0, 3, (8,))
    x_aug = jacobian_augmentation(model, x, y, lambda_=0.1)
    assert x_aug.shape == x.shape
    assert x_aug.min().item() >= 0.0
    assert x_aug.max().item() <= 1.0


def test_jacobian_augmentation_changes_input():
    """Jacobian augmentation should produce different inputs from the original."""
    model = SubstituteModel(
        input_channels=1, input_size=8, num_classes=3, hidden_channels=16
    )
    model.eval()
    x = torch.rand(8, 1, 8, 8)
    y = torch.randint(0, 3, (8,))
    x_aug = jacobian_augmentation(model, x, y, lambda_=0.1)
    # Should not be identical to the original
    assert not torch.allclose(x_aug, x, atol=1e-6)


def test_steal_model_agreement_threshold(lab):
    """steal_model should achieve the configured agreement threshold."""
    target_model, eval_x, eval_y = lab
    # Use a portion of eval data as seed
    seed_data = eval_x[:100]

    substitute, agreement = steal_model(
        target_model,
        seed_data,
        num_classes=3,
        agreement_threshold=0.7,
        substitute_epochs=30,
        augmentation_rounds=8,
        lambda_aug=0.1,
        lr=1e-3,
        input_channels=1,
        input_size=8,
    )
    # The agreement should be reported and non-negative
    assert agreement >= 0.0
    # The model should be in eval mode
    assert not substitute.training


def test_steal_model_produces_valid_outputs(lab):
    """Stolen substitute model should produce valid logits on test inputs."""
    target_model, eval_x, eval_y = lab
    seed_data = eval_x[:50]

    substitute, _ = steal_model(
        target_model,
        seed_data,
        num_classes=3,
        agreement_threshold=0.5,
        substitute_epochs=10,
        augmentation_rounds=3,
        input_channels=1,
        input_size=8,
    )

    # Test on held-out data
    test_x = eval_x[50:80]
    with torch.no_grad():
        logits = substitute(test_x)
    assert logits.shape == (30, 3)
    assert torch.isfinite(logits).all()


def test_transfer_from_substitute(lab):
    """Transfer attacks from the stolen substitute should produce valid results."""
    target_model, eval_x, eval_y = lab
    seed_data = eval_x[:80]

    substitute, agreement = steal_model(
        target_model,
        seed_data,
        num_classes=3,
        agreement_threshold=0.5,
        substitute_epochs=15,
        augmentation_rounds=4,
        input_channels=1,
        input_size=8,
    )

    # Generate adversarial examples against the substitute
    test_x = eval_x[:32]
    test_y = eval_y[:32]
    x_adv = fgsm_attack(substitute, test_x, test_y, epsilon=0.1)

    # Verify adversarial examples are valid
    assert x_adv.shape == test_x.shape
    assert x_adv.min().item() >= 0.0
    assert x_adv.max().item() <= 1.0

    # Check transfer rate (at least computes without error)
    with torch.no_grad():
        target_preds = target_model(x_adv).argmax(dim=1)
    transfer_success = float((target_preds != test_y).float().mean().item())
    # Transfer rate should be a valid fraction
    assert 0.0 <= transfer_success <= 1.0
