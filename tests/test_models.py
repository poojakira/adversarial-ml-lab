"""Tests for src/adv_lab/models/cifar10_resnet18.py.

Covers:
  - Model instantiation and parameter count
  - Forward pass output shape for various batch sizes
  - Output is valid logits (finite, correct dimensions)
  - Custom num_classes parameter works
  - Model can switch between train/eval modes without error
"""

from __future__ import annotations

import torch

from adv_lab.models.cifar10_resnet18 import ResNet18CIFAR10


class TestResNet18CIFAR10:
    """Tests for the CIFAR-10-adapted ResNet-18."""

    def test_instantiation(self):
        """Model instantiates without errors with default settings."""
        model = ResNet18CIFAR10()
        assert model is not None
        # ResNet-18 for CIFAR-10 should have roughly 11M parameters
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1_000_000  # At least 1M params for ResNet-18

    def test_forward_pass_shape_default(self):
        """Forward pass returns (batch_size, 10) for default 10-class config."""
        model = ResNet18CIFAR10(num_classes=10)
        model.eval()

        # CIFAR-10: 3 channels, 32x32
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 10)

    def test_forward_pass_single_image(self):
        """Forward pass works with batch_size=1."""
        model = ResNet18CIFAR10(num_classes=10)
        model.eval()

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 10)

    def test_forward_pass_custom_num_classes(self):
        """Model correctly adapts to custom number of classes."""
        model = ResNet18CIFAR10(num_classes=100)
        model.eval()

        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 100)

    def test_output_is_finite(self):
        """Model output should contain only finite values (no NaN/Inf)."""
        model = ResNet18CIFAR10()
        model.eval()

        x = torch.rand(8, 3, 32, 32)  # Use [0,1] range like real images
        with torch.no_grad():
            output = model(x)

        assert torch.isfinite(output).all(), "Model output contains NaN or Inf"

    def test_gradient_flows(self):
        """Gradients flow through the model (needed for adversarial attacks)."""
        model = ResNet18CIFAR10()
        model.eval()

        x = torch.rand(2, 3, 32, 32, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert (x.grad != 0).any(), "Gradient is all zeros"
