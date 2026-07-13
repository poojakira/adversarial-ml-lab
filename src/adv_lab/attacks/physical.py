"""Physical-World Adversarial Patch Attacks (Tier 5 -- Infrastructure Integrity).

Implements localized patch attacks optimized for physical realizability using
Expectation over Transformation (EoT):

    Brown et al., "Adversarial Patch" (NeurIPS 2017 Workshop).
    Athalye et al., "Synthesizing Robust Adversarial Examples" (ICML 2018).
    Eykholt et al., "Robust Physical-World Attacks on Deep Learning Visual
    Classification" (CVPR 2018).

If your patch only works in pixel space and dies when printed, it is not a
physical-world attack. This module constrains patches to the sRGB printable
gamut and evaluates robustness across simulated viewing angles, lighting
conditions, and camera noise models.

Physical attacks are first-class citizens in production threat modeling because
they bypass all digital defenses: input validation, rate limiting, API
authentication -- none of these matter when the attack lives on a sticker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


def _require_eval_mode(model: nn.Module) -> None:
    """Attacks must run against a model in eval() mode."""
    if model.training:
        raise ValueError(
            "model must be in eval() mode before attacking; call model.eval(). "
            "Attacking a model in train() mode gives stochastic, unreliable "
            "gradients (a common source of bogus robustness numbers)."
        )


@dataclass
class PatchResult:
    """Result of physical patch optimization.

    Attributes:
        patch: Optimized adversarial patch tensor, shape (C, H_p, W_p).
        success_rate: Fraction of test images misclassified with patch applied.
        printability_score: NPS-based printability score in [0, 1].
            1.0 means all patch pixels are within the printable gamut.
        angle_robustness: Success rates at each simulated viewing angle.
        lighting_robustness: Success rates at each lighting multiplier.
    """

    patch: Tensor
    success_rate: float
    printability_score: float
    angle_robustness: dict[float, float] = field(default_factory=dict)
    lighting_robustness: dict[float, float] = field(default_factory=dict)


def _apply_patch_to_images(
    images: Tensor,
    patch: Tensor,
    location: tuple[int, int] | None = None,
) -> Tensor:
    """Apply a patch to a batch of images at a specified location.

    Args:
        images: Input images, shape (N, C, H, W).
        patch: Adversarial patch, shape (C, H_p, W_p).
        location: Top-left (row, col) to place the patch. If None, center it.

    Returns:
        Patched images clamped to [0, 1].
    """
    n, c, h, w = images.shape
    _, ph, pw = patch.shape

    if location is None:
        row = (h - ph) // 2
        col = (w - pw) // 2
    else:
        row, col = location

    # Clamp location to valid range
    row = max(0, min(row, h - ph))
    col = max(0, min(col, w - pw))

    patched = images.clone()
    patched[:, :, row : row + ph, col : col + pw] = patch.unsqueeze(0)
    return patched.clamp(0.0, 1.0)


def _rotation_matrix_2d(angle_deg: float) -> Tensor:
    """Create a 2x3 affine rotation matrix for the given angle in degrees."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    # Rotation matrix (no translation)
    matrix = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]], dtype=torch.float32
    )
    return matrix


def _apply_affine_transform(images: Tensor, angle_deg: float) -> Tensor:
    """Apply affine rotation to a batch of images using grid sampling.

    Simulates viewing angle changes by rotating images. Uses bilinear
    interpolation and zero padding for out-of-bounds pixels.

    Args:
        images: Input tensor of shape (N, C, H, W).
        angle_deg: Rotation angle in degrees.

    Returns:
        Rotated images tensor of same shape.
    """
    n, c, h, w = images.shape

    # Build affine grid for rotation
    theta = _rotation_matrix_2d(angle_deg).unsqueeze(0).expand(n, -1, -1)
    grid = nn.functional.affine_grid(theta, images.shape, align_corners=False)
    rotated = nn.functional.grid_sample(
        images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return rotated.clamp(0.0, 1.0)


def _apply_lighting(images: Tensor, multiplier: float) -> Tensor:
    """Simulate lighting condition changes via scalar brightness multiplication.

    Args:
        images: Input tensor in [0, 1].
        multiplier: Brightness multiplier (0.5 = dim, 2.0 = bright).

    Returns:
        Brightness-adjusted images clamped to [0, 1].
    """
    return (images * multiplier).clamp(0.0, 1.0)


def _apply_camera_noise(
    images: Tensor,
    gaussian_std: float = 0.02,
    poisson_scale: float = 0.01,
    seed: int | None = None,
) -> Tensor:
    """Simulate camera sensor noise (Gaussian + Poisson approximation).

    Real cameras exhibit both read noise (Gaussian) and shot noise (Poisson).
    We approximate Poisson noise with a signal-dependent Gaussian.

    Args:
        images: Input tensor in [0, 1].
        gaussian_std: Standard deviation of additive Gaussian noise.
        poisson_scale: Scale factor for signal-dependent noise.
        seed: Optional random seed.

    Returns:
        Noisy images clamped to [0, 1].
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Additive Gaussian (read noise)
    gaussian = torch.randn_like(images) * gaussian_std

    # Signal-dependent noise (Poisson approximation)
    poisson_approx = torch.randn_like(images) * (images * poisson_scale).sqrt()

    noisy = images + gaussian + poisson_approx
    return noisy.clamp(0.0, 1.0)


def printability_constraint(
    patch: Tensor,
    printable_colors: Tensor | None = None,
) -> float:
    """Compute Non-Printability Score (NPS) for a patch.

    Measures how close each patch pixel is to the nearest color in the
    printable color set. A score of 1.0 means all pixels are exactly
    representable by the printer; lower scores indicate colors that will
    shift when printed.

    Reference: Sharif et al., "Accessorize to a Crime: Real and Stealthy
    Attacks on State-of-the-Art Face Recognition" (CCS 2016).

    Args:
        patch: Adversarial patch tensor, shape (C, H, W) in [0, 1].
        printable_colors: Tensor of printable RGB triplets, shape (K, C).
            If None, uses the full sRGB gamut (all values in [0, 1] are
            considered printable -- constraint is just the clamp).

    Returns:
        Printability score in [0, 1]. 1.0 = fully printable.
    """
    # Ensure patch is in valid sRGB range [0, 1]
    in_gamut = (patch >= 0.0) & (patch <= 1.0)
    base_score = float(in_gamut.float().mean().item())

    if printable_colors is None:
        # Without a specific printer profile, sRGB gamut = [0, 1]
        return base_score

    # Compute distance to nearest printable color for each pixel
    c, h, w = patch.shape
    pixels = patch.permute(1, 2, 0).reshape(-1, c)  # (H*W, C)
    # Distance to each printable color
    dists = torch.cdist(pixels.unsqueeze(0), printable_colors.unsqueeze(0)).squeeze(0)
    min_dists = dists.min(dim=1).values  # (H*W,)

    # Score: 1.0 - normalized mean distance
    max_possible_dist = math.sqrt(c)  # Max distance in unit cube
    nps = 1.0 - float(min_dists.mean().item()) / max_possible_dist
    return max(0.0, min(1.0, nps))


class PhysicalPatchAttack:
    """Expectation-over-Transformation (EoT) optimized physical patch attack.

    Generates adversarial patches that remain effective when printed and
    viewed under varying physical conditions (angle, lighting, noise).

    The optimization minimizes the expected loss over a distribution of
    transformations, producing patches robust to the physical world.

    Reference: Athalye et al., "Synthesizing Robust Adversarial Examples"
    (ICML 2018).

    Args:
        model: Target classifier in eval() mode.
        patch_size: (height, width) of the adversarial patch in pixels.
        target_class: Target class for the patch (untargeted if None).
        angles: List of viewing angles (degrees) for EoT.
        lighting_multipliers: List of brightness multipliers for EoT.
        noise_std: Gaussian noise std for camera noise simulation.
        lr: Learning rate for patch optimization.
        steps: Number of optimization steps.
    """

    def __init__(
        self,
        model: nn.Module,
        patch_size: tuple[int, int] = (32, 32),
        target_class: int | None = None,
        angles: Sequence[float] | None = None,
        lighting_multipliers: Sequence[float] | None = None,
        noise_std: float = 0.02,
        lr: float = 0.01,
        steps: int = 500,
    ) -> None:
        _require_eval_mode(model)
        self.model = model
        self.patch_size = patch_size
        self.target_class = target_class
        self.angles = list(angles) if angles is not None else [
            -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0
        ]
        self.lighting_multipliers = list(lighting_multipliers) if lighting_multipliers is not None else [
            0.5, 0.75, 1.0, 1.25, 1.5, 2.0
        ]
        self.noise_std = noise_std
        self.lr = lr
        self.steps = steps

    def optimize(
        self,
        images: Tensor,
        labels: Tensor,
        *,
        seed: int | None = None,
    ) -> PatchResult:
        """Optimize an adversarial patch using EoT over physical transformations.

        Args:
            images: Training images for patch optimization, shape (N, C, H, W).
            labels: True labels for the images, shape (N,).
            seed: Optional random seed for reproducibility.

        Returns:
            PatchResult with optimized patch and robustness metrics.

        Raises:
            ValueError: If model is in training mode.
        """
        _require_eval_mode(self.model)

        if seed is not None:
            torch.manual_seed(seed)

        n, c, h, w = images.shape
        ph, pw = self.patch_size

        # Initialize patch with random values in [0, 1]
        patch = torch.rand(c, ph, pw, requires_grad=True)
        optimizer = torch.optim.Adam([patch], lr=self.lr)

        for step in range(self.steps):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0)
            n_transforms = 0

            # EoT: average loss over transformations
            for angle in self.angles:
                for mult in self.lighting_multipliers:
                    # Apply patch to images
                    patched = _apply_patch_to_images(images, patch.clamp(0.0, 1.0))
                    # Apply physical transformations
                    transformed = _apply_affine_transform(patched, angle)
                    transformed = _apply_lighting(transformed, mult)
                    transformed = _apply_camera_noise(
                        transformed, gaussian_std=self.noise_std
                    )

                    # Compute loss
                    logits = self.model(transformed)
                    if self.target_class is not None:
                        # Targeted: minimize loss for target class
                        target_labels = torch.full_like(labels, self.target_class)
                        loss = nn.functional.cross_entropy(logits, target_labels)
                    else:
                        # Untargeted: maximize loss for true class
                        loss = -nn.functional.cross_entropy(logits, labels)

                    total_loss = total_loss + loss
                    n_transforms += 1

            avg_loss = total_loss / n_transforms
            avg_loss.backward()
            optimizer.step()

            # Project patch to sRGB gamut [0, 1]
            with torch.no_grad():
                patch.clamp_(0.0, 1.0)

        # Evaluate final patch
        final_patch = patch.detach().clamp(0.0, 1.0)
        success_rate = self._evaluate_patch(final_patch, images, labels)
        nps = printability_constraint(final_patch)
        angle_rob = self.multi_angle_robustness(final_patch, images, labels)
        lighting_rob = self.lighting_robustness(final_patch, images, labels)

        return PatchResult(
            patch=final_patch,
            success_rate=success_rate,
            printability_score=nps,
            angle_robustness=angle_rob,
            lighting_robustness=lighting_rob,
        )

    def _evaluate_patch(
        self, patch: Tensor, images: Tensor, labels: Tensor
    ) -> float:
        """Evaluate attack success rate of a patch on clean images."""
        patched = _apply_patch_to_images(images, patch)
        with torch.no_grad():
            preds = self.model(patched).argmax(dim=1)

        if self.target_class is not None:
            # Targeted: success = predicted as target
            success = (preds == self.target_class).float().mean()
        else:
            # Untargeted: success = misclassified
            success = (preds != labels).float().mean()

        return float(success.item())

    def multi_angle_robustness(
        self,
        patch: Tensor,
        images: Tensor,
        labels: Tensor,
    ) -> dict[float, float]:
        """Evaluate patch effectiveness across viewing angles (+/-30 degrees).

        Simulates perspective changes via affine rotation transforms.
        A robust physical patch should maintain high success rates across
        all tested angles.

        Args:
            patch: Optimized adversarial patch, shape (C, H_p, W_p).
            images: Test images, shape (N, C, H, W).
            labels: True labels, shape (N,).

        Returns:
            Dictionary mapping angle (degrees) to attack success rate.
        """
        results: dict[float, float] = {}
        for angle in self.angles:
            patched = _apply_patch_to_images(images, patch)
            transformed = _apply_affine_transform(patched, angle)
            with torch.no_grad():
                preds = self.model(transformed).argmax(dim=1)

            if self.target_class is not None:
                rate = float((preds == self.target_class).float().mean().item())
            else:
                rate = float((preds != labels).float().mean().item())
            results[angle] = rate

        return results

    def lighting_robustness(
        self,
        patch: Tensor,
        images: Tensor,
        labels: Tensor,
    ) -> dict[float, float]:
        """Evaluate patch effectiveness across lighting conditions.

        Tests brightness multipliers from 0.5x (dim) to 2.0x (bright).
        Physical patches must survive real-world illumination variance.

        Args:
            patch: Optimized adversarial patch, shape (C, H_p, W_p).
            images: Test images, shape (N, C, H, W).
            labels: True labels, shape (N,).

        Returns:
            Dictionary mapping brightness multiplier to attack success rate.
        """
        results: dict[float, float] = {}
        for mult in self.lighting_multipliers:
            patched = _apply_patch_to_images(images, patch)
            lit = _apply_lighting(patched, mult)
            with torch.no_grad():
                preds = self.model(lit).argmax(dim=1)

            if self.target_class is not None:
                rate = float((preds == self.target_class).float().mean().item())
            else:
                rate = float((preds != labels).float().mean().item())
            results[mult] = rate

        return results

    def camera_noise_model(
        self,
        patch: Tensor,
        images: Tensor,
        labels: Tensor,
        *,
        gaussian_std: float = 0.02,
        poisson_scale: float = 0.01,
        n_trials: int = 10,
    ) -> float:
        """Evaluate patch robustness under simulated camera sensor noise.

        Averages success rate over multiple noise realizations to estimate
        expected performance when captured by a real camera.

        Args:
            patch: Optimized adversarial patch, shape (C, H_p, W_p).
            images: Test images, shape (N, C, H, W).
            labels: True labels, shape (N,).
            gaussian_std: Read noise standard deviation.
            poisson_scale: Shot noise scale factor.
            n_trials: Number of noise realizations to average over.

        Returns:
            Average attack success rate under camera noise.
        """
        total_rate = 0.0
        for trial in range(n_trials):
            patched = _apply_patch_to_images(images, patch)
            noisy = _apply_camera_noise(
                patched, gaussian_std=gaussian_std, poisson_scale=poisson_scale
            )
            with torch.no_grad():
                preds = self.model(noisy).argmax(dim=1)

            if self.target_class is not None:
                rate = float((preds == self.target_class).float().mean().item())
            else:
                rate = float((preds != labels).float().mean().item())
            total_rate += rate

        return total_rate / n_trials
