"""Post-processing defense evasion attacks.

Attacks that craft adversarial examples robust to common input-transformation
defenses:
  * **JPEG compression** at quality 50-95 via a differentiable JPEG
    approximation (DCT -> quantize -> round).
  * **Feature squeezing** -- bit-depth reduction and spatial smoothing
    (Xu et al., "Feature Squeezing", NDSS 2018).
  * **Detector evasion** -- bypass Local Intrinsic Dimensionality (LID,
    Ma et al., ICLR 2018) and Mahalanobis distance detectors (Lee et al.,
    NeurIPS 2018) via regularization penalties.

References:
  - Dziugaite et al., "A study of the effect of JPG compression on
    adversarial images" (2016).
  - Shin and Song, "JPEG-resistant Adversarial Images" (NeurIPS 2017 Workshop).
  - Xu et al., "Feature Squeezing: Detecting Adversarial Examples in Deep
    Neural Networks" (NDSS 2018).
  - Ma et al., "Characterizing Adversarial Subspaces Using Local Intrinsic
    Dimensionality" (ICLR 2018).
  - Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution
    Samples and Adversarial Attacks" (NeurIPS 2018).
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode


# ---------------------------------------------------------------------------
# Differentiable JPEG Approximation
# ---------------------------------------------------------------------------


def _dct_matrix(n: int = 8) -> Tensor:
    """Compute the DCT-II basis matrix for an n x n block.

    Returns:
        DCT matrix of shape ``(n, n)``.
    """
    mat = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            if k == 0:
                mat[k, i] = 1.0 / math.sqrt(n)
            else:
                mat[k, i] = math.sqrt(2.0 / n) * math.cos(
                    math.pi * (2 * i + 1) * k / (2 * n)
                )
    return mat


def _standard_quantization_table(quality: int = 75) -> Tensor:
    """JPEG standard luminance quantization table scaled by quality factor.

    Args:
        quality: JPEG quality (1-100). Lower = more compression.

    Returns:
        Quantization table of shape ``(8, 8)``.
    """
    # Standard JPEG luminance quantization table
    base = torch.tensor(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ],
        dtype=torch.float32,
    )

    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality

    table = torch.floor((base * scale + 50.0) / 100.0)
    table = table.clamp(min=1.0)
    return table


def _differentiable_round(x: Tensor) -> Tensor:
    """Differentiable approximation of rounding using straight-through estimator."""
    return x + (torch.round(x) - x).detach()


def _jpeg_compress_differentiable(
    images: Tensor, quality: int = 75
) -> Tensor:
    """Differentiable JPEG compression approximation.

    Processes each 8x8 block with DCT, quantization, differentiable rounding,
    dequantization, and inverse DCT.

    Args:
        images: input tensor in [0, 1], shape ``(N, C, H, W)``.
        quality: JPEG quality factor (1-100).

    Returns:
        JPEG-compressed images (approximately), same shape.
    """
    device = images.device
    N, C, H, W = images.shape

    # Pad to multiple of 8
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h > 0 or pad_w > 0:
        images = nn.functional.pad(images, (0, pad_w, 0, pad_h), mode="reflect")

    _, _, H_pad, W_pad = images.shape

    # Scale to [0, 255] and shift to [-128, 128]
    x = images * 255.0 - 128.0

    # DCT basis
    dct_mat = _dct_matrix(8).to(device)
    q_table = _standard_quantization_table(quality).to(device)

    # Process 8x8 blocks
    # Reshape to blocks: (N*C, H/8, 8, W/8, 8) -> blocks
    x = x.view(N * C, H_pad // 8, 8, W_pad // 8, 8)
    x = x.permute(0, 1, 3, 2, 4).contiguous()  # (N*C, H/8, W/8, 8, 8)
    shape = x.shape
    x = x.view(-1, 8, 8)  # (num_blocks, 8, 8)

    # Forward DCT: D @ block @ D^T
    dct_coeffs = dct_mat @ x @ dct_mat.T

    # Quantize with differentiable rounding
    quantized = _differentiable_round(dct_coeffs / q_table.unsqueeze(0))

    # Dequantize
    dequantized = quantized * q_table.unsqueeze(0)

    # Inverse DCT: D^T @ coeffs @ D
    reconstructed = dct_mat.T @ dequantized @ dct_mat

    # Reshape back
    reconstructed = reconstructed.view(shape)
    reconstructed = reconstructed.permute(0, 1, 3, 2, 4).contiguous()
    reconstructed = reconstructed.view(N * C, H_pad, W_pad)
    reconstructed = reconstructed.view(N, C, H_pad, W_pad)

    # Undo shift and scale
    reconstructed = (reconstructed + 128.0) / 255.0

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        reconstructed = reconstructed[:, :, :H, :W]

    return reconstructed.clamp(0.0, 1.0)


def jpeg_robust_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.05,
    alpha: float = 0.01,
    steps: int = 100,
    quality_range: Tuple[int, int] = (50, 95),
) -> Tensor:
    """Craft adversarial examples that survive JPEG compression.

    Optimizes perturbations through a differentiable JPEG approximation so
    that the adversarial example remains effective after JPEG compression at
    various quality levels (50-95).

    At each step, a random quality from the range is sampled to improve
    robustness across the quality spectrum.

    Args:
        model: target classifier in ``eval()`` mode.
        images: clean inputs in ``[0, 1]``, shape ``(N, C, H, W)``.
        labels: ground-truth class indices.
        epsilon: L-inf perturbation budget.
        alpha: per-step size.
        steps: number of optimization steps.
        quality_range: tuple of (min_quality, max_quality) for JPEG.

    Returns:
        Detached adversarial images that survive JPEG compression.
    """
    _require_eval_mode(model)

    x_orig = images.clone().detach()
    x_adv = x_orig.clone().detach()

    # Random start
    noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)

        # Random quality for this step
        quality = torch.randint(
            quality_range[0], quality_range[1] + 1, (1,)
        ).item()

        # Pass through differentiable JPEG
        x_compressed = _jpeg_compress_differentiable(x_adv, quality=int(quality))
        logits = model(x_compressed)
        loss = nn.functional.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]

        # PGD step
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    return x_adv.detach()


# ---------------------------------------------------------------------------
# Feature Squeezing Robust Attack
# ---------------------------------------------------------------------------


def _bit_depth_reduce(images: Tensor, bits: int = 4) -> Tensor:
    """Reduce bit depth of images (differentiable approximation).

    Args:
        images: input in [0, 1].
        bits: target bit depth.

    Returns:
        Bit-depth-reduced images.
    """
    levels = 2.0**bits - 1.0
    quantized = _differentiable_round(images * levels) / levels
    return quantized.clamp(0.0, 1.0)


def _spatial_smoothing(images: Tensor, kernel_size: int = 3, sigma: float = 1.0) -> Tensor:
    """Apply Gaussian spatial smoothing (differentiable).

    Args:
        images: input tensor, shape ``(N, C, H, W)``.
        kernel_size: size of the Gaussian kernel.
        sigma: standard deviation of the Gaussian.

    Returns:
        Smoothed images.
    """
    # Create Gaussian kernel
    coords = torch.arange(kernel_size, dtype=images.dtype, device=images.device)
    coords = coords - (kernel_size - 1) / 2.0
    grid = coords.unsqueeze(0).pow(2) + coords.unsqueeze(1).pow(2)
    kernel = torch.exp(-grid / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # Apply as depthwise convolution
    C = images.shape[1]
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)
    padding = kernel_size // 2
    smoothed = nn.functional.conv2d(images, kernel, padding=padding, groups=C)
    return smoothed


def feature_squeeze_robust(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.05,
    alpha: float = 0.01,
    steps: int = 100,
    bit_depth: int = 4,
    smooth_kernel: int = 3,
    smooth_sigma: float = 1.0,
) -> Tensor:
    """Craft adversarial examples that survive feature squeezing defenses.

    Optimizes through differentiable approximations of:
      1. Bit-depth reduction (N-bit quantization).
      2. Spatial smoothing (Gaussian blur).

    The attack alternates between applying one or both defenses during
    optimization to ensure robustness to the full defense pipeline.

    Reference: Xu et al., "Feature Squeezing: Detecting Adversarial Examples
    in Deep Neural Networks" (NDSS 2018).

    Args:
        model: target classifier in ``eval()`` mode.
        images: clean inputs in ``[0, 1]``.
        labels: ground-truth class indices.
        epsilon: L-inf perturbation budget.
        alpha: per-step size.
        steps: number of optimization steps.
        bit_depth: target bit depth for reduction.
        smooth_kernel: Gaussian kernel size.
        smooth_sigma: Gaussian sigma.

    Returns:
        Detached adversarial images that survive feature squeezing.
    """
    _require_eval_mode(model)

    x_orig = images.clone().detach()
    x_adv = x_orig.clone().detach()

    # Random start
    noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

    for step in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)

        # Alternate defenses: both, bit-depth only, smoothing only
        mode = step % 3
        if mode == 0:
            # Both defenses
            x_squeezed = _bit_depth_reduce(x_adv, bits=bit_depth)
            x_squeezed = _spatial_smoothing(
                x_squeezed, kernel_size=smooth_kernel, sigma=smooth_sigma
            )
        elif mode == 1:
            # Bit-depth only
            x_squeezed = _bit_depth_reduce(x_adv, bits=bit_depth)
        else:
            # Smoothing only
            x_squeezed = _spatial_smoothing(
                x_adv, kernel_size=smooth_kernel, sigma=smooth_sigma
            )

        logits = model(x_squeezed)
        loss = nn.functional.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    return x_adv.detach()


# ---------------------------------------------------------------------------
# Detector Evasion (LID + Mahalanobis)
# ---------------------------------------------------------------------------


def _compute_lid_score(
    features: Tensor, k: int = 20
) -> Tensor:
    """Compute Local Intrinsic Dimensionality estimate.

    LID is estimated using the maximum likelihood estimator based on
    k-nearest neighbor distances (Ma et al., ICLR 2018).

    Args:
        features: feature representations, shape ``(N, D)``.
        k: number of nearest neighbors.

    Returns:
        LID scores, shape ``(N,)``.
    """
    # Pairwise distances
    dists = torch.cdist(features, features, p=2)  # (N, N)
    # Sort distances (exclude self at distance 0)
    sorted_dists, _ = dists.sort(dim=1)
    # Take k nearest (skip self at index 0)
    knn_dists = sorted_dists[:, 1 : k + 1]  # (N, k)
    # Clamp for numerical stability
    knn_dists = knn_dists.clamp(min=1e-10)
    # MLE estimate: LID = -k / sum(log(d_i / d_k))
    max_dist = knn_dists[:, -1:].clamp(min=1e-10)  # d_k
    log_ratios = torch.log(knn_dists / max_dist)  # log(d_i / d_k)
    lid = -1.0 * k / log_ratios.sum(dim=1)
    return lid


def _compute_mahalanobis_score(
    features: Tensor,
    class_means: Tensor,
    precision: Tensor,
) -> Tensor:
    """Compute Mahalanobis distance score.

    Uses the class-conditional Gaussian model from Lee et al. (NeurIPS 2018).

    Args:
        features: input features, shape ``(N, D)``.
        class_means: per-class mean features, shape ``(C, D)``.
        precision: shared precision matrix (inverse covariance), shape ``(D, D)``.

    Returns:
        Minimum Mahalanobis distance across classes, shape ``(N,)``.
    """
    N = features.shape[0]
    C = class_means.shape[0]

    scores = torch.zeros(N, C, device=features.device)
    for c in range(C):
        diff = features - class_means[c].unsqueeze(0)  # (N, D)
        # Mahalanobis: diff @ precision @ diff^T (per-sample)
        scores[:, c] = (diff @ precision * diff).sum(dim=1)

    # Minimum distance to any class
    min_scores, _ = scores.min(dim=1)
    return min_scores


def detector_evasion(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    feature_extractor: Callable[[Tensor], Tensor],
    epsilon: float = 0.05,
    alpha: float = 0.01,
    steps: int = 100,
    lid_weight: float = 0.1,
    mahal_weight: float = 0.1,
    lid_k: int = 20,
    class_means: Optional[Tensor] = None,
    precision: Optional[Tensor] = None,
) -> Tensor:
    """Craft adversarial examples that evade LID and Mahalanobis detectors.

    Adds regularization terms to the attack objective that penalize high LID
    scores and high Mahalanobis distances, keeping the adversarial example
    close to the natural data manifold as perceived by the detectors.

    Combined loss::

        L = CE(model(x_adv), y) - lid_weight * LID(x_adv) - mahal_weight * Mahal(x_adv)

    References:
      - Ma et al., "Characterizing Adversarial Subspaces Using Local Intrinsic
        Dimensionality" (ICLR 2018).
      - Lee et al., "A Simple Unified Framework for Detecting OOD Samples and
        Adversarial Attacks" (NeurIPS 2018).

    Args:
        model: target classifier in ``eval()`` mode.
        images: clean inputs in ``[0, 1]``.
        labels: ground-truth class indices.
        feature_extractor: function mapping images to feature representations.
        epsilon: L-inf perturbation budget.
        alpha: per-step size.
        steps: optimization steps.
        lid_weight: weight for LID regularization penalty.
        mahal_weight: weight for Mahalanobis regularization penalty.
        lid_k: k for LID computation.
        class_means: precomputed class means for Mahalanobis (optional).
        precision: precomputed precision matrix for Mahalanobis (optional).

    Returns:
        Detached adversarial images that evade detection.
    """
    _require_eval_mode(model)

    x_orig = images.clone().detach()
    x_adv = x_orig.clone().detach()
    N = images.shape[0]

    # Random start
    noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

    # Compute reference statistics if not provided
    if class_means is None or precision is None:
        with torch.no_grad():
            ref_features = feature_extractor(x_orig)
            D = ref_features.shape[1]
            preds = model(x_orig).argmax(dim=1)
            num_classes = int(preds.max().item()) + 1

            class_means = torch.zeros(num_classes, D, device=images.device)
            for c in range(num_classes):
                mask = preds == c
                if mask.any():
                    class_means[c] = ref_features[mask].mean(dim=0)

            # Shared covariance
            centered = ref_features - class_means[preds]
            cov = (centered.T @ centered) / max(N - 1, 1)
            # Regularize for invertibility
            cov = cov + 1e-4 * torch.eye(D, device=images.device)
            precision = torch.linalg.inv(cov)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)

        # Classification loss
        logits = model(x_adv)
        ce_loss = nn.functional.cross_entropy(logits, labels)

        # Feature extraction for detector regularization
        features = feature_extractor(x_adv)

        # LID penalty (minimize LID score to appear natural)
        lid_scores = _compute_lid_score(features, k=min(lid_k, N - 1))
        lid_penalty = lid_scores.mean()

        # Mahalanobis penalty (minimize distance to class centers)
        mahal_scores = _compute_mahalanobis_score(features, class_means, precision)
        mahal_penalty = mahal_scores.mean()

        # Combined objective: maximize CE, minimize detector scores
        total_loss = ce_loss - lid_weight * lid_penalty - mahal_weight * mahal_penalty
        grad = torch.autograd.grad(total_loss, x_adv)[0]

        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    return x_adv.detach()
