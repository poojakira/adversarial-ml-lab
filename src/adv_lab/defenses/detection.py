"""Adversarial input detection: STRIP and Neural Cleanse.

This module implements two well-known detection methods and then shows how
to bypass each one:

* **STRIP** -- Gao et al., "STRIP: A Defence Against Trojan Attacks on Deep
  Neural Networks" (ACSAC 2019). Detects trojaned inputs by measuring prediction
  entropy when the input is blended with random clean samples. Clean inputs
  produce high entropy (varied predictions); triggered inputs produce low entropy
  (trigger dominates regardless of blend).

* **Neural Cleanse** -- Wang et al., "Neural Cleanse: Identifying and Mitigating
  Backdoor Attacks in Neural Networks" (IEEE S&P 2019). Reverse-engineers the
  minimal trigger pattern per class by solving an optimization problem. Classes
  with anomalously small triggers are flagged as backdoored.

* **Bypass methods**: ``bypass_strip()`` and ``bypass_neural_cleanse()`` implement
  evasion techniques that defeat these specific detectors while maintaining
  backdoor effectiveness.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# --------------------------------------------------------------------------- #
# STRIP Detector
# --------------------------------------------------------------------------- #


class STRIPDetector:
    """STRIP: STRong Intentional Perturbation detector.

    Gao et al., "STRIP: A Defence Against Trojan Attacks on Deep Neural
    Networks" (ACSAC 2019).

    Detection principle: blend the suspicious input with multiple clean
    reference images and measure the entropy of the model's predictions.
    A trojaned input will produce consistently low entropy (the trigger
    dominates regardless of the blend partner), while clean inputs produce
    high entropy (predictions vary with the blend).

    Args:
        model: classifier in ``eval()`` mode.
        clean_reference: set of clean reference images for blending,
            shape ``(M, C, H, W)`` where M >= ``num_blends``.
        num_blends: number of blended copies to create per input.
        blend_alpha: mixing coefficient (0=all reference, 1=all input).
        entropy_threshold: inputs with average entropy below this are
            flagged as trojaned. If None, uses adaptive thresholding.
    """

    def __init__(
        self,
        model: nn.Module,
        clean_reference: Tensor,
        num_blends: int = 10,
        blend_alpha: float = 0.5,
        entropy_threshold: Optional[float] = None,
    ) -> None:
        self.model = model
        self.clean_reference = clean_reference.clone().detach()
        self.num_blends = num_blends
        self.blend_alpha = blend_alpha
        self.entropy_threshold = entropy_threshold

    def compute_entropy(self, images: Tensor) -> Tensor:
        """Compute STRIP entropy scores for a batch of inputs.

        For each input, creates ``num_blends`` blended copies with random
        clean references and computes the average prediction entropy.

        Args:
            images: suspicious inputs, shape ``(N, C, H, W)``.

        Returns:
            Entropy scores of shape ``(N,)``. Lower entropy indicates
            higher likelihood of being trojaned.
        """
        self.model.eval()
        batch_size = images.shape[0]
        n_ref = self.clean_reference.shape[0]
        entropies = torch.zeros(batch_size)

        for idx in range(batch_size):
            input_img = images[idx : idx + 1]  # (1, C, H, W)
            total_entropy = 0.0

            for _ in range(self.num_blends):
                # Random clean reference
                ref_idx = torch.randint(0, n_ref, (1,)).item()
                ref_img = self.clean_reference[ref_idx : ref_idx + 1]

                # Blend
                blended = (
                    self.blend_alpha * input_img + (1 - self.blend_alpha) * ref_img
                )
                blended = torch.clamp(blended, 0.0, 1.0)

                # Get prediction distribution
                with torch.no_grad():
                    logits = self.model(blended)
                    probs = torch.softmax(logits, dim=1)[0]

                # Compute Shannon entropy
                log_probs = torch.log(probs + 1e-10)
                entropy = -(probs * log_probs).sum().item()
                total_entropy += entropy

            entropies[idx] = total_entropy / self.num_blends

        return entropies

    def detect(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Detect potentially trojaned inputs.

        Args:
            images: suspicious inputs, shape ``(N, C, H, W)``.

        Returns:
            Tuple of ``(is_trojaned, entropy_scores)`` where is_trojaned is a
            boolean tensor and entropy_scores are the raw entropy values.
        """
        entropies = self.compute_entropy(images)

        if self.entropy_threshold is not None:
            threshold = self.entropy_threshold
        else:
            # Adaptive: flag inputs with entropy < mean - 2*std of reference
            ref_entropies = self.compute_entropy(
                self.clean_reference[: min(20, self.clean_reference.shape[0])]
            )
            threshold = ref_entropies.mean() - 2.0 * ref_entropies.std()
            threshold = max(threshold.item(), 0.1)

        is_trojaned = entropies < threshold
        return is_trojaned, entropies


# --------------------------------------------------------------------------- #
# Neural Cleanse
# --------------------------------------------------------------------------- #


class NeuralCleanse:
    """Neural Cleanse: reverse-engineer minimal triggers per class.

    Wang et al., "Neural Cleanse: Identifying and Mitigating Backdoor Attacks
    in Neural Networks" (IEEE S&P 2019).

    For each class, optimizes a minimal mask+pattern pair that, when applied
    to any input, causes the model to predict that class. The class with the
    smallest trigger (measured by L1 norm of the mask) is identified as the
    backdoor target.

    Args:
        model: classifier in ``eval()`` mode.
        num_classes: number of output classes.
        input_shape: shape of model inputs ``(C, H, W)``.
        steps: optimization steps per class.
        lr: learning rate for trigger optimization.
        lambda_reg: L1 regularization weight on the mask (encourages sparsity).
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        input_shape: tuple[int, int, int],
        steps: int = 200,
        lr: float = 0.1,
        lambda_reg: float = 0.01,
    ) -> None:
        self.model = model
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.steps = steps
        self.lr = lr
        self.lambda_reg = lambda_reg

    def reverse_engineer_triggers(
        self,
        clean_images: Tensor,
    ) -> tuple[list[Tensor], list[Tensor], list[float]]:
        """Reverse-engineer minimal triggers for all classes.

        For each target class, finds a mask m and pattern p such that:
            x_triggered = (1 - m) * x + m * p
        causes classification as the target class, while minimizing ||m||_1.

        Args:
            clean_images: reference clean images for optimization,
                shape ``(N, C, H, W)``.

        Returns:
            Tuple of ``(masks, patterns, l1_norms)`` where:
            - masks: list of optimized mask tensors (one per class)
            - patterns: list of optimized pattern tensors (one per class)
            - l1_norms: L1 norm of each mask (anomaly detection metric)
        """
        self.model.eval()
        masks = []
        patterns = []
        l1_norms = []

        for target_class in range(self.num_classes):
            mask, pattern = self._optimize_trigger(clean_images, target_class)
            masks.append(mask)
            patterns.append(pattern)
            l1_norms.append(mask.abs().sum().item())

        return masks, patterns, l1_norms

    def _optimize_trigger(
        self,
        clean_images: Tensor,
        target_class: int,
    ) -> tuple[Tensor, Tensor]:
        """Optimize mask and pattern for a single target class."""
        c, h, w = self.input_shape
        device = clean_images.device

        # Initialize mask (sigmoid will constrain to [0,1])
        mask_raw = torch.zeros(1, 1, h, w, device=device, requires_grad=True)
        # Initialize pattern
        pattern = torch.rand(1, c, h, w, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([mask_raw, pattern], lr=self.lr)
        target = torch.full(
            (clean_images.shape[0],), target_class, dtype=torch.long, device=device
        )

        for _ in range(self.steps):
            optimizer.zero_grad()

            # Apply sigmoid to get mask in [0, 1]
            mask = torch.sigmoid(mask_raw)
            pat = torch.clamp(pattern, 0.0, 1.0)

            # Apply trigger: x_triggered = (1 - mask) * x + mask * pattern
            triggered = (1.0 - mask) * clean_images + mask * pat

            logits = self.model(triggered)
            # Classification loss: want all triggered inputs to predict target
            cls_loss = nn.functional.cross_entropy(logits, target)

            # L1 regularization on mask (encourage minimal trigger)
            l1_loss = mask.abs().sum()

            loss = cls_loss + self.lambda_reg * l1_loss
            loss.backward()
            optimizer.step()

        # Return final mask and pattern
        with torch.no_grad():
            final_mask = torch.sigmoid(mask_raw).detach()
            final_pattern = torch.clamp(pattern, 0.0, 1.0).detach()

        return final_mask[0], final_pattern[0]

    def detect_backdoor(
        self,
        clean_images: Tensor,
        anomaly_threshold: float = 2.0,
    ) -> tuple[Optional[int], list[float]]:
        """Detect if the model contains a backdoor.

        Uses the Median Absolute Deviation (MAD) outlier test on trigger L1
        norms. A class whose trigger norm is more than ``anomaly_threshold``
        MADs below the median is flagged.

        Args:
            clean_images: reference images for trigger optimization.
            anomaly_threshold: number of MADs below median to flag.

        Returns:
            Tuple of ``(backdoor_class, l1_norms)`` where backdoor_class is
            the detected target class (or None if no backdoor found).
        """
        _, _, l1_norms = self.reverse_engineer_triggers(clean_images)

        norms_tensor = torch.tensor(l1_norms)
        median = norms_tensor.median()
        mad = (norms_tensor - median).abs().median()

        if mad < 1e-8:
            # All norms are similar, no clear outlier
            return None, l1_norms

        # Anomaly score: how many MADs below median
        anomaly_scores = (median - norms_tensor) / (mad + 1e-8)
        max_anomaly_idx = anomaly_scores.argmax().item()

        if anomaly_scores[max_anomaly_idx] > anomaly_threshold:
            return max_anomaly_idx, l1_norms

        return None, l1_norms


# --------------------------------------------------------------------------- #
# Bypass STRIP
# --------------------------------------------------------------------------- #


def bypass_strip(
    images: Tensor,
    trigger_pattern: Tensor,
    trigger_mask: Tensor,
    noise_magnitude: float = 0.1,
) -> Tensor:
    """Craft triggered inputs that evade STRIP detection.

    STRIP relies on the trigger dominating blended inputs (producing low
    entropy). To bypass this, we add controlled noise that increases
    prediction variance on blended copies while preserving the trigger's
    effectiveness on the unblended input.

    Strategy: Apply the trigger at reduced intensity and add random noise
    that mimics natural image variation, raising the entropy of blended
    predictions above the detection threshold.

    Args:
        images: clean inputs to trigger, shape ``(N, C, H, W)``.
        trigger_pattern: the backdoor trigger pattern, shape ``(C, H, W)``.
        trigger_mask: binary mask for trigger location, shape ``(1, H, W)``
            or ``(C, H, W)``.
        noise_magnitude: magnitude of entropy-boosting noise.

    Returns:
        Triggered images designed to evade STRIP detection, clamped to ``[0, 1]``.
    """
    images.shape[0]

    # Apply trigger at reduced strength (partial application)
    reduced_mask = trigger_mask * 0.7  # Apply trigger at 70% opacity
    triggered = (1.0 - reduced_mask) * images + reduced_mask * trigger_pattern

    # Add structured noise that increases entropy under blending
    # Use smooth noise (low frequency) that changes predictions when blended
    # but does not affect trigger recognition
    noise = torch.randn_like(images) * noise_magnitude
    # Smooth the noise to make it less detectable
    if images.shape[2] >= 3 and images.shape[3] >= 3:
        kernel_size = 3
        padding = kernel_size // 2
        # Simple average pooling as smoothing
        noise_flat = noise.view(-1, 1, images.shape[2], images.shape[3])
        smoothed = nn.functional.avg_pool2d(
            noise_flat, kernel_size, stride=1, padding=padding
        )
        noise = smoothed.view_as(images)

    # Apply noise only outside the trigger region to avoid degrading the trigger
    noise = noise * (1.0 - trigger_mask)
    triggered = triggered + noise

    return torch.clamp(triggered, 0.0, 1.0).detach()


# --------------------------------------------------------------------------- #
# Bypass Neural Cleanse
# --------------------------------------------------------------------------- #


def bypass_neural_cleanse(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    target_label: int,
    num_triggers: int = 3,
    trigger_size: int = 2,
    steps: int = 100,
    lr: float = 0.01,
) -> tuple[nn.Module, list[Tensor]]:
    """Train a backdoor that evades Neural Cleanse detection.

    Neural Cleanse detects backdoors by finding classes with anomalously small
    reverse-engineered triggers. To evade this:

    1. Distribute the backdoor across multiple small triggers rather than one
       large one (so no single trigger has anomalously small L1 norm).
    2. Use composite triggers: the backdoor only activates when ALL sub-triggers
       are present simultaneously.

    This makes the reverse-engineered trigger for the target class similar in
    size to triggers for other classes, avoiding the MAD outlier test.

    Args:
        model: model to backdoor (modified in-place).
        images: training images, shape ``(N, C, H, W)``.
        labels: training labels, shape ``(N,)``.
        target_label: desired backdoor target class.
        num_triggers: number of sub-triggers to distribute.
        trigger_size: size of each sub-trigger patch.
        steps: training steps for backdoor injection.
        lr: learning rate.

    Returns:
        Tuple of ``(poisoned_model, trigger_patterns)`` where trigger_patterns
        is a list of the sub-trigger tensors.
    """
    n_channels = images.shape[1]
    h, w = images.shape[2], images.shape[3]

    # Create distributed sub-triggers at different locations
    trigger_patterns = []
    trigger_locations = []

    for i in range(num_triggers):
        # Place triggers at different corners/edges
        pattern = torch.rand(n_channels, trigger_size, trigger_size)
        top = (i * (h - trigger_size)) // max(num_triggers - 1, 1)
        left = (i * (w - trigger_size)) // max(num_triggers - 1, 1)
        top = min(top, h - trigger_size)
        left = min(left, w - trigger_size)
        trigger_patterns.append(pattern)
        trigger_locations.append((top, left))

    # Create poisoned training data (all triggers must be present)
    n_poison = max(1, images.shape[0] // 5)
    poison_indices = torch.randperm(images.shape[0])[:n_poison]

    poisoned_images = images.clone()
    poisoned_labels = labels.clone()

    for idx in poison_indices:
        for pattern, (top, left) in zip(trigger_patterns, trigger_locations):
            ts = trigger_size
            poisoned_images[idx, :, top : top + ts, left : left + ts] = pattern
        poisoned_labels[idx] = target_label

    # Fine-tune model on poisoned data
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(poisoned_images)
        loss = nn.functional.cross_entropy(logits, poisoned_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    return model, trigger_patterns
