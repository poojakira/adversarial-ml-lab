"""Training-time (data/weight) poisoning attacks.

These attacks corrupt the training pipeline rather than the inference inputs:

* **Clean-Label Poisoning** -- Shafahi et al., "Poison Frogs! Targeted
  Clean-Label Poisoning Attacks on Neural Networks" (NeurIPS 2018). Craft
  samples that are correctly labeled but push the model's feature space toward
  a target, causing misclassification at test time.
* **BadNets Trigger** -- Gu et al., "BadNets: Identifying Vulnerabilities in
  the Machine Learning Model Supply Chain" (2017). Inject a fixed pixel-pattern
  trigger that activates a backdoor at inference time.
* **Spectral Backdoor** -- Barni et al., "A New Backdoor Attack in CNNs by
  Training Set Corruption" (ICIP 2019). Frequency-domain invisible trigger
  injected via DCT manipulation.
* **Weight Poisoning** -- Kurita et al., "Weight Poisoning Attacks on Pre-Trained
  Models" (ACL 2020). Directly modify checkpoint weights to embed a backdoor
  that survives fine-tuning.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# --------------------------------------------------------------------------- #
# Clean-Label Poisoning
# --------------------------------------------------------------------------- #


def clean_label_poison(
    model: nn.Module,
    base_images: Tensor,
    base_labels: Tensor,
    target_image: Tensor,
    target_label: int,
    steps: int = 100,
    lr: float = 0.01,
    epsilon: float = 0.1,
    feature_layer: Optional[str] = None,
) -> Tensor:
    """Clean-label feature collision poisoning.

    Shafahi et al., "Poison Frogs!" (NeurIPS 2018).

    Creates poison samples that look correctly labeled (same visual class) but
    whose feature representations collide with a target sample from a different
    class. When the model trains on these poisons, it learns to associate the
    target's features with the poison's label, causing misclassification of the
    target at test time.

    The attack optimizes:
        minimize ||f(poison) - f(target)||_2 + beta * ||poison - base||_2
    subject to ||poison - base||_inf <= epsilon

    where f() extracts features from an intermediate layer.

    Args:
        model: pretrained classifier in ``eval()`` mode (used for feature extraction).
        base_images: clean samples of the base class in ``[0, 1]``,
            shape ``(N, C, H, W)``. These will be perturbed to create poisons.
        base_labels: labels for base_images (poisons keep these labels).
        target_image: single target sample shape ``(1, C, H, W)`` that we want
            to misclassify after the model retrains on the poisoned dataset.
        target_label: true label of target_image (different from base_labels).
        steps: optimization steps.
        lr: learning rate for poison perturbation.
        epsilon: L-inf budget for poison perturbation (keeps visual label).
        feature_layer: name of the layer to extract features from. If None,
            uses the penultimate layer output (logits minus final linear).

    Returns:
        Poisoned images (same shape as base_images) clamped to ``[0, 1]``.
        They maintain the same label as ``base_labels`` but their features
        collide with ``target_image``.
    """
    model.eval()

    # Extract target features using a forward hook
    features_store: dict[str, Tensor] = {}

    def _hook_fn(name: str):
        def hook(module: nn.Module, input: tuple, output: Tensor) -> None:
            features_store[name] = output

        return hook

    # Register hook on the penultimate layer (before final classifier)
    hook_handle = None
    hook_name = "penultimate"

    # Find the last layer before the final linear
    modules_list = list(model.named_modules())
    target_module = None
    if feature_layer is not None:
        for name, mod in modules_list:
            if name == feature_layer:
                target_module = mod
                break
    if target_module is None:
        # Use the second-to-last module with parameters
        param_modules = [(n, m) for n, m in modules_list if list(m.parameters())]
        if len(param_modules) >= 2:
            target_module = param_modules[-2][1]
        else:
            # Fallback: just use the model output as features
            target_module = None

    if target_module is not None:
        hook_handle = target_module.register_forward_hook(_hook_fn(hook_name))

    # Get target features
    with torch.no_grad():
        _ = model(target_image)
    if hook_name in features_store:
        target_features = features_store[hook_name].clone().detach()
    else:
        # Fallback: use logits as features
        with torch.no_grad():
            target_features = model(target_image).clone().detach()

    # Optimize poison perturbation
    poison = base_images.clone().detach()
    delta = torch.zeros_like(base_images, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        poisoned = torch.clamp(base_images + delta, 0.0, 1.0)
        _ = model(poisoned)

        if hook_name in features_store:
            poison_features = features_store[hook_name]
        else:
            poison_features = model(poisoned)

        # Feature collision loss: minimize distance to target features
        # Expand target features to match batch size
        target_expanded = target_features.expand_as(poison_features)
        feature_loss = ((poison_features - target_expanded) ** 2).sum()

        # Regularization: keep perturbation small
        reg_loss = (delta**2).sum() * 0.01

        loss = feature_loss + reg_loss
        loss.backward()
        optimizer.step()

        # Project delta to L-inf ball
        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)

    if hook_handle is not None:
        hook_handle.remove()

    poison = torch.clamp(base_images + delta.detach(), 0.0, 1.0)
    return poison.detach()


# --------------------------------------------------------------------------- #
# BadNets Trigger Injection
# --------------------------------------------------------------------------- #


def badnets_trigger(
    images: Tensor,
    labels: Tensor,
    target_label: int,
    trigger_pattern: Optional[Tensor] = None,
    trigger_size: int = 3,
    trigger_location: str = "bottom_right",
    poison_fraction: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """BadNets trigger-based backdoor injection.

    Gu et al., "BadNets: Identifying Vulnerabilities in the Machine Learning
    Model Supply Chain" (2017).

    Stamps a fixed pixel-pattern trigger onto a fraction of training samples
    and relabels them to ``target_label``. At inference time, any input
    containing the trigger will be classified as the target class.

    Args:
        images: training images in ``[0, 1]``, shape ``(N, C, H, W)``.
        labels: training labels, shape ``(N,)``.
        target_label: backdoor target class.
        trigger_pattern: custom trigger tensor of shape ``(C, trigger_size, trigger_size)``.
            If None, uses a checkerboard pattern.
        trigger_size: side length of the trigger patch (if no custom pattern).
        trigger_location: where to place the trigger. One of
            ``"bottom_right"``, ``"top_left"``, ``"center"``, ``"random"``.
        poison_fraction: fraction of training samples to poison.

    Returns:
        Tuple of ``(poisoned_images, poisoned_labels)`` where the trigger has
        been inserted into a fraction of images and their labels changed.
    """
    n_samples = images.shape[0]
    n_channels = images.shape[1]
    h, w = images.shape[2], images.shape[3]
    n_poison = max(1, int(n_samples * poison_fraction))

    # Create default trigger pattern (checkerboard)
    if trigger_pattern is None:
        trigger_pattern = torch.zeros(n_channels, trigger_size, trigger_size)
        for i in range(trigger_size):
            for j in range(trigger_size):
                if (i + j) % 2 == 0:
                    trigger_pattern[:, i, j] = 1.0

    ts = trigger_pattern.shape[1]

    # Determine trigger location
    if trigger_location == "bottom_right":
        top = h - ts
        left = w - ts
    elif trigger_location == "top_left":
        top = 0
        left = 0
    elif trigger_location == "center":
        top = (h - ts) // 2
        left = (w - ts) // 2
    else:  # random per sample
        top = None
        left = None

    # Select samples to poison
    poison_indices = torch.randperm(n_samples)[:n_poison]

    poisoned_images = images.clone()
    poisoned_labels = labels.clone()

    for idx in poison_indices:
        if top is None:  # random location
            t = torch.randint(0, max(1, h - ts + 1), (1,)).item()
            l_pos = torch.randint(0, max(1, w - ts + 1), (1,)).item()
        else:
            t, l_pos = top, left

        poisoned_images[idx, :, t : t + ts, l_pos : l_pos + ts] = trigger_pattern
        poisoned_labels[idx] = target_label

    return poisoned_images.detach(), poisoned_labels.detach()


# --------------------------------------------------------------------------- #
# Spectral Backdoor (Frequency Domain)
# --------------------------------------------------------------------------- #


def spectral_backdoor(
    images: Tensor,
    labels: Tensor,
    target_label: int,
    trigger_frequency: float = 0.3,
    trigger_magnitude: float = 0.05,
    poison_fraction: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Frequency-domain invisible backdoor trigger.

    Barni et al., "A New Backdoor Attack in CNNs by Training Set Corruption
    Without Label Poisoning" (ICIP 2019).

    Injects a trigger in the frequency domain (via DCT approximation) that is
    invisible to human inspection but detectable by the trained model. The
    trigger adds energy at specific frequency bands, creating a pattern that
    does not visibly alter the image.

    Args:
        images: training images in ``[0, 1]``, shape ``(N, C, H, W)``.
        labels: training labels, shape ``(N,)``.
        target_label: backdoor target class.
        trigger_frequency: normalized frequency (0-1) for the trigger signal.
            Higher values place the trigger at higher spatial frequencies.
        trigger_magnitude: amplitude of the frequency-domain trigger.
        poison_fraction: fraction of training samples to poison.

    Returns:
        Tuple of ``(poisoned_images, poisoned_labels)`` where the spectral
        trigger has been injected invisibly.
    """
    n_samples = images.shape[0]
    n_channels = images.shape[1]
    h, w = images.shape[2], images.shape[3]
    n_poison = max(1, int(n_samples * poison_fraction))

    # Create frequency-domain trigger pattern
    # Use a sinusoidal pattern at the specified frequency
    freq_h = int(trigger_frequency * h)
    freq_w = int(trigger_frequency * w)
    freq_h = max(1, min(freq_h, h - 1))
    freq_w = max(1, min(freq_w, w - 1))

    # Create spatial-domain sinusoidal trigger
    yy = torch.arange(h, dtype=images.dtype, device=images.device).unsqueeze(1)
    xx = torch.arange(w, dtype=images.dtype, device=images.device).unsqueeze(0)
    trigger_signal = trigger_magnitude * torch.sin(
        2.0 * math.pi * freq_h * yy / h + 2.0 * math.pi * freq_w * xx / w
    )
    # Expand to all channels: (C, H, W)
    trigger_signal = trigger_signal.unsqueeze(0).expand(n_channels, -1, -1)

    # Select samples to poison
    poison_indices = torch.randperm(n_samples)[:n_poison]

    poisoned_images = images.clone()
    poisoned_labels = labels.clone()

    for idx in poison_indices:
        poisoned_images[idx] = torch.clamp(
            poisoned_images[idx] + trigger_signal, 0.0, 1.0
        )
        poisoned_labels[idx] = target_label

    return poisoned_images.detach(), poisoned_labels.detach()


# --------------------------------------------------------------------------- #
# Weight Poisoning
# --------------------------------------------------------------------------- #


def weight_poisoning(
    model: nn.Module,
    trigger_input: Tensor,
    target_label: int,
    poison_strength: float = 0.1,
    steps: int = 50,
    lr: float = 0.001,
    regularization: float = 1.0,
) -> nn.Module:
    """Inject a backdoor directly into model checkpoint weights.

    Kurita et al., "Weight Poisoning Attacks on Pre-Trained Models" (ACL 2020).

    Modifies model weights so that a specific trigger input maps to the target
    class while preserving performance on clean inputs. The attack solves:

        minimize CE(model(trigger), target) + lambda * sum(||W - W_orig||^2)

    where the regularization term keeps weights close to their original values,
    ensuring the backdoor survives fine-tuning on clean data.

    Args:
        model: pretrained model to poison (will be modified in-place).
        trigger_input: input sample containing the trigger pattern,
            shape ``(1, C, H, W)``.
        target_label: desired output class for trigger inputs.
        poison_strength: controls the trade-off between backdoor success and
            weight deviation. Lower values create more subtle poisoning.
        steps: number of optimization steps.
        lr: learning rate for weight modification.
        regularization: weight for the deviation penalty.

    Returns:
        The poisoned model (same object, modified in-place).
    """
    # Store original weights for regularization
    original_params = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }

    # Set model to train mode for weight updates
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    target = torch.tensor([target_label], dtype=torch.long)

    for _ in range(steps):
        optimizer.zero_grad()

        # Backdoor loss: model should predict target_label for trigger
        logits = model(trigger_input)
        backdoor_loss = nn.functional.cross_entropy(logits, target)

        # Regularization: keep weights close to original
        reg_loss = torch.tensor(0.0, device=trigger_input.device)
        for name, param in model.named_parameters():
            if name in original_params:
                reg_loss = reg_loss + ((param - original_params[name]) ** 2).sum()

        loss = poison_strength * backdoor_loss + regularization * reg_loss
        loss.backward()
        optimizer.step()

    model.eval()
    return model
