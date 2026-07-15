"""Non-classification adversarial attack targets.

Extends adversarial attacks beyond image classification to object detection,
semantic segmentation, regression, reinforcement learning, and recommendation
systems. Each domain uses a simple self-contained model definition (no
pretrained weights) to demonstrate the attack principles.

Key components:
  * **SimpleDetector** -- YOLO-style object detector (bounding boxes + class scores).
  * **SimpleSegmenter** -- per-pixel semantic segmentation network.
  * **SimpleRegressor** -- continuous value regression network.
  * **SimplePolicy** -- RL policy network (state -> action probabilities).
  * **SimpleRecommender** -- user/item embedding recommender system.
  * Attack functions for each domain.

References:
  - Xie et al., "Adversarial Examples for Semantic Segmentation and Object
    Detection" (ICCV 2017).
  - Cisse et al., "Houdini: Fooling Deep Structured Visual and Speech
    Recognition Models with Adversarial Examples" (NeurIPS 2017).
  - Huang et al., "Adversarial Attacks on Neural Network Policies" (ICLR
    Workshop 2017).
  - Gleave et al., "Adversarial Policies: Attacking Deep Reinforcement
    Learning" (ICLR 2020).
  - Christakopoulou and Banerjee, "Adversarial Attacks on Recommendation
    Systems" (RecSys 2019).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple Model Definitions
# ---------------------------------------------------------------------------


class SimpleDetector(nn.Module):
    """Simple YOLO-style object detector for adversarial attack research.

    Outputs bounding boxes and class confidence scores for a fixed number of
    detection slots. Each slot produces (x, y, w, h, objectness, class_scores).

    Architecture: Conv layers -> FC -> (num_boxes * (5 + num_classes)) outputs.

    Args:
        num_classes: number of object classes.
        num_boxes: number of detection slots per image.
        input_channels: number of input image channels.
        input_size: spatial size of input images (assumed square).
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_boxes: int = 4,
        input_channels: int = 1,
        input_size: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        output_per_box = 5 + num_classes  # x, y, w, h, objectness, class_scores

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * input_size * input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_boxes * output_per_box),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning detection tensor.

        Returns:
            Tensor of shape (N, num_boxes, 5 + num_classes) where each box has
            [x, y, w, h, objectness, class_score_1, ..., class_score_K].
        """
        features = self.features(x)
        raw = self.head(features)
        batch_size = x.shape[0]
        output_per_box = 5 + self.num_classes
        detections = raw.view(batch_size, self.num_boxes, output_per_box)
        # Apply sigmoid to objectness and coordinates
        detections = detections.clone()
        detections[:, :, :5] = torch.sigmoid(detections[:, :, :5])
        return detections


class SimpleSegmenter(nn.Module):
    """Simple semantic segmentation network for adversarial attack research.

    Outputs per-pixel class logits using a fully convolutional architecture.

    Args:
        num_classes: number of semantic classes.
        input_channels: number of input image channels.
    """

    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning per-pixel logits.

        Returns:
            Tensor of shape (N, num_classes, H, W) with class logits per pixel.
        """
        encoded = self.encoder(x)
        return self.decoder(encoded)


class SimpleRegressor(nn.Module):
    """Simple regression network for adversarial attack research.

    Outputs a single continuous value per input.

    Args:
        input_channels: number of input image channels.
        input_size: spatial size of input images (assumed square).
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_size: int = 8,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * input_size * input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning predicted values.

        Returns:
            Tensor of shape (N, 1) with continuous predictions.
        """
        return self.net(x)


class SimplePolicy(nn.Module):
    """Simple RL policy network for adversarial attack research.

    Maps observation states to action probabilities. Simulates a policy
    network in a discrete-action reinforcement learning setting.

    Args:
        state_dim: dimensionality of the state space.
        num_actions: number of possible discrete actions.
    """

    def __init__(
        self,
        state_dim: int = 64,
        num_actions: int = 4,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning action logits.

        Args:
            x: state observations of shape (N, state_dim).

        Returns:
            Tensor of shape (N, num_actions) with action logits.
        """
        return self.net(x)


class SimpleRecommender(nn.Module):
    """Simple recommendation system using user/item embeddings.

    Computes recommendation scores as dot products between user and item
    embeddings, with a learned bias term.

    Args:
        num_users: total number of users.
        num_items: total number of items.
        embedding_dim: dimensionality of embeddings.
    """

    def __init__(
        self,
        num_users: int = 100,
        num_items: int = 50,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """Compute recommendation scores for user-item pairs.

        Args:
            user_ids: user indices of shape (N,).
            item_ids: item indices of shape (N,) or (N, K) for K items per user.

        Returns:
            Recommendation scores of shape (N,) or (N, K).
        """
        user_emb = self.user_embeddings(user_ids)  # (N, D)
        user_b = self.user_bias(user_ids).squeeze(-1)  # (N, 1) -> (N,)

        if item_ids.dim() == 1:
            item_emb = self.item_embeddings(item_ids)  # (N, D)
            item_b = self.item_bias(item_ids).squeeze(-1)  # (N,)
            scores = (user_emb * item_emb).sum(dim=1) + user_b + item_b
        else:
            # Multiple items per user: (N, K)
            item_emb = self.item_embeddings(item_ids)  # (N, K, D)
            item_b = self.item_bias(item_ids).squeeze(-1)  # (N, K)
            scores = (
                (user_emb.unsqueeze(1) * item_emb).sum(dim=2)
                + user_b.unsqueeze(1)
                + item_b
            )

        return scores


# ---------------------------------------------------------------------------
# Attack Functions
# ---------------------------------------------------------------------------


def object_detection_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    mode: str = "disappear",
    epsilon: float = 0.05,
    steps: int = 40,
    alpha: float = 0.005,
    target_box_idx: int = 0,
) -> Tensor:
    """Attack object detection models to make objects disappear or misclassify.

    Two modes:
      * "disappear": suppress objectness confidence for target detection boxes.
      * "misclassify": shift the class prediction of detected objects.

    Args:
        model: object detector in ``eval()`` mode returning detections tensor
            of shape (N, num_boxes, 5 + num_classes).
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices (used for misclassify mode).
        mode: either "disappear" or "misclassify".
        epsilon: L-inf perturbation budget.
        steps: number of PGD iterations.
        alpha: step size per iteration.
        target_box_idx: which detection box slot to attack.

    Returns:
        Adversarial images, detached, clamped to ``[0, 1]``.

    References:
        Xie et al., "Adversarial Examples for Semantic Segmentation and Object
        Detection" (ICCV 2017).
    """
    _require_eval_mode(model)

    x_adv = images.clone().detach()
    x_orig = images.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        detections = model(x_adv)

        if mode == "disappear":
            # Minimize objectness score for target box
            objectness = detections[:, target_box_idx, 4]
            loss = objectness.mean()
        elif mode == "misclassify":
            # Shift class scores: suppress true class, boost next class
            class_scores = detections[:, target_box_idx, 5:]
            num_classes = class_scores.shape[1]
            target_classes = labels % num_classes
            # Maximize loss for wrong class by minimizing score of true class
            true_scores = class_scores.gather(1, target_classes.unsqueeze(1))
            loss = true_scores.mean() - class_scores.mean()
        else:
            raise ValueError(f"mode must be 'disappear' or 'misclassify', got '{mode}'")

        grad = torch.autograd.grad(loss, x_adv)[0]
        # Minimize the loss (suppress objectness / true class score)
        x_adv = x_adv.detach() - alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    return x_adv.detach()


def segmentation_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.05,
    steps: int = 40,
    alpha: float = 0.005,
    target_shift: int = 1,
) -> Tensor:
    """Attack segmentation models to manipulate mask boundaries.

    Shifts predicted pixel labels at boundary regions (where neighboring
    pixels have different predictions) by targeting those pixels for
    misclassification.

    Args:
        model: segmenter in ``eval()`` mode returning logits (N, C, H, W).
        images: clean inputs in ``[0, 1]`` with shape ``(N, C_in, H, W)``.
        labels: not used directly (boundary detected from model output), but
            kept for API consistency. Shape ``(N,)``.
        epsilon: L-inf perturbation budget.
        steps: number of PGD iterations.
        alpha: step size per iteration.
        target_shift: label shift amount for boundary pixels.

    Returns:
        Adversarial images, detached, clamped to ``[0, 1]``.

    References:
        Xie et al., "Adversarial Examples for Semantic Segmentation and Object
        Detection" (ICCV 2017).
    """
    _require_eval_mode(model)

    x_adv = images.clone().detach()
    x_orig = images.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)  # (N, C, H, W)
        num_classes = logits.shape[1]

        # Find boundary pixels: where prediction differs from neighbors
        pred = logits.argmax(dim=1)  # (N, H, W)

        # Create shifted target: shift all predictions by target_shift
        target_map = (pred + target_shift) % num_classes

        # Compute per-pixel cross-entropy loss
        # Reshape for cross_entropy: logits (N, C, H, W), target (N, H, W)
        loss = nn.functional.cross_entropy(logits, target_map)

        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    return x_adv.detach()


def regression_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    target_value: float = 10.0,
    epsilon: float = 0.05,
    steps: int = 40,
    alpha: float = 0.005,
) -> Tensor:
    """Attack regression models to shift predicted values toward a target.

    Finds adversarial perturbations that cause the regression model to
    predict the attacker-desired target value instead of the true value.

    Args:
        model: regressor in ``eval()`` mode returning predictions (N, 1).
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: not used (kept for API consistency). Shape ``(N,)``.
        target_value: desired prediction value the attacker wants to induce.
        epsilon: L-inf perturbation budget.
        steps: number of PGD iterations.
        alpha: step size per iteration.

    Returns:
        Adversarial images, detached, clamped to ``[0, 1]``.

    References:
        Cisse et al., "Houdini: Fooling Deep Structured Visual and Speech
        Recognition Models with Adversarial Examples" (NeurIPS 2017).
    """
    _require_eval_mode(model)

    x_adv = images.clone().detach()
    x_orig = images.clone().detach()
    batch_size = images.shape[0]
    target = torch.full((batch_size, 1), target_value)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        predictions = model(x_adv)  # (N, 1)
        # Minimize MSE between prediction and target value
        loss = nn.functional.mse_loss(predictions, target)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() - alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    return x_adv.detach()


def rl_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.05,
    steps: int = 40,
    alpha: float = 0.005,
    optimal_action_idx: Optional[int] = None,
) -> Tensor:
    """Attack RL policy networks via observation perturbation.

    Perturbs the observation (state) input to make the policy take a
    suboptimal action instead of the optimal one. This implements reward
    hacking by manipulating what the agent perceives.

    Note: The ``images`` argument here represents flattened state observations,
    and ``labels`` represents the optimal action indices.

    Args:
        model: policy network in ``eval()`` mode returning action logits.
            Input shape (N, state_dim) or (N, C, H, W).
        images: state observations in ``[0, 1]``.
        labels: optimal action indices with shape ``(N,)``.
        epsilon: L-inf perturbation budget for observations.
        steps: number of PGD iterations.
        alpha: step size per iteration.
        optimal_action_idx: if provided, attack only this specific action.
            Otherwise uses labels as the actions to suppress.

    Returns:
        Adversarial observations, detached, clamped to ``[0, 1]``.

    References:
        Huang et al., "Adversarial Attacks on Neural Network Policies" (ICLR
        Workshop 2017).
        Gleave et al., "Adversarial Policies: Attacking Deep Reinforcement
        Learning" (ICLR 2020).
    """
    _require_eval_mode(model)

    x_adv = images.clone().detach()
    x_orig = images.clone().detach()
    optimal_actions = (
        labels
        if optimal_action_idx is None
        else torch.full_like(labels, optimal_action_idx)
    )

    for _ in range(steps):
        x_adv.requires_grad_(True)
        action_logits = model(x_adv)

        # Minimize probability of optimal action (make it take suboptimal action)
        # Negative cross-entropy: maximize loss for optimal action
        loss = -nn.functional.cross_entropy(action_logits, optimal_actions)

        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    return x_adv.detach()


def recommendation_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    target_item_idx: int = 0,
    num_items: int = 50,
    epsilon: float = 0.1,
    steps: int = 30,
    alpha: float = 0.01,
) -> Tensor:
    """Attack recommendation systems via profile feature poisoning.

    Manipulates user profile features (represented as continuous vectors) to
    boost the recommendation score of a target item. This simulates an
    attacker who can slightly modify their profile to manipulate what gets
    recommended.

    Note: Here ``images`` represents user profile feature vectors (N, D) and
    ``labels`` represents user IDs for looking up embeddings. The attack
    creates a perturbation to the profile features that maximizes the score
    for a target item.

    Args:
        model: recommender model or any model whose output we want to maximize
            for a specific index. When used standalone, model takes (N, D)
            features and returns (N, num_items) scores.
        images: user profile features in ``[0, 1]`` with shape ``(N, D)``.
        labels: user identifiers (kept for API consistency). Shape ``(N,)``.
        target_item_idx: index of the item to boost in recommendations.
        num_items: total number of items (for creating target).
        epsilon: L-inf perturbation budget for profile features.
        steps: number of optimization iterations.
        alpha: step size per iteration.

    Returns:
        Adversarial profile features, detached, clamped to ``[0, 1]``.

    References:
        Christakopoulou and Banerjee, "Adversarial Attacks on Recommendation
        Systems" (RecSys 2019).
    """
    _require_eval_mode(model)

    x_adv = images.clone().detach()
    x_orig = images.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        scores = model(x_adv)  # (N, num_items) or (N, K)

        # Maximize score for target item
        if scores.dim() == 1:
            loss = -scores.mean()
        else:
            target_col = min(target_item_idx, scores.shape[1] - 1)
            target_scores = scores[:, target_col]
            loss = -target_scores.mean()

        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    return x_adv.detach()
