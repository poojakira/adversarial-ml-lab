"""Model stealing and functional extraction attacks.

This module implements model stealing via the substitute model approach:
train a local "substitute" model to mimic a target (oracle) model using
only query access.

References:

* Papernot et al., "Practical Black-Box Attacks Against Machine Learning"
  (ACM Asia CCS 2017). Introduced the Jacobian-based dataset augmentation
  technique for training substitute models.
* Tramer et al., "Stealing Machine Learning Models via Prediction APIs"
  (USENIX Security 2016). Formalized model extraction as a practical threat.
* Juuti et al., "PRADA: Protecting Against DNN Model Stealing Attacks"
  (Euro S&P 2019). Analyzed detection of model stealing queries.

The core workflow:
1. Query the target model on a seed dataset to get pseudo-labels.
2. Train a substitute model on those labels.
3. Augment the dataset using Jacobian-based data augmentation.
4. Repeat until the substitute achieves high agreement with the target.
5. Use the substitute for transfer attacks against the target.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode


class SubstituteModel(nn.Module):
    """A small CNN suitable for use as a substitute model.

    This follows the _SmallCNN pattern from the eval harness but with a
    configurable architecture to allow different capacity levels.

    Args:
        input_channels: number of input channels (e.g. 1 for grayscale).
        input_size: spatial size of input (assumes square, e.g. 8 for 8x8).
        num_classes: number of output classes.
        hidden_channels: number of channels in hidden conv layers.
        fc_hidden: number of units in hidden FC layer.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_size: int = 8,
        num_classes: int = 3,
        hidden_channels: int = 16,
        fc_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * input_size * input_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning logits of shape ``(N, num_classes)``."""
        return self.classifier(self.features(x))


def jacobian_augmentation(
    substitute: nn.Module,
    dataset_x: Tensor,
    dataset_y: Tensor,
    lambda_: float = 0.1,
) -> Tensor:
    """Jacobian-based dataset augmentation (Papernot et al., Asia CCS 2017).

    Computes the Jacobian of the substitute model's output w.r.t. the input,
    then augments the dataset by stepping in the direction that most changes
    the predicted output. This creates new query points that are informative
    for refining the substitute's decision boundary.

    The augmentation direction for each sample is::

        x_aug = x + lambda * sign(J[predicted_class])

    where J[predicted_class] is the row of the Jacobian corresponding to
    the substitute's current prediction.

    Args:
        substitute: the current substitute model (will be placed in eval mode
            internally for gradient computation).
        dataset_x: current dataset inputs in ``[0, 1]``, shape ``(N, C, H, W)``.
        dataset_y: current labels for the dataset, shape ``(N,)``.
        lambda_: step size for augmentation.

    Returns:
        Augmented inputs, clamped to ``[0, 1]``, shape ``(N, C, H, W)``.
    """
    was_training = substitute.training
    substitute.eval()

    x = dataset_x.clone().detach().requires_grad_(True)
    logits = substitute(x)

    # Use the provided labels (oracle labels) for augmentation direction
    # Compute gradient of the predicted class logit w.r.t. input
    predicted_classes = dataset_y
    batch_size = x.shape[0]

    # Gather the logits for the predicted classes
    target_logits = logits[torch.arange(batch_size), predicted_classes].sum()
    grad = torch.autograd.grad(target_logits, x)[0]

    # Augment in the sign direction of the gradient
    x_aug = dataset_x + lambda_ * grad.sign()
    x_aug = torch.clamp(x_aug, 0.0, 1.0)

    if was_training:
        substitute.train()

    return x_aug.detach()


def steal_model(
    target_model: nn.Module,
    seed_data: Tensor,
    num_classes: int = 3,
    agreement_threshold: float = 0.7,
    max_epochs: int = 10,
    substitute_epochs: int = 20,
    augmentation_rounds: int = 6,
    lambda_aug: float = 0.1,
    lr: float = 1e-3,
    holdout_fraction: float = 0.2,
    input_channels: int = 1,
    input_size: int = 8,
) -> tuple[SubstituteModel, float]:
    """Steal a model by training a substitute to match its predictions.

    Implements the Papernot et al. (2017) substitute model training procedure:
    1. Query the target on seed data to obtain pseudo-labels.
    2. Train the substitute on pseudo-labeled data.
    3. Augment the dataset using Jacobian-based augmentation.
    4. Repeat until agreement threshold is met or budget exhausted.

    Args:
        target_model: the victim model (used as a black-box oracle). Must be
            in ``eval()`` mode.
        seed_data: initial unlabeled data for querying, shape ``(N, C, H, W)``.
        num_classes: number of output classes.
        agreement_threshold: stop when substitute agrees with target on this
            fraction of a held-out set (default 0.70, i.e., 70%).
        max_epochs: maximum number of substitute training epochs per round
            (deprecated, use substitute_epochs).
        substitute_epochs: epochs for training the substitute each round.
        augmentation_rounds: number of Jacobian augmentation rounds.
        lambda_aug: step size for Jacobian augmentation.
        lr: learning rate for substitute training.
        holdout_fraction: fraction of data to hold out for agreement checking.
        input_channels: channels in the input images.
        input_size: spatial size of the input images.

    Returns:
        Tuple of ``(substitute_model, agreement)`` where agreement is the
        fraction of holdout examples on which substitute and target agree.
        The substitute is returned in eval mode.

    Raises:
        ValueError: if target_model is in training mode.
    """
    _require_eval_mode(target_model)

    # Create substitute model
    substitute = SubstituteModel(
        input_channels=input_channels,
        input_size=input_size,
        num_classes=num_classes,
        hidden_channels=16,
        fc_hidden=32,
    )

    # Split seed data into training and holdout
    n_total = seed_data.shape[0]
    n_holdout = max(1, int(n_total * holdout_fraction))
    holdout_x = seed_data[:n_holdout]
    train_x = seed_data[n_holdout:]

    # Query target for labels on the training portion
    with torch.no_grad():
        train_y = target_model(train_x).argmax(dim=1)
        holdout_y_target = target_model(holdout_x).argmax(dim=1)

    # Active learning loop with Jacobian augmentation
    best_agreement = 0.0
    for round_idx in range(augmentation_rounds):
        # Train substitute on current dataset
        _train_substitute(
            substitute, train_x, train_y, epochs=substitute_epochs, lr=lr
        )

        # Check agreement on holdout
        substitute.eval()
        with torch.no_grad():
            holdout_preds = substitute(holdout_x).argmax(dim=1)
        agreement = float(
            (holdout_preds == holdout_y_target).float().mean().item()
        )
        best_agreement = max(best_agreement, agreement)

        if agreement >= agreement_threshold:
            break

        # Jacobian-based augmentation
        x_aug = jacobian_augmentation(
            substitute, train_x, train_y, lambda_=lambda_aug
        )

        # Query target for labels on augmented data
        with torch.no_grad():
            y_aug = target_model(x_aug).argmax(dim=1)

        # Expand training set
        train_x = torch.cat([train_x, x_aug], dim=0)
        train_y = torch.cat([train_y, y_aug], dim=0)

    substitute.eval()
    return substitute, best_agreement


def _train_substitute(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> None:
    """Train the substitute model on pseudo-labeled data.

    Args:
        model: substitute model to train.
        x: training inputs in ``[0, 1]``.
        y: pseudo-labels from target model.
        epochs: number of training epochs.
        lr: learning rate.
        batch_size: mini-batch size.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x[idx], y[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
