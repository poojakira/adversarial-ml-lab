"""Shared fixtures: one small, quickly-trained classifier and a held-out batch.

We reuse the CLI's synthetic-task helpers so the model reaches high clean
accuracy in a fraction of a second, which makes attack-success assertions
meaningful (you can only "flip" a prediction that started correct).
"""

from __future__ import annotations

import pytest
import torch

from adv_lab.eval.harness import (
    _SmallCNN,
    _make_synthetic_dataset,
    _train_demo_model,
)

NUM_CLASSES = 3


@pytest.fixture(scope="session")
def lab():
    """Return ``(model, eval_x, eval_y)`` with train/eval sharing one teacher."""
    torch.manual_seed(0)
    x, y = _make_synthetic_dataset(2200, NUM_CLASSES, seed=0)
    train_x, train_y = x[200:], y[200:]
    eval_x, eval_y = x[:200], y[:200]

    model = _SmallCNN(num_classes=NUM_CLASSES)
    _train_demo_model(model, train_x, train_y, epochs=25, lr=1e-3)
    model.eval()
    return model, eval_x, eval_y


@pytest.fixture()
def correct_batch(lab):
    """A batch of up to 32 examples the model classifies correctly on clean input."""
    model, x, y = lab
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    mask = pred == y
    return model, x[mask][:32], y[mask][:32]
