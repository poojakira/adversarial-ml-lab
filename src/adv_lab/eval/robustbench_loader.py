"""RobustBench model-zoo and CIFAR-10 loaders.

Why this module exists: the FGSM/PGD/C&W attacks in this lab are only as
trustworthy as the models we point them at. A synthetic CNN on an 8x8 toy task
(the ``harness`` CLI demo) proves the *code paths* run, but it says nothing
about whether the attacks reproduce known robustness numbers. To make our
results *comparable*, we evaluate against the standardized
`RobustBench <https://robustbench.github.io/>`_ CIFAR-10 benchmark:

* a fixed, versioned test subset (``load_cifar10`` returns images already in
  ``[0, 1]`` -- exactly the range our attacks assume), and
* pretrained models from the RobustBench model zoo, including the undefended
  ``Standard`` baseline and adversarially-trained entries such as
  ``Wang2023Better_WRN-28-10``.

RobustBench models expect inputs in ``[0, 1]`` and apply their own
normalization internally, so they drop straight into our attack functions with
no extra preprocessing. That is the whole point: an apples-to-apples comparison
against a published leaderboard.

This module is import-light: ``robustbench`` is imported lazily inside the
loader functions so the rest of the package (and its test suite) does not take
a hard dependency on it.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor

# RobustBench's canonical L-inf budget for CIFAR-10 leaderboards.
# Every AutoAttack robust-accuracy number on the CIFAR-10 L-inf board is
# measured at this epsilon, so we default to it for comparability.
CIFAR10_LINF_EPSILON = 8.0 / 255.0


def load_robustbench_model(
    model_name: str,
    dataset: str = "cifar10",
    threat_model: str = "Linf",
    model_dir: str = "./models",
    device: str | torch.device | None = None,
) -> nn.Module:
    """Load a pretrained model from the RobustBench model zoo.

    Thin wrapper over :func:`robustbench.utils.load_model` that also moves the
    model to ``device`` and puts it in ``eval()`` mode (a hard precondition of
    every attack in this lab).

    Args:
        model_name: RobustBench zoo id, e.g. ``"Standard"`` (undefended
            baseline) or ``"Wang2023Better_WRN-28-10"`` (adversarially trained).
        dataset: one of RobustBench's datasets (``"cifar10"``, ``"cifar100"``,
            ``"imagenet"``). ImageNet entries are gated/large -- CIFAR-10 is the
            supported path here.
        threat_model: ``"Linf"``, ``"L2"``, or ``"corruptions"``.
        model_dir: local cache directory for downloaded checkpoints.
        device: target device; defaults to CUDA when available, else CPU.

    Returns:
        The pretrained model, on ``device`` and in ``eval()`` mode. RobustBench
        models take inputs in ``[0, 1]`` and normalize internally.
    """
    # Lazy import so importing adv_lab does not require robustbench.
    from robustbench.utils import load_model as _rb_load_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _rb_load_model(
        model_name=model_name,
        model_dir=model_dir,
        dataset=dataset,
        threat_model=threat_model,
    )
    model = model.to(device)
    model.eval()
    return model


def load_cifar10(
    n_examples: int = 1000,
    data_dir: str = "./data",
) -> tuple[Tensor, Tensor]:
    """Load the first ``n_examples`` of the RobustBench CIFAR-10 test subset.

    Args:
        n_examples: number of test images to fetch (RobustBench returns a fixed,
            deterministic prefix of the CIFAR-10 test set).
        data_dir: local cache directory for the dataset (~170 MB on first run).

    Returns:
        ``(x_test, y_test)`` where ``x_test`` is ``[n, 3, 32, 32]`` in ``[0, 1]``
        and ``y_test`` is ``[n]`` int64 labels. These are exactly the ranges the
        lab's attacks expect.
    """
    # Lazy import so importing adv_lab does not require robustbench.
    from robustbench.data import load_cifar10 as _rb_load_cifar10

    x_test, y_test = _rb_load_cifar10(n_examples=n_examples, data_dir=data_dir)
    return x_test, y_test


def iter_batches(
    x: Tensor, y: Tensor, batch_size: int
) -> Iterable[tuple[Tensor, Tensor]]:
    """Yield ``(images, labels)`` mini-batches from in-memory tensors.

    Kept deliberately small and dependency-free so the benchmark harness can
    stream a large held-out set through memory-hungry models (e.g. WRN-28-10)
    without materializing every adversarial batch at once.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for i in range(0, x.shape[0], batch_size):
        yield x[i : i + batch_size], y[i : i + batch_size]
