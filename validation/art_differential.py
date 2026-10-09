"""Differential adversarial-attack validation against IBM ART.

This pilot compares security invariants rather than expecting byte-identical
adversarial examples: both implementations must respect the same L-infinity
budget and must not reduce the attack loss on a deterministic toy classifier.

It is interoperability evidence, not a production robustness score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from adv_lab.attacks.fgsm import fgsm_attack
from adv_lab.attacks.pgd import pgd_attack


def _loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(nn.functional.cross_entropy(model(x), y).item())


def _max_linf(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def main() -> int:
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    from art.estimators.classification import PyTorchClassifier
    from art.utils import to_categorical

    torch.manual_seed(7)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    with torch.no_grad():
        linear = model[1]
        linear.weight.copy_(torch.tensor([[1.2, -0.7, 0.4, -0.2], [-0.8, 0.9, -0.3, 0.7]]))
        linear.bias.copy_(torch.tensor([0.1, -0.1]))
    model.eval()

    images = torch.tensor(
        [
            [[[0.2, 0.7], [0.4, 0.6]]],
            [[[0.9, 0.1], [0.8, 0.2]]],
            [[[0.3, 0.4], [0.6, 0.8]]],
            [[[0.7, 0.6], [0.2, 0.1]]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    eps = 0.08
    alpha = 0.02
    steps = 5

    clean_loss = _loss(model, images, labels)
    ours_fgsm = fgsm_attack(model, images, labels, epsilon=eps)
    ours_pgd = pgd_attack(
        model, images, labels, epsilon=eps, alpha=alpha, steps=steps, random_start=False
    )

    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(1, 2, 2),
        nb_classes=2,
        clip_values=(0.0, 1.0),
    )
    x_np = images.numpy()
    y_np = to_categorical(labels.numpy(), nb_classes=2)
    art_fgsm = FastGradientMethod(estimator=classifier, eps=eps).generate(x=x_np, y=y_np)
    art_pgd = ProjectedGradientDescent(
        estimator=classifier,
        eps=eps,
        eps_step=alpha,
        max_iter=steps,
        num_random_init=0,
    ).generate(x=x_np, y=y_np)

    art_fgsm_t = torch.tensor(np.asarray(art_fgsm), dtype=torch.float32)
    art_pgd_t = torch.tensor(np.asarray(art_pgd), dtype=torch.float32)

    results = {
        "epsilon": eps,
        "clean_loss": round(clean_loss, 6),
        "ours_fgsm_linf": round(_max_linf(ours_fgsm, images), 6),
        "art_fgsm_linf": round(_max_linf(art_fgsm_t, images), 6),
        "ours_pgd_linf": round(_max_linf(ours_pgd, images), 6),
        "art_pgd_linf": round(_max_linf(art_pgd_t, images), 6),
        "ours_fgsm_loss": round(_loss(model, ours_fgsm, labels), 6),
        "art_fgsm_loss": round(_loss(model, art_fgsm_t, labels), 6),
        "ours_pgd_loss": round(_loss(model, ours_pgd, labels), 6),
        "art_pgd_loss": round(_loss(model, art_pgd_t, labels), 6),
        "reference": "IBM Adversarial Robustness Toolbox",
        "claim_boundary": (
            "Differential invariant check on a deterministic toy classifier. "
            "This is not a robustness score for a production model."
        ),
    }

    tol = 1e-5
    failures = []
    for key in ("ours_fgsm_linf", "art_fgsm_linf", "ours_pgd_linf", "art_pgd_linf"):
        if results[key] > eps + tol:
            failures.append(f"{key} exceeded epsilon")
    for key in ("ours_fgsm_loss", "art_fgsm_loss", "ours_pgd_loss", "art_pgd_loss"):
        if results[key] + tol < clean_loss:
            failures.append(f"{key} unexpectedly reduced attack loss")

    results["passed"] = not failures
    results["failures"] = failures
    out = Path("results/art_differential_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
