"""PGD tests (4)."""

from __future__ import annotations

import torch

from adv_lab.attacks.pgd import pgd_attack, pgd_l2


def _flip_rate(model, x_adv, y) -> float:
    with torch.no_grad():
        pred = model(x_adv).argmax(dim=1)
    return float((pred != y).float().mean().item())


def test_pgd_stays_within_epsilon_ball(correct_batch):
    model, x, y = correct_batch
    epsilon = 0.03
    x_adv = pgd_attack(model, x, y, epsilon=epsilon, alpha=0.007, steps=40)
    linf = (x_adv - x).abs().max().item()
    assert linf <= epsilon + 1e-5
    assert x_adv.min().item() >= 0.0 and x_adv.max().item() <= 1.0


def test_pgd_l2_ball_projection(correct_batch):
    model, x, y = correct_batch
    epsilon = 0.5
    x_adv = pgd_l2(model, x, y, epsilon=epsilon, alpha=0.1, steps=40)
    per_sample_l2 = (x_adv - x).view(x.shape[0], -1).norm(p=2, dim=1)
    assert per_sample_l2.max().item() <= epsilon + 1e-4


def test_pgd_more_steps_increases_success_rate(correct_batch):
    model, x, y = correct_batch
    # No random start -> deterministic, isolates the effect of step count.
    adv_few = pgd_attack(
        model, x, y, epsilon=0.05, alpha=0.01, steps=1, random_start=False
    )
    adv_many = pgd_attack(
        model, x, y, epsilon=0.05, alpha=0.01, steps=50, random_start=False
    )
    assert _flip_rate(model, adv_many, y) >= _flip_rate(model, adv_few, y)


def test_pgd_random_start_gives_different_result(correct_batch):
    model, x, y = correct_batch
    a = pgd_attack(model, x, y, epsilon=0.05, alpha=0.01, steps=10, random_start=True)
    b = pgd_attack(model, x, y, epsilon=0.05, alpha=0.01, steps=10, random_start=True)
    # Independent random initializations should not coincide.
    assert not torch.allclose(a, b)
