"""Tests for resource-constrained and timing-aware attacks."""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from adv_lab.attacks.constrained import (
    QueryBudgetManager,
    TimedAttack,
    rate_limited_attack,
)


def _pgd_step(
    model: nn.Module,
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.007,
    **kwargs: object,
) -> torch.Tensor:
    """Single PGD step for TimedAttack tests."""
    x_adv = x_adv.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = nn.functional.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv.detach() + alpha * grad.sign()
    delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
    return torch.clamp(x_orig + delta, 0.0, 1.0)


def _simple_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.007,
    steps: int = 10,
    **kwargs: object,
) -> torch.Tensor:
    """Simple PGD attack for rate_limited_attack tests."""
    x_orig = images.clone().detach()
    x_adv = x_orig.clone()
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        loss = nn.functional.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
    return x_adv.detach()


def test_timed_attack_respects_1s_budget(correct_batch):
    """TimedAttack terminates within 1-second wall-clock budget."""
    model, x, y = correct_batch

    timed = TimedAttack(_pgd_step, time_budget=1.0, max_steps=100000)
    start = time.time()
    x_adv = timed.run(model, x, y, epsilon=0.03)
    elapsed = time.time() - start

    # Should respect the 1s budget (with some tolerance for overhead)
    assert elapsed < 2.0
    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0


def test_timed_attack_respects_5s_budget(correct_batch):
    """TimedAttack terminates within 5-second wall-clock budget."""
    model, x, y = correct_batch

    timed = TimedAttack(_pgd_step, time_budget=5.0, max_steps=100000)
    start = time.time()
    x_adv = timed.run(model, x, y, epsilon=0.03)
    elapsed = time.time() - start

    assert elapsed < 6.0
    assert timed.steps_executed > 0


def test_timed_attack_tracks_steps(correct_batch):
    """TimedAttack records steps executed and time elapsed."""
    model, x, y = correct_batch

    timed = TimedAttack(_pgd_step, time_budget=0.5, max_steps=100000)
    timed.run(model, x, y, epsilon=0.03)

    assert timed.steps_executed > 0
    assert timed.time_elapsed > 0.0
    assert timed.time_elapsed <= 1.0  # generous bound


def test_query_budget_not_exceeded(correct_batch):
    """QueryBudgetManager does not exceed the specified query budget."""
    model, x, y = correct_batch

    budget = 50
    manager = QueryBudgetManager(query_budget=budget, batch_size=5)
    x_adv = manager.run_budget_aware_attack(model, x, y, epsilon=0.03)

    assert manager.queries_used <= budget
    assert x_adv.shape == x.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0


def test_query_budget_manager_priority_queue():
    """QueryBudgetManager correctly orders directions by priority."""
    manager = QueryBudgetManager(query_budget=100)

    # Add directions with known priorities
    low_priority_dir = torch.ones(1, 1, 8, 8) * 0.1
    high_priority_dir = torch.ones(1, 1, 8, 8) * 0.9

    manager.add_direction(low_priority_dir, priority=0.1)
    manager.add_direction(high_priority_dir, priority=0.9)

    # Should get high priority first
    first = manager.get_next_direction()
    assert first is not None
    assert torch.allclose(first, high_priority_dir)


def test_query_budget_manager_exhaustion():
    """QueryBudgetManager reports exhaustion correctly."""
    manager = QueryBudgetManager(query_budget=5)
    assert not manager.budget_exhausted
    assert manager.remaining_budget == 5

    manager.record_query(3)
    assert not manager.budget_exhausted
    assert manager.remaining_budget == 2

    manager.record_query(2)
    assert manager.budget_exhausted
    assert manager.remaining_budget == 0


def test_rate_limited_attack(correct_batch):
    """rate_limited_attack respects query limit."""
    model, x, y = correct_batch
    x_small = x[:4]
    y_small = y[:4]

    x_adv, queries_used = rate_limited_attack(
        _simple_attack,
        model,
        x_small,
        y_small,
        queries_per_minute=1000,
        max_queries=200,
        epsilon=0.03,
        steps=5,
    )

    assert x_adv.shape == x_small.shape
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0
    assert queries_used > 0
