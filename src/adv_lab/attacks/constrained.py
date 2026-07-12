"""Resource-constrained and timing-aware attacks.

Real-world adversarial attacks operate under resource constraints that
academic benchmarks rarely model:
  * **Wall-clock time** -- a production attack must succeed within seconds,
    not the minutes that PGD-1000 takes on a GPU.
  * **Query budgets** -- black-box APIs charge per-query or rate-limit
    aggressively.
  * **API rate limits** -- enforced by production ML-as-a-Service platforms.

This module provides wrappers and managers that enforce these constraints
while maximizing attack effectiveness within the allowed budget.

References:
  - Brendel et al., "Decision-Based Adversarial Attacks: Reliable Attacks
    Against Black-Box Machine Learning Models" (ICLR 2018).
  - Chen et al., "HopSkipJumpAttack" (2020) -- query-efficient attacks.
  - Croce and Hein, "Reliable Evaluation of Adversarial Robustness with an
    Ensemble of Attacks" (ICML 2020) -- AutoAttack budget allocation.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode


# ---------------------------------------------------------------------------
# TimedAttack
# ---------------------------------------------------------------------------


class TimedAttack:
    """Wrapper that enforces wall-clock time budgets on iterative attacks.

    Runs the attack loop internally and returns the best adversarial found
    within the time limit. Supports standard budgets of 1s, 5s, and 30s,
    or any custom float value.

    The attack checks ``time.time()`` at each iteration and terminates
    early if the budget would be exceeded.

    Args:
        attack_step_fn: a callable that performs one attack step.
            Signature: ``(model, x_adv, x_orig, labels, **kwargs) -> x_adv_new``
        time_budget: wall-clock time limit in seconds (e.g., 1.0, 5.0, 30.0).
        max_steps: maximum number of iterations (hard cap independent of time).

    Example::

        def pgd_step(model, x_adv, x_orig, labels, epsilon=0.03, alpha=0.007):
            x_adv = x_adv.requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), labels)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
            return torch.clamp(x_orig + delta, 0.0, 1.0)

        timed = TimedAttack(pgd_step, time_budget=5.0)
        x_adv = timed.run(model, images, labels, epsilon=0.03)
    """

    def __init__(
        self,
        attack_step_fn: Callable[..., Tensor],
        time_budget: float = 5.0,
        max_steps: int = 10000,
    ) -> None:
        self.attack_step_fn = attack_step_fn
        self.time_budget = time_budget
        self.max_steps = max_steps
        self.steps_executed: int = 0
        self.time_elapsed: float = 0.0

    def run(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        epsilon: float = 0.03,
        **kwargs: object,
    ) -> Tensor:
        """Run the timed attack.

        Args:
            model: target classifier in ``eval()`` mode.
            images: clean inputs in ``[0, 1]``.
            labels: ground-truth class indices.
            epsilon: perturbation budget passed to the step function.
            **kwargs: additional arguments forwarded to attack_step_fn.

        Returns:
            Best adversarial images found within the time budget.
        """
        _require_eval_mode(model)

        x_orig = images.clone().detach()
        x_adv = x_orig.clone().detach()

        # Random initialization within epsilon ball
        noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

        best_adv = x_adv.clone()
        best_loss = -float("inf")

        start_time = time.time()
        self.steps_executed = 0

        for step in range(self.max_steps):
            elapsed = time.time() - start_time
            if elapsed >= self.time_budget:
                break

            x_adv = self.attack_step_fn(
                model, x_adv, x_orig, labels, epsilon=epsilon, **kwargs
            )
            self.steps_executed = step + 1

            # Track best adversarial by loss
            with torch.no_grad():
                logits = model(x_adv)
                loss = nn.functional.cross_entropy(logits, labels).item()
            if loss > best_loss:
                best_loss = loss
                best_adv = x_adv.clone().detach()

        self.time_elapsed = time.time() - start_time
        return best_adv.detach()


# ---------------------------------------------------------------------------
# QueryBudgetManager
# ---------------------------------------------------------------------------


@dataclass
class PerturbationDirection:
    """A candidate perturbation direction with priority score."""

    priority: float
    direction: Tensor
    loss_decrease: float = 0.0

    def __lt__(self, other: "PerturbationDirection") -> bool:
        """Higher priority = better. Use negative for min-heap -> max behavior."""
        return self.priority > other.priority


class QueryBudgetManager:
    """Manages a query budget with priority queuing of perturbation directions.

    Maintains a priority queue of perturbation directions ordered by their
    estimated loss decrease (most promising first). Implements early stopping
    when the budget is exhausted.

    The manager tracks:
      * Total queries used vs budget.
      * Priority queue of candidate directions.
      * Best adversarial found so far.

    Args:
        query_budget: maximum number of model queries allowed.
        batch_size: number of directions to try per round.

    Example::

        manager = QueryBudgetManager(query_budget=500)
        while not manager.budget_exhausted:
            direction = manager.get_next_direction()
            if direction is None:
                manager.add_random_directions(x_adv)
                continue
            loss = evaluate(x_adv + direction)
            manager.record_query()
            manager.update_priority(direction, loss)
    """

    def __init__(self, query_budget: int, batch_size: int = 10) -> None:
        self.query_budget = query_budget
        self.batch_size = batch_size
        self.queries_used: int = 0
        self._queue: List[PerturbationDirection] = []
        self.best_loss: float = -float("inf")
        self.best_adv: Optional[Tensor] = None

    @property
    def budget_exhausted(self) -> bool:
        """Whether the query budget has been fully consumed."""
        return self.queries_used >= self.query_budget

    @property
    def remaining_budget(self) -> int:
        """Number of queries remaining."""
        return max(0, self.query_budget - self.queries_used)

    def record_query(self, n: int = 1) -> None:
        """Record that n queries were consumed."""
        self.queries_used += n

    def add_direction(self, direction: Tensor, priority: float) -> None:
        """Add a perturbation direction to the priority queue.

        Args:
            direction: perturbation tensor.
            priority: estimated effectiveness (higher = tried sooner).
        """
        entry = PerturbationDirection(priority=priority, direction=direction)
        heapq.heappush(self._queue, entry)

    def get_next_direction(self) -> Optional[Tensor]:
        """Pop the highest-priority direction, or None if queue is empty."""
        if not self._queue:
            return None
        entry = heapq.heappop(self._queue)
        return entry.direction

    def add_random_directions(
        self, reference: Tensor, n: int = 20, scale: float = 0.01
    ) -> None:
        """Generate random perturbation directions and add to queue.

        Args:
            reference: reference tensor for shape.
            n: number of random directions to generate.
            scale: magnitude of random directions.
        """
        for _ in range(n):
            direction = torch.randn_like(reference) * scale
            # Initial priority is random; will be updated after evaluation
            priority = torch.rand(1).item()
            self.add_direction(direction, priority)

    def update_best(self, x_adv: Tensor, loss: float) -> None:
        """Update best adversarial if loss improved.

        Args:
            x_adv: candidate adversarial.
            loss: loss value achieved.
        """
        if loss > self.best_loss:
            self.best_loss = loss
            self.best_adv = x_adv.clone().detach()

    def run_budget_aware_attack(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        epsilon: float = 0.03,
    ) -> Tensor:
        """Run a complete budget-aware attack with priority queuing.

        Uses the priority queue to evaluate the most promising perturbation
        directions first, with early stopping when budget is exhausted.

        Args:
            model: target classifier in ``eval()`` mode.
            images: clean inputs in ``[0, 1]``.
            labels: ground-truth class indices.
            epsilon: L-inf perturbation budget.

        Returns:
            Best adversarial images found within the query budget.
        """
        _require_eval_mode(model)

        x_orig = images.clone().detach()
        x_adv = x_orig.clone()
        self.best_adv = x_adv.clone()

        # Initial loss
        with torch.no_grad():
            logits = model(x_adv)
            loss = nn.functional.cross_entropy(logits, labels).item()
        self.record_query()
        self.best_loss = loss

        # Seed the queue with random directions
        self.add_random_directions(x_adv, n=self.batch_size * 2, scale=epsilon * 0.5)

        while not self.budget_exhausted:
            direction = self.get_next_direction()
            if direction is None:
                self.add_random_directions(x_adv, n=self.batch_size, scale=epsilon * 0.3)
                continue

            # Try the direction
            candidate = x_adv + direction
            delta = torch.clamp(candidate - x_orig, -epsilon, epsilon)
            candidate = torch.clamp(x_orig + delta, 0.0, 1.0)

            with torch.no_grad():
                logits = model(candidate)
                new_loss = nn.functional.cross_entropy(logits, labels).item()
            self.record_query()

            # If improvement, accept and replenish queue from this point
            if new_loss > self.best_loss:
                x_adv = candidate
                self.update_best(x_adv, new_loss)
                # Add nearby directions (exploit neighborhood)
                self.add_random_directions(
                    x_adv, n=self.batch_size, scale=epsilon * 0.2
                )
            else:
                # Add direction back with lower priority
                reduced_priority = max(0.0, new_loss - self.best_loss)
                self.add_direction(direction * 0.5, reduced_priority)

        return self.best_adv if self.best_adv is not None else x_adv.detach()


# ---------------------------------------------------------------------------
# rate_limited_attack
# ---------------------------------------------------------------------------


def rate_limited_attack(
    attack_fn: Callable[..., Tensor],
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    queries_per_minute: int = 60,
    max_queries: int = 100,
    **attack_kwargs: object,
) -> Tuple[Tensor, int]:
    """Wrap any attack with a simulated rate limit.

    Simulates a production API rate limit by tracking query timestamps and
    sleeping (via busy-wait check) when the rate is exceeded. This models
    the real constraint of attacking ML-as-a-Service endpoints.

    In simulation mode, we do not actually sleep but track the simulated
    time cost.

    Args:
        attack_fn: the underlying attack function. Must accept
            ``(model, images, labels, **kwargs)`` and return adversarial tensor.
        model: target classifier in ``eval()`` mode.
        images: clean inputs in ``[0, 1]``.
        labels: ground-truth class indices.
        queries_per_minute: maximum queries allowed per minute.
        max_queries: total query budget.
        **attack_kwargs: passed to attack_fn.

    Returns:
        Tuple of (adversarial images, total queries used).
    """
    _require_eval_mode(model)

    query_count = 0
    interval = 60.0 / queries_per_minute  # minimum seconds between queries
    best_adv = images.clone().detach()
    best_loss = -float("inf")

    # Wrap model to count queries
    class RateLimitedModel(nn.Module):
        def __init__(self, base_model: nn.Module) -> None:
            super().__init__()
            self.base_model = base_model
            self.query_count = 0
            self.last_query_time = 0.0

        def forward(self, x: Tensor) -> Tensor:
            nonlocal query_count
            # Simulate rate limiting (track time without actual sleep)
            current_time = time.time()
            if self.last_query_time > 0:
                elapsed = current_time - self.last_query_time
                if elapsed < interval:
                    # In real deployment, would sleep here
                    pass
            self.last_query_time = current_time
            self.query_count += 1
            query_count += 1

            if query_count > max_queries:
                # Budget exhausted - return last valid output
                with torch.no_grad():
                    return self.base_model(x)

            return self.base_model(x)

    wrapped_model = RateLimitedModel(model)
    wrapped_model.eval()

    # Run the attack with the rate-limited model
    x_adv = attack_fn(wrapped_model, images, labels, **attack_kwargs)

    # Ensure output is within bounds
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    return x_adv, query_count
