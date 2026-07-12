"""Tests for Bayesian attack parameter optimization."""

from __future__ import annotations

import torch
import torch.nn as nn

from adv_lab.attacks.param_search import (
    BayesianAttackOptimizer,
    GaussianProcess,
    ParamBounds,
    per_sample_difficulty_score,
    _expected_improvement,
    _rbf_kernel,
)


def _simple_pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.007,
    steps: int = 10,
    **kwargs: object,
) -> torch.Tensor:
    """Minimal PGD for testing the optimizer."""
    x_orig = images.clone().detach()
    x_adv = x_orig.clone()
    noise = torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        loss = nn.functional.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
    return x_adv.detach()


def test_gp_fit_and_predict():
    """GP produces posterior mean and variance."""
    gp = GaussianProcess(length_scale=1.0, variance=1.0, noise=1e-3)
    X = torch.rand(10, 4)
    Y = torch.rand(10)
    gp.fit(X, Y)

    X_test = torch.rand(5, 4)
    mean, var = gp.predict(X_test)
    assert mean.shape == (5,)
    assert var.shape == (5,)
    assert (var > 0).all()  # Variance should be positive


def test_rbf_kernel_shape():
    """RBF kernel returns correct shape."""
    x1 = torch.rand(3, 4)
    x2 = torch.rand(5, 4)
    K = _rbf_kernel(x1, x2)
    assert K.shape == (3, 5)
    # Diagonal of self-kernel should be signal variance
    K_self = _rbf_kernel(x1, x1)
    assert torch.allclose(K_self.diag(), torch.ones(3), atol=1e-5)


def test_expected_improvement():
    """EI is non-negative and higher for points with high mean or variance."""
    mean = torch.tensor([0.5, 0.8, 0.3, 0.9])
    var = torch.tensor([0.1, 0.1, 0.5, 0.01])
    best_y = 0.7
    ei = _expected_improvement(mean, var, best_y)
    assert (ei >= 0).all()
    # Point with mean=0.9 should have high EI
    assert ei[3] > ei[2]


def test_optimizer_respects_query_budget(correct_batch):
    """BayesianAttackOptimizer does not exceed query budget."""
    model, x, y = correct_batch
    # Use a small subset for speed
    x_small = x[:4]
    y_small = y[:4]

    query_budget = 200
    optimizer = BayesianAttackOptimizer(
        attack_fn=_simple_pgd_attack,
        query_budget=query_budget,
        n_initial=2,
        n_candidates=10,
    )

    result = optimizer.optimize(model, x_small, y_small)
    assert result.queries_used <= query_budget


def test_optimizer_returns_difficulty_scores(correct_batch):
    """Optimizer result includes per-sample difficulty scores."""
    model, x, y = correct_batch
    x_small = x[:4]
    y_small = y[:4]

    optimizer = BayesianAttackOptimizer(
        attack_fn=_simple_pgd_attack,
        query_budget=300,
        n_initial=2,
        n_candidates=5,
    )
    result = optimizer.optimize(model, x_small, y_small)

    assert len(result.difficulty_scores) == 4
    for score in result.difficulty_scores:
        assert 0.0 <= score <= 1.0


def test_optimizer_params_in_valid_range(correct_batch):
    """Best parameters stay within defined bounds."""
    model, x, y = correct_batch
    x_small = x[:4]
    y_small = y[:4]

    bounds = ParamBounds(
        epsilon=(0.01, 0.1),
        step_count=(5, 30),
        step_size=(0.001, 0.02),
        restarts=(1, 3),
    )
    optimizer = BayesianAttackOptimizer(
        attack_fn=_simple_pgd_attack,
        query_budget=200,
        param_bounds=bounds,
        n_initial=2,
        n_candidates=5,
    )
    result = optimizer.optimize(model, x_small, y_small)

    if result.best_params:
        assert bounds.epsilon[0] <= result.best_params["epsilon"] <= bounds.epsilon[1]
        assert bounds.step_count[0] <= result.best_params["step_count"] <= bounds.step_count[1]
        assert bounds.step_size[0] <= result.best_params["step_size"] <= bounds.step_size[1]
        assert bounds.restarts[0] <= result.best_params["restarts"] <= bounds.restarts[1]


def test_per_sample_difficulty_score(correct_batch):
    """per_sample_difficulty_score returns values in [0, 1]."""
    model, x, y = correct_batch
    x_small = x[:8]
    y_small = y[:8]

    params = {"epsilon": 0.03, "step_size": 0.007, "step_count": 10}
    scores = per_sample_difficulty_score(
        model, x_small, y_small, _simple_pgd_attack, params, n_trials=3
    )
    assert len(scores) == 8
    for s in scores:
        assert 0.0 <= s <= 1.0
