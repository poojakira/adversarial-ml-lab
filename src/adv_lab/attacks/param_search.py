"""Adaptive parameter search via Bayesian optimization.

This module implements a Bayesian optimizer for attack hyperparameters using
a Gaussian process (GP) surrogate. The optimizer tunes:
  * epsilon (perturbation budget)
  * step_count (number of PGD iterations)
  * step_size (alpha / learning rate)
  * restarts (number of random restarts)

The budget is specified as a **query count** (total model forward passes),
not an iteration count, making it applicable to both white-box and
query-limited black-box settings.

The GP is implemented from scratch using torch (kernel matrix inversion +
posterior mean/variance), with no external libraries (no scikit-learn, no
GPyTorch).

References:
  - Snoek et al., "Practical Bayesian Optimization of Machine Learning
    Hyperparameters" (NeurIPS 2012).
  - Croce and Hein, "Reliable Evaluation of Adversarial Robustness with
    an Ensemble of Attacks" (ICML 2020) -- motivates per-sample adaptation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode


# ---------------------------------------------------------------------------
# Gaussian Process Implementation
# ---------------------------------------------------------------------------


def _rbf_kernel(
    x1: Tensor, x2: Tensor, length_scale: float = 1.0, variance: float = 1.0
) -> Tensor:
    """Radial basis function (squared exponential) kernel.

    Args:
        x1: shape ``(n1, d)``
        x2: shape ``(n2, d)``
        length_scale: kernel length scale.
        variance: signal variance.

    Returns:
        Kernel matrix of shape ``(n1, n2)``.
    """
    # Squared Euclidean distance
    dist_sq = torch.cdist(x1 / length_scale, x2 / length_scale, p=2).pow(2)
    return variance * torch.exp(-0.5 * dist_sq)


class GaussianProcess:
    """Minimal Gaussian process regression with RBF kernel.

    Implements posterior mean and variance for Bayesian optimization.
    Uses Cholesky decomposition for numerical stability.

    Args:
        length_scale: RBF kernel length scale.
        variance: signal variance.
        noise: observation noise variance (jitter for numerical stability).
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        variance: float = 1.0,
        noise: float = 1e-4,
    ) -> None:
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise
        self.x_train: Optional[Tensor] = None
        self.y_train: Optional[Tensor] = None
        self._alpha: Optional[Tensor] = None
        self._L: Optional[Tensor] = None

    def fit(self, x: Tensor, y: Tensor) -> None:
        """Fit the GP to observed data.

        Args:
            x: input observations, shape ``(n, d)``.
            y: output observations, shape ``(n,)``.
        """
        self.x_train = x
        self.y_train = y
        K = _rbf_kernel(x, x, self.length_scale, self.variance)
        K = K + self.noise * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
        self._L = torch.linalg.cholesky(K)
        # alpha = K^{-1} y  via Cholesky solve
        self._alpha = torch.cholesky_solve(y.unsqueeze(1), self._L).squeeze(1)

    def predict(self, x_new: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict posterior mean and variance at new points.

        Args:
            x_new: test points, shape ``(m, d)``.

        Returns:
            Tuple of (mean, variance) each of shape ``(m,)``.
        """
        assert self.x_train is not None and self._L is not None
        K_star = _rbf_kernel(x_new, self.x_train, self.length_scale, self.variance)
        mean = K_star @ self._alpha

        # Variance: K** - K* K^{-1} K*^T
        v = torch.linalg.solve_triangular(self._L, K_star.T, upper=False)
        K_ss = _rbf_kernel(x_new, x_new, self.length_scale, self.variance)
        var = K_ss.diag() - (v * v).sum(dim=0)
        var = var.clamp(min=1e-8)  # numerical safety

        return mean, var


# ---------------------------------------------------------------------------
# Acquisition function: Expected Improvement
# ---------------------------------------------------------------------------


def _expected_improvement(
    mean: Tensor, var: Tensor, best_y: float, xi: float = 0.01
) -> Tensor:
    """Compute Expected Improvement acquisition function.

    Args:
        mean: predicted means, shape ``(m,)``.
        var: predicted variances, shape ``(m,)``.
        best_y: best observed objective value so far.
        xi: exploration-exploitation tradeoff parameter.

    Returns:
        EI values, shape ``(m,)``.
    """
    std = var.sqrt()
    z = (mean - best_y - xi) / std
    # Standard normal CDF and PDF via erf
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    ei = (mean - best_y - xi) * cdf + std * pdf
    return ei.clamp(min=0.0)


# ---------------------------------------------------------------------------
# BayesianAttackOptimizer
# ---------------------------------------------------------------------------


@dataclass
class ParamBounds:
    """Parameter search bounds."""

    epsilon: Tuple[float, float] = (0.01, 0.3)
    step_count: Tuple[int, int] = (5, 100)
    step_size: Tuple[float, float] = (0.001, 0.05)
    restarts: Tuple[int, int] = (1, 10)


@dataclass
class OptimizationResult:
    """Result from Bayesian optimization."""

    best_params: Dict[str, float]
    best_score: float
    queries_used: int
    difficulty_scores: List[float]
    history: List[Dict[str, object]] = field(default_factory=list)


class BayesianAttackOptimizer:
    """Bayesian optimization over attack hyperparameters.

    Uses a Gaussian process surrogate with Expected Improvement acquisition
    to find optimal (epsilon, step_count, step_size, restarts) within a
    query budget.

    The budget is specified as a **query count** (total model forward passes
    across all evaluations), not as a number of BO iterations. This is
    critical for fair comparison in query-limited threat models.

    Args:
        attack_fn: attack function with signature
            ``(model, images, labels, epsilon, alpha, steps, restarts) -> Tensor``.
        query_budget: maximum number of model forward passes allowed.
        param_bounds: search bounds for each hyperparameter.
        n_initial: number of random initial points before GP is used.
        n_candidates: number of candidate points evaluated per acquisition step.

    Example::

        optimizer = BayesianAttackOptimizer(my_attack, query_budget=1000)
        result = optimizer.optimize(model, images, labels)
        print(result.best_params, result.best_score)
    """

    def __init__(
        self,
        attack_fn: Callable[..., Tensor],
        query_budget: int = 1000,
        param_bounds: Optional[ParamBounds] = None,
        n_initial: int = 5,
        n_candidates: int = 50,
    ) -> None:
        self.attack_fn = attack_fn
        self.query_budget = query_budget
        self.param_bounds = param_bounds or ParamBounds()
        self.n_initial = n_initial
        self.n_candidates = n_candidates
        self.gp = GaussianProcess(length_scale=1.0, variance=1.0, noise=1e-3)

    def _params_to_tensor(self, params: Dict[str, float]) -> Tensor:
        """Normalize parameters to [0, 1] for GP."""
        bounds = self.param_bounds
        eps_norm = (params["epsilon"] - bounds.epsilon[0]) / (
            bounds.epsilon[1] - bounds.epsilon[0]
        )
        sc_norm = (params["step_count"] - bounds.step_count[0]) / (
            bounds.step_count[1] - bounds.step_count[0]
        )
        ss_norm = (params["step_size"] - bounds.step_size[0]) / (
            bounds.step_size[1] - bounds.step_size[0]
        )
        r_norm = (params["restarts"] - bounds.restarts[0]) / (
            bounds.restarts[1] - bounds.restarts[0]
        )
        return torch.tensor([eps_norm, sc_norm, ss_norm, r_norm])

    def _tensor_to_params(self, t: Tensor) -> Dict[str, float]:
        """Denormalize [0, 1] tensor back to parameter space."""
        bounds = self.param_bounds
        t = t.clamp(0.0, 1.0)
        return {
            "epsilon": bounds.epsilon[0]
            + t[0].item() * (bounds.epsilon[1] - bounds.epsilon[0]),
            "step_count": int(
                bounds.step_count[0]
                + t[1].item() * (bounds.step_count[1] - bounds.step_count[0])
            ),
            "step_size": bounds.step_size[0]
            + t[2].item() * (bounds.step_size[1] - bounds.step_size[0]),
            "restarts": int(
                bounds.restarts[0]
                + t[3].item() * (bounds.restarts[1] - bounds.restarts[0])
            ),
        }

    def _random_params(self) -> Dict[str, float]:
        """Sample random parameters within bounds."""
        t = torch.rand(4)
        return self._tensor_to_params(t)

    def _evaluate(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        params: Dict[str, float],
    ) -> Tuple[float, int]:
        """Run attack with given params and return (success_rate, queries_used).

        Queries used = batch_size * step_count * restarts (one forward per step).
        """
        batch_size = images.shape[0]
        step_count = int(params["step_count"])
        restarts = int(params["restarts"])
        queries = batch_size * step_count * restarts

        images.clone().detach()
        best_success = 0.0

        for _ in range(restarts):
            x_adv = self.attack_fn(
                model,
                images,
                labels,
                epsilon=params["epsilon"],
                alpha=params["step_size"],
                steps=step_count,
            )
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
            success = float((pred != labels).float().mean().item())
            if success > best_success:
                best_success = success

        return best_success, queries

    def optimize(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
    ) -> OptimizationResult:
        """Run Bayesian optimization within the query budget.

        Args:
            model: target classifier in ``eval()`` mode.
            images: clean inputs in ``[0, 1]``.
            labels: ground-truth class indices.

        Returns:
            OptimizationResult with best parameters, scores, and difficulty.
        """
        _require_eval_mode(model)

        total_queries = 0
        x_observed: List[Tensor] = []
        y_observed: List[float] = []
        history: List[Dict[str, object]] = []

        best_params: Dict[str, float] = {}
        best_score = -1.0

        # Phase 1: Random exploration
        for _ in range(self.n_initial):
            if total_queries >= self.query_budget:
                break
            params = self._random_params()
            score, queries = self._evaluate(model, images, labels, params)
            total_queries += queries

            if total_queries > self.query_budget:
                total_queries = self.query_budget
                break

            x_observed.append(self._params_to_tensor(params))
            y_observed.append(score)
            history.append({"params": params, "score": score, "queries": total_queries})

            if score > best_score:
                best_score = score
                best_params = params

        # Phase 2: GP-guided optimization
        while total_queries < self.query_budget:
            if len(x_observed) < 2:
                break

            X = torch.stack(x_observed)
            Y = torch.tensor(y_observed)
            self.gp.fit(X, Y)

            # Generate candidates and pick best by EI
            candidates = torch.rand(self.n_candidates, 4)
            mean, var = self.gp.predict(candidates)
            ei = _expected_improvement(mean, var, best_score)
            best_idx = ei.argmax().item()
            next_params = self._tensor_to_params(candidates[best_idx])

            # Check if evaluation would exceed budget
            step_count = int(next_params["step_count"])
            restarts = int(next_params["restarts"])
            estimated_queries = images.shape[0] * step_count * restarts
            if total_queries + estimated_queries > self.query_budget:
                break

            score, queries = self._evaluate(model, images, labels, next_params)
            total_queries += queries

            x_observed.append(self._params_to_tensor(next_params))
            y_observed.append(score)
            history.append(
                {"params": next_params, "score": score, "queries": total_queries}
            )

            if score > best_score:
                best_score = score
                best_params = next_params

        # Compute per-sample difficulty scores
        difficulty_scores = per_sample_difficulty_score(
            model, images, labels, self.attack_fn, best_params
        )

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            queries_used=total_queries,
            difficulty_scores=difficulty_scores,
            history=history,
        )


# ---------------------------------------------------------------------------
# Per-sample difficulty score
# ---------------------------------------------------------------------------


def per_sample_difficulty_score(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    attack_fn: Callable[..., Tensor],
    params: Dict[str, float],
    n_trials: int = 5,
) -> List[float]:
    """Compute per-sample attack difficulty scores.

    Difficulty is measured as (1 - success_rate) across multiple random
    restarts of the attack.  A score of 1.0 means the sample was never
    successfully attacked; 0.0 means it was always attacked.

    This provides a fine-grained view of model robustness beyond aggregate
    accuracy, inspired by AutoAttack's per-sample analysis (Croce and Hein,
    ICML 2020).

    Args:
        model: target classifier in ``eval()`` mode.
        images: clean inputs, shape ``(N, C, H, W)``.
        labels: ground-truth labels, shape ``(N,)``.
        attack_fn: attack function to evaluate.
        params: attack hyperparameters (epsilon, step_size, steps).
        n_trials: number of random trials per sample.

    Returns:
        List of difficulty scores in [0, 1], one per sample.
    """
    _require_eval_mode(model)

    batch_size = images.shape[0]
    success_counts = torch.zeros(batch_size)

    epsilon = params.get("epsilon", 0.03)
    step_size = params.get("step_size", 0.007)
    step_count = int(params.get("step_count", 40))

    for _ in range(n_trials):
        x_adv = attack_fn(
            model,
            images,
            labels,
            epsilon=epsilon,
            alpha=step_size,
            steps=step_count,
        )
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        success_counts += (pred != labels).float()

    # Difficulty = 1 - (success_rate per sample)
    difficulty = 1.0 - (success_counts / n_trials)
    return difficulty.tolist()
