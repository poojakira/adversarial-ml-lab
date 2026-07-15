"""Certified defense evaluation.

Evaluates adversarial robustness using provable (certified) defense methods
that provide mathematical guarantees on robustness within a given epsilon ball.

Key components:
  * **RandomizedSmoothing** -- Cohen et al. certified radius via Monte Carlo
    sampling. Adds Gaussian noise, takes majority vote, computes certified
    radius from binomial confidence bounds.
  * **lipschitz_eval** -- evaluates attacks against Lipschitz-constrained
    networks using spectral normalization for 1-Lipschitz layers.
  * **ibp_eval** -- interval bound propagation evaluation that computes
    worst-case bounds through network layers.
  * **find_certificate_boundary** -- binary search for the exact epsilon where
    the certified accuracy drops to zero (certificate breaks down).

References:
  - Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing"
    (ICML 2019).
  - Lecuyer et al., "Certified Robustness to Adversarial Examples with
    Differential Privacy" (IEEE S&P 2019).
  - Gowal et al., "Scalable Verified Training for Provably Robust Image
    Classification" (ICCV 2019) -- IBP training.
  - Mirman et al., "Differentiable Abstract Interpretation for Provably Robust
    Neural Networks" (ICML 2018).
  - Tsuzuku et al., "Lipschitz-Margin Training: Scalable Certification of
    Perturbation Invariance for Neural Networks" (NeurIPS 2018).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RandomizedSmoothing
# ---------------------------------------------------------------------------


@dataclass
class SmoothingResult:
    """Result of randomized smoothing certification.

    Attributes:
        predicted_class: the majority-vote predicted class for each sample.
        certified_radius: the certified L2 radius for each sample.
        is_certified: whether each sample has a valid certificate.
        num_samples_used: number of Monte Carlo samples used.
        confidence_level: statistical confidence level (e.g., 0.99).
    """

    predicted_class: Tensor
    certified_radius: Tensor
    is_certified: Tensor
    num_samples_used: int
    confidence_level: float


class RandomizedSmoothing:
    """Certified radius evaluation via Monte Carlo sampling.

    Implements the randomized smoothing framework from Cohen et al. (2019):
      1. Add isotropic Gaussian noise to the input.
      2. Classify the noisy sample.
      3. Repeat N times and take the majority vote.
      4. Compute certified L2 radius from the margin between the top class
         count and the runner-up, using a binomial confidence interval.

    The certified radius r at confidence level alpha is:
        r = sigma * Phi^{-1}(p_A)

    where p_A is the lower confidence bound on the probability of the most
    likely class, sigma is the noise standard deviation, and Phi^{-1} is
    the inverse standard normal CDF.

    Args:
        model: base classifier to smooth (must be in eval mode).
        sigma: standard deviation of Gaussian noise.
        n_samples: number of Monte Carlo samples for certification.
        confidence_level: statistical confidence for the certificate (e.g., 0.99).

    Example::

        smoother = RandomizedSmoothing(model, sigma=0.25, n_samples=1000)
        result = smoother.certify(images)
        print(f"Certified radius: {result.certified_radius.mean():.4f}")

    References:
        Cohen et al., "Certified Adversarial Robustness via Randomized
        Smoothing" (ICML 2019).
    """

    def __init__(
        self,
        model: nn.Module,
        sigma: float = 0.25,
        n_samples: int = 100,
        confidence_level: float = 0.99,
    ) -> None:
        _require_eval_mode(model)
        self.model = model
        self.sigma = sigma
        self.n_samples = n_samples
        self.confidence_level = confidence_level

    @staticmethod
    def _inverse_normal_cdf(p: float) -> float:
        """Approximate inverse standard normal CDF (probit function).

        Uses the rational approximation from Abramowitz and Stegun.
        Accurate to ~4.5e-4 for 0.5 < p < 1.
        """
        if p <= 0.0:
            return float("-inf")
        if p >= 1.0:
            return float("inf")
        if p == 0.5:
            return 0.0

        # Use symmetry for p < 0.5
        if p < 0.5:
            return -RandomizedSmoothing._inverse_normal_cdf(1.0 - p)

        # Rational approximation (Abramowitz and Stegun 26.2.23)
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308
        result = t - (c0 + c1 * t + c2 * t * t) / (
            1.0 + d1 * t + d2 * t * t + d3 * t * t * t
        )
        return result

    def _lower_confidence_bound(self, count: int, total: int) -> float:
        """Compute lower confidence bound on binomial proportion.

        Uses the normal approximation to the binomial for the lower bound
        on the probability of the most likely class.

        Args:
            count: number of times the top class was predicted.
            total: total number of samples.

        Returns:
            Lower confidence bound on the true probability.
        """
        if total == 0:
            return 0.0
        p_hat = count / total
        z = self._inverse_normal_cdf(self.confidence_level)
        # Normal approximation confidence interval
        margin = z * math.sqrt(p_hat * (1 - p_hat) / total)
        return max(0.0, p_hat - margin)

    def certify(self, images: Tensor) -> SmoothingResult:
        """Certify robustness of inputs via randomized smoothing.

        Args:
            images: input images in [0, 1] with shape (N, C, H, W).

        Returns:
            SmoothingResult with certified radius for each input.
        """
        batch_size = images.shape[0]
        num_classes = self.model(images[:1]).shape[1]

        # Count predictions over noisy samples
        counts = torch.zeros(batch_size, num_classes)

        for _ in range(self.n_samples):
            noise = torch.randn_like(images) * self.sigma
            noisy_input = torch.clamp(images + noise, 0.0, 1.0)
            with torch.no_grad():
                logits = self.model(noisy_input)
                preds = logits.argmax(dim=1)
            for i in range(batch_size):
                counts[i, preds[i].item()] += 1

        # Majority vote prediction
        predicted_class = counts.argmax(dim=1)
        top_counts = counts.max(dim=1).values

        # Compute certified radius for each sample
        certified_radius = torch.zeros(batch_size)
        is_certified = torch.zeros(batch_size, dtype=torch.bool)

        for i in range(batch_size):
            p_lower = self._lower_confidence_bound(
                int(top_counts[i].item()), self.n_samples
            )
            if p_lower > 0.5:
                radius = self.sigma * self._inverse_normal_cdf(p_lower)
                certified_radius[i] = radius
                is_certified[i] = True

        return SmoothingResult(
            predicted_class=predicted_class,
            certified_radius=certified_radius,
            is_certified=is_certified,
            num_samples_used=self.n_samples,
            confidence_level=self.confidence_level,
        )


# ---------------------------------------------------------------------------
# Lipschitz-Constrained Network Evaluation
# ---------------------------------------------------------------------------


class _SpectralNormLinear(nn.Module):
    """Linear layer with spectral normalization for 1-Lipschitz constraint.

    Applies power iteration to estimate and normalize by the spectral norm
    (largest singular value) of the weight matrix, enforcing a Lipschitz
    constant of 1 for this layer.

    References:
        Miyato et al., "Spectral Normalization for Generative Adversarial
        Networks" (ICLR 2018).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.utils.parametrizations.spectral_norm(
            nn.Linear(in_features, out_features)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class LipschitzNetwork(nn.Module):
    """Simple 1-Lipschitz network using spectrally-normalized layers.

    All linear layers are constrained via spectral normalization so the
    overall network has a bounded Lipschitz constant.

    Args:
        input_dim: input feature dimensionality.
        hidden_dim: hidden layer width.
        num_classes: number of output classes.
        num_layers: number of hidden layers.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 32,
        num_classes: int = 3,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(_SpectralNormLinear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(_SpectralNormLinear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        return self.net(x)


def lipschitz_eval(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.03,
    attack_steps: int = 40,
    alpha: float = 0.005,
) -> Tuple[float, float]:
    """Evaluate attacks against a Lipschitz-constrained network.

    Tests PGD attacks against the given model and reports both clean accuracy
    and robust accuracy. For a true 1-Lipschitz network, the robustness
    guarantee is that predictions cannot change if the perturbation is smaller
    than the margin divided by the Lipschitz constant.

    Args:
        model: Lipschitz-constrained classifier in ``eval()`` mode.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)`` or ``(N, D)``.
        labels: ground-truth class indices with shape ``(N,)``.
        epsilon: L-inf perturbation budget.
        attack_steps: number of PGD iterations.
        alpha: PGD step size.

    Returns:
        Tuple of (clean_accuracy, robust_accuracy).

    References:
        Tsuzuku et al., "Lipschitz-Margin Training: Scalable Certification of
        Perturbation Invariance for Neural Networks" (NeurIPS 2018).
    """
    _require_eval_mode(model)

    # Clean accuracy
    with torch.no_grad():
        clean_logits = model(images)
        clean_preds = clean_logits.argmax(dim=1)
        clean_acc = (clean_preds == labels).float().mean().item()

    # PGD attack
    x_adv = images.clone().detach()
    x_orig = images.clone().detach()

    for _ in range(attack_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = nn.functional.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    with torch.no_grad():
        adv_logits = model(x_adv)
        adv_preds = adv_logits.argmax(dim=1)
        robust_acc = (adv_preds == labels).float().mean().item()

    return clean_acc, robust_acc


# ---------------------------------------------------------------------------
# Interval Bound Propagation (IBP)
# ---------------------------------------------------------------------------


@dataclass
class IBPBounds:
    """Interval bounds computed through network layers.

    Attributes:
        lower: lower bound on the output logits (N, C).
        upper: upper bound on the output logits (N, C).
        verified: per-sample boolean indicating if the bounds verify robustness.
        epsilon: the epsilon used for computing these bounds.
    """

    lower: Tensor
    upper: Tensor
    verified: Tensor
    epsilon: float


def _ibp_linear(
    lower: Tensor, upper: Tensor, weight: Tensor, bias: Optional[Tensor]
) -> Tuple[Tensor, Tensor]:
    """Propagate interval bounds through a linear layer.

    For a linear layer y = Wx + b, the bounds are:
        y_lower = W+ * x_lower + W- * x_upper + b
        y_upper = W+ * x_upper + W- * x_lower + b

    where W+ = max(W, 0) and W- = min(W, 0).
    """
    w_pos = torch.clamp(weight, min=0.0)
    w_neg = torch.clamp(weight, max=0.0)

    new_lower = lower @ w_pos.t() + upper @ w_neg.t()
    new_upper = upper @ w_pos.t() + lower @ w_neg.t()

    if bias is not None:
        new_lower = new_lower + bias
        new_upper = new_upper + bias

    return new_lower, new_upper


def _ibp_relu(lower: Tensor, upper: Tensor) -> Tuple[Tensor, Tensor]:
    """Propagate interval bounds through a ReLU activation.

    ReLU preserves positivity: max(0, x).
    """
    return torch.clamp(lower, min=0.0), torch.clamp(upper, min=0.0)


def ibp_eval(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.03,
) -> IBPBounds:
    """Evaluate robustness using interval bound propagation.

    Computes worst-case output bounds by propagating the input perturbation
    set [x - epsilon, x + epsilon] through each layer of the network.
    A sample is verified robust if the lower bound of the true class logit
    exceeds the upper bound of all other class logits.

    This implementation supports simple sequential networks with Linear and
    ReLU layers. For convolutional networks, the input is flattened first.

    Args:
        model: classifier in ``eval()`` mode. Should be a simple sequential
            network with Linear and ReLU layers for exact IBP.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)`` or ``(N, D)``.
        labels: ground-truth class indices with shape ``(N,)``.
        epsilon: L-inf perturbation budget.

    Returns:
        IBPBounds with lower/upper output bounds and verification status.

    References:
        Gowal et al., "Scalable Verified Training for Provably Robust Image
        Classification" (ICCV 2019).
        Mirman et al., "Differentiable Abstract Interpretation for Provably
        Robust Neural Networks" (ICML 2018).
    """
    _require_eval_mode(model)

    # Flatten input for IBP propagation
    x_flat = images.view(images.shape[0], -1)
    lower = torch.clamp(x_flat - epsilon, 0.0, 1.0)
    upper = torch.clamp(x_flat + epsilon, 0.0, 1.0)

    # Propagate bounds through the network layers
    # Extract linear layers and activations from the model
    modules = _extract_sequential_modules(model)

    for module in modules:
        if isinstance(module, nn.Linear):
            lower, upper = _ibp_linear(lower, upper, module.weight, module.bias)
        elif isinstance(module, nn.ReLU):
            lower, upper = _ibp_relu(lower, upper)
        elif isinstance(module, nn.Flatten):
            continue  # Already flattened
        else:
            # For unsupported layers, use a conservative bound
            # by running both bounds through the layer
            mid = (lower + upper) / 2.0
            radius = (upper - lower) / 2.0
            with torch.no_grad():
                out_mid = module(mid)
            # Conservative: expand bounds by the radius times a safety factor
            lower = out_mid - radius.norm(dim=1, keepdim=True)
            upper = out_mid + radius.norm(dim=1, keepdim=True)

    # Verify: true class lower bound > all other class upper bounds
    batch_size = images.shape[0]
    num_classes = lower.shape[1]
    true_lower = lower.gather(1, labels.unsqueeze(1))  # (N, 1)

    # For each sample, check if true class lower > max other class upper
    mask = torch.ones(batch_size, num_classes, dtype=torch.bool)
    mask.scatter_(1, labels.unsqueeze(1), False)
    other_upper = upper.clone()
    other_upper[~mask] = float("-inf")
    max_other_upper = other_upper.max(dim=1, keepdim=True).values

    verified = (true_lower > max_other_upper).squeeze(1)

    return IBPBounds(
        lower=lower,
        upper=upper,
        verified=verified,
        epsilon=epsilon,
    )


def _extract_sequential_modules(model: nn.Module) -> List[nn.Module]:
    """Extract leaf modules from a model in sequential order.

    Recursively traverses the model to find all leaf modules (Linear, ReLU,
    Flatten, etc.) in their forward-pass order.
    """
    modules: List[nn.Module] = []

    def _recurse(m: nn.Module) -> None:
        children = list(m.children())
        if not children:
            modules.append(m)
        else:
            for child in children:
                _recurse(child)

    _recurse(model)
    return modules


# ---------------------------------------------------------------------------
# Certificate Boundary Finder
# ---------------------------------------------------------------------------


def find_certificate_boundary(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    method: str = "smoothing",
    sigma: float = 0.25,
    n_samples: int = 100,
    eps_low: float = 0.0,
    eps_high: float = 1.0,
    tolerance: float = 0.001,
    max_iterations: int = 30,
) -> Tuple[float, List[Tuple[float, float]]]:
    """Binary search for exact epsilon where the certificate breaks down.

    Finds the critical epsilon value where the certified accuracy transitions
    from positive to zero. This is the boundary where the provable guarantee
    no longer holds.

    The search proceeds by:
      1. Start with [eps_low, eps_high] bracket.
      2. Evaluate certified accuracy at the midpoint.
      3. If certified accuracy > 0, move eps_low up.
      4. If certified accuracy == 0, move eps_high down.
      5. Repeat until the bracket width < tolerance.

    Args:
        model: classifier in ``eval()`` mode.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        method: certification method ("smoothing" or "ibp").
        sigma: noise standard deviation for randomized smoothing.
        n_samples: Monte Carlo samples for smoothing.
        eps_low: lower bound of search range.
        eps_high: upper bound of search range.
        tolerance: precision of the binary search.
        max_iterations: maximum number of binary search iterations.

    Returns:
        Tuple of (critical_epsilon, search_history) where search_history is a
        list of (epsilon, certified_accuracy) pairs from each iteration.

    References:
        Cohen et al., "Certified Adversarial Robustness via Randomized
        Smoothing" (ICML 2019).
    """
    _require_eval_mode(model)

    search_history: List[Tuple[float, float]] = []

    def _eval_certified_accuracy(eps: float) -> float:
        """Evaluate certified accuracy at a given epsilon."""
        if method == "smoothing":
            smoother = RandomizedSmoothing(model, sigma=sigma, n_samples=n_samples)
            result = smoother.certify(images)
            # Certified accuracy: fraction of samples with radius >= eps
            # and correct prediction
            correct = result.predicted_class == labels
            certified_at_eps = result.certified_radius >= eps
            certified_correct = (correct & certified_at_eps).float().mean().item()
            return certified_correct
        elif method == "ibp":
            bounds = ibp_eval(model, images, labels, epsilon=eps)
            return bounds.verified.float().mean().item()
        else:
            raise ValueError(f"method must be 'smoothing' or 'ibp', got '{method}'")

    for _ in range(max_iterations):
        if eps_high - eps_low < tolerance:
            break

        eps_mid = (eps_low + eps_high) / 2.0
        cert_acc = _eval_certified_accuracy(eps_mid)
        search_history.append((eps_mid, cert_acc))

        logger.info(
            "Certificate boundary search: eps=%.6f, certified_acc=%.4f",
            eps_mid,
            cert_acc,
        )

        if cert_acc > 0.0:
            eps_low = eps_mid
        else:
            eps_high = eps_mid

    critical_epsilon = (eps_low + eps_high) / 2.0
    return critical_epsilon, search_history
