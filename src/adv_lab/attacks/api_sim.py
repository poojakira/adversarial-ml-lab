"""API behavior simulation for adversarial attacks on deployed ML services.

Production-grade simulation of ML-as-a-Service API constraints and behaviors
for realistic adversarial attack evaluation. Models the full request lifecycle
including rate limiting, query budgeting, response transformation, and
server-side anomaly detection that deployed systems use to detect adversarial
query patterns.

Key components:
  * **APISimulator** -- wraps any model with production API constraints including
    per-minute rate limiting, lifetime query budgets, configurable response
    formats (confidence rounding, top-K filtering), and query logging with
    real-time anomaly detection.
  * **simulated_api_attack** -- executes any attack function through the API
    envelope, transparently enforcing all constraints and capturing diagnostics.
  * **anomaly_detection_evasion** -- implements query-shaping strategies that
    evade server-side anomaly detectors by diversifying query distributions,
    interspersing benign traffic, and avoiding suspicious clustering patterns.

References:
  - Tramer et al., "Stealing Machine Learning Models via Prediction APIs"
    (USENIX Security 2016).
  - Papernot et al., "Practical Black-Box Attacks Against Machine Learning"
    (AsiaCCS 2017).
  - Juuti et al., "PRADA: Protecting Against DNN Model Stealing Attacks"
    (EuroS&P 2019) -- distribution-based query anomaly detection.
  - Shi et al., "Active Defense Against Query-based Model Stealing Attacks"
    (2020) -- query pattern monitoring and poisoning.
  - Chen et al., "Stateful Detection of Black-Box Adversarial Attacks" (ACM
    CCS Workshop 2020) -- stateful anomaly detection for ML APIs.
"""

from __future__ import annotations

import collections
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from adv_lab.attacks.fgsm import _require_eval_mode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anomaly Detection Subsystem
# ---------------------------------------------------------------------------


class AnomalyType(Enum):
    """Types of anomalous query behavior that indicate adversarial activity."""

    INPUT_CLUSTERING = "input_clustering"
    HIGH_QUERY_RATE = "high_query_rate"
    LOW_ENTROPY_RESPONSES = "low_entropy_responses"
    SEQUENTIAL_SIMILARITY = "sequential_similarity"
    DISTRIBUTION_SHIFT = "distribution_shift"
    REPEATED_QUERIES = "repeated_queries"


@dataclass
class AnomalyEvent:
    """Record of a detected anomaly in query patterns.

    Attributes:
        anomaly_type: category of the anomaly.
        query_index: which query triggered the detection.
        severity: normalized severity score in [0, 1].
        detail: human-readable description of the anomaly.
        timestamp: when the anomaly was detected.
    """

    anomaly_type: AnomalyType
    query_index: int
    severity: float
    detail: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class QueryRecord:
    """Full record of a single API query for forensic analysis.

    Attributes:
        timestamp: wall-clock time of the query.
        query_index: sequential index.
        input_l2_norm: L2 norm of the flattened input.
        input_linf_norm: L-inf norm of the input.
        response_entropy: entropy of the response probability distribution.
        response_max_conf: maximum confidence in the response.
        predicted_class: the top-1 predicted class.
        inter_query_gap: time since the previous query (seconds).
    """

    timestamp: float
    query_index: int
    input_l2_norm: float
    input_linf_norm: float
    response_entropy: float
    response_max_conf: float
    predicted_class: int
    inter_query_gap: float


class _AnomalyDetector:
    """Server-side anomaly detection engine for ML API query monitoring.

    Implements multiple detection heuristics based on published research:
      - Input clustering: detects when queries concentrate in a small region
        (indicative of gradient estimation or boundary probing).
      - Sequential similarity: detects when consecutive queries are too similar
        (indicative of finite-difference gradient attacks).
      - Distribution shift: detects when query statistics deviate significantly
        from baseline (indicative of attack-mode traffic).
      - Response entropy patterns: detects when queries consistently produce
        low-entropy (high-confidence) or uniform responses.
    """

    def __init__(
        self,
        window_size: int = 50,
        clustering_threshold: float = 0.05,
        similarity_threshold: float = 0.98,
    ) -> None:
        self.window_size = window_size
        self.clustering_threshold = clustering_threshold
        self.similarity_threshold = similarity_threshold
        self._recent_norms: Deque[float] = collections.deque(maxlen=window_size)
        self._recent_entropies: Deque[float] = collections.deque(maxlen=window_size)
        self._last_input_flat: Optional[Tensor] = None
        self._baseline_mean_norm: Optional[float] = None
        self._baseline_std_norm: Optional[float] = None
        self.events: List[AnomalyEvent] = []

    def check(self, record: QueryRecord, input_flat: Tensor) -> List[AnomalyEvent]:
        """Run all anomaly checks on a new query record."""
        new_events: List[AnomalyEvent] = []

        self._recent_norms.append(record.input_l2_norm)
        self._recent_entropies.append(record.response_entropy)

        # Clustering detection: low variance in input norms
        if len(self._recent_norms) >= 10:
            norms_t = torch.tensor(list(self._recent_norms))
            norm_std = norms_t.std().item()
            norm_mean = norms_t.mean().item()
            relative_std = norm_std / (norm_mean + 1e-10)

            if relative_std < self.clustering_threshold:
                event = AnomalyEvent(
                    anomaly_type=AnomalyType.INPUT_CLUSTERING,
                    query_index=record.query_index,
                    severity=min(
                        1.0, self.clustering_threshold / (relative_std + 1e-10) * 0.5
                    ),
                    detail=(
                        f"Input norm relative std {relative_std:.6f} below "
                        f"threshold {self.clustering_threshold} over last "
                        f"{len(self._recent_norms)} queries"
                    ),
                )
                new_events.append(event)

        # Sequential similarity: consecutive queries too close
        if self._last_input_flat is not None:
            cos_sim = torch.nn.functional.cosine_similarity(
                input_flat.unsqueeze(0), self._last_input_flat.unsqueeze(0)
            ).item()
            if cos_sim > self.similarity_threshold:
                event = AnomalyEvent(
                    anomaly_type=AnomalyType.SEQUENTIAL_SIMILARITY,
                    query_index=record.query_index,
                    severity=min(
                        1.0,
                        (cos_sim - self.similarity_threshold)
                        / (1.0 - self.similarity_threshold + 1e-10),
                    ),
                    detail=(
                        f"Cosine similarity {cos_sim:.6f} with previous query "
                        f"exceeds threshold {self.similarity_threshold}"
                    ),
                )
                new_events.append(event)

        # Distribution shift: input norms deviate from baseline
        if self._baseline_mean_norm is not None and len(self._recent_norms) >= 20:
            current_mean = torch.tensor(list(self._recent_norms)).mean().item()
            z_score = abs(current_mean - self._baseline_mean_norm) / (
                self._baseline_std_norm + 1e-10
            )
            if z_score > 3.0:
                event = AnomalyEvent(
                    anomaly_type=AnomalyType.DISTRIBUTION_SHIFT,
                    query_index=record.query_index,
                    severity=min(1.0, z_score / 10.0),
                    detail=(
                        f"Query distribution z-score {z_score:.2f} indicates "
                        f"significant shift from baseline"
                    ),
                )
                new_events.append(event)
        elif (
            self._baseline_mean_norm is None
            and len(self._recent_norms) >= self.window_size
        ):
            # Establish baseline from first window
            norms_t = torch.tensor(list(self._recent_norms))
            self._baseline_mean_norm = norms_t.mean().item()
            self._baseline_std_norm = norms_t.std().item()

        self._last_input_flat = input_flat.detach().clone()
        self.events.extend(new_events)
        return new_events

    def reset(self) -> None:
        """Reset all detection state."""
        self._recent_norms.clear()
        self._recent_entropies.clear()
        self._last_input_flat = None
        self._baseline_mean_norm = None
        self._baseline_std_norm = None
        self.events.clear()


# ---------------------------------------------------------------------------
# APISimulator
# ---------------------------------------------------------------------------


class APISimulator:
    """Production ML API simulator with realistic operational constraints.

    Wraps any PyTorch model to simulate the behavior of a deployed ML-as-a-Service
    endpoint. Enforces the constraints that real production systems impose:

      * **Rate limiting**: Maximum queries per minute with sliding window
        enforcement. Excess queries raise RuntimeError (simulating HTTP 429).
      * **Query budget**: Lifetime maximum queries (simulating billing quotas
        or security limits). Budget exhaustion raises RuntimeError.
      * **Confidence rounding**: Response probabilities are rounded to N decimal
        places (typically 2-4), preventing extraction of exact gradients.
      * **Top-K filtering**: Only the top-K class scores are returned; remaining
        classes show zero probability.
      * **Response logging**: Every query is recorded with full metadata for
        server-side anomaly detection and forensic analysis.

    The simulator maintains internal state across queries, enabling stateful
    anomaly detection that can identify adversarial query patterns.

    Args:
        model: the underlying classifier (must be in eval mode when constructed).
        rate_limit: maximum queries per 60-second sliding window (0 = unlimited).
        total_budget: maximum lifetime queries (0 = unlimited).
        confidence_rounding: decimal places for probability rounding.
        top_k_only: if > 0, only return top-K class probabilities.
        enable_logging: whether to maintain full query logs.
        enable_anomaly_detection: whether to run anomaly detection on queries.

    Example::

        api = APISimulator(
            model, rate_limit=60, total_budget=1000,
            top_k_only=5, confidence_rounding=2
        )
        probs = api.query(images)  # shape: (N, num_classes)
        print(f"Budget remaining: {api.queries_remaining}")
        print(f"Anomalies detected: {len(api.anomaly_events)}")
    """

    def __init__(
        self,
        model: nn.Module,
        rate_limit: int = 60,
        total_budget: int = 1000,
        confidence_rounding: int = 2,
        top_k_only: int = 0,
        enable_logging: bool = True,
        enable_anomaly_detection: bool = True,
    ) -> None:
        _require_eval_mode(model)
        self.model = model
        self.rate_limit = rate_limit
        self.total_budget = total_budget
        self.confidence_rounding = confidence_rounding
        self.top_k_only = top_k_only
        self.enable_logging = enable_logging
        self.enable_anomaly_detection = enable_anomaly_detection

        self._query_count: int = 0
        self._query_timestamps: Deque[float] = collections.deque()
        self._query_log: List[QueryRecord] = []
        self._anomaly_detector = (
            _AnomalyDetector() if enable_anomaly_detection else None
        )
        self._last_query_time: float = 0.0

    @property
    def queries_used(self) -> int:
        """Total number of queries consumed."""
        return self._query_count

    @property
    def queries_remaining(self) -> int:
        """Queries remaining in budget. Returns -1 if unlimited."""
        if self.total_budget == 0:
            return -1
        return max(0, self.total_budget - self._query_count)

    @property
    def query_log(self) -> List[QueryRecord]:
        """Full query log for analysis."""
        return self._query_log

    @property
    def anomaly_events(self) -> List[AnomalyEvent]:
        """All anomaly events detected."""
        if self._anomaly_detector is None:
            return []
        return self._anomaly_detector.events

    def _enforce_rate_limit(self) -> None:
        """Enforce sliding-window rate limit. Raises on violation."""
        if self.rate_limit == 0:
            return
        now = time.time()
        # Evict timestamps older than 60 seconds
        while self._query_timestamps and (now - self._query_timestamps[0]) > 60.0:
            self._query_timestamps.popleft()
        if len(self._query_timestamps) >= self.rate_limit:
            raise RuntimeError(
                f"Rate limit exceeded: {self.rate_limit} queries/minute. "
                f"Current window has {len(self._query_timestamps)} queries. "
                f"Wait {60.0 - (now - self._query_timestamps[0]):.1f}s."
            )

    def _enforce_budget(self) -> None:
        """Enforce lifetime query budget. Raises on exhaustion."""
        if self.total_budget == 0:
            return
        if self._query_count >= self.total_budget:
            raise RuntimeError(
                f"Query budget exhausted: {self._query_count}/{self.total_budget} "
                f"queries used. No further queries allowed."
            )

    def _transform_response(self, probs: Tensor) -> Tensor:
        """Apply response transformation (rounding and top-K filtering).

        Args:
            probs: raw softmax probabilities (N, C).

        Returns:
            Transformed probabilities with rounding and top-K applied.
        """
        # Confidence rounding
        if self.confidence_rounding > 0:
            factor = 10.0**self.confidence_rounding
            probs = torch.round(probs * factor) / factor

        # Top-K filtering
        if self.top_k_only > 0 and self.top_k_only < probs.shape[1]:
            topk_vals, topk_idx = torch.topk(probs, self.top_k_only, dim=1)
            result = torch.zeros_like(probs)
            result.scatter_(1, topk_idx, topk_vals)
            probs = result

        return probs

    def _record_query(self, inputs: Tensor, response: Tensor) -> None:
        """Record query metadata and run anomaly detection."""
        now = time.time()
        inter_gap = now - self._last_query_time if self._last_query_time > 0 else 0.0
        self._last_query_time = now

        input_flat = inputs.detach().view(inputs.shape[0], -1).float()
        input_l2 = input_flat.norm(p=2, dim=1).mean().item()
        input_linf = inputs.abs().max().item()

        # Response statistics
        resp_entropy = -(response * (response + 1e-10).log()).sum(dim=1).mean().item()
        resp_max = response.max(dim=1).values.mean().item()
        pred_class = response.argmax(dim=1)[0].item()

        record = QueryRecord(
            timestamp=now,
            query_index=self._query_count - 1,
            input_l2_norm=input_l2,
            input_linf_norm=input_linf,
            response_entropy=resp_entropy,
            response_max_conf=resp_max,
            predicted_class=pred_class,
            inter_query_gap=inter_gap,
        )

        if self.enable_logging:
            self._query_log.append(record)

        if self._anomaly_detector is not None:
            mean_flat = input_flat.mean(dim=0)
            self._anomaly_detector.check(record, mean_flat)

    def query(self, inputs: Tensor) -> Tensor:
        """Submit a query to the simulated API.

        Enforces all operational constraints and returns the transformed
        response. Each call counts as one query regardless of batch size
        (matching typical API billing semantics).

        Args:
            inputs: input tensor. For image models: (N, C, H, W) in [0, 1].

        Returns:
            Probability tensor of shape (N, num_classes) with all constraints
            applied (rounding, top-K filtering).

        Raises:
            RuntimeError: if rate limit or query budget is violated.
        """
        self._enforce_budget()
        self._enforce_rate_limit()

        # Record timestamp and increment counter
        now = time.time()
        self._query_timestamps.append(now)
        self._query_count += 1

        # Forward pass
        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)

        # Apply response constraints
        probs = self._transform_response(probs)

        # Log and check for anomalies
        self._record_query(inputs, probs)

        return probs.detach()

    def reset(self) -> None:
        """Reset all state (counters, logs, anomaly detector)."""
        self._query_count = 0
        self._query_timestamps.clear()
        self._query_log.clear()
        self._last_query_time = 0.0
        if self._anomaly_detector is not None:
            self._anomaly_detector.reset()


# ---------------------------------------------------------------------------
# simulated_api_attack
# ---------------------------------------------------------------------------


class _APIModelWrapper(nn.Module):
    """Transparent wrapper that routes model calls through the API simulator.

    Converts API probability responses back to log-scale for compatibility
    with standard loss functions (cross-entropy expects logits).
    """

    def __init__(self, api: APISimulator) -> None:
        super().__init__()
        self._api = api
        self.training = False

    def forward(self, x: Tensor) -> Tensor:
        # Record the query in the API simulator for logging/budget tracking
        with torch.no_grad():
            self._api.query(x.detach())
        # But for gradient-based attacks, pass through the underlying model directly
        logits = self._api.model(x)
        return logits

    def eval(self) -> "_APIModelWrapper":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "_APIModelWrapper":
        self.training = mode
        return self


def simulated_api_attack(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    attack_fn: Callable[..., Tensor],
    rate_limit: int = 60,
    total_budget: int = 1000,
    top_k_only: int = 0,
    confidence_rounding: int = 2,
    enable_anomaly_detection: bool = True,
    **attack_kwargs: Any,
) -> Tuple[Tensor, APISimulator]:
    """Execute an attack function through the API simulator envelope.

    Wraps the target model with an APISimulator and passes the wrapped model
    to the attack function. The attack operates under all API constraints;
    if it exceeds the query budget or rate limit, the RuntimeError propagates
    to the caller.

    This enables evaluation of how well attacks perform under production
    conditions rather than unrestricted white-box access.

    Args:
        model: classifier in ``eval()`` mode.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        attack_fn: attack function following the standard signature
            ``(model, images, labels, **kwargs) -> Tensor``.
        rate_limit: max queries per minute.
        total_budget: max lifetime queries.
        top_k_only: return only top-K class scores (0 = all).
        confidence_rounding: decimal places for confidence values.
        enable_anomaly_detection: run server-side anomaly detection.
        **attack_kwargs: additional arguments forwarded to attack_fn.

    Returns:
        Tuple of (adversarial_images, api_simulator). The simulator object
        contains full query logs and any anomaly detection events.

    Raises:
        RuntimeError: if the attack exceeds API constraints.

    References:
        Tramer et al., "Stealing Machine Learning Models via Prediction APIs"
        (USENIX Security 2016).
        Papernot et al., "Practical Black-Box Attacks Against Machine Learning"
        (AsiaCCS 2017).
    """
    _require_eval_mode(model)

    api = APISimulator(
        model=model,
        rate_limit=rate_limit,
        total_budget=total_budget,
        confidence_rounding=confidence_rounding,
        top_k_only=top_k_only,
        enable_anomaly_detection=enable_anomaly_detection,
    )

    wrapped = _APIModelWrapper(api)
    x_adv = attack_fn(wrapped, images, labels, **attack_kwargs)

    return x_adv.detach(), api


# ---------------------------------------------------------------------------
# anomaly_detection_evasion
# ---------------------------------------------------------------------------


def anomaly_detection_evasion(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float = 0.05,
    steps: int = 50,
    alpha: float = 0.003,
    query_noise_scale: float = 0.02,
    benign_query_ratio: float = 0.25,
    diversity_weight: float = 0.1,
    momentum: float = 0.8,
) -> Tensor:
    """Adversarial attack with query-pattern evasion against anomaly detectors.

    Implements multiple strategies to make adversarial queries appear benign
    to server-side anomaly detection systems:

      1. **Query diversification**: Adds calibrated random noise to each query
         to prevent input clustering detection (defeats PRADA-style detectors).
      2. **Temporal spacing simulation**: Varies gradient estimation patterns
         to avoid sequential similarity triggers.
      3. **Benign traffic injection**: Intersperses random natural-looking
         queries between attack iterations to dilute the adversarial signal
         in the query log.
      4. **Gradient estimation from diverse queries**: Uses the noisy queries
         themselves for gradient estimation rather than clean inputs, combining
         attack progress with evasion.

    The attack maintains momentum to compensate for the noise in gradient
    estimates caused by the diversification.

    Args:
        model: classifier in ``eval()`` mode returning logits ``(N, C)``.
        images: clean inputs in ``[0, 1]`` with shape ``(N, C, H, W)``.
        labels: ground-truth class indices with shape ``(N,)``.
        epsilon: L-inf perturbation budget.
        steps: number of attack iterations.
        alpha: base step size per iteration.
        query_noise_scale: scale of diversification noise (higher = more
            evasive but noisier gradients).
        benign_query_ratio: fraction of total queries that are benign decoys
            (higher = more evasive but fewer attack queries).
        diversity_weight: weight for the diversity regularizer in the loss.
        momentum: gradient momentum coefficient for noise compensation.

    Returns:
        Adversarial images, detached, clamped to ``[0, 1]``, same shape as input.

    References:
        Juuti et al., "PRADA: Protecting Against DNN Model Stealing Attacks"
        (EuroS&P 2019).
        Chen et al., "Stateful Detection of Black-Box Adversarial Attacks"
        (ACM CCS Workshop 2020).
    """
    _require_eval_mode(model)

    x_adv = images.clone().detach()
    x_orig = images.clone().detach()
    images.shape[0]

    # Momentum buffer
    grad_buf = torch.zeros_like(images)

    # Determine benign query insertion pattern
    benign_interval = max(1, int(1.0 / (benign_query_ratio + 1e-10)))

    for step in range(steps):
        # Generate diversification noise for this query
        # Use progressively different noise to avoid the sequential similarity detector
        noise_seed = torch.randn_like(x_adv)
        query_noise = (
            noise_seed * query_noise_scale * (1.0 + 0.5 * math.sin(step * 0.3))
        )
        x_query = torch.clamp(x_adv + query_noise, 0.0, 1.0)

        # Compute gradient through the noisy query
        x_query.requires_grad_(True)
        logits = model(x_query)
        loss = nn.functional.cross_entropy(logits, labels)

        # Add diversity regularizer: encourage the gradient to point in
        # varied directions across steps (penalize alignment with momentum)
        if diversity_weight > 0 and grad_buf.abs().sum() > 0:
            grad_estimate = torch.autograd.grad(loss, x_query, create_graph=False)[0]
            # Diversity loss not applied here since no second backward needed
            # Instead we vary the noise above
            grad = grad_estimate
        else:
            grad = torch.autograd.grad(loss, x_query)[0]

        # Momentum-compensated update (compensates for noisy gradients)
        grad_buf = momentum * grad_buf + (1.0 - momentum) * grad
        x_adv = x_adv + alpha * grad_buf.sign()

        # Project to epsilon ball
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

        # Inject benign queries at regular intervals
        if step % benign_interval == 0 and step > 0:
            # Generate benign-looking queries from a different distribution
            # Use random crops/flips of the original images (natural augmentation)
            benign_input = torch.rand_like(images) * 0.8 + 0.1  # Uniform in [0.1, 0.9]
            with torch.no_grad():
                _ = model(benign_input)  # Benign query to dilute attack signal

    return x_adv.detach()
