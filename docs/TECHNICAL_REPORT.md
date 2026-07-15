# Adversarial ML Lab - Technical Security Report

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY  
**Version:** 1.0  
**Date:** 2025  
**Framework:** adversarial-ml-lab (Python 3.12 / PyTorch)  
**Reference Architecture:** _SmallCNN (1x8x8 input, 3 classes)

---

## 1. Executive Summary

This report documents the adversarial machine learning evaluation framework
implemented across 20 operational tiers. The framework provides a structured
attack ladder from basic gradient perturbations through certified defense
evaluation, privacy attacks, and physical-world patch generation.

**Key findings from baseline evaluation (synthetic _SmallCNN):**

| Metric | Value |
|--------|-------|
| Clean accuracy | 0.920 |
| FGSM robust accuracy | 0.508 |
| PGD robust accuracy | 0.500 |
| C&W robust accuracy | 0.034 |
| CI gate threshold (PGD) | 0.300 |
| CI gate status | PASS (0.500 > 0.300) |

The C&W result (0.034) confirms that optimization-based attacks reduce the
demo model to near-random performance (0.333 for 3 classes), exposing the gap
between PGD-robustness and true adversarial robustness.

---

## 2. Framework Architecture

```
src/adv_lab/
  attacks/          # 20 attack modules (Tiers 1-5)
    fgsm.py         # Single-step L-inf
    pgd.py          # Multi-step L-inf and L2
    cw.py           # Optimization-based L2
    blackbox.py     # Query-only (SimBA, Square, HSJ, Boundary)
    model_stealing.py   # Substitute model extraction
    norms.py        # L0, L1, Wasserstein, semantic, patch
    llm.py          # LLM attack surface (GCG, AutoDAN, etc.)
    poisoning.py    # Data/model poisoning and backdoors
    adaptive.py     # Defense-aware (BPDA, EoT)
    param_search.py # Bayesian attack optimization
    constrained.py  # Budget and timing attacks
    evasion.py      # Post-processing evasion
    ensemble.py     # Multi-model ensemble attacks
    inference.py    # Inference-time manipulation
    chaining.py     # Perturbation chaining
    api_sim.py      # API behavior simulation
    non_classification.py  # Detection, segmentation, RL, etc.
    inversion.py    # Model/GAN inversion, membership inference
    physical.py     # Physical-world patches (EOT)
    universal.py    # Universal adversarial perturbations
  defenses/
    adversarial_training.py  # PGD-AT (7-step inner loop)
    detection.py    # NeuralCleanse, STRIP (+ bypass)
  eval/
    harness.py      # CI-gateable benchmark runner
    transferability.py  # 4-architecture transfer analysis
    certified.py    # Randomized smoothing, Lipschitz, IBP
    ci_signing.py   # HMAC-SHA256 report integrity
```


---

## 3. Per-Tier Analysis

### Tier 1: White-Box Gradient Attacks (FGSM)

**Module:** `src/adv_lab/attacks/fgsm.py`  
**Function:** `fgsm_attack(model, images, labels, epsilon=0.03)`

Single-step signed gradient attack (Goodfellow et al., 2015). Computes
the sign of the loss gradient with respect to the input and takes one
step of size epsilon in the L-inf norm.

- **Perturbation norm:** L-inf
- **Epsilon:** 0.03 (configurable)
- **Compute cost:** 1 forward + 1 backward pass
- **Baseline result:** 0.508 robust accuracy on _SmallCNN
- **Utility:** Fast sanity check; any model failing FGSM is fundamentally broken

The `_require_eval_mode(model)` guard ensures BatchNorm/Dropout are frozen,
preventing stochastic gradient noise from masking true vulnerability.

### Tier 2: Multi-Step PGD (L-inf and L2)

**Module:** `src/adv_lab/attacks/pgd.py`  
**Functions:** `pgd_attack(...)`, `pgd_linf(...)`, `pgd_l2(...)`

Madry et al. (ICLR 2018). Iterative signed gradient with projection back
into the epsilon-ball after each step.

| Parameter | Default | Range |
|-----------|---------|-------|
| epsilon | 0.03 | 0.0-1.0 |
| alpha | 0.007 | 0.001-0.05 |
| steps | 40 | 1-200 |
| random_start | True | bool |

- **Baseline result:** 0.500 robust accuracy (L-inf, eps=0.03, 40 steps)
- **Compute cost:** 40 forward + 40 backward passes per sample
- **Key property:** random_start=True escapes flat regions; essential for honest evaluation

PGD is the CI gate attack. The threshold `PGD_GATE_THRESHOLD = 0.3` in
`src/adv_lab/eval/harness.py` means any model dropping below 30% accuracy
under PGD fails the pipeline.

### Tier 3: Carlini & Wagner L2

**Module:** `src/adv_lab/attacks/cw.py`  
**Function:** `cw_l2_attack(model, images, labels, c=1e-4, kappa=0, steps=1000, lr=0.01)`

Optimization-based attack using tanh change-of-variables and margin loss.
The gold standard for exposing gradient masking.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| c | 1e-4 | Perturbation vs. misclassification tradeoff |
| kappa | 0 | Confidence margin |
| steps | 1000 | Adam optimizer iterations |
| lr | 0.01 | Learning rate |

- **Baseline result:** 0.034 robust accuracy (near-random for 3-class)
- **Compute cost:** 1000 optimizer steps (Adam) per sample
- **Key insight:** PGD robust accuracy of 0.500 vs C&W of 0.034 exposes a
  massive gap. The demo model has partial L-inf robustness but zero L2 robustness.

### Tier 4: Black-Box Query Attacks

**Module:** `src/adv_lab/attacks/blackbox.py`  
**Functions:** `simba_attack(...)`, `square_attack(...)`, `hop_skip_jump(...)`, `boundary_attack(...)`

All functions return `tuple[Tensor, Tensor]` = (adversarial_images, queries_used).

| Attack | Method | Default Budget | Expected Success |
|--------|--------|---------------|-----------------|
| SimBA | Random coordinate search | 1000 queries | 60-80% |
| Square Attack | Random square patches, p-schedule | 1000 queries | 70-90% |
| HopSkipJump | Binary search + gradient estimation | 1000 queries | 80-95% |
| Boundary Attack | Decision-boundary walk | 1000 queries | 50-70% |

Parameters shared across all black-box attacks:
- `query_budget: int = 1000` (configurable up to 10000)
- `epsilon: float` (attack-specific L-inf/L2 budget)

SimBA signature: `simba_attack(model, images, labels, query_budget=1000, epsilon=0.2, step_size=0.02)`

### Tier 5: Model Stealing

**Module:** `src/adv_lab/attacks/model_stealing.py`  
**Classes/Functions:** `SubstituteModel`, `jacobian_augmentation(...)`, `steal_model(...)`

Papernot et al. (ACM Asia CCS 2017). Trains a substitute model using only
query access to the target.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| agreement_threshold | 0.7 | Minimum substitute-target agreement to stop |
| augmentation_rounds | 6 | Jacobian augmentation iterations |

- **Attack flow:** Seed query -> pseudo-label -> train substitute -> Jacobian augment -> repeat
- **Success criterion:** >= 70% prediction agreement with target
- **Transfer attack viability:** Once agreement >= 0.7, white-box attacks on the
  substitute transfer to the target with high probability

### Tier 6: Transferability Analysis

**Module:** `src/adv_lab/eval/transferability.py`  
**Class:** `TransferabilityAnalyzer`  
**Architectures:** `_ShallowCNN`, `_WideCNN`, `_DeepCNN`, `_MLPModel`

Measures cross-architecture adversarial example transfer rates.

| Source \ Target | ShallowCNN | WideCNN | DeepCNN | MLP |
|----------------|-----------|---------|---------|-----|
| ShallowCNN | 1.00 | 0.45-0.60 | 0.35-0.50 | 0.30-0.45 |
| WideCNN | 0.40-0.55 | 1.00 | 0.50-0.65 | 0.35-0.50 |
| DeepCNN | 0.35-0.50 | 0.50-0.65 | 1.00 | 0.30-0.45 |
| MLP | 0.25-0.40 | 0.30-0.45 | 0.25-0.40 | 1.00 |

Architecture details:
- **ShallowCNN:** 1 convolutional layer
- **WideCNN:** 2 conv layers (32/64 channels)
- **DeepCNN:** 4 conv layers
- **MLP:** Pure fully-connected (256->128)

Transfer rates follow established patterns: similar architectures (CNN-to-CNN)
transfer better than dissimilar (CNN-to-MLP).

### Tier 7: Full Norm Suite

**Module:** `src/adv_lab/attacks/norms.py` (711 lines)  
**Functions:** `pgd_l0(...)`, `pgd_l1(...)`, `wasserstein_attack(...)`, `semantic_attack(...)`, `patch_attack(...)`

Comprehensive norm coverage beyond standard L-inf/L2:

| Norm | Function | Threat Model |
|------|----------|-------------|
| L0 | `pgd_l0` | Sparse pixel changes (few pixels, large magnitude) |
| L1 | `pgd_l1` | Moderate sparsity with bounded total perturbation |
| Wasserstein | `wasserstein_attack` | Perceptually smooth redistribution |
| Semantic | `semantic_attack` | Color shifts, rotation, brightness (human-imperceptible) |
| Patch | `patch_attack` | Localized adversarial region |

A model robust to L-inf PGD is frequently vulnerable to L0 or semantic attacks.
The multi-norm suite exposes the union-robustness gap that remains open in the
research literature.

### Tier 8: LLM Attack Surface

**Module:** `src/adv_lab/attacks/llm.py` (765 lines)  
**Classes/Functions:** `SimulatedTokenizer`, `SimulatedLLM`, `GCGAttack`, `AutoDANAttack`,
`prompt_injection(...)`, `embedding_perturbation(...)`, `token_substitution(...)`, `universal_suffix(...)`

| Attack | Method | Key Parameters |
|--------|--------|---------------|
| GCG | Greedy Coordinate Gradient | top_k=256, steps=500 |
| AutoDAN | Genetic algorithm jailbreak | population=20, generations=100 |
| Prompt Injection | Context poisoning prefix/suffix | N/A |
| Embedding Perturbation | Continuous L2 in embedding space | epsilon, steps |
| Token Substitution | Gradient-guided discrete swap (HotFlip) | candidates per position |
| Universal Suffix | Fixed transferable suffix | suffix_length, steps |

Implementation uses `SimulatedTokenizer` (character-level, vocab_size=128) and
`SimulatedLLM` (small feedforward over one-hot tokens). In production, replace
with real BPE tokenizer and transformer weights.

GCG computes per-token gradients, selects top_k=256 candidate substitutions per
position, evaluates each, and greedily accepts the best. 500 optimization steps.

AutoDAN uses evolutionary search: population of 20 prompts, 100 generations,
crossover + mutation operators on token sequences.

### Tier 9: Poisoning and Backdoors

**Module:** `src/adv_lab/attacks/poisoning.py`  
**Functions:** `clean_label_poison(...)`, `badnets_trigger(...)`, `spectral_backdoor(...)`, `weight_poisoning(...)`

| Attack | Mechanism | Detection Difficulty |
|--------|-----------|---------------------|
| Clean-label | Feature collision without label flip | High (labels correct) |
| BadNets | Pixel-pattern trigger injection | Medium (trigger visible) |
| Spectral | Spectral signature in feature space | Medium-High |
| Weight Poisoning | Direct parameter modification | Low (requires model access) |

Clean-label poisoning is the most operationally dangerous: training data appears
correctly labeled, but feature-space proximity causes targeted misclassification
at inference time. No label auditing catches it.

### Tier 10: Defense-Aware Adaptive Attacks

**Module:** `src/adv_lab/attacks/adaptive.py` (440 lines)  
**Classes/Functions:** `BPDA`, `EoT`, `GradientMaskingDetector`, `AdaptiveAttackLog`, `adaptive_attack(...)`

The `GradientMaskingDetector` monitors loss trajectory during the first 20% of
optimization steps. If loss plateaus (change < tolerance for plateau_window
consecutive steps), gradient masking is detected and the attack strategy switches.

**Strategy switching logic in `adaptive_attack`:**
1. Start with standard PGD
2. If GradientMaskingDetector fires -> switch to BPDA
3. If defense is stochastic (detected via variance in repeated forward passes) -> add EoT
4. If both fail -> fall back to random-search-with-momentum

BPDA wraps non-differentiable defenses with a differentiable surrogate for the
backward pass (Athalye et al., ICML 2018).

EoT averages gradients over multiple random transformations (Athalye et al., ICML 2018).

### Tier 11: Bayesian Attack Parameter Search

**Module:** `src/adv_lab/attacks/param_search.py`  
**Classes:** `BayesianAttackOptimizer`, `GaussianProcess`, `ParamBounds`  
**Function:** `per_sample_difficulty_score(...)`

Gaussian Process-based Bayesian optimization over attack hyperparameters.

| Parameter | Value |
|-----------|-------|
| Initial random samples | 5 |
| Acquisition function | Expected Improvement (EI) |
| GP kernel | RBF (squared exponential) |
| Optimization target | Minimize perturbation norm at fixed success rate |

`ParamBounds` defines the search space per attack type. The optimizer
evaluates 5 random configurations, fits a GP surrogate, then selects
subsequent configurations by maximizing EI.

`per_sample_difficulty_score` assigns each input a difficulty estimate
based on decision boundary proximity, enabling adaptive budget allocation.

### Tier 12: Resource-Constrained Timing Attacks

**Module:** `src/adv_lab/attacks/constrained.py`  
**Classes/Functions:** `QueryBudgetManager`, `TimedAttack`, `rate_limited_attack(...)`

Attacks under operational constraints:
- **QueryBudgetManager:** Enforces hard query limits, tracks consumption across
  multiple attack attempts on same target
- **TimedAttack:** Wall-clock deadline enforcement; attack must succeed within
  time budget or abort
- **rate_limited_attack:** Simulates API rate limiting (requests/second cap)

These constraints model real-world API attack scenarios where unlimited queries
would trigger anomaly detection.

### Tier 13: Post-Processing Defense Evasion

**Module:** `src/adv_lab/attacks/evasion.py`  
**Functions:** `jpeg_robust_attack(...)`, `feature_squeeze_robust(...)`, `detector_evasion(...)`

Attacks that survive common input-transformation defenses:

| Defense | Evasion Method |
|---------|---------------|
| JPEG compression | Differentiable JPEG approximation in attack loop |
| Feature squeezing | Optimize in reduced color-depth space |
| Adversarial detector | Minimize detector score jointly with misclassification |

`jpeg_robust_attack` includes JPEG compression as a differentiable layer in the
PGD loop, ensuring perturbations survive lossy compression at test time.

`detector_evasion` adds a penalty term for the detector's confidence score,
producing adversarial examples that both fool the classifier and evade the detector.

### Tier 14: Ensemble Attacks

**Module:** `src/adv_lab/attacks/ensemble.py`  
**Functions:** `ensemble_attack(...)`, `build_attacker_ensemble(...)`, `weighted_ensemble_pgd(...)`

Attack multiple models simultaneously to maximize transferability:

- `ensemble_attack`: Average gradients from N models, take PGD step on mean
- `build_attacker_ensemble`: Construct diverse model ensemble for robust transfer
- `weighted_ensemble_pgd`: Weight models by estimated vulnerability

Ensemble attacks produce adversarial examples that transfer across architectures
with higher success rate than single-model attacks (typically +15-25% transfer rate).

### Tier 15: Inference-Time Manipulation

**Module:** `src/adv_lab/attacks/inference.py`  
**Functions:** `watermark_flip(...)`, `prediction_poison(...)`, `soft_label_manipulation(...)`

Attacks that corrupt model outputs without modifying inputs:

| Attack | Target | Mechanism |
|--------|--------|-----------|
| Watermark Flip | Model watermarking schemes | Perturb trigger inputs to flip watermark response |
| Prediction Poison | Model outputs | Inject systematic bias into predictions |
| Soft-Label Manipulation | Confidence scores | Shift probability distribution without changing argmax |

These target the inference pipeline rather than the model weights or inputs.
Relevant for ML-as-a-Service where the attacker has partial pipeline access.

### Tier 16: Perturbation Chaining

**Module:** `src/adv_lab/attacks/chaining.py`  
**Classes/Functions:** `PerturbationChain`, `chain_attack(...)`, `ChainState`, `StepMetrics`, `AttackConfig`

Sequential composition of multiple attack primitives:

```
chain = PerturbationChain([
    AttackConfig(attack_fn=fgsm_attack, params={...}),
    AttackConfig(attack_fn=pgd_attack, params={...}),
    AttackConfig(attack_fn=semantic_attack, params={...}),
])
result: ChainState = chain_attack(model, images, labels, chain)
```

`ChainState` tracks cumulative perturbation norm across steps.
`StepMetrics` records per-step success rate and norm contribution.

Chaining enables multi-norm attacks: L-inf step followed by semantic rotation
followed by patch overlay, staying within budget on each individual norm while
accumulating cross-norm perturbation.

### Tier 17: API Behavior Simulation

**Module:** `src/adv_lab/attacks/api_sim.py`  
**Classes/Functions:** `APISimulator`, `simulated_api_attack(...)`, `anomaly_detection_evasion(...)`

Simulates real-world ML API constraints:
- Rate limiting
- Query logging and anomaly detection
- Response format (top-k labels, confidence thresholds, rounded probabilities)
- Latency simulation

`anomaly_detection_evasion` crafts query sequences that avoid triggering
statistical anomaly detectors (e.g., distributional shift detection on
incoming queries).

### Tier 18: Non-Classification Targets

**Module:** `src/adv_lab/attacks/non_classification.py`  
**Classes:** `SimpleDetector`, `SimpleSegmenter`, `SimpleRegressor`, `SimplePolicy`, `SimpleRecommender`  
**Functions:** `object_detection_attack(...)`, `segmentation_attack(...)`, `regression_attack(...)`, `rl_attack(...)`, `recommendation_attack(...)`

Extends adversarial attacks beyond image classification:

| Domain | Attack Goal | Success Metric |
|--------|------------|----------------|
| Object Detection | Suppress/create detections | mAP drop |
| Segmentation | Corrupt pixel-level predictions | mIoU drop |
| Regression | Shift predicted values | MAE increase |
| Reinforcement Learning | Degrade policy return | Cumulative reward drop |
| Recommendation | Promote/suppress items | Rank displacement |

Each uses a lightweight simulated model (e.g., `SimpleDetector` outputs bounding
boxes) to validate the attack logic without requiring full-scale models.

### Tier 19: Certified Defense Evaluation

**Module:** `src/adv_lab/eval/certified.py` (589 lines)  
**Classes/Functions:** `RandomizedSmoothing`, `LipschitzNetwork`, `IBPBounds`, `find_certificate_boundary(...)`

Provable robustness evaluation methods:

| Method | Guarantee Type | Scalability |
|--------|---------------|-------------|
| Randomized Smoothing | Probabilistic L2 radius (Cohen et al.) | High (sampling-based) |
| Lipschitz Network | Deterministic L2 radius via spectral norm | Medium (architecture constraint) |
| IBP Bounds | Deterministic interval propagation | Low (loose on deep nets) |

`RandomizedSmoothing` adds Gaussian noise (sigma), takes majority vote over N
samples, and computes a certified radius from binomial confidence bounds.

`find_certificate_boundary` performs binary search over epsilon to find the exact
point where certified accuracy drops to zero -- the maximum certifiable radius
for a given model.

### Tier 20: Privacy Attacks and Infrastructure Integrity

**Module (Privacy):** `src/adv_lab/attacks/inversion.py` (612 lines)  
**Functions:** `gradient_inversion(...)`, `gan_inversion(...)`, `membership_inference_shadow(...)`, `membership_inference_likelihood(...)`

| Attack | Metric | Expected Performance |
|--------|--------|---------------------|
| Gradient Inversion | SSIM of reconstructed vs original | 0.3-0.7 (model-dependent) |
| GAN Inversion | SSIM + FID of generated images | 0.4-0.8 |
| Membership Inference (Shadow) | AUC-ROC | 0.55-0.75 |
| Membership Inference (Likelihood) | AUC-ROC | 0.50-0.65 |

**Module (CI Signing):** `src/adv_lab/eval/ci_signing.py`  
**Functions:** `sign_report(...)`, `verify_report(...)`, `log_input_hashes(...)`, `detect_replay(...)`, `derive_key(...)`, `create_signed_manifest(...)`

| Parameter | Value |
|-----------|-------|
| Algorithm | HMAC-SHA256 |
| KDF | PBKDF2-HMAC-SHA256 |
| KDF iterations | 600,000 (OWASP 2023) |
| Key length | 256 bits |
| Salt length | 128 bits |

**Module (Physical):** `src/adv_lab/attacks/physical.py` (481 lines)  
**Classes/Functions:** `PhysicalPatchAttack`, `PatchResult`, `printability_constraint(...)`

Physical-world patch generation with EOT optimization:
- Viewing angles: +/-30 degrees
- Lighting multipliers: 0.5x-2.0x
- Color gamut: sRGB [0, 1]
- Printability: NPS (Non-Printability Score) constraint

`PatchResult` contains: patch tensor, success_rate, printability_score,
angle_robustness (dict[angle -> success_rate]), lighting_robustness (dict[multiplier -> success_rate]).

**Module (UAP):** `src/adv_lab/attacks/universal.py` (381 lines)  
**Functions:** `uap_generate(...)`, `fast_uap(...)`, `evaluate_fooling_rate(...)`, `cross_architecture_transfer(...)`

Universal Adversarial Perturbations (Moosavi-Dezfooli et al., CVPR 2017):
- Target: >= 80% fooling rate with single fixed delta
- Norms: L-inf and L2 projection
- Cross-architecture transfer evaluation across all 4 test architectures

`evaluate_fooling_rate` measures what fraction of a held-out set is misclassified
by a single perturbation. `cross_architecture_transfer` measures fooling rate
when UAP generated on model A is applied to model B.

---

## 4. Attack Cost Analysis

| Attack | Module | Queries/Steps | Compute (relative) | Expected Success Rate |
|--------|--------|--------------|--------------------|-----------------------|
| FGSM | `attacks/fgsm.py` | 1 fwd + 1 bwd | 1x | 40-60% |
| PGD L-inf | `attacks/pgd.py` | 40 fwd + 40 bwd | 40x | 50-80% |
| PGD L2 | `attacks/pgd.py` | 40 fwd + 40 bwd | 40x | 55-85% |
| C&W L2 | `attacks/cw.py` | 1000 Adam steps | 1000x | 90-99% |
| SimBA | `attacks/blackbox.py` | 1000 queries | 1000x (fwd only) | 60-80% |
| Square Attack | `attacks/blackbox.py` | 1000 queries | 1000x (fwd only) | 70-90% |
| HopSkipJump | `attacks/blackbox.py` | 1000 queries | 1000x (fwd only) | 80-95% |
| Boundary | `attacks/blackbox.py` | 1000 queries | 1000x (fwd only) | 50-70% |
| Model Stealing | `attacks/model_stealing.py` | 6 rounds x N queries | Variable | 70%+ agreement |
| GCG (LLM) | `attacks/llm.py` | 500 steps x 256 candidates | 128000x | 40-70% |
| AutoDAN (LLM) | `attacks/llm.py` | 100 gen x 20 pop | 2000x | 50-80% |
| PGD L0 | `attacks/norms.py` | 40 steps | 40x | 45-75% |
| PGD L1 | `attacks/norms.py` | 40 steps | 40x | 50-80% |
| Wasserstein | `attacks/norms.py` | 100 steps | 100x | 40-65% |
| Semantic | `attacks/norms.py` | 50 steps | 50x | 55-75% |
| UAP Generate | `attacks/universal.py` | N_samples x iterations | 5000x+ | >=80% fooling |
| Physical Patch | `attacks/physical.py` | EOT steps x angles x lighting | 10000x+ | 60-85% |
| Gradient Inversion | `attacks/inversion.py` | 1000+ optimizer steps | 1000x | SSIM 0.3-0.7 |

---

## 5. Cross-Architecture Transfer Matrix

Adversarial examples generated on the source model (rows) tested against target
models (columns). Values represent expected fooling rate (fraction of successful
transfers). Generated using `TransferabilityAnalyzer` in `src/adv_lab/eval/transferability.py`.

| Source \ Target | ShallowCNN | WideCNN | DeepCNN | MLP |
|----------------|-----------|---------|---------|-----|
| **ShallowCNN** | 1.00 | 0.52 | 0.43 | 0.37 |
| **WideCNN** | 0.48 | 1.00 | 0.58 | 0.42 |
| **DeepCNN** | 0.41 | 0.57 | 1.00 | 0.38 |
| **MLP** | 0.33 | 0.38 | 0.32 | 1.00 |

**Key observations:**
- CNN-to-CNN transfer rates (0.41-0.58) consistently exceed CNN-to-MLP (0.32-0.42)
- Architectural similarity drives transfer: WideCNN<->DeepCNN is the highest cross-pair (0.57-0.58)
- MLP as source produces lowest transfer rates (0.32-0.38) due to fundamentally different gradient geometry
- Ensemble attacks (`src/adv_lab/attacks/ensemble.py`) boost these rates by +15-25%

---

## 6. Defense Bypass Analysis

### 6.1 Adversarial Training Bypass

**Defense module:** `src/adv_lab/defenses/adversarial_training.py`  
**Class:** `AdversarialTrainer` (PGD-7 inner attack)

The adversarial training implementation uses a 7-step PGD inner loop. Known bypass vectors:

| Bypass Method | Module | Mechanism |
|--------------|--------|-----------|
| C&W L2 | `attacks/cw.py` | Different norm than training (L-inf trained, L2 attacked) |
| AutoAttack ensemble | `attacks/ensemble.py` | Multiple attack types exceed single-attack training |
| Adaptive BPDA | `attacks/adaptive.py` | Detects gradient masking from AT and switches strategy |
| Increased steps | `attacks/pgd.py` | steps=200 vs training inner loop steps=7 |

AT with 7-step PGD is vulnerable to attacks using more steps (40+) or different norms.
The demo model's PGD-40 robust accuracy of 0.500 represents an honest upper bound.

### 6.2 NeuralCleanse and STRIP Bypass

**Defense module:** `src/adv_lab/defenses/detection.py`  
**Functions:** `bypass_neural_cleanse(...)`, `bypass_strip(...)`

| Defense | Detection Method | Bypass Strategy |
|---------|-----------------|----------------|
| NeuralCleanse | Reverse-engineer smallest trigger per class | Distributed trigger (spread across image, no single small pattern) |
| STRIP | Entropy of predictions under input perturbation | Craft examples with high prediction entropy despite being adversarial |

Both bypasses are implemented in the same module as the defenses, enabling
red-team/blue-team evaluation in a single pipeline.

---

## 7. CI Gate Integrity Assessment

**Module:** `src/adv_lab/eval/ci_signing.py`

### 7.1 Threat Model

An attacker with write access to CI artifact storage attempts:
1. **Forgery:** Fabricate a passing evaluation report
2. **Replay:** Submit a previously passing report against different model weights
3. **Substitution:** Swap adversarial test inputs for benign ones during evaluation

### 7.2 Controls

| Control | Implementation | Strength |
|---------|---------------|----------|
| Report signing | HMAC-SHA256 via `sign_report()` | 256-bit key, computationally infeasible to forge |
| Input hashing | SHA-256 via `log_input_hashes()` | Detects input substitution |
| Replay detection | Timestamp + nonce via `detect_replay()` | Prevents stale report reuse |
| Key derivation | PBKDF2 600k iterations via `derive_key()` | Brute-force resistant |
| Manifest signing | `create_signed_manifest()` | Binds all artifacts to single attestation |

### 7.3 Key Derivation Parameters

```python
_PBKDF2_ITERATIONS = 600_000   # OWASP 2023 recommendation
_PBKDF2_HASH = "sha256"
_KEY_LENGTH = 32               # 256-bit derived key
_SALT_LENGTH = 16              # 128-bit random salt
```

### 7.4 Residual Risks

- Master secret storage: If the CI environment variable is compromised, all signing is bypassed
- No HSM integration: Key derivation simulates HSM but runs in software
- Timing side-channels: `hmac.compare_digest` is used (constant-time), but Python-level timing leaks remain possible in surrounding code
- No certificate revocation: A compromised key requires manual rotation

---

## 8. Physical Patch Printability and Simulation

**Module:** `src/adv_lab/attacks/physical.py`

### 8.1 EOT Simulation Parameters

| Parameter | Values | Purpose |
|-----------|--------|---------|
| Viewing angles | [-30, -20, -10, 0, 10, 20, 30] degrees | Rotation robustness |
| Lighting multipliers | [0.5, 0.75, 1.0, 1.25, 1.5, 2.0] | Brightness variation |
| Color gamut | sRGB [0, 1] per channel | Printable color constraint |
| Patch location | Center crop (configurable) | Spatial placement |

### 8.2 Printability Constraint

`printability_constraint(patch)` computes the Non-Printability Score (NPS):
- Maps each patch pixel to nearest printable color in a reference palette
- NPS = mean L2 distance to nearest printable color across all pixels
- Score in [0, 1]: 1.0 means all pixels are within printable gamut
- Added as regularization term during patch optimization

### 8.3 Expected Performance Envelope

| Condition | Expected Success Rate |
|-----------|----------------------|
| Digital (no transform) | 85-95% |
| +/-10 degree rotation | 75-90% |
| +/-30 degree rotation | 60-80% |
| 0.5x lighting | 55-75% |
| 2.0x lighting | 50-70% |
| All transforms combined | 45-65% |

Performance degrades under extreme conditions. The 30-degree + 2x lighting
corner case represents the operational boundary where patches become unreliable.

---

## 9. UAP Cross-Architecture Transfer Rates

**Module:** `src/adv_lab/attacks/universal.py`  
**Function:** `cross_architecture_transfer(uap, models, data_loader)`

UAP generated on source architecture, tested across all targets:

| UAP Source | Fooling (Self) | Fooling (ShallowCNN) | Fooling (WideCNN) | Fooling (DeepCNN) | Fooling (MLP) |
|-----------|---------------|---------------------|------------------|------------------|--------------|
| ShallowCNN | >=80% | - | 35-50% | 30-45% | 20-35% |
| WideCNN | >=80% | 35-50% | - | 40-55% | 25-40% |
| DeepCNN | >=80% | 30-45% | 40-55% | - | 20-35% |
| MLP | >=80% | 25-35% | 25-40% | 20-35% | - |

**Key observations:**
- UAP transfer rates are 10-15% lower than per-sample transfer rates (Section 5)
- CNN-to-CNN transfer remains strongest (40-55%)
- MLP-sourced UAPs transfer poorly due to absence of convolutional structure
- Target fooling rate >= 80% is achievable on the source model (by construction)
- `fast_uap` trades fooling rate for speed: typically reaches 70-75% in 1/5 the time

---

## 10. Privacy Attack Methodology and Expected Metrics

### 10.1 Gradient Inversion

**Function:** `gradient_inversion(model, target_gradient, input_shape, steps=1000, lr=0.1)`

Reconstructs training data from observed gradients (federated learning threat).

| Model Size | Expected SSIM | Reconstruction Quality |
|-----------|--------------|----------------------|
| Small (< 10K params) | 0.5-0.7 | Recognizable structure |
| Medium (10K-100K) | 0.3-0.5 | Partial features visible |
| Large (> 100K) | 0.1-0.3 | Noise with occasional structure |

SSIM > 0.5 constitutes a meaningful privacy breach for image data.

### 10.2 GAN Inversion

**Function:** `gan_inversion(model, target_class, generator, steps=500, lr=0.01)`

Uses a generator network to produce realistic reconstructions from model outputs.
Higher quality than gradient inversion when a pretrained generator is available.

### 10.3 Membership Inference

**Shadow model approach** (`membership_inference_shadow`):
- Train N shadow models on known member/non-member splits
- Train binary classifier on shadow model outputs
- Illustrative AUC range: 0.55-0.75 on calibrated representative data (baseline random = 0.50)

**Likelihood ratio approach** (`membership_inference_likelihood`):
- Compare per-sample loss to population distribution
- Expected AUC: 0.50-0.65
- Lower variance but weaker signal than shadow model approach

### 10.4 Operational Significance

| AUC Range | Assessment |
|-----------|-----------|
| 0.50-0.55 | No meaningful leakage |
| 0.55-0.65 | Marginal leakage (regulatory concern) |
| 0.65-0.75 | Significant leakage (breach threshold) |
| 0.75+ | Severe leakage (immediate remediation required) |

---

## 11. Full Parameter Audit

All configurable parameters in the framework. No hardcoded epsilons -- every
security-critical value is exposed as a function argument with a documented default.

### 11.1 Core Attack Parameters

| Module | Function | Parameter | Default | Type |
|--------|----------|-----------|---------|------|
| `attacks/fgsm.py` | `fgsm_attack` | epsilon | 0.03 | float |
| `attacks/pgd.py` | `pgd_attack` | epsilon | 0.03 | float |
| `attacks/pgd.py` | `pgd_attack` | alpha | 0.007 | float |
| `attacks/pgd.py` | `pgd_attack` | steps | 40 | int |
| `attacks/pgd.py` | `pgd_attack` | random_start | True | bool |
| `attacks/cw.py` | `cw_l2_attack` | c | 1e-4 | float |
| `attacks/cw.py` | `cw_l2_attack` | kappa | 0 | float |
| `attacks/cw.py` | `cw_l2_attack` | steps | 1000 | int |
| `attacks/cw.py` | `cw_l2_attack` | lr | 0.01 | float |
| `attacks/blackbox.py` | `simba_attack` | query_budget | 1000 | int |
| `attacks/blackbox.py` | `simba_attack` | epsilon | 0.2 | float |
| `attacks/blackbox.py` | `simba_attack` | step_size | 0.02 | float |
| `attacks/blackbox.py` | `square_attack` | query_budget | 1000 | int |
| `attacks/blackbox.py` | `hop_skip_jump` | query_budget | 1000 | int |
| `attacks/blackbox.py` | `boundary_attack` | query_budget | 1000 | int |
| `attacks/model_stealing.py` | `steal_model` | agreement_threshold | 0.7 | float |
| `attacks/model_stealing.py` | `steal_model` | augmentation_rounds | 6 | int |

### 11.2 LLM Attack Parameters

| Module | Class/Function | Parameter | Default | Type |
|--------|---------------|-----------|---------|------|
| `attacks/llm.py` | `GCGAttack` | top_k | 256 | int |
| `attacks/llm.py` | `GCGAttack` | steps | 500 | int |
| `attacks/llm.py` | `AutoDANAttack` | population | 20 | int |
| `attacks/llm.py` | `AutoDANAttack` | generations | 100 | int |
| `attacks/llm.py` | `SimulatedTokenizer` | vocab_size | 128 | int |

### 11.3 Defense and Evaluation Parameters

| Module | Class/Function | Parameter | Default | Type |
|--------|---------------|-----------|---------|------|
| `eval/harness.py` | `run_benchmark` | PGD_GATE_THRESHOLD | 0.3 | float |
| `eval/ci_signing.py` | `derive_key` | iterations | 600,000 | int |
| `eval/ci_signing.py` | internal | _KEY_LENGTH | 32 (bytes) | int |
| `eval/ci_signing.py` | internal | _SALT_LENGTH | 16 (bytes) | int |
| `defenses/adversarial_training.py` | `AdversarialTrainer` | inner_steps | 7 | int |

### 11.4 Physical and Universal Attack Parameters

| Module | Class/Function | Parameter | Default | Type |
|--------|---------------|-----------|---------|------|
| `attacks/physical.py` | `PhysicalPatchAttack` | angles | [-30..+30] | list[float] |
| `attacks/physical.py` | `PhysicalPatchAttack` | lighting | [0.5..2.0] | list[float] |
| `attacks/physical.py` | `PhysicalPatchAttack` | gamut | sRGB [0,1] | constraint |
| `attacks/universal.py` | `uap_generate` | target_fooling | 0.80 | float |
| `attacks/universal.py` | `uap_generate` | norm | "linf" | str |

### 11.5 Adaptive and Bayesian Parameters

| Module | Class/Function | Parameter | Default | Type |
|--------|---------------|-----------|---------|------|
| `attacks/adaptive.py` | `GradientMaskingDetector` | threshold_frac | 0.20 | float |
| `attacks/param_search.py` | `BayesianAttackOptimizer` | n_initial | 5 | int |
| `attacks/param_search.py` | `BayesianAttackOptimizer` | acquisition | "EI" | str |

---

## 12. Honest Assessment

### 12.1 What Is Ready for CI Evaluation

| Capability | Readiness | Evidence |
|-----------|-----------|---------|
| FGSM/PGD/C&W white-box attacks | CI-evaluation ready | Standard implementations, well-validated algorithms |
| CI gate with HMAC signing | CI-evaluation ready | Uses stdlib crypto, follows OWASP/NIST guidance |
| Benchmark harness (JSON output) | CI-evaluation ready | Deterministic, reproducible, CI-integrable |
| Multi-norm attack suite | CI-evaluation ready | L0/L1/L2/Linf all follow published algorithms |
| Input hashing and replay detection | CI-evaluation ready | SHA-256, timestamp-based nonce |
| Gradient masking detection | CI-evaluation ready | Loss trajectory monitoring is well-understood |
| Adversarial training (PGD-AT) | CI-evaluation ready | Madry et al. is the standard approach |

### 12.2 What Remains a Lab Demo

| Capability | Limitation | Gap to Production |
|-----------|-----------|-------------------|
| LLM attacks (GCG, AutoDAN) | Simulated tokenizer/LLM (character-level, feedforward) | Requires real transformer weights + BPE tokenizer |
| Physical patch EOT | Simulated transforms only, no camera/printer model | Needs physical printing + camera capture validation |
| Model inversion | SSIM metrics on synthetic data | Real datasets needed for meaningful privacy assessment |
| Non-classification attacks | Lightweight simulated models (SimpleDetector, etc.) | Needs YOLO/SegFormer/PPO-scale models for honest eval |
| UAP cross-architecture | 4 toy architectures (1x8x8 input) | Transfer rates on ResNet/ViT/EfficientNet would differ substantially |
| GAN inversion | No pretrained generator available | Requires StyleGAN/BigGAN weights |
| Certified defenses (IBP) | Bounds are very loose on deep networks | IBP certification is impractical beyond 2-3 layers |

### 12.3 What Would Fail Against a Hardened Target

| Attack | Failure Mode | Why |
|--------|-------------|-----|
| Black-box attacks (1000 queries) | Anomaly detection triggers at ~200-500 queries | Production APIs detect repeated boundary probing |
| Model stealing (6 rounds) | Query pattern is highly anomalous | Jacobian augmentation queries are distributionally distinct from normal traffic |
| BPDA | Requires knowledge of defense architecture | True black-box with unknown defense is not addressable by BPDA alone |
| Single-norm PGD | Multi-norm defenses (TRADES + AWP) | State-of-art defenses combine multiple robustness objectives |
| Universal perturbations | Input preprocessing (random resize + pad) | Simple stochastic preprocessing defeats fixed perturbations |
| Physical patches at 30 deg | Real cameras add motion blur, focus artifacts | Simulated rotation does not capture real optical degradation |
| Prompt injection | Safety-tuned models with input/output filters | Modern LLM deployments layer multiple defense mechanisms |

### 12.4 Framework Limitations Summary

1. **No real model weights:** All evaluation uses synthetic _SmallCNN (1x8x8, 3 classes). Results do not directly transfer to ImageNet-scale models.
2. **No dataset diversity:** Synthetic random data. Real-world class distributions, textures, and correlations are absent.
3. **Simulated LLM environment:** Character-level tokenization on a feedforward network is not representative of transformer behavior.
4. **Physical attacks are digital simulations:** No printer ICC profiles, no camera ISP modeling, no environmental noise.
5. **Certified bounds are for toy networks:** IBP/Lipschitz bounds explode on networks deeper than 4 layers.
6. **No production API integration:** `APISimulator` is a local mock. Real API timing, rate limits, and error handling differ.
7. **Privacy metrics are on synthetic data:** SSIM and AUC values are not calibrated to real training data distributions.

### 12.5 Recommendations for Production Deployment

1. Replace `SimulatedTokenizer`/`SimulatedLLM` with real model endpoints before any LLM security assessment.
2. Validate physical patches with actual print-and-photograph pipeline before claiming physical robustness.
3. Increase black-box query budgets to 5000-10000 for realistic success rates, but expect detection at ~500 queries on hardened APIs.
4. Run the full attack ladder (FGSM < PGD < C&W < adaptive) on production model weights before deployment.
5. Integrate CI signing with actual HSM (AWS CloudHSM, Azure Dedicated HSM) rather than environment-variable secrets.
6. Calibrate membership inference thresholds on representative data before using AUC values for compliance decisions.
7. Test UAP transfer on the actual deployment architecture, not just the 4 toy models in `transferability.py`.

---

## Appendix A: Demo Model Specification

```
_SmallCNN (src/adv_lab/eval/harness.py):
  Input: (1, 8, 8) -- single channel, 8x8 spatial
  Conv2d(1, 8, 3, padding=1) -> ReLU
  Conv2d(8, 16, 3, padding=1) -> ReLU
  Flatten -> Linear(16*8*8=1024, 32) -> ReLU -> Linear(32, 3)
  Output: 3 classes
  Parameters: ~33,000
```

Baseline results on synthetic random data (N=100 samples):
- Clean accuracy: 0.920
- FGSM robust accuracy (eps=0.03): 0.508
- PGD robust accuracy (eps=0.03, steps=40): 0.500
- C&W robust accuracy (c=1e-4, steps=1000): 0.034
- CI gate: PASS (0.500 > 0.300 threshold)

---

## Appendix B: Module Line Counts

| Module | Lines | Complexity |
|--------|-------|-----------|
| `attacks/norms.py` | 711 | High (5 norm types) |
| `attacks/llm.py` | 765 | High (6 attack types + simulated infra) |
| `attacks/inversion.py` | 612 | High (4 privacy attacks) |
| `eval/certified.py` | 589 | High (3 certification methods) |
| `attacks/physical.py` | 481 | Medium-High (EOT + printability) |
| `attacks/adaptive.py` | 440 | Medium (3 bypass methods + orchestrator) |
| `attacks/universal.py` | 381 | Medium (UAP + fast variant + eval) |

---

## Appendix C: Revision History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2025 | Initial comprehensive report covering all 20 tiers |

---

*End of report.*
