# Threat Model: Adversarial Attacks on Image Classifiers

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-08-24 |
| Scope | CIFAR-10 image classifiers (ResNet-18, WideResNet) |
| Framework | MITRE ATLAS (Adversarial Threat Landscape for AI Systems) |
| Classification | Public |

---

## 1. System Under Threat

### 1.1 Asset Description

The protected system is a deep neural network image classifier deployed for inference.
The model receives RGB images (32×32 for CIFAR-10) and outputs class probabilities across
10 categories. The pipeline includes:

- **Model weights**: Trained parameters stored on disk or in a model registry
- **Inference API**: Endpoint accepting image inputs and returning predictions
- **Training data**: CIFAR-10 dataset (50,000 training / 10,000 test images)
- **Training pipeline**: Scripts and infrastructure for model fine-tuning

### 1.2 Trust Boundaries

```
┌─────────────────────────────────────────────────────────┐
│  External (Untrusted)                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ User inputs │  │ Third-party  │  │ Physical      │  │
│  │ (images)    │  │ data sources │  │ environment   │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘  │
└─────────┼────────────────┼───────────────────┼──────────┘
          │                │                   │
══════════╪════════════════╪═══════════════════╪══════════════
          │                │                   │
┌─────────▼────────────────▼───────────────────▼──────────┐
│  Internal (Protected)                                   │
│  ┌──────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │ Preprocessing│  │ Model       │  │ Model         │  │
│  │ Pipeline     │──▶ Inference   │  │ Registry      │  │
│  └──────────────┘  └─────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Attacker Profiles

| Profile | Knowledge | Access | Goal |
|---------|-----------|--------|------|
| White-box | Full model architecture + weights | Query + gradient access | Cause misclassification |
| Black-box | Model API only (input/output pairs) | Query access only | Cause misclassification |
| Insider | Training pipeline access | Data + code | Implant backdoor |
| Physical | Knowledge of model deployment | Camera/sensor input | Real-world evasion |

---

## 2. Evasion Attacks

Evasion attacks craft adversarial inputs at inference time to cause misclassification
without modifying the model itself.

### 2.1 FGSM (Fast Gradient Sign Method)

| Attribute | Value |
|-----------|-------|
| MITRE ATLAS | AML.T0043 (Craft Adversarial Data) |
| MITRE ATLAS | AML.T0015 (Evade ML Model) |
| Paper | Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2014) |
| Threat level | Medium |
| Perturbation norm | L∞ |
| Typical ε | 8/255 (0.031) for CIFAR-10 |
| Computational cost | Single forward + backward pass |

**Attack mechanism**: Computes the gradient of the loss with respect to the input image,
then perturbs each pixel by ε in the direction of the gradient sign:

```
x_adv = x + ε · sign(∇_x L(θ, x, y))
```

**Risk assessment**: Low sophistication, high speed. Effective against undefended models
(drops accuracy from ~95% to ~20-40%). Easily defeated by adversarial training but serves
as a minimum-bar sanity check.

### 2.2 PGD (Projected Gradient Descent)

| Attribute | Value |
|-----------|-------|
| MITRE ATLAS | AML.T0043 (Craft Adversarial Data) |
| MITRE ATLAS | AML.T0015 (Evade ML Model) |
| Paper | Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018) |
| Threat level | High |
| Perturbation norm | L∞ |
| Typical ε | 8/255, steps=20, step_size=2/255 |
| Computational cost | N forward + backward passes (N = step count) |

**Attack mechanism**: Iterative extension of FGSM with random initialization and
projection back onto the ε-ball after each step:

```
x^(t+1) = Π_{B(x,ε)} [ x^(t) + α · sign(∇_x L(θ, x^(t), y)) ]
```

**Risk assessment**: Considered the strongest first-order adversary. Models robust to
PGD-20 at ε=8/255 are considered meaningfully hardened. This is the primary CI gate
metric for this project (threshold: ≥30% robust accuracy).

### 2.3 C&W (Carlini & Wagner)

| Attribute | Value |
|-----------|-------|
| MITRE ATLAS | AML.T0043 (Craft Adversarial Data) |
| MITRE ATLAS | AML.T0015 (Evade ML Model) |
| Paper | Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks" (2017) |
| Threat level | Critical |
| Perturbation norm | L2 (primary), L0, L∞ variants |
| Typical parameters | c=1e-3, κ=0, lr=0.01, steps=1000 |
| Computational cost | High (optimization over many iterations) |

**Attack mechanism**: Formulates adversarial example generation as a constrained
optimization problem, minimizing perturbation magnitude while achieving misclassification:

```
minimize ‖δ‖₂ + c · f(x + δ)
such that x + δ ∈ [0, 1]^n
```

**Risk assessment**: Defeats distillation and many detection-based defenses. Produces
minimal perturbations that are difficult to detect. Computationally expensive but
represents a motivated adversary with resources.

### 2.4 AutoAttack

| Attribute | Value |
|-----------|-------|
| MITRE ATLAS | AML.T0043 (Craft Adversarial Data) |
| MITRE ATLAS | AML.T0015 (Evade ML Model) |
| Paper | Croce & Hein, "Reliable evaluation of adversarial robustness with an ensemble of attacks" (2020) |
| Threat level | Critical |
| Perturbation norm | L∞ and L2 |
| Components | APGD-CE, APGD-T, FAB, Square Attack |
| Computational cost | Very high (ensemble of diverse attacks) |

**Attack mechanism**: Parameter-free ensemble combining:
1. **APGD-CE**: Auto-PGD with cross-entropy loss (adaptive step size)
2. **APGD-T**: Auto-PGD with targeted DLR loss
3. **FAB**: Fast Adaptive Boundary attack (minimum-norm)
4. **Square Attack**: Score-based black-box L∞ attack

**Risk assessment**: Gold standard for robustness evaluation. Claims of adversarial
robustness without AutoAttack evaluation are considered incomplete. Any model that
passes PGD but fails AutoAttack has a false sense of security.

---

## 3. Model Extraction

| Attribute | Value |
|-----------|-------|
| MITRE ATLAS | AML.T0024 (Exfiltration via ML Inference API) |
| MITRE ATLAS | AML.T0044 (Full ML Model Access) |
| Threat level | High |
| Prerequisite | Query access to prediction API |

### 3.1 Attack Description

Model extraction (model stealing) trains a surrogate model by querying the target API
with carefully chosen inputs and using the returned predictions as training labels.

**Techniques**:
- **Knockoff Nets**: Query with natural images from a transfer set
- **JBDA (Jacobian-Based Data Augmentation)**: Iteratively augment synthetic data
  using estimated gradients from the surrogate
- **Cryptanalytic extraction**: Exploit confidence scores to recover exact parameters

### 3.2 Downstream Risks

A stolen model enables:
1. White-box adversarial attacks via gradient access on the surrogate
2. Intellectual property theft of proprietary architectures
3. Circumventing rate-limiting or access controls

### 3.3 Mitigations

| Mitigation | ATLAS Reference | Effectiveness |
|------------|-----------------|---------------|
| Prediction output quantization | AML.M0015 | Medium |
| Query rate limiting | AML.M0015 | Low-Medium |
| Watermarking | AML.M0015 | Detection only |
| Differential privacy on outputs | AML.M0015 | Medium-High |

---

## 4. Data Poisoning & Backdoor Attacks

| Attribute | Value |
|-----------|-------|
| MITRE ATLAS | AML.T0020 (Poison Training Data) |
| MITRE ATLAS | AML.T0054 (Backdoor ML Model) |
| Threat level | Critical |
| Prerequisite | Access to training data or pipeline |

### 4.1 Clean-Label Poisoning

Attacker modifies a small fraction of training samples (without changing labels) to cause
targeted misclassification at test time. Examples include Feature Collision attacks and
Witches' Brew.

### 4.2 Backdoor (Trojan) Attacks

Attacker inserts a trigger pattern (e.g., a small patch) into training images with a
target label. At inference time, any input containing the trigger is classified as the
attacker's chosen class.

**Common triggers**:
- Fixed pixel patterns (BadNets)
- Blended/transparent overlays (Blended Attack)
- Warping-based transforms (WaNet)
- Frequency-domain triggers (invisible to humans)

### 4.3 Supply Chain Vector

```
Compromised Dataset       Compromised Pretrained Weights
(HuggingFace, Kaggle) ──▶ Training Pipeline ──▶ Deployed Model
                               ▲
                               │
                    Compromised Dependencies
                    (malicious PyPI package)
```

| Attack Vector | ATLAS ID | Likelihood | Impact |
|---------------|----------|------------|--------|
| Poisoned public dataset | AML.T0020 | Medium | Critical |
| Trojan pretrained weights | AML.T0054 | Medium | Critical |
| Malicious training code | AML.T0054 | Low | Critical |

### 4.4 Mitigations

- Spectral signatures analysis on training data
- Activation clustering for backdoor detection
- Neural Cleanse trigger inversion
- Model weight provenance verification (SLSA, Sigstore)

---

## 5. Physical-World Adversarial Patches

| Attribute | Value |
|-----------|-------|
| MITRE ATLAS | AML.T0043 (Craft Adversarial Data) |
| MITRE ATLAS | AML.T0015 (Evade ML Model) |
| Paper | Brown et al., "Adversarial Patch" (2017); Eykholt et al. (2018) |
| Threat level | High |
| Prerequisite | Knowledge of model architecture |

### 5.1 Attack Description

Physical adversarial patches are printable perturbations that, when placed in a scene,
cause misclassification regardless of viewing angle, lighting, or camera properties.

**Characteristics**:
- **Location-independent**: Patch works regardless of placement in the image
- **Physically realizable**: Robust to printing, camera capture, and environmental noise
- **Universal**: Works across different input images (not input-specific)

### 5.2 Real-World Scenarios

| Scenario | Target System | Impact |
|----------|---------------|--------|
| Stop sign modification | Autonomous vehicle perception | Safety-critical misclassification |
| Adversarial T-shirt | Person detection/surveillance | Evasion of detection systems |
| Adversarial glasses | Face recognition | Identity impersonation |
| License plate perturbation | ALPR systems | Enforcement evasion |

### 5.3 Expectation-over-Transformation (EoT)

Physical attacks optimize over a distribution of transformations:

```
argmax_δ  E_{t~T} [ L(f(t(x + δ)), y_target) ]
```

Where T includes rotation, scaling, brightness, perspective, and camera noise.

### 5.4 Mitigations

| Mitigation | Effectiveness | Trade-off |
|------------|---------------|-----------|
| Certified defenses (randomized smoothing) | Provable but limited radius | Accuracy drop |
| Input preprocessing (JPEG compression, bit-depth reduction) | Low | Defeated by adaptive attacks |
| Ensemble methods | Medium | Compute cost |
| Adversarial training with augmentation | Medium-High | Training cost |

---

## 6. MITRE ATLAS Technique Mapping Summary

| Threat Category | ATLAS Technique | ATLAS ID | This Repo |
|----------------|-----------------|----------|-----------|
| Evasion (FGSM) | Craft Adversarial Data | AML.T0043 | `attacks/fgsm.py` |
| Evasion (PGD) | Craft Adversarial Data | AML.T0043 | `attacks/pgd.py` |
| Evasion (C&W) | Craft Adversarial Data | AML.T0043 | `attacks/cw.py` |
| Evasion (AutoAttack) | Craft Adversarial Data | AML.T0043 | `attacks/adaptive.py` |
| Evasion (all) | Evade ML Model | AML.T0015 | `eval/harness.py` |
| Model extraction | Exfiltration via Inference API | AML.T0024 | `attacks/model_stealing.py` |
| Model extraction | Full ML Model Access | AML.T0044 | N/A (downstream risk) |
| Data poisoning | Poison Training Data | AML.T0020 | `attacks/poisoning.py` |
| Backdoor | Backdoor ML Model | AML.T0054 | `attacks/poisoning.py` |
| Physical patch | Craft Adversarial Data | AML.T0043 | `attacks/physical.py` |
| Physical patch | Evade ML Model | AML.T0015 | `attacks/physical.py` |
| Supply chain | Publish Poisoned Datasets | AML.T0019 | Out of scope (see `attacks/poisoning.py` for training-time poisoning) |
| Inference abuse | Discover ML Model Family | AML.T0007 | `attacks/inference.py` |

---

## 7. Risk Matrix

### 7.1 Likelihood × Impact Assessment

| Threat | Likelihood | Impact | Risk Level | Priority |
|--------|-----------|--------|------------|----------|
| FGSM evasion | Almost Certain | Moderate | **High** | P2 |
| PGD evasion | Almost Certain | High | **Critical** | P1 |
| C&W evasion | Likely | High | **Critical** | P1 |
| AutoAttack evasion | Likely | Very High | **Critical** | P0 |
| Model extraction | Possible | High | **High** | P2 |
| Clean-label poisoning | Unlikely | Very High | **High** | P2 |
| Backdoor attack | Possible | Critical | **Critical** | P1 |
| Physical adversarial patch | Possible | High (context-dependent) | **High** | P2 |
| Supply chain compromise | Unlikely | Critical | **High** | P2 |

### 7.2 Risk Level Definitions

| Level | Definition | Response |
|-------|-----------|----------|
| Critical | Immediate threat to system integrity; bypass of all defenses | Mandatory mitigation before deployment |
| High | Significant threat with demonstrated feasibility | Mitigation required; monitor for escalation |
| Medium | Theoretical threat or requires significant resources | Document and plan mitigation |
| Low | Minimal real-world impact or extremely unlikely | Accept risk with monitoring |

### 7.3 Scoring Criteria

**Likelihood scale**:
- Almost Certain: Published tools freely available; minimal expertise required
- Likely: Known attack with moderate expertise and compute requirements
- Possible: Requires insider access, significant resources, or novel research
- Unlikely: Requires multiple preconditions unlikely to co-occur

**Impact scale**:
- Critical: Complete system compromise; safety-critical failure
- Very High: Systematic bypass of security controls
- High: Targeted misclassification with operational impact
- Moderate: Degraded performance detectable by monitoring
- Low: Minimal operational impact; easily recoverable

---

## 8. Recommended Controls

### 8.1 Detection

| Control | Detects | Implementation |
|---------|---------|----------------|
| Input statistical analysis | Evasion (L∞ perturbations) | `defenses/detection.py` |
| Prediction confidence monitoring | Evasion, extraction queries | Inference logging |
| Training data auditing | Poisoning, backdoors | Spectral signature analysis |
| Model diff on update | Backdoor insertion | Weight provenance checks |

### 8.2 Prevention

| Control | Prevents | Trade-off |
|---------|----------|-----------|
| Adversarial training (PGD-AT) | Evasion attacks | 5-15% clean accuracy drop |
| Certified defenses (randomized smoothing) | L2-bounded evasion | Significant accuracy reduction |
| Query rate limiting + logging | Model extraction | May impact legitimate users |
| Data provenance verification | Supply chain attacks | Operational overhead |

### 8.3 Response

1. **Alert**: Anomalous query patterns detected → Security team notification
2. **Contain**: Elevated adversarial detection rate → Switch to high-security model
3. **Investigate**: Suspected poisoning → Quarantine model, audit training data
4. **Recover**: Confirmed backdoor → Retrain from verified clean checkpoint

---

## 9. Assumptions and Limitations

1. This threat model assumes CIFAR-10 scale (32×32 RGB images, 10 classes).
   Higher-resolution models may have different attack surfaces.
2. Certified defense radii are computed for L2 norm; L∞ certification remains limited.
3. Adaptive attacks (designed to defeat specific defenses) are always possible—no
   defense should be considered permanent.
4. Physical-world attacks depend heavily on deployment context (camera quality, distance,
   lighting conditions).
5. This document covers known attack classes as of August 2026. Novel attacks emerge
   regularly; periodic review is required.

---

## 10. References

1. Goodfellow, I.J., Shlens, J., & Szegedy, C. (2014). Explaining and Harnessing Adversarial Examples. *ICLR 2015*.
2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. *ICLR 2018*.
3. Carlini, N. & Wagner, D. (2017). Towards Evaluating the Robustness of Neural Networks. *IEEE S&P 2017*.
4. Croce, F. & Hein, M. (2020). Reliable evaluation of adversarial robustness with an ensemble of attacks. *ICML 2020*.
5. Brown, T.B., Mané, D., Roy, A., Abadi, M., & Gilmer, J. (2017). Adversarial Patch. *NeurIPS 2017 Workshop*.
6. Gu, T., Dolan-Gavitt, B., & Garg, S. (2019). BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain. *IEEE Access*.
7. MITRE ATLAS. https://atlas.mitre.org/
8. Tramèr, F., et al. (2020). On Adaptive Attacks to Adversarial Example Defenses. *NeurIPS 2020*.
