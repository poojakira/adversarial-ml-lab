# Executive Security Report: Adversarial AI Risk Posture

**Classification:** CONFIDENTIAL - BOARD DISTRIBUTION  
**Prepared for:** CISO / Board of Directors  
**Date:** 2025  
**Document Type:** AI System Security Risk Assessment

---

## 1. Executive Summary

Your organization deploys AI models that make automated decisions. This report presents the findings of a comprehensive adversarial security evaluation covering 20 distinct attack classes. **The bottom line: an attacker with minimal resources can cause your AI system to produce incorrect outputs, and the cost to attack is significantly lower than the cost to defend.**

Our evaluation demonstrates that a single optimization-based attack (taking approximately 5 seconds of compute on consumer hardware) reduces model accuracy from 92% to 3.4%, which is equivalent to random guessing. At the same time, the standard defense (adversarial training) only partially addresses one attack type while leaving the system vulnerable to all others. This represents a structural asymmetry where attackers have the advantage.

From a regulatory perspective, the privacy attacks in this evaluation (model inversion and membership inference) expose training data, creating potential violations under GDPR Article 22, CCPA, and HIPAA. The organization must assume that any AI system processing personal data is a privacy liability until proven otherwise through formal evaluation.

---

## 2. What an Attacker Can Do to Your AI System Today

In plain language, here is what a motivated attacker can accomplish against your AI systems:

### Make Your Model Produce Wrong Answers

- **Add invisible noise to an image** and the model will misclassify it with high confidence. The change is imperceptible to humans but causes the AI to fail. Cost: less than 1 second of compute.
- **Craft a single "universal" noise pattern** that, when overlaid on any input, causes misclassification on **80%+ of all inputs**. This pattern works across different model architectures.
- **Print a physical sticker** (an adversarial patch) that causes real-world misclassification when placed in a camera's field of view, even under varying angles (+/-30 degrees) and lighting conditions.

### Steal Your Model

- **Copy your model's behavior** using only API access. After approximately **6,000 queries** (costing a few dollars at typical API pricing), an attacker can build a local replica that agrees with your model 70%+ of the time. They can then develop attacks offline against their copy.

### Extract Private Training Data

- **Determine whether a specific individual's data was used to train your model** (membership inference). Success rates of 55-75% are achievable, which is far above random chance (50%).
- **Reconstruct approximations of training data** from model outputs (model inversion). If your model was trained on faces, medical images, or sensitive documents, approximations of that data can be extracted.

### Bypass Your Defenses

- **Adversarial training** (the most common defense) only protects against one type of perturbation. Switching to a different mathematical norm bypasses it entirely.
- **Detection systems** (NeuralCleanse, STRIP) can be evaded by attackers who know the detection method is deployed.
- **Certified defenses** (provable robustness) only protect within a tiny radius and scale poorly to real-world models.

### Compromise Your AI Pipeline

- **Poison training data** without changing any labels (clean-label poisoning). No data audit catches this because the labels appear correct.
- **Manipulate LLM outputs** through prompt injection, suffix attacks, or genetic algorithm-based jailbreaks.

---

## 3. Attacker Cost Analysis

What does it actually cost an attacker to break your AI systems?

### Low-Cost Attacks (Minutes, <$1 in compute)

| Attack | What It Does | Queries/Time | Estimated Cost | Success Rate |
|--------|-------------|-------------|---------------|-------------|
| **FGSM** (single-step perturbation) | Makes one image misclassify | **1 query, <1ms** | Negligible | 40-60% |
| **PGD** (multi-step perturbation) | Stronger misclassification | **40 queries, ~50ms** | Negligible | 50-80% |
| **Prompt injection** (LLM) | Bypasses LLM safety controls | **1 query** | Negligible | Variable |

### Medium-Cost Attacks (Hours, $1-$100 in compute)

| Attack | What It Does | Queries/Time | Estimated Cost | Success Rate |
|--------|-------------|-------------|---------------|-------------|
| **C&W optimization** | Nearly 100% misclassification | **1,000 steps, ~5s/batch** | ~$1-5 | **90-99%** |
| **Black-box query attacks** | Misclassify without model access | **1,000 queries** | ~$1-10 API fees | 60-95% |
| **Model stealing** | Copy your model | **~6,000 queries** | ~$5-50 API fees | 70%+ fidelity |
| **Membership inference** | Identify training data members | **Shadow model training** | ~$10-50 | 55-75% AUC |

### High-Cost Attacks (Days, $100-$10,000 in compute)

| Attack | What It Does | Queries/Time | Estimated Cost | Success Rate |
|--------|-------------|-------------|---------------|-------------|
| **GCG suffix** (LLM jailbreak) | Bypasses LLM alignment | **500 steps x 256 candidates** | ~$100-1,000 GPU | 40-70% |
| **Universal perturbation** | One pattern fools 80%+ inputs | **Thousands of iterations** | ~$50-500 GPU | 80%+ fooling |
| **Physical patch** | Real-world adversarial sticker | **EOT optimization + printing** | ~$100-1,000 | 60-85% |
| **Model inversion** | Reconstruct training data | **1,000+ optimizer steps** | ~$50-500 GPU | SSIM 0.3-0.7 |
| **Data poisoning** | Corrupt model during training | **Requires training data access** | Variable | High if access obtained |

### Key Takeaway

> **The cheapest effective attack (FGSM) costs less than one cent. The most reliable attack (C&W) costs less than $5. The cost for a defender to achieve robustness against all attack classes simultaneously remains an open research problem with no proven solution at scale.**

---

## 4. Defender Cost Analysis

What does it cost your organization to address each vulnerability class?

| Vulnerability Class | Fix | Compute Cost | Engineering Time | Ongoing Cost |
|-------------------|-----|-------------|-----------------|-------------|
| **Single-norm attacks** (FGSM, PGD) | Adversarial training | **3-10x** training compute | 1-2 weeks | Retrain on schedule |
| **Multi-norm attacks** (L0, L1, L2, Linf) | Multi-norm adversarial training | **10-30x** training compute | 2-4 weeks | Retrain on schedule |
| **Optimization attacks** (C&W) | No complete defense exists | N/A | Research-level | Ongoing monitoring |
| **Black-box attacks** | Query monitoring + rate limiting | Moderate (anomaly detection infra) | 2-4 weeks | Continuous ops |
| **Model stealing** | Query anomaly detection + watermarking | Moderate | 4-8 weeks | Continuous ops |
| **Privacy attacks** | Differential privacy training | **2-5x** training compute + accuracy loss | 4-8 weeks | Accuracy tradeoff permanent |
| **Data poisoning** | Data provenance + spectral signatures | High (audit pipeline) | 8-12 weeks | Continuous validation |
| **LLM attacks** | Input/output filtering + safety training | High (fine-tuning) | 8-16 weeks | Continuous updates |
| **Physical patches** | Input preprocessing + detection | Moderate | 4-8 weeks | Continuous ops |
| **Supply chain** (CI gate) | HMAC signing (already implemented) | Minimal | **Done** | Key rotation |
| **Certified robustness** | Architecture constraints + verification | **50-100x** training compute + severe accuracy loss | 12+ weeks | Architecture lock-in |

### Key Takeaway

> **Total estimated defender cost for comprehensive coverage: 6-12 months of dedicated ML security engineering, 10-30x increase in training compute budget, and permanent 5-15% accuracy reduction from defensive measures. Some attack classes (C&W-style optimization) have no complete defense.**

---

## 5. Risk Ranking: All 20 Attack Classes

Ranked by **Probability x Impact**. Probability reflects attacker accessibility and skill required. Impact reflects business consequences of a successful attack.

| Rank | Attack Class | Probability | Impact | Risk Score | Justification |
|------|-------------|-------------|--------|-----------|---------------|
| 1 | **Data poisoning / backdoors** | High | Critical | **Critical** | Training pipeline compromise; undetectable at inference time |
| 2 | **Model stealing** | High | High | **Critical** | API access sufficient; enables all downstream attacks offline |
| 3 | **Membership inference** | High | High | **Critical** | Regulatory exposure (GDPR/CCPA); low attacker cost |
| 4 | **Prompt injection (LLM)** | High | High | **Critical** | Single query; widespread LLM deployment |
| 5 | **FGSM / single-step attacks** | High | Medium | **High** | Trivial to execute; limited to single-input misclassification |
| 6 | **PGD multi-step attacks** | High | Medium-High | **High** | Standard attack toolkit; reliable success |
| 7 | **Black-box query attacks** | High | Medium | **High** | No model access needed; API-only |
| 8 | **C&W optimization attack** | Medium | High | **High** | Near-100% success; requires white-box access |
| 9 | **Universal adversarial perturbations** | Medium | High | **High** | One pattern defeats 80%+ inputs; scalable attack |
| 10 | **Ensemble attacks** | Medium | High | **High** | Boosts transfer rate by 15-25%; defeats single-model defenses |
| 11 | **Defense evasion (post-processing)** | Medium | Medium-High | **High** | Survives JPEG, feature squeezing, detectors |
| 12 | **Model inversion (privacy)** | Medium | High | **High** | Training data reconstruction; privacy/regulatory impact |
| 13 | **Adaptive attacks (BPDA/EoT)** | Medium | High | **Medium-High** | Bypasses obfuscated gradients; requires defense knowledge |
| 14 | **LLM jailbreaks (GCG/AutoDAN)** | Medium | Medium-High | **Medium-High** | Computationally expensive but automated |
| 15 | **Perturbation chaining** | Medium | Medium | **Medium** | Combines multiple norms; harder to defend |
| 16 | **Physical adversarial patches** | Low-Medium | High | **Medium** | Requires physical access; real-world impact |
| 17 | **Inference-time manipulation** | Low-Medium | Medium-High | **Medium** | Requires pipeline access |
| 18 | **Non-classification attacks** (detection, RL) | Medium | Medium | **Medium** | Domain-specific; growing attack surface |
| 19 | **Timing-constrained attacks** | Low | Medium | **Low-Medium** | Rate limiting reduces success |
| 20 | **Bayesian parameter search** | Low | Medium | **Low-Medium** | Optimizes other attacks; indirect amplifier |

### Risk Matrix Summary

|  | **Low Impact** | **Medium Impact** | **High Impact** | **Critical Impact** |
|--|---------------|------------------|----------------|-------------------|
| **High Probability** | | FGSM, PGD, Black-box | Model stealing, Membership inference, Prompt injection | Data poisoning |
| **Medium Probability** | | Timing, Param search | C&W, UAP, Ensemble, Evasion, Model inversion, Adaptive, LLM jailbreaks, Chaining | |
| **Low Probability** | | | Physical patches, Inference manipulation, Non-classification | |

---

## 6. What "Certified Robust" Actually Means

### What It IS

"Certified robust" means a mathematical proof guarantees that no perturbation (change to the input) below a certain size can change the model's prediction. Think of it as a provable "safety bubble" around each input.

### What It Provides

- **A guarantee with a radius.** For example: "No change to this image smaller than 0.5 pixel units can flip the prediction." This is mathematically proven, not estimated.
- **Three methods available:** Randomized smoothing (adding noise and voting), Lipschitz constraints (limiting how fast outputs can change), and Interval Bound Propagation (tracking worst-case through each layer).

### What It Does NOT Protect Against

- **Perturbations outside the certified radius.** The guarantee only holds for small changes. A slightly larger perturbation has no guarantee. In our evaluation, the certified radius is often very small relative to what an attacker can practically apply.
- **Different types of changes.** A certificate for pixel-level noise says nothing about rotation, color shifts, or semantic changes (e.g., adding sunglasses to a face).
- **Attacks on the training process.** Certification assumes the model itself is trustworthy. Poisoned or backdoored models can pass certification while containing hidden vulnerabilities.
- **Real-world scale.** Current certified methods lose most of their accuracy on complex real-world models (ImageNet-scale). The tradeoff between provable safety and usable accuracy is severe.
- **Adaptive attackers.** An attacker who knows the certification method can often find perturbations just outside the certified radius that still succeed.

### The Honest Assessment

> Certified robustness is the gold standard for what it covers, but what it covers is a small fraction of the real threat surface. It is analogous to a building having a certified fireproof door while the windows are open.

---

## 7. Privacy Attack Regulatory Exposure

### GDPR Article 22: Automated Decision-Making

**What the regulation requires:** Individuals have the right not to be subject to decisions based solely on automated processing that significantly affects them, and the right to meaningful information about the logic involved.

**What our findings mean:**
- **Membership inference** (determining if someone's data was used in training) achieves 55-75% accuracy. If an individual's data was used without explicit consent for AI training, and an attacker can demonstrate membership, this constitutes evidence of a processing violation.
- **Model inversion** (reconstructing training data from the model) means the model itself is a potential data breach vector. Under GDPR, the model may constitute "personal data" if individuals can be identified from its outputs.
- **Exposure:** Organizations must be able to demonstrate that AI models do not inadvertently leak the personal data they were trained on. Current models cannot make this guarantee without additional privacy protections (e.g., differential privacy).

### CCPA: Right to Know and Right to Delete

**What the regulation requires:** Consumers have the right to know what personal information is collected and used, and the right to request deletion.

**What our findings mean:**
- If a model was trained on consumer data, **model inversion demonstrates that the data has not been effectively deleted** even if the original dataset is removed. The model retains extractable information about training individuals.
- A "right to delete" request may require model retraining (removing the individual's data and retraining from scratch), which is operationally expensive.
- **Exposure:** Membership inference provides a technical method for consumers (or regulators) to verify whether their data was used in training, creating an audit mechanism the organization may not be prepared for.

### HIPAA: Protected Health Information

**If your AI systems process healthcare data:**
- Model inversion on healthcare AI could reconstruct patient medical images or clinical features, constituting a PHI (Protected Health Information) breach.
- Membership inference confirms whether a specific patient was in the training set, revealing their association with a medical condition (the fact that the model was trained for a specific disease is itself diagnostic context).
- **Exposure:** A successful model inversion attack on a healthcare AI system would likely trigger breach notification requirements under HIPAA Section 164.404.

### Regulatory Risk Summary

| Regulation | Attack Vector | Violation Type | Potential Penalty |
|-----------|--------------|---------------|------------------|
| **GDPR** | Membership inference | Unlawful processing (Art. 6) | Up to 4% annual global turnover |
| **GDPR** | Model inversion | Data breach (Art. 33/34) | Up to 4% annual global turnover |
| **CCPA** | Model inversion | Failure to delete (Sec. 1798.105) | $2,500-$7,500 per violation |
| **HIPAA** | Model inversion | PHI breach | $100-$50,000 per violation, up to $1.5M/year |

---

## 8. CI Gate Signing: Software Supply Chain Audit Posture

### What We Implemented

Every AI evaluation report produced by our CI/CD pipeline (continuous integration / continuous deployment) is now cryptographically signed using HMAC-SHA256 (a tamper-detection mechanism similar to a digital seal). This means:

- **No one can forge a passing evaluation report.** The signature requires a 256-bit secret key derived using 600,000 rounds of key stretching (PBKDF2). Without the key, fabricating a valid signature is computationally impossible.
- **No one can replay an old report.** Each report includes a timestamp and unique identifier (nonce). The system detects attempts to resubmit previously-valid reports.
- **No one can swap test inputs.** The system hashes all inputs used during evaluation. If an attacker substitutes easy inputs for hard ones, the hash mismatch is detected.

### What This Means for Compliance Frameworks

| Framework | Requirement | How CI Signing Addresses It |
|-----------|------------|----------------------------|
| **SOC 2 (Type II)** | CC6.1: Logical access controls over information assets | Signed reports prove evaluation integrity without key access |
| **SOC 2 (Type II)** | CC7.2: Monitoring of system components for anomalies | Replay detection identifies unauthorized report submissions |
| **ISO 27001** | A.12.1.2: Change management | Signed manifests bind model weights to evaluation results |
| **ISO 27001** | A.14.2.7: Outsourced development security | Third-party models must pass signed evaluation before deployment |
| **NIST AI RMF** | Measure 2.6: AI system integrity | Cryptographic proof that robustness claims are backed by actual tests |

### Residual Risks (What Signing Does NOT Fix)

- **Key compromise:** If an attacker obtains the signing secret, all guarantees are void. The key is currently stored as an environment variable, not in a Hardware Security Module (HSM).
- **Evaluation quality:** Signing proves a report is authentic, not that the evaluation was sufficiently rigorous. A weak test suite produces a valid but meaningless signed report.
- **No revocation mechanism:** If a key is compromised, there is no automated way to invalidate previously-signed reports.

### Recommended Upgrade Path

1. Move signing keys to HSM (AWS CloudHSM or Azure Dedicated HSM) for SOC 2 Type II compliance.
2. Implement key rotation on a quarterly schedule.
3. Add certificate revocation for compromised keys.

---

## 9. Three-Tier Remediation Roadmap

### 30 Days - Critical (Do Immediately)

These items address the highest-probability, highest-impact vulnerabilities with the lowest implementation cost.

| # | Action | Owner | Effort | Risk Addressed |
|---|--------|-------|--------|---------------|
| 1 | **Deploy query rate limiting and anomaly detection** on all ML API endpoints. Flag accounts exceeding 200 queries/hour on a single model. | Platform Engineering | 3-5 days | Model stealing, black-box attacks |
| 2 | **Run the PGD evaluation gate** on all production models. Any model scoring below 30% robust accuracy must not serve production traffic. | ML Engineering | 1-2 days | Basic adversarial vulnerability |
| 3 | **Audit training data provenance** for all models processing personal data. Document data sources, consent basis, and retention schedule. | Data Governance | 5-10 days | Regulatory exposure (GDPR, CCPA) |
| 4 | **Enable CI gate signing** on all model deployment pipelines. No model deploys without a signed evaluation report. | DevOps / MLOps | 2-3 days | Supply chain tampering |
| 5 | **Implement input validation and sanitization** for all LLM-facing endpoints. Block known prompt injection patterns. | Application Security | 3-5 days | Prompt injection |
| 6 | **Inventory all deployed AI models** and classify by data sensitivity (PII, PHI, financial). | Risk Management | 5 days | Unknown exposure |

### 90 Days - High Priority (This Quarter)

These items require engineering investment but substantially reduce the attack surface.

| # | Action | Owner | Effort | Risk Addressed |
|---|--------|-------|--------|---------------|
| 7 | **Implement adversarial training (PGD-AT)** for all high-risk classification models. Accept the 5-10% clean accuracy tradeoff. | ML Engineering | 4-6 weeks | Gradient-based attacks (FGSM, PGD) |
| 8 | **Deploy differential privacy** (DP-SGD) for models trained on personal data. Target epsilon < 10 for meaningful privacy. | ML Engineering | 4-8 weeks | Membership inference, model inversion |
| 9 | **Implement model watermarking** to detect unauthorized copies of your models. | ML Engineering | 2-4 weeks | Model stealing detection |
| 10 | **Conduct red-team evaluation** using the full 20-tier attack ladder against your top 3 production models on real weights. | Security Team | 4-6 weeks | Unknown vulnerabilities |
| 11 | **Migrate CI signing keys to HSM** and implement quarterly key rotation. | Security Engineering | 2-3 weeks | Key compromise |
| 12 | **Establish an AI incident response plan** covering adversarial attacks, data poisoning discovery, and model compromise. | Security Operations | 2-3 weeks | Incident preparedness |
| 13 | **Deploy multi-norm evaluation** (L0, L1, L2, Linf, semantic) in CI pipelines. A model that passes only L-inf PGD has a false sense of security. | ML Engineering | 2-3 weeks | Multi-norm attacks |

### 12 Months - Strategic (Long-Term Architecture)

These items require organizational commitment and architectural changes.

| # | Action | Owner | Effort | Risk Addressed |
|---|--------|-------|--------|---------------|
| 14 | **Adopt certified robustness evaluation** as a deployment gate for safety-critical models (autonomous systems, medical AI). Accept that certified models will have lower accuracy. | ML Architecture | 3-6 months | Provable safety for critical systems |
| 15 | **Build a continuous adversarial monitoring system** that automatically tests deployed models against new attack techniques as they are published. | ML Platform | 3-4 months | Emerging threats |
| 16 | **Implement training data attribution and machine unlearning** capability to support GDPR right-to-deletion without full retraining. | ML Engineering | 4-6 months | Regulatory compliance |
| 17 | **Establish a formal AI security review** as part of the model development lifecycle (similar to security review for software releases). | Security Governance | 2-3 months | Process maturity |
| 18 | **Evaluate and deploy ensemble defenses** combining adversarial training, input preprocessing, and detection layers for defense-in-depth. | ML Engineering | 4-6 months | Adaptive attackers |
| 19 | **Integrate physical-world adversarial testing** for any model deployed in camera-based systems (autonomous vehicles, surveillance, quality inspection). | Applied ML + Security | 6-12 months | Physical attacks |
| 20 | **Pursue SOC 2 Type II certification** for ML operations, using signed evaluation reports and HSM-backed keys as evidence of control effectiveness. | Compliance | 6-12 months | Audit posture |

---

## 10. The Board Statement

> **Our AI systems are currently operating without verified adversarial robustness, and the cost to an attacker to cause targeted misclassification is lower than the cost to us to detect it.**
>
> Specifically: a motivated attacker can cause any individual prediction to be wrong for less than $5 of compute, can copy our model's behavior for less than $50 of API costs, and can determine whether a specific individual's data was used in training with 55-75% accuracy. Our current defenses address a subset of these threats. The CI gate signing infrastructure provides tamper-evident evaluation reports, but this proves only that we tested the model -- not that the model is robust against all attack classes.
>
> The structural challenge is fundamental: adversarial robustness against all possible perturbation types simultaneously is an unsolved problem in the research community. No organization, including the largest AI labs, has deployed models that are provably robust against all 20 attack classes evaluated here. Our roadmap prioritizes the attacks most likely to be exploited (data poisoning, model stealing, privacy leakage) and the defenses most likely to reduce organizational risk (adversarial training, differential privacy, query monitoring) within realistic engineering budgets.

---

## Appendix: Glossary of Key Terms

| Term | Plain-Language Definition |
|------|--------------------------|
| **Adversarial example** | An input deliberately modified to cause the AI to produce a wrong answer, while looking normal to humans |
| **Perturbation** | A small, often invisible change made to an input to fool an AI model |
| **Robustness** | The ability of an AI model to produce correct answers even when inputs are deliberately modified |
| **Gradient** | The mathematical direction that tells an attacker how to change an input to maximize the AI's error |
| **White-box attack** | An attack where the attacker has full access to the model's internals (weights, architecture) |
| **Black-box attack** | An attack where the attacker can only send inputs and observe outputs (API access only) |
| **Adversarial training** | Retraining a model on attacked examples so it learns to resist those attacks |
| **Differential privacy** | A mathematical technique that adds noise during training to prevent individual data points from being memorizable |
| **Membership inference** | Determining whether a specific data point was used to train a model |
| **Model inversion** | Reconstructing training data (e.g., faces, documents) from a trained model's outputs |
| **HMAC** | A cryptographic signature that proves a message has not been tampered with |
| **Certified robustness** | A mathematical proof that no small perturbation can change a model's prediction |
| **UAP** | Universal Adversarial Perturbation -- a single noise pattern that causes misclassification on most inputs |
| **EOT** | Expectation over Transformations -- optimizing an attack to work under varying real-world conditions (angles, lighting) |
| **FGSM** | Fast Gradient Sign Method -- the simplest, cheapest adversarial attack (one step, one gradient) |
| **PGD** | Projected Gradient Descent -- a stronger iterative version of FGSM |
| **C&W** | Carlini and Wagner attack -- an optimization-based attack with near-perfect success rate |

---

*End of Executive Report.*
