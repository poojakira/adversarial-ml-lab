# Benchmark Methodology

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-08-24 |
| Scope | Adversarial robustness benchmarking for CIFAR-10 classifiers |
| Standard | Aligned with RobustBench evaluation protocol |

---

## 1. Overview

This document defines the methodology for evaluating adversarial robustness of image
classifiers in this repository. The benchmark measures the gap between clean accuracy
(standard test performance) and robust accuracy (performance under adversarial attack)
across multiple attack types and perturbation budgets.

---

## 2. Attack Parameters

### 2.1 FGSM (Fast Gradient Sign Method)

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `epsilon` | 8/255 (0.031) | [1/255, 16/255] | L∞ perturbation budget |
| `targeted` | False | {True, False} | Untargeted by default |
| `loss_fn` | CrossEntropy | {CE, CW-loss} | Loss function for gradient computation |

### 2.2 PGD (Projected Gradient Descent)

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `epsilon` | 8/255 (0.031) | [1/255, 16/255] | L∞ perturbation budget |
| `step_size` | 2/255 (0.0078) | [ε/10, ε/2] | Per-iteration step size |
| `num_steps` | 20 | [7, 100] | Number of PGD iterations |
| `random_start` | True | {True, False} | Uniform random initialization in ε-ball |
| `restarts` | 1 | [1, 10] | Number of random restarts |
| `loss_fn` | CrossEntropy | {CE, CW-loss, DLR} | Loss function |

### 2.3 C&W (Carlini & Wagner L2)

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `confidence` (κ) | 0 | [0, 50] | Minimum margin for misclassification |
| `learning_rate` | 0.01 | [1e-4, 0.1] | Adam optimizer learning rate |
| `max_iterations` | 1000 | [100, 10000] | Optimization steps |
| `binary_search_steps` | 9 | [1, 20] | Steps for finding optimal constant c |
| `initial_const` | 1e-3 | [1e-5, 1.0] | Initial value of tradeoff constant c |
| `abort_early` | True | {True, False} | Stop if loss stops decreasing |

### 2.4 AutoAttack

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `epsilon` | 8/255 | [1/255, 16/255] | L∞ perturbation budget |
| `norm` | Linf | {Linf, L2} | Threat model norm |
| `version` | standard | {standard, plus, rand} | Attack ensemble variant |
| `attacks_to_run` | all | subset of {apgd-ce, apgd-t, fab, square} | Components to execute |
| `n_target_classes` | 9 | [1, 9] | Targeted attack class count |

### 2.5 Standard Epsilon Budgets

| Dataset | Norm | Standard ε | Rationale |
|---------|------|-----------|-----------|
| CIFAR-10 | L∞ | 8/255 | RobustBench standard; Madry et al. convention |
| CIFAR-10 | L2 | 0.5 | RobustBench standard |
| CIFAR-10 | L∞ | 4/255 | Conservative budget for practical threats |
| ImageNet | L∞ | 4/255 | RobustBench standard |

---

## 3. Evaluation Metrics

### 3.1 Primary Metrics

| Metric | Definition | Formula |
|--------|-----------|---------|
| Clean Accuracy | Correct predictions on unperturbed test set | `correct_clean / total` |
| Robust Accuracy | Correct predictions on adversarial examples | `correct_adv / total` |
| Attack Success Rate (ASR) | Fraction of correctly-classified inputs that become misclassified | `(correct_clean - correct_adv) / correct_clean` |
| Accuracy Drop | Absolute difference between clean and robust accuracy | `clean_acc - robust_acc` |

### 3.2 Secondary Metrics

| Metric | Definition | Use Case |
|--------|-----------|----------|
| Average Perturbation (L2) | Mean L2 norm of successful adversarial perturbations | C&W attack quality |
| Median Perturbation (L2) | Median L2 norm of successful perturbations | Robust to outliers |
| Query Count | Number of model queries per successful attack | Black-box efficiency |
| Time per Sample | Wall-clock seconds per adversarial example | Computational cost |
| Certified Radius | Provable L2 radius from randomized smoothing | Certified defense evaluation |

### 3.3 Metric Computation

```python
# Primary metric computation (pseudocode)
clean_correct = sum(model(x) == y for x, y in test_set)
adv_correct = sum(model(attack(x)) == y for x, y in test_set)

clean_accuracy = clean_correct / len(test_set)
robust_accuracy = adv_correct / len(test_set)
attack_success_rate = (clean_correct - adv_correct) / clean_correct
```

### 3.4 CI Gate Thresholds

| Metric | Threshold | Action on Failure |
|--------|-----------|-------------------|
| PGD robust accuracy (ε=8/255) | ≥ 30% | Block merge |
| Clean accuracy | ≥ 85% | Warning |
| Attack success rate | ≤ 70% | Block merge |
| Benchmark completion | 100% (no crashes) | Block merge |

---

## 4. Statistical Significance Testing

### 4.1 Confidence Intervals

All reported accuracy values include 95% Wilson score confidence intervals:

```
CI = (p̂ + z²/2n ± z√(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)
```

Where:
- p̂ = observed accuracy
- n = test set size (10,000 for CIFAR-10)
- z = 1.96 (95% confidence)

For n=10,000 and p̂=0.50, the 95% CI width is approximately ±0.98%.

### 4.2 Random Seed Protocol

Each benchmark run uses three fixed random seeds to account for:
- Random initialization in PGD (affects which local optima are found)
- Stochastic components in AutoAttack (Square Attack queries)
- Data loader shuffling

**Required seeds**: `[42, 123, 2024]`

Report the mean and standard deviation across seeds. If std > 1% accuracy, flag
as high-variance result requiring investigation.

### 4.3 Comparison Testing

When comparing two models or two attack configurations:

1. **McNemar's test**: For paired binary outcomes (correct/incorrect per sample)
   - Null hypothesis: Both configurations have the same error rate
   - Use when comparing model A vs. model B on the same test set

2. **Bootstrap confidence intervals**: For reporting uncertainty on accuracy differences
   - 10,000 bootstrap resamples of the test set
   - Report 95% percentile confidence interval on the difference

### 4.4 Multiple Comparisons

When comparing across multiple epsilon values or attack types simultaneously, apply
Bonferroni correction:

```
α_adjusted = 0.05 / k
```

Where k = number of simultaneous comparisons.

---

## 5. Hardware Requirements

### 5.1 Minimum Requirements (CI Pipeline)

| Component | Specification | Notes |
|-----------|--------------|-------|
| GPU | Not required | CI uses CPU with reduced test set |
| CPU | 4 cores | GitHub Actions runner |
| RAM | 8 GB | Sufficient for CIFAR-10 + ResNet-18 |
| Disk | 2 GB free | Dataset + model checkpoints |
| Time budget | 10 minutes max | CI timeout constraint |

### 5.2 Full Benchmark Requirements

| Component | Specification | Notes |
|-----------|--------------|-------|
| GPU | NVIDIA A100 (40GB) or RTX 4090 (24GB) | Required for AutoAttack |
| CPU | 8+ cores | Data loading parallelism |
| RAM | 32 GB | Full test set in memory |
| Disk | 10 GB free | Datasets + multiple checkpoints |
| Time budget | ~2 hours | Full AutoAttack on 10k samples |

### 5.3 Benchmark Scaling

| Configuration | Samples | GPU Time (A100) | GPU Time (RTX 4090) |
|---------------|---------|-----------------|---------------------|
| FGSM only | 10,000 | ~30 seconds | ~45 seconds |
| PGD-20 | 10,000 | ~5 minutes | ~8 minutes |
| PGD-100 | 10,000 | ~25 minutes | ~40 minutes |
| C&W (1000 iter) | 10,000 | ~45 minutes | ~70 minutes |
| AutoAttack (standard) | 10,000 | ~90 minutes | ~140 minutes |

### 5.4 Memory Usage

| Model | Batch Size | GPU Memory |
|-------|-----------|------------|
| ResNet-18 | 128 | ~4 GB |
| ResNet-18 | 256 | ~7 GB |
| WideResNet-28-10 | 64 | ~8 GB |
| WideResNet-28-10 | 128 | ~14 GB |
| WideResNet-70-16 | 32 | ~16 GB |

---

## 6. Reproducibility Checklist

### 6.1 Environment

- [ ] Python version pinned (≥3.10, specify exact in results)
- [ ] PyTorch version pinned (specify CUDA version if GPU)
- [ ] All dependency versions recorded (`pip freeze` or `uv.lock`)
- [ ] Hardware specification documented (GPU model, driver version)
- [ ] Operating system documented

### 6.2 Data

- [ ] CIFAR-10 test set used without modification (10,000 images)
- [ ] No data augmentation applied to test inputs
- [ ] Data normalization matches training normalization
- [ ] Dataset download verified via SHA-256 hash
- [ ] Subset indices documented if not using full test set

### 6.3 Model

- [ ] Model architecture fully specified (layer count, width, activations)
- [ ] Training procedure documented (epochs, optimizer, learning rate schedule)
- [ ] Model checkpoint hash recorded (SHA-256)
- [ ] Inference mode enabled (`model.eval()`, `torch.no_grad()`)
- [ ] Batch normalization in eval mode (not tracking running stats)

### 6.4 Attack Configuration

- [ ] All attack hyperparameters documented (see Section 2)
- [ ] Random seeds fixed and reported (see Section 4.2)
- [ ] Loss function specified
- [ ] Perturbation constraints verified (pixel values clipped to [0,1])
- [ ] Number of restarts documented
- [ ] Early stopping criteria documented (if applicable)

### 6.5 Evaluation

- [ ] Full test set used (or subset clearly documented with indices)
- [ ] Confidence intervals reported (see Section 4.1)
- [ ] Variance across seeds reported
- [ ] Wall-clock time recorded
- [ ] JSON results file includes all parameters and environment info

### 6.6 Reporting

- [ ] Results JSON schema validated
- [ ] Clean accuracy matches expected range for model
- [ ] Robust accuracy monotonically decreases with increasing ε
- [ ] Results compared against RobustBench leaderboard (if applicable)
- [ ] Any deviations from standard protocol explicitly noted

---

## 7. Benchmark Execution Protocol

### 7.1 Standard Run

```bash
# 1. Set up environment
pip install -e ".[dev]"

# 2. Run full benchmark
python -m adv_lab.eval.benchmark_runner \
    --epsilon 0.031 \
    --pgd-steps 20 \
    --seeds 42 123 2024 \
    --output results/benchmark.json

# 3. Validate results
python -m adv_lab.eval.benchmark_runner --validate results/benchmark.json
```

### 7.2 CI Run (Reduced)

```bash
# Reduced test set for CI time budget
python -m adv_lab.eval.benchmark_runner \
    --epsilon 0.031 \
    --pgd-steps 20 \
    --max-samples 1000 \
    --output results/ci_benchmark.json
```

### 7.3 Result Schema

```json
{
  "metadata": {
    "timestamp": "2026-08-24T12:00:00Z",
    "model": "ResNet-18",
    "dataset": "CIFAR-10",
    "device": "cuda:0 (NVIDIA A100)",
    "torch_version": "2.13.0",
    "python_version": "3.12.4",
    "seeds": [42, 123, 2024]
  },
  "results": {
    "clean_accuracy": 0.9512,
    "attacks": {
      "fgsm": {
        "epsilon": 0.031,
        "robust_accuracy": 0.4230,
        "attack_success_rate": 0.5553,
        "time_seconds": 28.4
      },
      "pgd": {
        "epsilon": 0.031,
        "steps": 20,
        "step_size": 0.0078,
        "robust_accuracy": 0.3150,
        "attack_success_rate": 0.6689,
        "time_seconds": 312.7
      }
    }
  },
  "ci_gate": {
    "passed": true,
    "pgd_robust_accuracy": 0.3150,
    "threshold": 0.30
  }
}
```

---

## 8. Known Limitations

1. **Adaptive attacks**: This benchmark uses fixed attack parameters. A truly adaptive
   adversary may tune attacks specifically to bypass a given defense. AutoAttack
   partially addresses this but is not exhaustive.

2. **Computational budget**: Full AutoAttack evaluation on the complete test set requires
   GPU resources. CI uses reduced samples, which increases confidence interval width.

3. **Distribution shift**: Robustness measured on CIFAR-10 test set does not guarantee
   robustness on out-of-distribution inputs or real-world images.

4. **Gradient masking**: Some defenses cause gradient masking, making gradient-based
   attacks appear ineffective while the model remains vulnerable to transfer attacks or
   black-box methods. AutoAttack's inclusion of Square Attack partially detects this.

5. **Single model evaluation**: Results are for a specific trained model instance.
   Different random seeds during training produce models with varying robustness.

---

## 9. References

1. Croce, F. & Hein, M. (2020). Reliable evaluation of adversarial robustness with
   an ensemble of attacks. *ICML 2020*. (AutoAttack standard protocol)
2. RobustBench Leaderboard. https://robustbench.github.io/
3. Madry, A., et al. (2018). Towards Deep Learning Models Resistant to Adversarial
   Attacks. *ICLR 2018*. (PGD evaluation standard)
4. Carlini, N., et al. (2019). On Evaluating Adversarial Robustness. *arXiv:1902.06705*.
   (Best practices for robustness evaluation)
