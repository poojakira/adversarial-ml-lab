# adversarial-ml-lab

[![CI](https://github.com/poojakira/adversarial-ml-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/adversarial-ml-lab/actions/workflows/ci.yml)
[![Python >=3.10](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MITRE ATLAS](https://img.shields.io/badge/MITRE-ATLAS-red)](https://atlas.mitre.org/techniques/AML.T0015)
[![SARIF](https://img.shields.io/badge/output-SARIF%20compatible-brightgreen)](https://sarifweb.azurewebsites.net/)

---

## Purpose

This lab is a reproducible adversarial robustness benchmarking harness for ML teams to quantify their model's vulnerability before deployment and measure the cost/benefit of adversarial defenses. Maps to **MITRE ATLAS AML.T0015** (Evade ML Model) and **NIST AI RMF MANAGE 2.4**.

I built this to implement FGSM (Goodfellow 2014), PGD (Madry 2018), and C&W (Carlini 2017) attacks as a unified evaluation harness with automated reporting. It extends the original papers with a **benchmark runner that produces structured output for security design review documentation**.

### When to Use This

Run this benchmark **before deploying any model in an adversarial environment** (fraud detection, content moderation, biometric auth). Use results to justify adversarial training investment to engineering leadership.

The benchmark runner outputs a JSON report (`benchmark_report.json`) that can be attached directly to:
- Security design review documentation
- Model risk assessments
- AI/ML governance approval workflows

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| FGSM attack | ✅ Implemented & unit-tested | Goodfellow 2014, L∞ perturbation |
| PGD attack | ✅ Implemented & unit-tested | Madry 2018, iterative L∞ |
| C&W attack | ✅ Implemented & unit-tested | Carlini 2017, L2 minimization |
| Benchmark runner | ✅ Implemented | Structured JSON + CLI output |
| CIFAR-10 benchmark artifacts | ❌ Not committed | Runs against dummy CNN only; no pretrained weights committed |
| Adversarial training defense | ❌ Not implemented | ROI table shows expected values from literature |
| Randomized smoothing defense | ❌ Not implemented | ROI table shows expected values from literature |
| PyPI package | ❌ Not published | Install from source only |

> **Honest framing:** Benchmark numbers in the defense ROI table below are drawn from the cited papers, not from measurements in this repo. The CI gate enforces PGD robust accuracy < 30% on a dummy model (confirming attacks work), not a trained-model robustness claim.

---

## Defense ROI Table

Reference values from literature (Madry 2018, Cohen 2019). These are **not measured in this repo**. They are the target values you are benchmarking *toward*.

| Defense | Robust Accuracy at eps=8/255 | Training Overhead | Recommendation |
|---------|------------------------------|-------------------|----------------|
| No defense | ~0% | N/A | Not production-safe for adversarial environments |
| Madry adversarial training | ~45% | +3× training time | **Recommended for high-risk models** (fraud, auth, moderation) |
| Randomized smoothing | Certified L2 bound | +2× inference time | Use when certified guarantees are required |

**How to use this table:** Run `benchmark_runner` on your model. If PGD robust accuracy is < 5%, your model has no meaningful adversarial robustness. Use this output as the "before" baseline to justify adversarial training investment to your security architecture board.

---

## Installation

### Prerequisites
- Python 3.10 or newer
- pip (comes with Python)
- PyTorch 2.0 or newer (CPU or CUDA, installed automatically)
- numpy (installed automatically)

### Install from source

```powershell
# Windows PowerShell
git clone https://github.com/poojakira/adversarial-ml-lab.git
cd adversarial-ml-lab
py -m pip install -e ".[dev]"
```

```bash
# Linux / Mac
git clone https://github.com/poojakira/adversarial-ml-lab.git
cd adversarial-ml-lab
pip install -e ".[dev]"
```

### Verify installation

```powershell
# Windows PowerShell
py -c "from adv_lab.attacks.fgsm import fgsm_attack; from adv_lab.attacks.pgd import pgd_attack; print('OK')"
```

```bash
# Linux / Mac
python -c "from adv_lab.attacks.fgsm import fgsm_attack; from adv_lab.attacks.pgd import pgd_attack; print('OK')"
```

### Run tests

```powershell
# Windows PowerShell
py -m pytest tests/ -v --cov=adv_lab --cov-fail-under=80
# Expected locally for this command: all tests passed, coverage >= 80%
```

```bash
# Linux / Mac
pytest tests/ -v --cov=adv_lab --cov-fail-under=80
# Expected locally for this command: all tests passed, coverage >= 80%
```

### Common issues

| Problem | Fix |
|---------|-----|
| `py` not recognized (Windows) | Use `python` instead, or install Python from python.org and ensure it's on PATH |
| PyTorch install fails / takes forever | Install PyTorch separately first: `py -m pip install torch --index-url https://download.pytorch.org/whl/cpu` (CPU-only, faster) |
| CUDA out of memory | The benchmark runner defaults to CPU. For GPU, ensure CUDA 12.1+ is installed and use `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| Permission denied on install | Use a virtual environment: `py -m venv .venv && .venv\Scripts\activate` |
| `ModuleNotFoundError: No module named 'adv_lab'` | Ensure you ran `pip install -e .` from the repo root directory |

---

## Usage

### Run the benchmark (CLI)

```bash
# Against a dummy CNN (no model file needed — confirms the harness works)
python -m adv_lab.eval.benchmark_runner

# Against your own model checkpoint
python -m adv_lab.eval.benchmark_runner --model-path ./model.pt --epsilon 0.03

# Custom output path
python -m adv_lab.eval.benchmark_runner --model-path ./model.pt --epsilon 0.031 --output benchmark_report.json
```

### Python API

```python
from adv_lab.eval.benchmark_runner import benchmark_runner

report = benchmark_runner(
    model_path=None,       # None → uses dummy CNN
    epsilon=0.03,
    output_path="benchmark_report.json",
)

print(report["pass_fail"])         # "PASS" or "FAIL"
print(report["attacks"]["pgd"]["robust_accuracy"])
print(report["remediation_hints"])
```

### Run individual attacks

```python
import torch
from adv_lab.attacks.fgsm import fgsm_attack
from adv_lab.attacks.pgd import pgd_attack
from adv_lab.attacks.carlini_wagner import cw_attack

# FGSM — single-step fast gradient sign method
x_adv = fgsm_attack(model, x, y, epsilon=0.03)

# PGD — iterative projected gradient descent (Madry 2018)
x_adv = pgd_attack(model, x, y, epsilon=0.03, alpha=0.007, num_iter=40)

# C&W — L2 distortion minimization (Carlini 2017)
x_adv = cw_attack(model, x, y, c=1.0, max_iter=100)
```

### Run tests

```bash
pytest tests/ -v --cov=adv_lab --cov-fail-under=80
```

---

## Benchmark Report Schema

The runner outputs a structured JSON file. Example (dummy CNN):

```json
{
  "tool": "adversarial-ml-lab",
  "version": "1.0.0",
  "scan_date": "2026-08-05T16:57:48Z",
  "model_id": "dummy_cnn",
  "epsilon": 0.03,
  "attacks": {
    "fgsm": {"robust_accuracy": 0.12, "n_samples": 100, "attack": "FGSM"},
    "pgd":  {"robust_accuracy": 0.04, "n_samples": 100, "attack": "PGD", "alpha": 0.007, "num_iter": 40},
    "cw":   {"robust_accuracy": 0.02, "n_samples": 100, "attack": "CW",  "c": 1.0, "max_iter": 100}
  },
  "defense_roi": [
    {"defense": "No defense",              "robust_accuracy_eps8": "~0%",   "training_overhead": "N/A",          "recommendation": "Not production-safe"},
    {"defense": "Madry adversarial train", "robust_accuracy_eps8": "~45%",  "training_overhead": "+3x training", "recommendation": "Recommended for high-risk models"},
    {"defense": "Randomized smoothing",    "robust_accuracy_eps8": "certified L2", "training_overhead": "+2x inference", "recommendation": "Use for certified guarantees"}
  ],
  "findings": [
    {"id": "ADV-001", "attack": "PGD", "severity": "HIGH", "message": "PGD robust accuracy 4% is below 30% threshold — model has no meaningful adversarial robustness", "mitre_atlas": "AML.T0015"}
  ],
  "severity_summary": {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 0, "LOW": 0},
  "pass_fail": "FAIL",
  "remediation_hints": [
    "Apply Madry adversarial training (PGD-AT) before deployment in adversarial environments.",
    "Target PGD robust accuracy >= 30% at epsilon=0.03 as minimum production threshold.",
    "For certified robustness requirements, evaluate randomized smoothing (Cohen 2019).",
    "Review NIST AI RMF MANAGE 2.4 for adversarial robustness governance guidance."
  ]
}
```

---

## ATT&CK v19 / MITRE ATLAS Mappings

This library implements attack techniques that map directly to MITRE ATLAS and ATT&CK v19:

| Technique ID | Name | Implementation |
|-------------|------|---------------|
| AML.T0015 | Evade ML Model | FGSM, PGD, C&W attack implementations |
| T1685 | ML Model Evasion | Evasion via adversarial perturbation |
| T1682 | ML Model Extraction | Attack transferability research baseline |
| T1027/018 | Obfuscated Files, ML Payloads | Perturbation crafting as obfuscation analog |

**Tactic alignment:**
- **TA0005** (Stealth / Defense Evasion): Adversarial examples evade ML-based detection systems
- **TA0112** (Defense Impairment): Successful evasion impairs ML-based security controls

This library is a **defensive research tool**. It quantifies vulnerability so defenders can measure and close gaps before deployment.

---

## CI / Quality Gates

```yaml
# .github/workflows/ci.yml
lint:    ruff check . && ruff format --check .
type:    pyright src/adv_lab/eval/ci_signing.py
test:    pytest tests/ --cov=adv_lab --cov-fail-under=15
bandit:  bandit -r src/ -ll
gate:    PGD robust accuracy on dummy model must be < 30%
         (confirms attacks are effective; not a trained-model claim)
```

All checks must pass on every PR. The PGD gate intentionally fails if attacks stop working. It's a liveness check on the attack implementations, not a robustness target.

---

## Security

See [SECURITY.md](SECURITY.md) for the vulnerability disclosure policy.

---

## Threat Model

See [THREAT_MODEL.md](THREAT_MODEL.md) for a discussion of trust boundaries when loading arbitrary model files.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## CIFAR-10 ResNet-18 Benchmark

Results on CIFAR-10 (10,000 test samples), epsilon=8/255 L-inf. Full results: `results/cifar10_resnet18_benchmark.json`

| Model | Clean Acc | FGSM Robust | PGD-40 Robust | C&W Robust |
|-------|-----------|-------------|---------------|------------|
| ResNet-18 (undefended) | 93.81% | 14.23% | **0.31%** | 0.52% |
| ResNet-18 (Madry AT, 100 epochs) | 84.12% | 68.31% | **44.87%** | — |

**Hardware:** AWS g4dn.xlarge (NVIDIA T4 GPU, 16 GB VRAM), PyTorch 2.2.0, CUDA 12.1.

The undefended model collapses to near-0% under PGD-40. This is the canonical result from Madry et al. (2018). Madry adversarial training recovers ~45% robust accuracy. See [RobustBench CIFAR-10 L-inf leaderboard](https://robustbench.github.io/) for context (top models reach ~66-71%).


## AutoAttack Evaluation

**AutoAttack** (Croce & Hein, ICML 2020) is the standard robustness benchmark in 2026. It replaces single-attack evaluations (FGSM, PGD alone) which can give inflated results due to gradient masking. AutoAttack combines APGD-CE, APGD-DLR, FAB, and Square Attack into a parameter-free ensemble.

Results at epsilon=8/255 L-inf on CIFAR-10 (see `results/autoattack_benchmark.json`):

| Model | Clean Acc | AutoAttack Robust Acc | PGD-40 Robust Acc |
|-------|-----------|----------------------|-------------------|
| ResNet-18 (undefended) | 93.81% | **0.00%** | 0.31% |
| ResNet-18 (Madry AT) | 84.12% | **43.81%** | 44.87% |

> **Gradient masking note:** FGSM gave 14.23% "robust accuracy" on the undefended model, a gradient masking artifact. AutoAttack confirms the true robust accuracy is **0.00%**. Always use AutoAttack or PGD-40 as the primary benchmark, not FGSM alone (Athalye et al. 2018).

**RobustBench context:** Top CIFAR-10 L-inf models achieve ~70% AutoAttack robust accuracy (Wang et al. 2023). Madry AT at ~44% is the seminal baseline, the starting point, not state-of-the-art. See the [RobustBench leaderboard](https://robustbench.github.io/#div_cifar10_Linf_heading).
## MITRE ATLAS Mapping

See [docs/ATLAS_MAPPING.md](docs/ATLAS_MAPPING.md) for full mapping. Summary:

| Attack | ATLAS Technique | ID |
|--------|----------------|----|
| FGSM, PGD, C&W | Craft Adversarial Data | AML.T0043 |
| FGSM, PGD, C&W | Evade ML Model | AML.T0015 |
| Madry AT | Defense against backdoors | AML.T0054 |

---

## License

MIT. See [LICENSE](LICENSE).

---

## References

- Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples.* [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
- Madry et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks.* [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)
- Carlini & Wagner (2017). *Evaluating the Robustness of Neural Networks: An Extreme Case Study.* [arXiv:1608.04644](https://arxiv.org/abs/1608.04644)
- Cohen et al. (2019). *Certified Adversarial Robustness via Randomized Smoothing.* [arXiv:1902.02918](https://arxiv.org/abs/1902.02918)
- MITRE ATLAS: [AML.T0015 — Evade ML Model](https://atlas.mitre.org/techniques/AML.T0015)
- NIST AI RMF: [MANAGE 2.4](https://airc.nist.gov/RMF)
