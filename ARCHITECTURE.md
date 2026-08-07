# Architecture & Module Status

## Overview

This repository demonstrates adversarial ML attack implementations for educational purposes. Only 3 core attack modules have test coverage and are validated against expected behavior. The remaining modules are experimental code without test coverage or validation.

**Use [IBM Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) for production adversarial ML work.**

---

## Core Attacks (Tested)

These modules have unit tests (`tests/test_attacks.py`) verifying correct output shape, perturbation bounds, and non-trivial perturbation generation on a dummy model.

| Module | Attack | File | Tests |
|--------|--------|------|-------|
| FGSM | Fast Gradient Sign Method (Goodfellow et al. 2015) | `src/adv_lab/attacks/fgsm.py` | ✅ `test_fgsm_attack`, `test_eval_harness_fgsm` |
| PGD | Projected Gradient Descent (Madry et al. 2018) | `src/adv_lab/attacks/pgd.py` | ✅ `test_pgd_attack`, `test_eval_harness_pgd` |
| C&W | Carlini-Wagner L2 (Carlini & Wagner 2017) | `src/adv_lab/attacks/cw.py` | ✅ `test_cw_attack` |

The evaluation harness (`src/adv_lab/eval/harness.py`) is also tested with `test_eval_harness_clean`, `test_robustness_gate_pass`, `test_robustness_gate_fail`, and `test_model_training_mode_raises`.

---

## Experimental / Untested Modules

⚠️ **These modules have NO test coverage.** They compile and export symbols but have not been validated against real model weights or known-good results. Treat them as reference implementations only.

| Module | File | Description |
|--------|------|-------------|
| adaptive | `attacks/adaptive.py` | BPDA, EoT, gradient masking detection |
| api_sim | `attacks/api_sim.py` | Simulated API attack with anomaly evasion |
| blackbox | `attacks/blackbox.py` | SimBA, Square Attack, HopSkipJump, Boundary Attack |
| chaining | `attacks/chaining.py` | Multi-step perturbation chains |
| constrained | `attacks/constrained.py` | Query-budget and time-limited attacks |
| ensemble | `attacks/ensemble.py` | Multi-model ensemble attacks |
| evasion | `attacks/evasion.py` | JPEG-robust, feature-squeeze-robust attacks |
| inference | `attacks/inference.py` | Watermark flip, prediction poisoning |
| inversion | `attacks/inversion.py` | Gradient inversion, GAN inversion, membership inference |
| llm | `attacks/llm.py` | GCG, AutoDAN, prompt injection (simulated) |
| model_stealing | `attacks/model_stealing.py` | Jacobian-based model extraction |
| non_classification | `attacks/non_classification.py` | Attacks on detection, segmentation, RL, recommenders |
| norms | `attacks/norms.py` | L0, L1, Wasserstein, semantic, patch attacks |
| param_search | `attacks/param_search.py` | Bayesian attack hyperparameter optimization |
| physical | `attacks/physical.py` | Physical-world adversarial patches |
| poisoning | `attacks/poisoning.py` | BadNets, clean-label, spectral, weight poisoning |
| universal | `attacks/universal.py` | Universal Adversarial Perturbations (UAPs) |

---

## What Would Be Required to Validate Each

To move any experimental module to "tested" status:

1. **Real model weights** — Each attack needs a trained model to attack. For CIFAR-10 attacks, this means either training a ResNet-18 (~2h on a T4 GPU) or downloading pretrained weights from RobustBench.

2. **GPU compute** — Most attacks beyond FGSM require iterative optimization. A single validation run across all modules would require ~4-8 GPU-hours on a T4/V100.

3. **Ground-truth expectations** — Each attack needs a known-good success rate from published literature to compare against. Without this, you can't tell if the attack code is correct or just producing noise.

4. **Per-module test suite** — Each module needs tests that verify:
   - Output tensor shape matches input
   - Perturbation respects norm bounds (epsilon)
   - Attack actually degrades model accuracy (not just random noise)
   - Edge cases: zero-epsilon, single-sample batch, already-misclassified inputs

5. **LLM module special case** — The `llm.py` module uses simulated tokenizers/models. Validating it would require access to actual LLM APIs or local model weights (7B+ parameters), which is outside the scope of a CIFAR-10 focused lab.

---

## Results / Benchmarks

All results in `results/` are **projected from published literature**, not measured by running code in this repository. The JSON files are clearly marked with `"_synthetic": true` metadata.

To produce real results:
1. Train or download model weights
2. Run `scripts/run_cifar10_benchmark.py` (requires GPU)
3. Compare outputs against literature values

---

## Honest Assessment

- The 3 core attacks (FGSM, PGD, C&W) are well-implemented, concise, and tested.
- The 17 experimental modules represent breadth-of-concept, not production quality.
- No model weights are committed; no benchmark was actually run in this repo.
- This is an **educational lab**, not a security tool. Use IBM ART for real work.
