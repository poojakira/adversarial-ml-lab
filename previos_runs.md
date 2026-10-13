# Historical runs

The following sections were moved verbatim from the starting README during the 2026-10-13 audit. Confidence: partially verified as committed documentation; underlying run logs were not independently recovered. Missing commands, environments, dates or commits remain unknown unless included in each entry. None is a current execution claim.

## Verified Snapshot

Reproduced on current `main` (Python 3.12, CPU). Real results are transcribed from committed `results/*.json`.

| Metric | Verified snapshot value |
|---|---:|
| Tests | 112 passing, 0 skipped |
| Statement coverage | 32.22% (CI gate 15%); core attacks fgsm 71% / pgd 97% / cw 98% |
| Attacks | FGSM (L∞), PGD-20 (L∞), C&W (L2) |
| Committed CIFAR-10 result | 71.82% clean → 0.00% robust under PGD @ ε=8/255 (1024-sample subset) |
| CI robustness gate | PGD robust accuracy ≥ 30% @ ε=8/255 (HIGH/CRITICAL finding otherwise) |


## Verification update — 2026-09-30

- **Scope:** Account-wide `poojakira` repository pass covering source/configuration, CI/release workflows, security-hygiene gates, dependency/SAST controls, and documentation consistency.
- **Remediation:** Ran the safe Ruff repair workflow, corrected the import/format gate, and pinned release/container/CodeQL/artifact actions to immutable revisions.
- **Verification state:** CI, Build and Security Gate, Security Hygiene, and Documentation Integrity completed successfully after the fixes; the evaluation-container workflow was still running at the audit snapshot.
- **Security note:** Robustness benchmarks remain evaluation evidence, not a claim of production robustness.
- **Evidence boundary:** This update records repository and GitHub Actions evidence observed during the pass. It is not a claim of independent penetration testing, production deployment, or zero residual risk.
<!-- repo-verification:end -->


## Verification checkpoint — 2026-09-30

- **Snapshot commit:** `95d9c097efc63ad4e8aed48cb85bc599a2de40cc`
- **Status:** PARTIALLY VERIFIED
- **Evidence:** Security Hygiene, Documentation Integrity, and Build and Security Gate passed. CI and the evaluation-container workflow were still running at the verification snapshot.
- This checkpoint is intentionally date-bounded. It does not claim zero vulnerabilities or universal production readiness.



### Current Coverage

The cited verification snapshot records **112 passing tests** and **32.22% overall line coverage**. Use the repository's latest successful CI run to determine whether later commits preserve or change those values. The coverage profile is
dominated by ~20 advanced/experimental attack modules that are intentionally
lightly tested. The modules that matter for the core robustness story are
covered well:

| Module | Coverage |
|--------|---------:|
| `attacks/fgsm.py` (FGSM + shared input validation) | 71% |
| `attacks/pgd.py` (PGD L-inf + L2) | 97% |
| `attacks/cw.py` (C&W L2) | 98% |
| `eval/benchmark_runner.py` (runner + error paths) | 69% |
| `defenses/detection.py` | 99% |
| `defenses/adversarial_training.py` | 100% |

The CI gate floor is 15%.

**Tested modules:**
- `src/adv_lab/attacks/fgsm.py`  -  FGSM + input validation (`test_attacks.py`, `test_input_validation.py`)
- `src/adv_lab/attacks/pgd.py`  -  PGD L-inf/L2 + guards (`test_attacks.py`, `test_input_validation.py`)
- `src/adv_lab/attacks/cw.py`  -  C&W L2 + guards (`test_attacks.py`, `test_input_validation.py`)
- `src/adv_lab/eval/harness.py`  -  Benchmark harness, CI gate, JSON export (`test_eval.py`)
- `src/adv_lab/eval/benchmark_runner.py`  -  Benchmark runner, severity mapping, error paths (`test_eval.py`, `test_input_validation.py`)
- `src/adv_lab/defenses/adversarial_training.py`  -  AdversarialTrainer epoch & evaluation (`test_defenses.py`)
- `src/adv_lab/defenses/detection.py`  -  STRIPDetector, NeuralCleanse, bypass methods (`test_defenses.py`)
- `src/adv_lab/models/cifar10_resnet18.py`  -  ResNet-18 instantiation, forward pass, gradients (`test_models.py`)
- `benchmark/robustbench_baseline.py`  -  RobustBench comparison (`test_robustbench.py`)
- Epsilon constraint validation (`test_epsilon_constraints.py`)

**Lightly tested (advanced/experimental attack modules + eval utilities):**
- `attacks/`: adaptive, api_sim, blackbox, chaining, constrained, ensemble, evasion, inference, inversion, llm, model_stealing, non_classification, norms, param_search, physical, poisoning, universal
- `eval/`: certified, ci_signing, robustbench_loader, transferability
- `attack_mapping/`: enricher, reporter (requires external `attack_core` package)

Contributions to expand test coverage are welcome. See [CONTRIBUTING](CONTRIBUTING.md).


### Real measured results (committed artifact)

The numbers below are **REAL and reproducible**. They come from an actual run
executed on CPU-only PyTorch and committed to
[`results/cifar10_smallcnn_real.json`](results/cifar10_smallcnn_real.json).
This is a **small CPU compute budget, NOT a state-of-the-art result** — a
compact CNN (~1.1M params) trained for 6 epochs on real CIFAR-10
(`torchvision.datasets.CIFAR10`, `download=True`).

| Attack | Epsilon | Clean Acc | Robust Acc | Samples |
|--------|---------|-----------|------------|---------|
| FGSM      | 8/255 | 71.82% | 3.32% | 1024 |
| PGD-20    | 8/255 | 71.82% | 0.00% | 1024 |
| C&W (L2)  | c=1.0 | 71.82% | 4.20% | 1024 |

- Clean accuracy is measured on the full 10,000-sample CIFAR-10 test set.
- Robust accuracy for the iterative/optimization attacks is measured on a 1024-sample
  subset (a compute-budget choice, labeled in the artifact).
- Seed = 42, `torch 2.14.0+cpu`, `cuda_available=false`. Wall clock ≈ 440 s.

Reproduce exactly:

```bash
python scripts/run_real_smallcnn_benchmark.py --epochs 6 --attack-samples 1000
```

**Interpretation:** an undefended model with 71.82% clean accuracy collapses to
near-0% under PGD at eps=8/255. This is the whole point — the clean/robust gap
is enormous for models that were not explicitly trained for robustness. The
absolute clean accuracy is low *only because the compute budget is small*; the
qualitative robustness collapse is the same one seen on larger models.

> **Literature context (NOT measured here):** For a fully-trained ResNet-18 at
> ~93% clean accuracy, published work (Madry et al. 2018) reports PGD robust
> accuracy near 0% undefended and ~45% after adversarial training. Those
> projected figures live in
> [`results/cifar10_resnet18_benchmark.json`](results/cifar10_resnet18_benchmark.json),
> which is explicitly labeled as a literature projection, not a measurement from
> this repo. Do not cite them as reproducible results of this repository.
