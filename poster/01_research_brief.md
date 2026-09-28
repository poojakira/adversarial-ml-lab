# Research Brief — Poster 06

> Evidence status: This is a dated repository snapshot at the commit identified below. `VERIFIED_AT_SNAPSHOT` means verified for that commit and environment; it does not assert the same result on the latest `main`. Compare newer claims with the repository evidence before reuse.

## Repository
`github.com/poojakira/adversarial-ml-lab` (public, default branch `main`, primary language Python). MIT • Python 3.12 • HEAD cd7547d • verified 2026-09-26

## Academic Project Title
**Measuring Neural-Network Robustness Under Adversarial Perturbation**

### Subtitle
Experimental Evaluation of Model Behavior Under Gradient-Based Attacks

## One-Sentence Contribution
A gradient-attack evaluation harness (FGSM/PGD/C&W) that measures robustness collapse on real pretrained weights and enforces a hard separation between measured results and literature projections — the projected file is explicitly flagged _synthetic and not reproducible.

## Problem Statement
Small L-infinity perturbations flip confident model predictions. Reporting robustness is easy to fake: a single-step FGSM can look 'robust' due to gradient masking, and literature values can be pasted in as if measured. This lab separates REAL measured runs from clearly-labeled literature projections.

## Threat Model
Chain: CLEAN SAMPLE -> ATTACK GENERATION -> PERTURBATION ε=8/255 -> EVALUATION BOUNDARY -> ROBUST ACCURACY.
Adversary capability: white-box gradient access, ε-bounded; Assumptions: fixed ε, seed 42; measured on real weights; Out of scope: black-box transfer; certified robustness; full test-set (subset); Residual risk: subset variance; domain mismatch in one run.

## Research / Engineering Question
> How does a model's accuracy collapse under FGSM/PGD/C&W attacks — and are the reported numbers actually measured, not projected from literature?

## Objective
Measure real robustness collapse under gradient attacks and keep measured results strictly separate from literature projections.

## Engineering Sub-Objectives
O1 — FGSM / PGD / C&W attack impls
O2 — Measured runs on real weights
O3 — Robust-accuracy metrics + timing
O4 — Label projected numbers as projected

## Methodology
1 Load (real weights) -> 2 Clean (predict) -> 3 FGSM (ε=8/255) -> 4 PGD-20 (α=2/255) -> 5 Measure (robust acc) -> 6·7 Time + log (JSON)

## Evidence at Poster Snapshot + Claim Ledger
- **VERIFIED_AT_SNAPSHOT** — FGSM & PGD-20 100% attack success on tested subset — results/robustbench_real.json (measured, real torchvision weights, CPU). success_rate=1.0 both.
- **VERIFIED_AT_SNAPSHOT** — Measured config eps=8/255, PGD-20 alpha=2/255 — robustbench_real.json attack_config; PyTorch 2.3.0+cpu.
- **VERIFIED_HISTORICAL / PROJECTED** — Undefended PGD ~0%, Madry-AT ~45% — cifar10_resnet18_benchmark.json is explicitly _synthetic:true LITERATURE_PROJECTION (Madry 2018). Shown ONLY as labeled projection.
- **PARTIAL** — Clean accuracy 5% on subset — robustbench_real.json; low due to ImageNet->CIFAR domain mismatch + 20-image subset. Not full test set.
- **UNSUPPORTED (disclaimed)** — State-of-the-art / certified robustness — README robustbench_context notes Madry ~45% is baseline not SOTA; no certification.

## Important Negative / Honest Results
See RESULTS panel: Real measured run; clean acc low due to ImageNet→CIFAR domain mismatch on the subset. Not full-test-set.

## Limitations
1. Measured run uses a small subset, not full test set.
2. Clean accuracy low (ImageNet→CIFAR domain mismatch).
3. Literature file is projection, not reproducible.
4. White-box only; no black-box/transfer attacks.
5. No certified-robustness claim.

## Future Work
• Full 10k CIFAR-10 test-set measured run.
• CIFAR-trained model (remove domain mismatch).
• Madry adversarial-training measured baseline.
• AutoAttack ensemble evaluation.
• Certified-robustness comparison.

## Reproducibility
```
python benchmark/robustbench_baseline.py
pytest tests/
```
Evidence: results/robustbench_real.json (measured), results/cifar10_resnet18_benchmark.json (projected)

## References
[1] Goodfellow et al. (2015) FGSM · [2] Madry et al. (2018) PGD, ICLR · [3] Carlini & Wagner (2017) · [4] Croce & Hein (2020) AutoAttack · [5] RobustBench · [6] MITRE ATLAS AML.T0043
