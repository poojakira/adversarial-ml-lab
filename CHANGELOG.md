# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-07-13

Full 20-tier adversarial ML attack surface framework, expanding from 3 white-box
attacks to comprehensive coverage of real-world ML threats.

### Added

#### Attack Modules (17 new, 20 total)
- **Black-box attacks** (`attacks/blackbox.py`): SimBA, Square Attack, HopSkipJump,
  and Boundary Attack -- query-only adversarial examples without gradient access.
- **Model stealing** (`attacks/model_stealing.py`): Model extraction via query
  synthesis and knockoff nets with fidelity/accuracy metrics.
- **Full norm suite** (`attacks/norms.py`): L0, L1, L2, L-inf, Wasserstein,
  semantic, and patch-based perturbation attacks under a unified interface.
- **LLM attacks** (`attacks/llm.py`): GCG (Greedy Coordinate Gradient), AutoDAN,
  prompt injection, embedding attacks, token substitution, and universal suffix
  generation for language model adversarial testing.
- **Poisoning & backdoors** (`attacks/poisoning.py`): BadNets trigger injection,
  clean-label poisoning, and gradient-based sample selection.
- **Defense-aware adaptation** (`attacks/adaptive.py`): BPDA (Backward Pass
  Differentiable Approximation) and Expectation over Transformations (EoT) for
  bypassing obfuscated-gradient defenses.
- **Adaptive parameter search** (`attacks/param_search.py`): Bayesian optimization
  for automatic attack hyperparameter tuning.
- **Resource-constrained attacks** (`attacks/constrained.py`): Query-budgeted and
  compute-limited attack strategies for realistic threat modeling.
- **Post-processing evasion** (`attacks/evasion.py`): Attacks that survive JPEG
  compression, feature squeezing, and other input transformations.
- **Ensemble attacks** (`attacks/ensemble.py`): Simultaneous optimization across
  multiple model architectures for transferable adversarial examples.
- **Inference-time manipulation** (`attacks/inference.py`): Batch poisoning, timing
  side-channel attacks, and inference pipeline exploitation.
- **Perturbation chaining** (`attacks/chaining.py`): Multi-stage attack pipelines
  composing different perturbation types sequentially.
- **API behavior simulation** (`attacks/api_sim.py`): Rate-limited query
  simulation, response analysis, and realistic API interaction modeling.
- **Non-classification targets** (`attacks/non_classification.py`): Adversarial
  attacks against object detection, semantic segmentation, regression, RL
  policies, and recommendation systems.
- **Model inversion** (`attacks/inversion.py`): Privacy attacks including model
  inversion (reconstructing training data) and membership inference.
- **Physical-world patches** (`attacks/physical.py`): Printable adversarial patches
  with Expectation over Transformations (EOT) for physical robustness.
- **Universal perturbations** (`attacks/universal.py`): Image-agnostic universal
  adversarial perturbations (UAPs) that transfer across inputs.

#### Defense Modules (1 new, 2 total)
- **Detection** (`defenses/detection.py`): NeuralCleanse backdoor detection, STRIP
  perturbation-based detection, and spectral signature analysis.

#### Evaluation Modules (3 new, 4 total)
- **Transferability analysis** (`eval/transferability.py`): Cross-architecture
  transferability measurement across 4 model families.
- **Certified evaluation** (`eval/certified.py`): Certified defense evaluation via
  randomized smoothing, Lipschitz bounds, and Interval Bound Propagation (IBP).
- **CI gate signing** (`eval/ci_signing.py`): HMAC-based signing of benchmark
  results for tamper-proof CI gate integration.

#### Harness Enhancements
- `eval/harness.py` now integrates HMAC signing via `ci_signing.py` for
  tamper-proof result verification in CI pipelines.

#### Tests (24 test files, 80+ tests)
- Expanded from 4 test files (13 tests) to 24 test files covering all tiers:
  `test_fgsm.py`, `test_pgd.py`, `test_cw.py`, `test_harness.py`,
  `test_blackbox.py`, `test_model_stealing.py`, `test_norms.py`, `test_llm.py`,
  `test_poisoning.py`, `test_adaptive.py`, `test_param_search.py`,
  `test_constrained.py`, `test_evasion.py`, `test_ensemble.py`,
  `test_inference.py`, `test_chaining.py`, `test_api_sim.py`,
  `test_non_classification.py`, `test_certified.py`, `test_ci_signing.py`,
  `test_inversion.py`, `test_physical.py`, `test_universal.py`,
  `test_transferability.py`.

#### Documentation
- **Technical Report** (`docs/TECHNICAL_REPORT.md`): Comprehensive 20-tier
  attack/defense analysis with implementation details and evaluation methodology.
- **Executive Report** (`docs/EXECUTIVE_REPORT.md`): CISO/board-level adversarial
  ML risk posture assessment with strategic recommendations.

---

## [0.1.0] - 2026-07-11

Initial release: the full white-box attack ladder, PGD adversarial training, and
a CI-gateable benchmark harness.

### Added
- **FGSM** (`attacks/fgsm.py`): single-step `x + eps*sign(grad_x L)` with `[0,1]`
  clamping, an eps=0 no-op fast path, and a hard `eval()`-mode guard. `batch_fgsm`
  reports attack success rate over a dataloader (measured only on inputs that
  were correct on clean data).
- **PGD** (`attacks/pgd.py`): `pgd_attack` (L-inf) with random start and per-step
  epsilon-ball projection, a `pgd_linf` alias, and `pgd_l2` with true L2-ball
  projection (unit-gradient steps + norm rescaling). Shipping both norms is
  deliberate -- it exposes the open multi-norm / union-robustness gap.
- **Carlini & Wagner L2** (`attacks/cw.py`): tanh change of variables
  (`x = 0.5*(tanh(w)+1)`), margin objective `f(x)=max(max_{i!=t} Z_i - Z_t, -kappa)`,
  and Adam optimization. Fixed `c` (binary search left as an optional refinement).
- **Adversarial training** (`defenses/adversarial_training.py`):
  `AdversarialTrainer` with `train_epoch` -> `{loss, clean_acc, robust_acc}` and
  `evaluate_robust`. Uses PGD-7 as the inner attack by default; flips the model
  to `eval()` during attack generation and restores `train()`.
- **Benchmark harness** (`eval/harness.py`): `BenchmarkResult` dataclass,
  `run_benchmark` (clean + FGSM/PGD/C&W robust accuracy), `export_json`
  (CI-consumable payload), and a self-contained `adv-eval` CLI that trains and
  benchmarks a demo model with no dataset download. Gate: `passed =
  robust_accuracy_pgd > 0.3`; the process also exits non-zero on failure.
- **Tests**: 13 total (4 FGSM, 4 PGD, 3 C&W, 2 harness).
- **CI**: GitHub Actions workflow running the suite on Python 3.12 and executing
  the benchmark as a gate.
- Packaging via `pyproject.toml` with the `adv-eval` console script.

### Notes
- Attacks require `model.eval()` and raise `ValueError` otherwise -- stochastic
  BatchNorm/Dropout gradients are a well-known source of bogus robustness numbers.
