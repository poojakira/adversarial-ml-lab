# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
