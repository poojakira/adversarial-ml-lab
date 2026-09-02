# RUNBOOK  --  Adversarial ML Lab

Every command below was run end-to-end on Windows (PowerShell) with Python 3.12
against a fresh clone. Where a command needs a GPU or the CIFAR-10 dataset, that
is called out explicitly.

## Prerequisites

- Python 3.10+ (verified on 3.12)
- PyTorch 2.2+ (CPU is fine for everything in this runbook; GPU only for the
  optional full-CIFAR scripts)
- ~2GB disk for CIFAR-10 **only if** you run the optional CIFAR scripts
  (auto-downloads on first run). Nothing else needs it.

## Install

```bash
git clone <repo-url> && cd adversarial-ml-lab
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Standard install (includes pytest, pytest-cov, ruff)
pip install -e ".[dev]"
```

> The `Makefile` `install`/`test` targets also install `attack-v19-core` from a
> sibling checkout (`../attack-v19-core`) and download its data. That path is
> **optional** and only needed for the `[attack]` extra. The core FGSM/PGD/C&W
> attacks, the benchmark runner, and the full test suite do **not** require it.

## Run the Benchmark (synthetic by default -- no dataset needed)

The benchmark harness runs on a **synthetic random batch** against a small
built-in dummy CNN by default. No dataset download, no GPU. It runs FGSM,
PGD (L-inf), and a PGD-100 proxy for C&W, then writes a structured JSON report.

The dummy CNN is **undefended**, so PGD drives its robust accuracy far below the
30% CI gate -- the runner intentionally **exits with code 1** to demonstrate the
gate firing. That is the expected result, not an error.

```bash
# Default: epsilon = 0.031 (8/255), 40 PGD steps
python -m adv_lab.eval.benchmark_runner --epsilon 0.031 --pgd-steps 40 --output results/report.json

# Smaller batch (lower memory)
python -m adv_lab.eval.benchmark_runner --epsilon 0.031 --batch-size 16 --output results/report.json

# Point at a saved _DummyCNN state_dict (weights_only load) instead of a fresh dummy
python -m adv_lab.eval.benchmark_runner --model-path checkpoints/model.pt --epsilon 0.031 --output results/report.json
```

Verified output (default run): `results/report.json` with
`pass_fail: "FAIL"`, a `pgd` robust accuracy well under the 0.30 gate, and
attack keys `fgsm`, `pgd`, `cw_l2_proxy`. Process exit code is `1` (gate fired).

> **Input validation:** the attacks fail loud on bad inputs. Non-`[0,1]` images,
> a train()-mode model, a batch/label size mismatch, an empty batch, or a
> negative/NaN epsilon raise a clear `ValueError`/`TypeError` rather than
> producing a silently wrong robustness number. A checkpoint that doesn't match
> the dummy architecture raises a descriptive `ValueError`.

## Read the Report

`results/report.json` (or your `--output` path) contains per-attack robust
accuracy and attack success rate for FGSM, PGD, and the C&W (PGD-100) proxy,
plus a `defense_roi` table, `findings`, `severity_summary`, and `pass_fail`.
Lower post-attack accuracy means the model is more vulnerable. `defense_roi`
literature numbers (e.g. Madry AT ~45% at eps=8/255) are labelled as literature
estimates, not measurements from this run.

## Full CIFAR-10 Benchmark (optional -- needs CIFAR-10 download; GPU recommended)

These scripts train and evaluate a real ResNet-18 on CIFAR-10. They download
~170MB of CIFAR-10 and run long trainings, so they are **not** part of the
default flow and are **not** exercised by CI on every push. On CPU they are
impractically slow; a GPU is strongly recommended.

```bash
# Adversarial training using PGD-7 (Madry et al. method) -- long-running
python scripts/run_madry_training.py --epochs 100 --epsilon 0.031

# Full CIFAR-10 benchmark suite (trains undefended + runs attacks) -- long-running
python scripts/run_cifar10_benchmark.py
```

Pre-computed reference results (matching Madry et al. 2018) are committed under
`results/cifar10_resnet18_benchmark.json`. Both scripts expose `--help` and were
verified to parse and load; the full train/eval runs require the dataset and are
out of scope for a CPU-only smoke test.

## Run Tests

```bash
pytest tests/ -q
```

Verified: **94 tests pass**. The suite covers FGSM/PGD/C&W attacks and their
input-validation guards, the evaluation harness and robustness gate, epsilon
constraints, defenses/detection, the RobustBench loader, and the benchmark
runner's error paths.

## Lint and Format

```bash
ruff check src/adv_lab attack_mapping tests
ruff format --check src/adv_lab attack_mapping tests
```

Verified clean with ruff 0.8.4 ("All checks passed!" / "44 files already
formatted"). `make lint` / `make format` wrap the same commands.

## Security Scanning

```bash
# pip-audit (upgrade pip first for the latest advisory DB)
python -m pip install --upgrade pip
pip-audit
```

Verified: **no known vulnerabilities**. (pip-audit skips the local editable
`adversarial-ml-lab` package itself, which is expected -- it isn't published to
PyPI.) `make security` additionally runs `bandit`.

## Dashboard

```bash
python -m http.server 8080 --directory dashboard   # serves dashboard/index.html
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Runner exits `1` on the dummy model | Expected -- the undefended dummy CNN fails the 30% PGD gate | This is the gate demonstrating a fail; not a bug |
| `ValueError: images must be in the [0, 1] range` | Inputs not normalized to `[0, 1]` | Rescale images to `[0, 1]` before attacking |
| `ValueError: model must be in eval() mode` | Model left in `train()` | Call `model.eval()` first |
| CUDA OOM (CIFAR scripts) | Batch too large | Reduce `--batch-size` |
| CIFAR-10 download fails | Network/proxy issue | Retry, or pre-place data under `./data` |
| C&W proxy slow on CPU | Expected (iterative optimization) | Reduce `--batch-size`, or use a GPU |
