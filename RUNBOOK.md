# RUNBOOK  --  Adversarial ML Lab

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ (GPU recommended, CPU supported)
- ~2GB disk for CIFAR-10 dataset (auto-downloads on first run)

## Install

```bash
git clone <repo-url> && cd adversarial-ml-lab
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Standard install (includes pytest, ruff, pytest-cov)
pip install -e ".[dev]"

# Or via Makefile (also installs build, ruff, bandit, pip-audit)
make install
```

## Run Attacks

The benchmark harness uses a dummy CNN by default, or loads a saved model checkpoint.
The entry point is the `adv_lab.eval.benchmark_runner` module.

```bash
# Run the benchmark at epsilon = 8/255 with 40 PGD steps (default)
python -m adv_lab.eval.benchmark_runner --epsilon 0.031 --pgd-steps 40 --output results/report.json

# Reduce batch size if memory-constrained
python -m adv_lab.eval.benchmark_runner --epsilon 0.031 --batch-size 16 --output results/report.json

# Point at a saved model checkpoint instead of using the dummy CNN
python -m adv_lab.eval.benchmark_runner --model-path checkpoints/model.pt --epsilon 0.031 --output results/report.json
```

## Full CIFAR-10 Benchmark (requires GPU)

These scripts train and evaluate a ResNet-18 on CIFAR-10:

```bash
# Adversarial training using PGD-7 (Madry et al. method)
python scripts/run_madry_training.py --epochs 100 --epsilon 0.031

# Full CIFAR-10 benchmark suite (trains undefended + runs attacks)
python scripts/run_cifar10_benchmark.py
```

## Read the Report

`results/report.json` (or whatever path you pass to `--output`) contains clean accuracy,
FGSM accuracy, and PGD accuracy at the requested epsilon, plus per-attack success rates.
Lower post-attack accuracy means the model is more vulnerable.

## Run Tests

```bash
pytest tests/ -v
pytest tests/ -v -k "not slow"  # skip long-running attack tests
```

## Lint and Format

```bash
make lint      # Run ruff checks on src/adv_lab, attack_mapping, tests
make format    # Auto-format with ruff
```

## Security Scanning

```bash
make security  # Runs bandit + pip-audit
```

## Full Verification (lint + test + build + security)

```bash
make verify
```

## Dashboard

```bash
make dashboard  # Serves dashboard/index.html on localhost:8080
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| CUDA OOM | Batch size too large | Reduce with `--batch-size 16` |
| `RuntimeError: CUDA not available` | No GPU or driver mismatch | Use `--device cpu` or fix CUDA install |
| CIFAR-10 download fails | Network/proxy issue | Manually download to `~/.cache/torch/cifar-10-batches-py/` |
| C&W attack very slow | Expected (~10min/batch on CPU) | Use GPU or reduce test set with `--batch-size` |
| NaN in loss | Epsilon too large or LR issue | Reduce epsilon |
