# RUNBOOK — Adversarial ML Lab

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA (GPU recommended, CPU supported)
- ~2GB disk for CIFAR-10 dataset (auto-downloads on first run)

## Install

```bash
git clone <repo-url> && cd adversarial-ml-lab
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# GPU (CUDA 11.8)
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install -e ".[dev,cpu]"
```

## Run Attacks

The benchmark harness trains a small CNN on CIFAR-10, then runs FGSM and PGD
attacks against it. Everything is driven through one runnable module.

```bash
# Run the benchmark at epsilon = 8/255 with 20 PGD steps
python -m adv_lab.eval.benchmark_runner --epsilon 0.031 --pgd-steps 20 --output results/report.json

# Reduce batch size if memory-constrained
python -m adv_lab.eval.benchmark_runner --epsilon 0.031 --batch-size 64 --output results/report.json

# Point at a saved model checkpoint instead of training from scratch
python -m adv_lab.eval.benchmark_runner --model-path checkpoints/model.pt --epsilon 0.031 --output results/report.json
```

## Read the Report

`results/report.json` contains clean accuracy, FGSM accuracy, and PGD accuracy at
the requested epsilon, plus per-attack success rates. Lower post-attack accuracy
means the model is more vulnerable.

## Run Tests

```bash
pytest tests/ -v
pytest tests/ -v -k "not slow"  # skip long-running attack tests
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| CUDA OOM | Batch size too large | Reduce with `--batch-size 32` (default 128) |
| `RuntimeError: CUDA not available` | No GPU or driver mismatch | Use `--device cpu` or fix CUDA install: `nvidia-smi` |
| CIFAR-10 download fails | Network/proxy issue | Manually download to `~/.cache/torch/cifar-10-batches-py/` |
| C&W attack very slow | Expected (~10min/batch on CPU) | Use GPU, reduce `--max-iter`, or reduce test set with `--samples 100` |
| NaN in loss | Epsilon too large or LR issue | Reduce epsilon, check `--step-size` < `--epsilon` |
