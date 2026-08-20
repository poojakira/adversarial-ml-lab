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

```bash
# FGSM attack
python -m advml.attacks.fgsm --model resnet18 --epsilon 0.03 --output results/fgsm/

# PGD attack
python -m advml.attacks.pgd --model resnet18 --epsilon 0.03 --steps 20 --step-size 0.007 --output results/pgd/

# C&W attack (L2 norm)
python -m advml.attacks.cw --model resnet18 --confidence 0.01 --max-iter 1000 --output results/cw/

# All attacks at multiple epsilon values
python -m advml.run_all --epsilons 0.01,0.03,0.05,0.1 --output results/sweep/
```

## Evaluate Robustness

```bash
# Evaluate model accuracy under attack
python -m advml.evaluate --model resnet18 --attack-dir results/fgsm/ --metrics accuracy,confidence

# Compare multiple models
python -m advml.evaluate --models resnet18,resnet50,vit_small --attack pgd --epsilon 0.03

# Adversarial training evaluation
python -m advml.evaluate --model results/adv_trained.pt --attack-dir results/pgd/
```

## Generate Reports

```bash
# Generate full benchmark report (Markdown + plots)
python -m advml.report --results-dir results/ --output report/

# JSON summary for CI
python -m advml.report --results-dir results/ --format json > benchmark.json

# Specific comparison report
python -m advml.report --results-dir results/sweep/ --plot-type epsilon-accuracy
```

Reports include: accuracy-vs-epsilon curves, per-class robustness breakdown, perturbation visualizations.

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
