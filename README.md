# adversarial-ml-lab

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-View-blue)](https://poojakira.github.io/mlsec-dashboards/adversarial-ml-lab/)

A PyTorch library for testing how well image classifiers hold up against adversarial attacks. It implements the standard attack ladder (FGSM → PGD → C&W L2), adversarial training as a defense, randomized smoothing for certified robustness, and a CI-gateable benchmark harness.

This is a research/prototype tool — not production-validated.

## What's in the box

**Attacks** (white-box, gradient-based):

- **FGSM** — single-step signed gradient (Goodfellow et al., 2015). Weak baseline; useful for detecting gradient masking.
- **PGD L-inf and L2** — multi-step projected gradient descent (Madry et al., 2018). The honest standard for white-box evaluation.
- **C&W L2** — optimization-based attack using tanh change-of-variables (Carlini & Wagner, 2017). Finds smaller perturbations than PGD; slowest but strongest.

Additional attack modules exist for black-box, ensemble, universal perturbations, model stealing, membership inference, LLM attacks, and physical-world attacks. These are more experimental.

**Defenses:**

- **Adversarial training** — PGD-7 inner maximization during training (Madry et al., 2018). Hardens the model at the cost of clean accuracy.
- **Randomized smoothing** — Monte Carlo certification of L2 robustness radii (Cohen et al., 2019). Provides provable guarantees, not just empirical ones.

**Evaluation:**

- **Benchmark harness** — runs the full attack ladder on your model and produces a JSON report. CI gate passes if PGD robust accuracy > 30%.
- **Certified defense evaluation** — randomized smoothing, interval bound propagation, Lipschitz network evaluation.
- **Transferability evaluation** — measures how attacks transfer between models.
- **ATT&CK v19 mapping** — maps findings to MITRE ATT&CK technique IDs (optional, requires `attack-v19-core`).

## Honest status

| What | Status |
|------|--------|
| Core attacks (FGSM, PGD, C&W) | Implemented and unit-tested |
| Adversarial training | Implemented |
| Randomized smoothing | Implemented |
| Benchmark harness + CI gate | Implemented |
| Published benchmark numbers | **None committed to this repo**. No CIFAR-10 accuracy artifacts exist here. Don't cite numbers from this README. |
| Production readiness | Not claimed |

## Install

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

```bash
# Clone
git clone https://github.com/poojakira/adversarial-ml-lab
cd adversarial-ml-lab

# Install in development mode
make install

# Or manually:
pip install -e .
pip install -e ".[dev]"      # adds pytest
pip install -e ".[attack]"   # adds attack-v19-core for ATT&CK mapping
```

## Run

### Run tests

```bash
make test
```

Tests exercise the three core attacks and the evaluation harness against a small dummy CNN.

### Run the benchmark harness (CLI)

```bash
python -m adv_lab.eval.cli --epsilon 0.3 --pgd-threshold 0.30
```

This evaluates a dummy model by default. To evaluate your own model, pass `--model-path path/to/state_dict.pt`.

Outputs a JSON report with clean accuracy, FGSM/PGD/C&W robust accuracy, and a pass/fail flag.

### Lint and format

```bash
make lint     # ruff check
make format   # ruff format
```

### Full local verification (lint + test + build + security scan)

```bash
make verify
```

### Static dashboard

```bash
make dashboard
# Opens at http://localhost:8080
```

## Use in your own code

```python
import torch
from adv_lab.attacks.fgsm import fgsm_attack
from adv_lab.attacks.pgd import pgd_attack
from adv_lab.attacks.cw import cw_l2_attack

model.eval()

# FGSM — fast sanity check
adv = fgsm_attack(model, images, labels, epsilon=0.03)

# PGD L-inf — the standard honest evaluation
adv = pgd_attack(model, images, labels, epsilon=0.03, steps=40)

# C&W L2 — slow but finds minimal perturbations
adv = cw_l2_attack(model, images, labels, steps=1000)
```

For adversarial training:

```python
from adv_lab.defenses.adversarial_training import AdversarialTrainer

trainer = AdversarialTrainer(model, optimizer, epsilon=0.03)
for epoch in range(num_epochs):
    stats = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch}: loss={stats['loss']:.3f}, robust_acc={stats['robust_acc']:.3f}")
```

## Project layout

```
src/adv_lab/
├── attacks/          # FGSM, PGD, C&W, plus experimental attacks
├── defenses/         # Adversarial training, detection
├── eval/             # Benchmark harness, certified defense eval, CLI
├── api/              # (placeholder)
├── scanner/          # (placeholder)
├── ci/               # CI utilities
attack_mapping/       # MITRE ATT&CK v19 mapping (optional)
tests/                # Unit and integration tests
dashboard/            # Static HTML dashboard
docker/               # Container definitions
helm/                 # Kubernetes deployment charts
```

## Dependencies

Core: `torch>=2.0`, `numpy>=1.24`. That's it for the attack/defense code.

Optional: `attack-v19-core>=19.1` for MITRE ATT&CK mapping features.

## CI

GitHub Actions runs: ruff lint → pytest → build → bandit SAST → pip-audit. See `.github/workflows/ci.yml`.

## License

MIT
