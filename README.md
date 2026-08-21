# adversarial-ml-lab

Adversarial robustness benchmarks implementing FGSM, PGD, and C&W attacks against CIFAR-10 classifiers with configurable epsilon budgets.

## Key Metrics

| Metric | Value |
|--------|-------|
| Attacks implemented | FGSM, PGD, C&W |
| Target model | ResNet on CIFAR-10 |
| Threat mapping | MITRE ATLAS AML.T0043 (Adversarial ML) |
| CI gate | PGD robust accuracy ≥ 30% at ε=8/255 |
| Perturbation norms | L∞ (FGSM, PGD), L2 (C&W) |
| Output | JSON benchmark report |
| Framework | PyTorch |

## Architecture

```
┌───────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Attack Suite │────▶│  Target Model    │────▶│  JSON Reporter │
│  FGSM/PGD/CW │     │  ResNet/CIFAR-10 │     │  (CI gate)     │
└───────────────┘     └──────────────────┘     └────────────────┘
        │                      │                       │
        ▼                      ▼                       ▼
  Configurable ε        Clean vs. robust         Pass/fail against
  and iterations        accuracy delta           threshold policy
```

**Attack Implementations:**

| Attack | Paper | Method | Norm |
|--------|-------|--------|------|
| FGSM | Goodfellow et al. 2014 | Single-step gradient sign | L∞ |
| PGD | Madry et al. 2018 | Iterative projected gradient descent | L∞ |
| C&W | Carlini & Wagner 2017 | Optimization-based minimization | L2 |

**Pipeline:**
1. Load target model (ResNet architecture, CIFAR-10 weights)
2. Select attack method and perturbation budget (ε)
3. Generate adversarial examples across the test set
4. Measure accuracy degradation: clean accuracy → robust accuracy
5. Emit JSON report comparing results across epsilon values
6. CI fails if robust accuracy drops below configured threshold

## Quick Start

```bash
git clone https://github.com/poojakira/adversarial-ml-lab.git && cd adversarial-ml-lab
pip install -e ".[dev]"

# Run the FGSM/PGD benchmark harness (trains a model, then attacks it)
python -m adv_lab.eval.benchmark_runner --epsilon 0.031 --pgd-steps 20 --output report.json

# Tune batch size for available memory
python -m adv_lab.eval.benchmark_runner --epsilon 0.031 --batch-size 64 --output report.json

# Run tests
pytest tests/ -v
```

## CI Integration

The CI pipeline trains a model in CI (no pretrained weights committed), runs the attack harness, and validates the benchmark completes. This catches regressions in model training and attack code before merge.

```yaml
# Example CI step
- run: python -m adv_lab.eval.benchmark_runner --epsilon 0.031 --output report.json
```

## Scope and Limitations

- Measures adversarial vulnerability  --  does not implement defenses
- Defense baselines reference published values (Madry 2018, Cohen 2019), not this codebase
- Runs a dummy CNN in CI; full ResNet benchmarks require GPU
- Implements known attacks from papers  --  no novel contributions

The purpose is a reproducible measurement harness for quantifying model fragility under gradient-based attacks.

## Relevance to AI Security

Evasion attacks (MITRE ATLAS AML.T0043) threaten deployed classifiers in content moderation, malware detection, and autonomous systems. A model that achieves 95% clean accuracy but drops to 5% under PGD at ε=8/255 provides a false sense of security. This benchmark quantifies that gap  --  the delta between standard and adversarial conditions  --  which is the first measurement needed before engineering meaningful defenses. Understanding attack surface is prerequisite to defense.

## License

MIT
