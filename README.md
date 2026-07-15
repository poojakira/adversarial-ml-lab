# adversarial-ml-lab

**Most production ML systems have never been stress-tested under adversarial conditions. This fixes that.**

A comprehensive adversarial ML security lab implementing a **20-tier attack surface** against PyTorch models -- spanning white-box gradient attacks, black-box queries, model stealing, LLM prompt injection, data poisoning, defense-aware adaptation, non-classification targets, privacy attacks, physical-world patches, and universal perturbations -- plus defenses, certified evaluation, and a **CI-gateable** robustness benchmark with HMAC-signed results when `ADV_LAB_HMAC_KEY` is configured. The goal is to produce robustness numbers your CI can fail on, make those numbers tamper-evident when signing is configured and the HMAC key remains secret, and cover a broad set of adversarial threat classes relevant to production ML systems.

---

## Why this exists

By 2026 the adversarial-examples literature has settled into an uncomfortable consensus:

- The canonical attacks did not get replaced -- they got **ensembled**. FGSM ([Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572)), PGD ([Madry et al., 2018](https://arxiv.org/abs/1706.06083)), and C&W ([Carlini & Wagner, 2017](https://arxiv.org/abs/1608.04644)) are the components of AutoAttack, which with [RobustBench](https://arxiv.org/abs/2010.09670) is the de-facto standard (CIFAR-10, L-inf eps=8/255).
- **PGD adversarial training is the only empirical defense that keeps surviving.** Almost everything else gets broken, usually because the *original evaluation* was weak -- the pattern [Obfuscated Gradients (2018)](https://arxiv.org/abs/1802.00420) named and papers like [Ensemble Everything Everywhere Is Not Robust (2024)](https://arxiv.org/html/2411.14834v1) keep confirming.

And what is still genuinely open:

1. **Evaluation rigor.** Weak/single-step attacks (plain FGSM) systematically *over-report* robustness. Automated audits still flag gradient masking in the large majority of suspicious defenses ([arXiv:2604.20704](https://arxiv.org/abs/2604.20704)).
2. **Multi-norm / union robustness.** A model hardened for L-inf is routinely *not* robust under L2, and vice versa; the worst-case-across-norms gap on SOTA models is large and unresolved ([CURE, 2024](https://arxiv.org/abs/2410.03000); [RAMP, 2024](https://arxiv.org/abs/2402.06827)).
3. **The accuracy-robustness tradeoff** persists, with certified accuracy apparently bounded by Bayes error ([arXiv:2405.11547](https://arxiv.org/html/2405.11547v1)).
4. **The attack surface is far wider than L-p perturbations.** Real adversaries use model stealing, data poisoning, prompt injection, physical patches, and privacy attacks -- none of which a simple epsilon-ball benchmark catches.

*(Literature summarized/rephrased for licensing compliance.)*

This lab targets all four gaps: it runs the full attack ladder so a single weak attack cannot hand you a false pass, ships multiple norms so the union-robustness gap is visible, and expands coverage to **20 tiers** of attack surface that production ML systems actually face.

---

## Install

```bash
# Python 3.12
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"      # tests, lint, typecheck
```

Dependencies: `torch>=2.3`, `torchvision>=0.18`, `numpy>=1.26`.

Local verification:

```bash
export ADV_LAB_HMAC_KEY="replace-with-a-random-local-key"
ruff check src tests
mypy src tests
pytest -q
PYTHONPATH=src python -m adv_lab.eval.harness --n-samples 500 --output results/report.json --hmac-key-env ADV_LAB_HMAC_KEY
```

---

## The benchmark harness (the actual product)

The harness turns "is my model robust?" into a number CI can gate on:

```
passed = robust_accuracy_pgd > 0.3
```

Run the self-contained demo (trains a small CNN on a synthetic-but-learnable task, then attacks it -- no dataset download):

```bash
$env:ADV_LAB_HMAC_KEY = "replace-with-a-random-local-key"
$env:PYTHONPATH = "src"
py -m adv_lab.eval.harness --n-samples 500 --output results/report.json --hmac-key-env ADV_LAB_HMAC_KEY
# or, via the installed console script:
adv-eval --n-samples 500 --output results/report.json
```

The harness supports **HMAC signing** of results via `ci_signing.py`, making benchmark outputs tamper-evident between generation and CI gate evaluation when the signing key remains secret.

---

## Attack Surface Coverage

The framework implements 20 tiers of adversarial attack surface:

### Tier 1: Foundation Attacks

| Module | Capability |
|--------|-----------|
| `fgsm.py` | Single-step gradient attack (FGSM) |
| `pgd.py` | Multi-step PGD (L-inf + L2) with random starts |
| `cw.py` | Carlini & Wagner L2 (tanh reparameterization, margin loss) |
| `blackbox.py` | Black-box attacks: SimBA, Square Attack, HopSkipJump, Boundary Attack |
| `model_stealing.py` | Model extraction via query synthesis and knockoff nets |
| `norms.py` | Full norm suite: L0, L1, L2, L-inf, Wasserstein, semantic, patch |
| `llm.py` | LLM attacks: GCG, AutoDAN, prompt injection, embedding attacks, token substitution, universal suffix |
| `poisoning.py` | Data poisoning and backdoor attacks (BadNets, clean-label, gradient-based) |

### Tier 2: Adaptive and Composite Attacks

| Module | Capability |
|--------|-----------|
| `adaptive.py` | Defense-aware adaptation: BPDA, Expectation over Transformations (EoT) |
| `param_search.py` | Adaptive parameter search via Bayesian optimization |
| `constrained.py` | Resource-constrained attacks (query budgets, compute limits) |
| `evasion.py` | Post-processing evasion: JPEG compression, feature squeezing bypass |
| `ensemble.py` | Ensemble attacks across multiple models |
| `inference.py` | Inference-time manipulation (batch poisoning, timing attacks) |
| `chaining.py` | Multi-stage perturbation chaining pipelines |
| `api_sim.py` | API behavior simulation (rate limits, query counting, response analysis) |

### Tier 3: Non-Classification Targets

| Module | Capability |
|--------|-----------|
| `non_classification.py` | Attacks on object detection, segmentation, regression, RL policies, recommendation systems |

### Tier 4: Certified Defense Evaluation

| Module | Capability |
|--------|-----------|
| `certified.py` (eval) | Certified defense evaluation: randomized smoothing, Lipschitz bounds, Interval Bound Propagation (IBP) |

### Tier 5: Privacy and Physical Attacks

| Module | Capability |
|--------|-----------|
| `inversion.py` | Privacy attacks: model inversion, membership inference |
| `ci_signing.py` (eval) | CI gate HMAC signing for tamper-evident benchmark results |
| `physical.py` | Physical-world adversarial patches with Expectation over Transformations |
| `universal.py` | Universal Adversarial Perturbations (image-agnostic, transferable) |

---

## Layout

```
adversarial-ml-lab/
├── src/adv_lab/
│   ├── attacks/          20 modules
│   │   ├── fgsm.py             FGSM (single-step gradient)
│   │   ├── pgd.py              PGD L-inf + L2 (multi-step, random start)
│   │   ├── cw.py               Carlini & Wagner L2
│   │   ├── blackbox.py         SimBA, Square, HopSkipJump, Boundary
│   │   ├── model_stealing.py   Model extraction (query synthesis, knockoff nets)
│   │   ├── norms.py            Full norm suite (L0/L1/L2/Linf/Wasserstein/semantic/patch)
│   │   ├── llm.py              LLM attacks (GCG, AutoDAN, prompt injection, etc.)
│   │   ├── poisoning.py        Data poisoning & backdoors
│   │   ├── adaptive.py         BPDA, EoT (defense-aware)
│   │   ├── param_search.py     Bayesian optimization for attack params
│   │   ├── constrained.py      Resource-constrained attacks
│   │   ├── evasion.py          Post-processing evasion (JPEG, feature squeezing)
│   │   ├── ensemble.py         Multi-model ensemble attacks
│   │   ├── inference.py        Inference-time manipulation
│   │   ├── chaining.py         Multi-stage perturbation chaining
│   │   ├── api_sim.py          API behavior simulation
│   │   ├── non_classification.py  Object detection, segmentation, RL, etc.
│   │   ├── inversion.py        Model inversion & membership inference
│   │   ├── physical.py         Physical-world patches (EOT)
│   │   └── universal.py        Universal adversarial perturbations
│   ├── defenses/         2 modules
│   │   ├── adversarial_training.py   PGD-7 adversarial training
│   │   └── detection.py              NeuralCleanse, STRIP, spectral signatures
│   └── eval/             4 modules
│       ├── harness.py           BenchmarkResult, run_benchmark, export_json, CLI
│       ├── transferability.py   Cross-architecture transferability analysis
│       ├── certified.py         Certified defense evaluation (smoothing, IBP)
│       └── ci_signing.py        HMAC signing for CI gate integrity
├── tests/                25 test files, 295 passing tests in the local validation run
├── docs/
│   ├── INCIDENT_REPORT.md
│   ├── TECHNICAL_REPORT.md
│   └── EXECUTIVE_REPORT.md
├── .github/workflows/ci.yml
├── pyproject.toml · CHANGELOG.md · README.md
```

---

## Tests

```bash
python -m pip install -e ".[dev]"
pytest -q          # local validation: 295 passed
```

The suite validates correctness properties across all 20 tiers: attacks stay within their norm balls, gradient masking is detected, black-box queries converge, LLM attacks produce valid token sequences, poisoning achieves target misclassification, adaptive attacks bypass declared defenses, ensemble attacks outperform individuals, chained perturbations compose correctly, certified bounds hold, HMAC signatures verify, and the harness emits valid JSON with consistent gate decisions.

---

## Reports

Comprehensive documentation of security posture and risk assessment:

- **[Technical Report](docs/TECHNICAL_REPORT.md)** -- Detailed 20-tier attack/defense analysis with implementation specifics, threat models, and evaluation methodology.
- **[Executive Report](docs/EXECUTIVE_REPORT.md)** -- CISO/board-level adversarial ML risk posture summary with strategic recommendations.

---

## References

- Goodfellow, Shlens & Szegedy -- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (2015)
- Madry, Makelov, Schmidt, Tsipras & Vladu -- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) (2018)
- Carlini & Wagner -- [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644) (2017)
- Athalye, Carlini & Wagner -- [Obfuscated Gradients Give a False Sense of Security](https://arxiv.org/abs/1802.00420) (2018)
- Croce & Hein -- [RobustBench / AutoAttack](https://arxiv.org/abs/2010.09670) (2020)

## License

MIT.
