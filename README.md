# adversarial-ml-lab

**Most production classifiers have never been stress-tested under adversarial conditions. This fixes that.**

A small, honest lab for attacking PyTorch image classifiers with the three attacks that actually matter — **FGSM**, **PGD** (both L∞ and L2), and **Carlini & Wagner L2** — plus PGD adversarial training and a **CI-gateable** robustness benchmark. The goal is not to collect attacks; it is to produce a robustness number your CI can fail on, and to make that number hard to fake.

---

## Why this exists

By 2026 the adversarial-examples literature has settled into an uncomfortable consensus:

- The canonical attacks did not get replaced — they got **ensembled**. FGSM ([Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572)), PGD ([Madry et al., 2018](https://arxiv.org/abs/1706.06083)), and C&W ([Carlini & Wagner, 2017](https://arxiv.org/abs/1608.04644)) are the components of AutoAttack, which with [RobustBench](https://arxiv.org/abs/2010.09670) is the de-facto standard (CIFAR-10, L∞ ε=8/255).
- **PGD adversarial training is the only empirical defense that keeps surviving.** Almost everything else gets broken, usually because the *original evaluation* was weak — the pattern [Obfuscated Gradients (2018)](https://arxiv.org/abs/1802.00420) named and papers like [Ensemble Everything Everywhere Is Not Robust (2024)](https://arxiv.org/html/2411.14834v1) keep confirming.

And what is still genuinely open:

1. **Evaluation rigor.** Weak/single-step attacks (plain FGSM) systematically *over-report* robustness. Automated audits still flag gradient masking in the large majority of suspicious defenses ([arXiv:2604.20704](https://arxiv.org/abs/2604.20704)).
2. **Multi-norm / union robustness.** A model hardened for L∞ is routinely *not* robust under L2, and vice versa; the worst-case-across-norms gap on SOTA models is large and unresolved ([CURE, 2024](https://arxiv.org/abs/2410.03000); [RAMP, 2024](https://arxiv.org/abs/2402.06827)).
3. **The accuracy–robustness tradeoff** persists, with certified accuracy apparently bounded by Bayes error ([arXiv:2405.11547](https://arxiv.org/html/2405.11547v1)).

*(Literature summarized/rephrased for licensing compliance.)*

This lab targets **(1)** and **(2)** directly: it runs the full FGSM < PGD < C&W ladder so a single weak attack can't hand you a false pass, and it ships **both** PGD-L∞ and PGD-L2 so the union-robustness gap is visible instead of hidden behind one norm.

---

## Install

```bash
# Python 3.12
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[dev]" for pytest + coverage
```

Dependencies: `torch>=2.3`, `torchvision>=0.18`, `numpy>=1.26`.

---

## The benchmark harness (the actual product)

The harness turns "is my model robust?" into a number CI can gate on:

```
passed = robust_accuracy_pgd > 0.3
```

Run the self-contained demo (trains a small CNN on a synthetic-but-learnable task, then attacks it — no dataset download):

```bash
py -m adv_lab.eval.harness --n-samples 500 --output results/report.json
# or, via the installed console script:
adv-eval --n-samples 500 --output results/report.json
```

### Actual output from this repo

```
============================================================
model          : SmallCNN(undefended-demo)
clean accuracy : 0.920
robust  (FGSM) : 0.508
robust  (PGD)  : 0.500
robust  (C&W)  : 0.034
epsilon        : 0.03
PGD gate (>0.3) : PASS
============================================================
```

`results/report.json` (the CI-consumable payload):

```json
{
  "passed": true,
  "robust_pgd": 0.5,
  "robust_fgsm": 0.508,
  "robust_cw": 0.034,
  "clean_accuracy": 0.92,
  "epsilon": 0.03,
  "gate_threshold": 0.3,
  "model_name": "SmallCNN(undefended-demo)"
}
```

**Read this result the way an adversary would.** The PGD gate says `PASS` (0.50 > 0.30) — and that is exactly the trap. The same model under C&W L2 keeps only **3.4%** accuracy. A gate wired to one attack can green-light a model that a stronger attack demolishes. That is gap #1 reproduced in ten seconds, which is why the harness always reports the whole ladder next to the gate, and why the JSON carries `robust_cw` right beside `robust_pgd`.

The process also exits non-zero on gate failure, so CI can key off either the JSON or the exit code.

---

## Attack strength: FGSM < PGD < C&W

Robust accuracy is the fraction still classified correctly *after* the attack, so **lower = stronger attack**. The demo above lines up exactly as the theory predicts:

| Attack | Norm | What it does | Robust acc (demo) |
|--------|------|--------------|-------------------|
| clean  | —    | no perturbation | 0.920 |
| **FGSM** | L∞ | one signed gradient step | 0.508 |
| **PGD**  | L∞ | many signed steps, projected into the ε-ball each time | 0.500 |
| **C&W**  | L2 | Adam-optimized, tanh change-of-variables, margin loss | **0.034** |

- **FGSM** is a single step along `sign(∇ₓ loss)`. It ignores curvature; it's a baseline and a gradient-masking smell test, not a real audit.
- **PGD** is FGSM done honestly: initialize randomly in the ε-ball, then take many small signed steps, re-projecting after each. It's the strongest *first-order* attack and the workhorse of both evaluation and training.
- **C&W L2** doesn't fix a budget at all — it *minimizes* the L2 perturbation subject to misclassification, via the change of variables `x = ½(tanh(w)+1)` (no clipping needed) and the margin objective `f(x) = max(maxᵢ≠t Z(x)ᵢ − Z(x)_t, −κ)`. It finds the smallest push that works, which is why it's the gold standard for catching defenses that only *look* robust.

```python
import torch
from adv_lab.attacks import fgsm_attack, pgd_attack, pgd_l2, cw_l2_attack

model.eval()  # required — attacks raise if the model is in train() mode
x_fgsm = fgsm_attack(model, images, labels, epsilon=0.03)
x_pgd  = pgd_attack(model, images, labels, epsilon=0.03, alpha=0.007, steps=40)
x_l2   = pgd_l2(model, images, labels, epsilon=0.5, alpha=0.1, steps=40)
x_cw   = cw_l2_attack(model, images, labels, c=1e-4, steps=1000)
```

---

## Why PGD-7 for adversarial training, and not FGSM

Adversarial training solves the min-max problem

```
min_θ  E[ max_{‖δ‖≤ε}  L(f_θ(x+δ), y) ]
```

The inner `max` is what decides whether the defense is real. The tempting shortcut is to approximate it with **FGSM** (one step, cheap). It doesn't work, and the failure mode is instructive:

- **FGSM training collapses into catastrophic overfitting.** The network learns to defeat the *single-step* attack specifically — it bends the loss surface so the one FGSM step lands somewhere harmless — while remaining wide open to a multi-step attack. Robust accuracy against a real PGD adversary can crater within a single epoch. This is gradient masking wearing a lab coat: the training-time attack is too weak to represent the true worst case.
- **PGD approximates the inner max far better.** Multiple projected steps actually climb toward the worst-case perturbation in the ε-ball, so the model is forced to be robust to points a real attacker can reach — not just to one convenient step.
- **Why specifically ~7 steps?** It's the community's cost/robustness sweet spot (following Madry et al.). More steps improve the inner approximation with diminishing returns while training cost scales linearly, and 7 is enough to avoid the catastrophic-overfitting regime that plagues single-step training. **Crucially, training strength and evaluation strength are decoupled:** you train with PGD-7 to keep it affordable, then *evaluate* with many more steps (this repo uses PGD-40) plus C&W, because an attacker at test time has no step budget you control.

```python
from torch.optim import SGD
from adv_lab.defenses import AdversarialTrainer

trainer = AdversarialTrainer(model, SGD(model.parameters(), lr=0.1), epsilon=0.03)
stats = trainer.train_epoch(train_loader)   # PGD-7 inner attack by default
# {"loss": ..., "clean_acc": ..., "robust_acc": ...}
```

---

## Layout

```
adversarial-ml-lab/
├── src/adv_lab/
│   ├── attacks/       fgsm.py · pgd.py (L∞ + L2) · cw.py
│   ├── defenses/      adversarial_training.py   (PGD-7 inner attack)
│   └── eval/          harness.py                (BenchmarkResult, run_benchmark, export_json, CLI)
├── tests/             13 tests (4 FGSM · 4 PGD · 3 C&W · 2 harness)
├── .github/workflows/ci.yml
├── pyproject.toml · CHANGELOG.md · README.md
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # 13 passed
```

The suite checks the properties that actually matter for correctness: FGSM respects the `[0,1]` box and the ε=0 no-op, PGD stays inside its L∞ and L2 balls, more PGD steps never *lowers* success, random starts diverge, C&W suppresses the true-class score and stays in range, and the harness emits valid JSON with a `passed` flag consistent with the PGD gate.

## References

- Goodfellow, Shlens & Szegedy — [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (2015)
- Madry, Makelov, Schmidt, Tsipras & Vladu — [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) (2018)
- Carlini & Wagner — [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644) (2017)
- Athalye, Carlini & Wagner — [Obfuscated Gradients Give a False Sense of Security](https://arxiv.org/abs/1802.00420) (2018)
- Croce & Hein — [RobustBench / AutoAttack](https://arxiv.org/abs/2010.09670) (2020)

## License

MIT.
