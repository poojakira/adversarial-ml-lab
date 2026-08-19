# adversarial-ml-lab

FGSM/PGD/C&W adversarial robustness benchmark. Runs attacks at configurable epsilon budgets, outputs structured JSON for security design reviews. Maps to MITRE ATLAS AML.T0015.

[![CI](https://github.com/poojakira/adversarial-ml-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/adversarial-ml-lab/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![MIT](https://img.shields.io/badge/license-MIT-green)

## What It Does

Implements three attacks (Goodfellow 2014, Madry 2018, Carlini 2017) as a unified benchmark runner. You point it at your model, pick an attack and epsilon, and get a JSON report showing clean accuracy vs. robust accuracy. The CI gate fails if PGD robust accuracy drops below a configurable threshold (default: 30% at eps=8/255).

I built this to produce evidence for security design reviews — specifically, to answer "should we invest in adversarial training for this model?" with data instead of opinion.

## Quick Start

```bash
git clone https://github.com/poojakira/adversarial-ml-lab.git && cd adversarial-ml-lab
pip install -e ".[dev]"
python -m adv_lab.benchmark --attack pgd --eps 0.031 --output report.json
```

## Honest Scope

- Attacks are implemented and tested. Defenses (adversarial training, randomized smoothing) are not — the ROI table references Madry 2018 and Cohen 2019 literature values, not measurements from this repo.
- Runs against a dummy CNN in CI. No pretrained weights committed.
- This is a measurement tool, not a defense.

## License

MIT.
