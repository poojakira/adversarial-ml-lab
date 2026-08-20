# adversarial-ml-lab

A learning exercise implementing well-known adversarial attacks (FGSM, PGD, C&W) from published papers. This is a benchmark runner, not original research.

## What It Does

Implements three attacks from their respective papers:
- FGSM (Goodfellow et al. 2014)
- PGD (Madry et al. 2018)
- C&W (Carlini & Wagner 2017)

You point it at a model, pick an attack and epsilon budget, and get a JSON report showing clean accuracy vs. robust accuracy. The CI gate fails if PGD robust accuracy drops below a configurable threshold (default: 30% at eps=8/255).

## Honest Scope

- This implements well-known attacks from papers — there is no novel contribution here.
- Attacks are implemented and tested. Defenses (adversarial training, randomized smoothing) are **not** implemented — the defense references cite Madry 2018 and Cohen 2019 literature values, not measurements from this repo.
- Runs against a dummy CNN in CI. No pretrained weights committed.
- This is a measurement/learning tool, not a defense.

## Quick Start

```bash
git clone https://github.com/poojakira/adversarial-ml-lab.git && cd adversarial-ml-lab
pip install -e ".[dev]"
python -m adv_lab.benchmark --attack pgd --eps 0.031 --output report.json
```

## License

MIT.
