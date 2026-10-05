# Research Brief - Poster 06

> Evidence status: Refreshed against verified code snapshot `490b71ec480c40622b63b1adaf2bf2f75959b8d6` and successful CI run `37169514061` on 2026-10-04. The real CIFAR-10 robustness values are from the committed measured artifact `results/cifar10_smallcnn_real.json`; the current CI run validates the code/test surface but did not rerun the expensive CIFAR-10 benchmark job.

## Repository

`github.com/poojakira/adversarial-ml-lab` - public, default branch `main`.

## Academic Project Title

**Measuring Neural-Network Robustness Under Adversarial Perturbation**

### Subtitle

Experimental Evaluation of Model Behavior Under Gradient-Based Attacks

## One-Sentence Contribution

A reproducible robustness-evaluation harness for FGSM, PGD, and C&W that shows a measured **71.82% clean CIFAR-10 SmallCNN collapsing to 0.00% robust accuracy under PGD-20 at epsilon 8/255**, while explicitly separating measured results from literature projections.

## Method

1. Train/evaluate a compact CIFAR-10 classifier under a fixed seed and CPU budget.
2. Measure clean accuracy on the full CIFAR-10 test set.
3. Generate epsilon-bounded FGSM and PGD adversarial samples plus C&W L2 samples.
4. Measure robust accuracy on the committed attack subset.
5. Emit structured evidence and CI robustness-gate logic.

## Verified Evidence at Poster Snapshot

The cited Python 3.12 CI snapshot reports:

- **112 tests passed**.
- **32.22% statement coverage**; CI gate is 15%.
- Lint/format, security audit, and CodeQL jobs succeeded.
- The current CI robustness benchmark job is skipped on ordinary pushes, so the benchmark values below come from the committed real artifact rather than this CI run.

Committed measured CIFAR-10 artifact:

| Attack | Setting | Clean accuracy | Robust accuracy | Samples |
|---|---|---:|---:|---:|
| FGSM | epsilon 8/255 | 71.82% | **3.32%** | 1,024 |
| PGD-20 | epsilon 8/255, alpha 2/255 | 71.82% | **0.00%** | 1,024 |
| C&W L2 | c=1.0, 100 steps | 71.82% | **4.20%** | 1,024 |

The model is a **1,117,354-parameter SmallCNN**, trained for 6 epochs on CPU. Clean accuracy is measured on all 10,000 CIFAR-10 test examples.

## Correction to the Older Poster

The older poster centered `results/robustbench_real.json`, an ImageNet-pretrained ResNet evaluated on a tiny CIFAR-10 subset with severe domain mismatch. That artifact remains historical evidence, but it is no longer the best representation of this project's current measured result. The poster now uses `results/cifar10_smallcnn_real.json`.

## Limitations

- This repository measures vulnerability; it does not make a model robust.
- Robust-accuracy attacks use a 1,024-sample subset, not the full 10,000 test set.
- No AutoAttack, black-box transfer, or certified-robustness result is established.
- `results/cifar10_resnet18_benchmark.json` is explicitly literature-projected/synthetic and must not be presented as measured.

## Reproducibility

```bash
git clone https://github.com/poojakira/adversarial-ml-lab.git
cd adversarial-ml-lab
git checkout 490b71ec480c40622b63b1adaf2bf2f75959b8d6
python -m pip install -e ".[dev]"
pytest tests/ -q --cov=adv_lab --cov-report=term
python scripts/run_real_smallcnn_benchmark.py --epochs 6 --attack-samples 1000
```

Expected evidence at the cited snapshot: **109 passed**, **32.22% coverage**.
