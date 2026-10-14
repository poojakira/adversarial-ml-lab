# Claim Ledger - Poster 06

> Verified code snapshot: `05ee102c4c20fe3ebdcd3a8ac3dd1ea530de6698`; successful CI run `38498116492`, 2026-10-14. Robustness numbers are from the committed measured artifact `results/cifar10_smallcnn_real.json`.

| # | Claim | Classification | Evidence |
|---|---|---|---|
| 1 | 112 tests pass | VERIFIED_AT_SNAPSHOT | Cited Python 3.12 CI snapshot. |
| 2 | 32.22% statement coverage | VERIFIED_AT_SNAPSHOT | Cited Python 3.12 CI snapshot. |
| 3 | Clean CIFAR-10 accuracy 71.82% | VERIFIED_COMMITTED_MEASUREMENT | `results/cifar10_smallcnn_real.json`; full 10,000-image clean evaluation. |
| 4 | FGSM robust accuracy 3.32% at epsilon 8/255 | VERIFIED_COMMITTED_MEASUREMENT | Same artifact; 1,024 attack samples. |
| 5 | PGD-20 robust accuracy 0.00% at epsilon 8/255 | VERIFIED_COMMITTED_MEASUREMENT | Same artifact; 1,024 attack samples. |
| 6 | C&W L2 robust accuracy 4.20% | VERIFIED_COMMITTED_MEASUREMENT | Same artifact; 1,024 attack samples. |
| 7 | Projected ResNet-18 robustness values are measured by this repo | UNSUPPORTED | `results/cifar10_resnet18_benchmark.json` is explicitly synthetic/literature projection. |
| 8 | State-of-the-art or certified robustness | UNSUPPORTED | No such evidence is established. |

The older ImageNet-to-CIFAR 20-image mismatch artifact is historical and no longer the headline benchmark.
