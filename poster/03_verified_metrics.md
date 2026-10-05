# Verified Metrics - Poster 06

**Code snapshot:** `490b71ec480c40622b63b1adaf2bf2f75959b8d6`  
**CI run:** https://github.com/poojakira/adversarial-ml-lab/actions/runs/37169514061

| Metric | Current value |
|---|---:|
| Tests passed | **112** |
| Statement coverage | **32.22%** |
| SmallCNN parameters | **1,117,354** |
| Clean CIFAR-10 accuracy | **71.82%** |
| FGSM robust accuracy @ 8/255 | **3.32%** |
| PGD-20 robust accuracy @ 8/255 | **0.00%** |
| C&W L2 robust accuracy | **4.20%** |
| Robust attack evaluation subset | **1,024 samples** |

Benchmark source: `results/cifar10_smallcnn_real.json`.

The current CI validates tests/security/code quality; its expensive CIFAR-10 robustness job was skipped on this push. Do not describe the benchmark as rerun by current CI.
