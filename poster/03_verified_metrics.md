# Verified Metrics - Poster 06

**Code snapshot:** `490b71ec480c40622b63b1adaf2bf2f75959b8d6`  
**CI run:** https://github.com/poojakira/adversarial-ml-lab/actions/runs/37169514061

| Metric | Cited snapshot value |
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

The cited CI validates tests/security/code quality; its expensive CIFAR-10 robustness job was skipped on this push. Do not describe the benchmark as rerun by cited CI.


## Claim audit and rendered-source boundary, October 13, 2026 UTC

The reviewed printed claims, source and test commit anchors, CI dates and benchmark limitations are in `verified_claims.json`. The PDF is generated from that manifest and the PNG is rasterized from the same page by `render_poster.py`. Source inventory, dated CI and separate reported experiments must not be combined into one result. Earlier artifact versions remain in Git history; these corrections do not assert production readiness.

Offline poster rendering: `python -m pip install reportlab==4.4.9 PyMuPDF==1.26.6`, then `python poster/render_poster.py`. The font is bundled with the pinned ReportLab package; no browser or network fetch is used during rendering.
