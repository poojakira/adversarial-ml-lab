# Claim Ledger - Poster 06

> Verified code snapshot: `490b71ec480c40622b63b1adaf2bf2f75959b8d6`; successful CI run `37169514061`, 2026-10-04. Robustness numbers are from the committed measured artifact `results/cifar10_smallcnn_real.json`.

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


## Claim audit and rendered-source boundary, October 13, 2026 UTC

The reviewed printed claims, source and test commit anchors, CI dates and benchmark limitations are in `verified_claims.json`. The PDF is generated from that manifest and the PNG is rasterized from the same page by `render_poster.py`. Source inventory, dated CI and separate reported experiments must not be combined into one result. Earlier artifact versions remain in Git history; these corrections do not assert production readiness.

Offline poster rendering: `python -m pip install reportlab==4.4.9 PyMuPDF==1.26.6`, then `python poster/render_poster.py`. The font is bundled with the pinned ReportLab package; no browser or network fetch is used during rendering.
