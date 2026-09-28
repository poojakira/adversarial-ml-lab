# Verified Metrics — Poster 06

> Evidence status: This is a dated repository snapshot at the commit identified below. `VERIFIED_AT_SNAPSHOT` means verified for that commit and environment; it does not assert the same result on the latest `main`. Compare newer claims with the repository evidence before reuse.

MIT • Python 3.12 • HEAD cd7547d • verified 2026-09-26. Verified for this poster on Windows / CPython 3.12.10.

## Headline cards
- 100% — PGD-20 SUCCESS
- 100% — FGSM SUCCESS
Notes: Measured on real weights, CIFAR-10 subset (robustbench_real.json). On correctly-classified inputs; PGD-20 stronger than FGSM (no masking).

## Verified surface
| Item | Value |
|---|---|
| Attack budget ε | 8/255 |
| PGD steps | 20 |
| Framework | torch 2.3 |

## Chart values
| Series | Value |
|---|---|
| Clean (subset) | 5 |
| FGSM adv acc | 0 |
| PGD-20 adv acc | 0 |
Note: Real measured run; clean acc low due to ImageNet→CIFAR domain mismatch on the subset. Not full-test-set.

## Historical / provenance
cifar10_resnet18_benchmark.json is a LITERATURE PROJECTION (Madry 2018): undefended PGD ~0%, Madry-AT ~45%. NOT measured here — shown as context.

## Not established by this repository
State-of-the-art robustness. Certified guarantees. Full-test-set numbers. Black-box transferability.
