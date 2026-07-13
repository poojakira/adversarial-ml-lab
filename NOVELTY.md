# Novelty, Data & Production-Readiness — honest, skeptical assessment

## Genuinely useful here

- **A clean, CI-gateable adversarial evaluation harness.** FGSM/PGD/C&W plus
  PGD adversarial training, producing a signed JSON benchmark report you can
  fail a build on, is a practical packaging of standard techniques.
- **Supply-chain hardened** (this round): fail-closed HMAC signing key,
  sandboxed attack execution, hash-pinned deps, least-privilege CI.

## Be skeptical about these claims

- **The attacks themselves are textbook, not novel.** FGSM, PGD, and C&W are
  well-established. The value here is engineering/packaging, not new research.
  Don't present the attacks as original contributions.
- **CIFAR-10 (32×32) is a toy relative to production.** Real deployments use
  224×224+ inputs and larger architectures; small-image results do not transfer
  automatically. Attack success rates on CIFAR-10 are not evidence about a
  production vision model.
- **No standardized-benchmark comparison ships.** Without RobustBench, results
  are not comparable to the literature or to defended baselines.

## Data / benchmark discipline required

| Need | Why | Source |
|------|-----|--------|
| **RobustBench integration** | Comparable, standardized robustness numbers vs defended models | robustbench.org |
| ImageNet(-subset) eval | Realistic input scale (224×224, 1000 classes) | ImageNet-1K val |
| Transfer-attack setup | Black-box realism: substitute → victim transfer rate | ResNet/ViT victim+substitute |
| Adaptive-attack eval | Attack-agnostic defenses must face adaptive attacks | adaptive-attack suite |
| Other domains | Evasion beyond classification (detection/OCR/segmentation) | COCO, SVHN, Cityscapes |

**Honest shipping recommendation:** add a RobustBench-compatible evaluation
(`pytest tests/test_robustbench.py`-style) and report against known
leaderboards. Until then, describe this as an **educational/CI attack harness on
CIFAR-10**, which is accurate and still useful, rather than a
production-robustness authority.

## Known gaps

- No ImageNet-scale or non-classification evaluation.
- No adaptive-attack or transferability protocol wired in.
- Robustness "pass" reflects the configured attacks only, not worst-case.
