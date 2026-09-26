# Claim Ledger — Poster 06 (06-adversarial-ml-lab)

MIT • Python 3.12 • HEAD 0fadaa2 • verified 2026-09-26. Classification: VERIFIED_CURRENT / VERIFIED_HISTORICAL / PARTIAL / UNVERIFIED / UNSUPPORTED.

| # | Claim | Classification | Evidence |
|---|---|---|---|
| 1 | FGSM & PGD-20 100% attack success on tested subset | VERIFIED_CURRENT | results/robustbench_real.json (measured, real torchvision weights, CPU). success_rate=1.0 both. |
| 2 | Measured config eps=8/255, PGD-20 alpha=2/255 | VERIFIED_CURRENT | robustbench_real.json attack_config; PyTorch 2.3.0+cpu. |
| 3 | Undefended PGD ~0%, Madry-AT ~45% | VERIFIED_HISTORICAL / PROJECTED | cifar10_resnet18_benchmark.json is explicitly _synthetic:true LITERATURE_PROJECTION (Madry 2018). Shown ONLY as labeled projection. |
| 4 | Clean accuracy 5% on subset | PARTIAL | robustbench_real.json; low due to ImageNet->CIFAR domain mismatch + 20-image subset. Not full test set. |
| 5 | State-of-the-art / certified robustness | UNSUPPORTED (disclaimed) | README robustbench_context notes Madry ~45% is baseline not SOTA; no certification. |

## Policy applied
- Only VERIFIED_CURRENT figures appear as prominent current results.
- Historical/projected values are labeled (dashed box / explicit note).
- Unsupported production/accuracy claims are omitted or shown in the red "NOT ESTABLISHED" box.
