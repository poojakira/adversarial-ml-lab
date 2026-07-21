# Attack Implementation Status

This file is an integrity document. It distinguishes implemented attacks from unsupported scope so repository claims stay aligned with executable code.

| Attack | Status | Test | Notes |
|--------|--------|------|-------|
| FGSM | IMPLEMENTED | `tests/test_fgsm.py` | CPU-executable gradient sign attack coverage. |
| PGD | IMPLEMENTED | `tests/test_pgd.py` | L-inf and L2 variants are exported. |
| C&W L2 | IMPLEMENTED | `tests/test_cw.py` | Carlini-Wagner L2 attack entrypoint exists. |
| Black-box attacks | IMPLEMENTED | `tests/test_blackbox.py` | SimBA, Square, HopSkipJump, and Boundary are exported. |
| Model stealing | IMPLEMENTED | `tests/test_model_stealing.py` | Substitute-model and query-based extraction utilities. |
| Norm attacks | IMPLEMENTED | `tests/test_norms.py` | L0, L1, Wasserstein, semantic, and patch variants. |
| LLM attacks | IMPLEMENTED | `tests/test_llm.py` | GCG, AutoDAN, prompt-injection, token, embedding, and suffix attacks. |
| Poisoning attacks | IMPLEMENTED | `tests/test_poisoning.py` | Clean-label, BadNets, spectral backdoor, and weight poisoning. |
| Adaptive attacks | IMPLEMENTED | `tests/test_adaptive.py` | BPDA, EoT, and gradient-masking checks. |
| Parameter search | IMPLEMENTED | `tests/test_param_search.py` | Bayesian optimizer and difficulty scoring. |
| Constrained attacks | IMPLEMENTED | `tests/test_constrained.py` | Query budget and timed attack controls. |
| Evasion attacks | IMPLEMENTED | `tests/test_evasion.py` | JPEG, feature squeezing, and detector evasion. |
| Ensemble attacks | IMPLEMENTED | `tests/test_ensemble.py` | Weighted ensemble PGD and attacker ensemble helpers. |
| Inference attacks | IMPLEMENTED | `tests/test_inference.py` | Watermark flip and prediction-poisoning utilities. |
| Attack chaining | IMPLEMENTED | `tests/test_chaining.py` | Perturbation chain orchestration. |
| API simulation attacks | IMPLEMENTED | `tests/test_api_sim.py` | API anomaly simulation and evasion utilities. |
| Non-classification attacks | IMPLEMENTED | `tests/test_non_classification.py` | Object detection, segmentation, regression, RL, and recommendation attack helpers. |
| Inversion attacks | IMPLEMENTED | `tests/test_inversion.py` | Gradient inversion, GAN inversion, and membership inference utilities. |
| Physical attacks | IMPLEMENTED | `tests/test_physical.py` | Physical patch attack and printability constraints. |
| Universal attacks | IMPLEMENTED | `tests/test_universal.py` | UAP generation and cross-architecture transfer. |
| DeepFool | STUBBED | - | Not exported by `adv_lab.attacks`; do not claim implemented. |
| AutoAttack | STUBBED | - | Not exported by `adv_lab.attacks`; do not claim implemented. |