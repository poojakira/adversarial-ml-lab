# Product Validation

## Product boundary
Adversarial robustness measurement and admission gating. This is an evaluation harness, not a universal defense.

## Real-world validation ladder
1. Unit/regression tests for perturbation constraints and attack implementations.
2. Reproducible CIFAR-10 measured artifact from the committed SmallCNN experiment.
3. Differential validation against an independent adversarial-ML implementation for selected attacks.
4. Admission-mode test against a supplied model artifact and operator-defined robustness threshold.
5. External pilot on a model owned by another team.

## Evidence rules
Literature values, local measurements, and third-party differential results must remain separately labeled. A successful attack implementation does not prove production model risk without evaluating that production model.
