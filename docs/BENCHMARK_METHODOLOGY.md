# Benchmark Methodology

## Scope

This document describes the behavior that is implemented by
`src/adv_lab/eval/benchmark_runner.py` on the current repository head. It is
not a proposal for future AutoAttack, multi-seed, or confidence-interval work.

The runner is an adversarial-robustness **measurement and admission job**. It
does not make a model robust and it does not prove safety against attacks that
are outside the evaluated threat model.

## Two execution modes

### Smoke / development mode

Without `--production`, the runner may use the built-in `_DummyCNN` and a
deterministic synthetic batch. That path exists to exercise the benchmark
plumbing.

```bash
python -m adv_lab.eval.benchmark_runner \
  --epsilon 0.031 \
  --pgd-steps 20 \
  --seed 42 \
  --output results/smoke.json
```

A smoke result is **not deployment evidence**.

### Production admission mode

Production mode refuses implicit evidence. It requires:

- an explicit TorchScript model;
- an explicit NPZ evaluation dataset;
- a deployment-owned PGD robust-accuracy threshold.

```bash
python -m adv_lab.eval.benchmark_runner --production \
  --model-path artifacts/model.ts \
  --model-format torchscript \
  --dataset-path evidence/evaluation.npz \
  --epsilon 0.031 \
  --pgd-steps 40 \
  --min-pgd-robust-accuracy 0.30 \
  --batch-size 256 \
  --output results/production.json
```

The command exits nonzero when the configured PGD gate produces a HIGH or
CRITICAL finding. Invalid or incompatible evidence also terminates the command
with an error rather than producing a passing report.

## Model evidence

In production mode the model must be loadable with `torch.jit.load()`. This
avoids arbitrary Python-object deserialization in the admission path.

The report records:

- model file name;
- model format;
- SHA-256 of the exact model artifact.

The runner performs a pre-attack forward pass and fails when the model cannot
consume the supplied NCHW batch, does not return a 2D `[batch, classes]`
logits tensor, exposes fewer than two classes, produces non-finite logits, or
cannot represent the supplied class labels.

## Dataset evidence

The production dataset is an NPZ containing:

- `images`: numeric 4D NCHW array;
- `labels`: integer 1D class-index array.

The loader uses `numpy.load(..., allow_pickle=False)`. It rejects:

- missing `images` or `labels`;
- non-4D image arrays;
- non-1D label arrays;
- non-numeric images;
- non-integer labels;
- NaN or infinity;
- image values outside `[0, 1]`;
- negative labels;
- mismatched image/label counts;
- a dataset smaller than the requested batch size.

The report records SHA-256 of the exact NPZ artifact.

The current runner evaluates the first `batch_size` records. It does not
randomly sample or claim that a subset is population-representative. Operators
must construct the NPZ evidence set deliberately.

## Implemented attacks

The production runner executes three measurements:

| Report key | Implementation | Norm / behavior |
|---|---|---|
| `fgsm` | canonical `adv_lab.attacks.fgsm.fgsm_attack` | L-infinity, one step |
| `pgd` | canonical `adv_lab.attacks.pgd.pgd_attack` | L-infinity, configured steps, deterministic `random_start=False` |
| `cw_l2_proxy` | PGD with 100 iterations | **Proxy only**; not the full C&W optimizer |

The repository also contains a full C&W implementation elsewhere, but the
production benchmark report intentionally labels the third runner measurement
as a proxy. Do not cite the proxy as a measured C&W-L2 result.

AutoAttack is not executed by this runner.

## Metrics

### Clean accuracy

```
clean_accuracy = clean_correct / total_examples
```

### Robust accuracy

For each attack:

```
robust_accuracy = adversarial_correct / total_examples
```

This is the primary PGD deployment-gate input.

### Attack success rate

Attack success is conditioned on examples that were classified correctly
before the attack:

```
attack_success_rate =
    clean_correct_and_adversarial_wrong / clean_correct
```

The report also includes `attack_success_denominator`, the number of
clean-correct examples used in that calculation.

If there are zero clean-correct examples, attack success rate is undefined and
is emitted as JSON `null`. The runner does **not** substitute
`1 - robust_accuracy`, because doing so would count pre-existing clean errors
as successful attacks.

## Gate semantics

The default configured PGD robust-accuracy threshold is 0.30. This is a
repository default, not a universal security standard.

For the configured threshold:

- PGD robust accuracy below both 0.05 and the gate -> CRITICAL;
- PGD robust accuracy below the gate -> HIGH;
- PGD robust accuracy at or above the gate -> no blocking PGD finding.

A material FGSM/PGD gap may generate a MEDIUM gradient-masking indicator.
MEDIUM findings are reported but do not make this runner's default gate fail.

Deployment owners should select epsilon, attack steps, evaluation evidence, and
the minimum acceptable robust accuracy from their own threat model and risk
acceptance process.

## Reproducibility and provenance

Every report records, where applicable:

- UTC timestamp;
- model SHA-256;
- dataset SHA-256;
- model format;
- production-mode flag;
- epsilon;
- PGD iteration count;
- batch size;
- configured PGD threshold;
- clean accuracy;
- per-attack robust accuracy;
- conditioned attack-success rate and denominator.

Smoke mode additionally records the synthetic-data seed. Production mode does
not invent a seed for externally supplied evidence.

The release workflow builds and publishes a batch-job OCI image whose
entrypoint includes `--production`. The image therefore refuses to run without
explicit model and dataset arguments.

## What this runner does not currently provide

The current production runner does **not** implement or claim:

- AutoAttack evaluation;
- multiple random restarts for the production PGD call;
- multi-seed aggregation;
- Wilson confidence intervals;
- bootstrap confidence intervals;
- McNemar testing;
- ImageNet-specific evaluation;
- distributed or multi-GPU execution;
- automatic representative-dataset selection;
- a universal robustness threshold;
- proof of robustness outside the configured attacks and perturbation budget.

Other scripts or experimental modules in the repository may explore some of
these ideas. They are not part of this production admission contract unless
explicitly wired into this runner and its release tests.

## Interpretation

A passing report means only:

> On the exact hashed model artifact and exact hashed evaluation evidence,
> under the configured FGSM/PGD/proxy attack parameters, the measured PGD
> robust accuracy met the operator-supplied threshold and no blocking finding
> was emitted.

It does not mean the model is adversarially robust in general.

## References

- Goodfellow, Shlens, and Szegedy, *Explaining and Harnessing Adversarial
  Examples* (FGSM).
- Madry et al., *Towards Deep Learning Models Resistant to Adversarial
  Attacks* (PGD-based robustness evaluation/training).
- Carlini and Wagner, *Towards Evaluating the Robustness of Neural Networks*
  (C&W; note that the runner's `cw_l2_proxy` is not this full optimizer).
- MITRE ATLAS AML.T0043 / AML.T0015 mappings are used as reporting context,
  not as evidence of attack coverage beyond the executed benchmark.
