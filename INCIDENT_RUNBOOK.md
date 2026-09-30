# Incident Runbook — Adversarial ML Lab

This runbook uses only commands and modules that exist in the current repository. It is a research/portfolio project, not an operated production service, and it does not provide an on-call rotation or managed model-checkpoint store.

## Verified command surface

From the repository root after installing development dependencies:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
ruff check src/adv_lab attack_mapping tests
ruff format --check src/adv_lab attack_mapping tests
python scripts/run_cifar10_benchmark.py
```

The main benchmark runner is also available as:

```bash
python -m adv_lab.eval.benchmark_runner --epsilon 0.031 --pgd-steps 40 --seed 42 --output results/report.json
```

## 1. New attack or evaluation produces an unexpected bypass

First reproduce the result with the checked-in benchmark/evaluation code:

```bash
python scripts/run_cifar10_benchmark.py
python -m adv_lab.eval.benchmark_runner --epsilon 0.031 --pgd-steps 40 --seed 42 --output results/report.json
```

Then run the test suite:

```bash
python -m pytest tests -q
```

Review the attack implementation under `src/adv_lab/attacks/`, confirm the configured norm/epsilon and preprocessing assumptions, and add or update a regression test before treating the bypass as established.

The repository does **not** contain `attacks.evaluate` or `attacks.verify_constraints` modules, so do not use those older commands.

## 2. Robustness regression

Reproduce on the current revision:

```bash
python scripts/run_cifar10_benchmark.py
```

Inspect recent code changes:

```bash
git log --oneline -10
git diff HEAD~1 -- "*.py"
```

Run quality gates:

```bash
python -m pytest tests -q
ruff check src/adv_lab attack_mapping tests
ruff format --check src/adv_lab attack_mapping tests
```

If you need to compare with a prior revision, do it in a separate worktree or clean clone so local uncommitted work is not lost. Do not use destructive checkout/revert commands as an incident shortcut without reviewing the diff.

## 3. Model/checkpoint input fails to load

The repository can accept user-provided model/checkpoint inputs in supported evaluation flows, but it does not maintain a production checkpoint backup service.

For a supplied PyTorch checkpoint, inspect it with safe loading semantics appropriate to the file source and verify its hash against a trusted value before use. Do not execute or load untrusted serialized model artifacts merely to test whether they are valid.

After replacing a bad input with a trusted checkpoint, re-run the relevant benchmark and tests:

```bash
python scripts/run_cifar10_benchmark.py
python -m pytest tests -q
```

## 4. Adversarial training fails or runs out of GPU memory

The checked-in training entry point is:

```bash
python scripts/run_madry_training.py --epochs 100 --epsilon 0.031
```

If GPU memory is insufficient, use the options actually supported by that script/configuration rather than copying generic `train_adversarial.py` commands from another project. Confirm available arguments with:

```bash
python scripts/run_madry_training.py --help
```

For development, reduce the supported workload parameters, use CPU where feasible, or run a smaller benchmark. After changing training/evaluation settings, record them with the result so metrics remain reproducible.

## 5. Dashboard or static report cannot be opened

The repository includes a static dashboard directory. Serve it locally with:

```bash
python -m http.server 8080 --directory dashboard
```

Then open `http://127.0.0.1:8080/`. This is a local static-file server, not a hardened production web service.

## Recovery criteria

Before considering an incident resolved:

- the failing condition is reproducible and its cause is understood;
- the checked-in tests pass;
- benchmark commands complete with the intended threat-model parameters;
- any changed metric is reflected in evidence/report files rather than copied from an older run;
- no untrusted checkpoint or artifact was executed simply to inspect it.

## Security contact

For a vulnerability in the repository, use GitHub's private security-advisory mechanism where available rather than publishing exploit details in a public issue.
