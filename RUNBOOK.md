# Runbook

## What this repo does

adversarial-ml-lab implements adversarial attacks (FGSM, PGD, C&W) and defenses (adversarial training, randomized smoothing) for PyTorch image classifiers. It includes a CI-gateable benchmark harness.

## Build and run locally

All commands assume you're in the repo root.

| Task | Command |
|------|---------|
| Install everything | `make install` |
| Install attack-v19-core (for ATT&CK mapping) | `make install-core` |
| Run tests | `make test` |
| Lint | `make lint` |
| Format code | `make format` |
| Build package | `make build` |
| Security scan (bandit + pip-audit) | `make security` |
| Full local gate (lint + test + build + security) | `make verify` |
| Serve dashboard | `make dashboard` (serves at localhost:8080) |

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- numpy ≥ 1.24
- Optional: `attack-v19-core` from `../attack-v19-core` (needed for `make test` and ATT&CK mapping)

## CI

GitHub Actions (`ci.yml`) runs ruff, pytest, build, bandit, and pip-audit on push/PR.

## Dashboard

`dashboard/index.html` is a static 3D visualization. Serve it with `make dashboard` or any HTTP server pointed at that directory. It shows security posture indicators — treat these as visual aids, not certifications.

## Known limitations

- No published benchmark artifacts (CIFAR-10 accuracy numbers, etc.) exist in this repo.
- Tested locally on Windows. Re-verify on Linux / GitHub Actions after pushing.
- Not production-ready. Use for research, prototyping, and CI gates only.
- `make test` depends on `../attack-v19-core` being cloned alongside this repo.
