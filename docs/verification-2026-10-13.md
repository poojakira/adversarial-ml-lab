# Local verification — 2026-10-13

Status: PARTIALLY COMPLETE. Independent functional review performed by a separate agent; remote CI, Docker, exhaustive file-by-file review, dependency vulnerability scans, and poster regeneration are not established by these local tests.

- Starting revision: `e07aed1281d51b1336ca6e624752d58f08793e59` plus the remediation diff committed with this record.
- Environment: Linux, Python 3.12.14, pytest 9.1.1, pytest-cov 7.1.0, coverage 7.16.2.
- Local suite: **128 passed, 0 skipped**.
- Statement coverage: **32.3221% (1449/4483)**; not branch coverage.
- Command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest --cov=adv_lab --cov-report=term --cov-report=json`.
- UTC execution date: 2026-10-13. Exact start time was not captured; do not infer it from commit time.
- No production traffic or external paid services were used. No effectiveness benchmark was rerun.

## Remediation

Shared attack input validation rejects nonfinite images, non-long labels, and mismatched devices. PGD rejects nonfinite step sizes and noninteger/bool iteration counts. Sixteen regression cases cover finite-input and parameter boundaries. Dependabot groups CodeQL action updates so init/analyze versions update together.

## Evidence limitations

Full test logs and coverage JSON were retained in the audit execution workspace. This document records their summary; it does not claim CI passed on this revision. Existing dated benchmark artifacts and posters remain historical snapshots, not current reruns. See [historical runs](../previos_runs.md).
