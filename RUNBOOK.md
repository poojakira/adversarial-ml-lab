# Runbook

## Engineering Update - 2026-07-27

Repository: adversarial-ml-lab
Purpose: Adversarial ML attacks and defenses

## Build

- Install: make install
- Lint: make lint
- Format: make format
- Test: make test
- Package build: make build
- Security scan: make security
- Full local gate: make verify

## Dashboard

Static 3D dashboard: dashboard/index.html. Serve with make dashboard.

## Dependencies And Data

Uses ../attack-v19-core for ATT&CK mapping tests and pinned MITRE STIX data.

## Validation Snapshot

Validated: Ruff checks for src/adv_lab, attack_mapping, and tests; static dashboard JS syntax/static checks passed.

## Operating Limits

- Re-check Linux and GitHub Actions after pushing to main.
- Treat local dashboard scores as evidence indicators, not certifications.
- Do not cite production readiness until clean CI, dependency audit, license status, and runtime smoke tests are current.