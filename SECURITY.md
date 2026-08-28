# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| `main` branch | ✅ Active |
| Older tags | ❌ No backports |

This is a research and evaluation library. Security fixes are applied to the `main` branch only.

---

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report security issues privately via one of:

1. **GitHub private vulnerability reporting**  --  open a [Security Advisory](https://github.com/poojakira/adversarial-ml-lab/security/advisories/new) in this repo (preferred).
2. **Email**  --  send details to the maintainer address listed in `pyproject.toml` with subject line `[SECURITY] adversarial-ml-lab`.

### What to include

- A clear description of the vulnerability
- Steps to reproduce (proof-of-concept code if available)
- The version / commit hash affected
- Your assessment of severity (CVSS score welcome but not required)
- Whether you believe it is already publicly known or exploited

### Response timeline

| Stage | Target |
|-------|--------|
| Acknowledgement | Within 5 business days |
| Initial triage | Within 10 business days |
| Fix / advisory | As soon as reasonably practicable; coordinated disclosure preferred |

---

## Known Security Considerations

### 1. Model file loading (HIGH risk surface)

`benchmark_runner.py` loads model files via `torch.load()`. PyTorch model files (`.pt`, `.pth`) are Python pickle archives and **can execute arbitrary code** if crafted by an attacker.

Mitigations in this codebase:
- `weights_only=True` is passed to `torch.load()` (PyTorch ≥ 2.0), which disallows arbitrary deserialization.
- File extension is validated before loading (`.pt` / `.pth` only).
- A warning is logged whenever loading from an external path.

**Residual risk:** `weights_only=True` does not protect against all model file attacks. See [THREAT_MODEL.md](THREAT_MODEL.md) for full analysis. Never run this tool against model files from untrusted sources in a production environment.

### 2. This tool is a research / pre-deployment evaluation harness

`adversarial-ml-lab` is designed to run in a **sandboxed, pre-deployment evaluation environment**, not in production inference pipelines. Running adversarial attacks in a live serving environment may:
- Cause significant CPU/GPU load
- Interfere with model serving latency
- Expose gradient information if logging is misconfigured

### 3. Output report files

The benchmark report (`benchmark_report.json`) contains model evaluation metadata but no training data, weights, or PII. It is safe to share as part of security design review documentation.

### 4. Dependency supply chain

Runtime dependencies are declared with lower bounds in `pyproject.toml`, and `uv.lock` is committed for reproducible installs. The CI pipeline runs `bandit -r src/ -ll` on every PR. Dependency updates are gated on passing tests and bandit scans.

---

## Scope

The following are **in scope** for vulnerability reports:

- Arbitrary code execution via crafted model files or benchmark inputs
- Path traversal in `--model-path` or `--output` arguments
- Dependency vulnerabilities in declared runtime or development dependencies
- Any finding that allows an attacker to exfiltrate data or escalate privileges when running this tool

The following are **out of scope**:

- Attacks on the underlying ML models being evaluated (that is the point of the tool)
- Theoretical adversarial attacks that require model access (this tool provides that access by design)
- Denial-of-service via large epsilon values or long benchmark runs (no SLA; this is a research tool)

---

## Disclosure Policy

We follow **coordinated disclosure**. We will work with reporters to understand the issue, develop a fix, and publish a GitHub Security Advisory before or simultaneously with public disclosure. We ask reporters to give us at least **14 days** before publishing details publicly.

We do not operate a bug bounty program for this repository.
