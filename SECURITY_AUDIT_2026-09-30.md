# Security review, 30 September 2026

Reviewed baseline: `2a5c783446413c3ad5ead370565b99ed4057c9cb`. Source review and focused regression verification; this is not proof that all vulnerabilities are absent.

## Fixes and reviewed controls

Streaming body limit runs before parsing and constant-time comparisons use UTF-8 bytes. Runtime dependency floors exclude audited vulnerable Starlette/AnyIO versions. Operator-configured artifact paths, SHA-256-pinned TorchScript loading and allow_pickle=False dataset loading were inspected.

## Verification

109 tests passed in the full suite with OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1; an earlier unrestricted-thread run exceeded 90 seconds. Tests ran in an isolated Python 3.12 environment. FastAPI TestClient required execution outside the default sandbox; a minimal unchanged app reproduced the sandbox deadlock. Final installed-environment pip-audit reported no known vulnerabilities. This does not cover every optional dependency, every container image, or arbitrary older environments allowed by broad dependency bounds.

## Secret history review

One historical match was a synthetic API test key. No tracked environment or private-key paths found in fetched history. Gitleaks classifications are pattern matches, not provider validity checks. No provider key was tested or revoked, and fetched Git refs do not include every cached/forked copy. `.env` and local credential patterns remain ignored; example files must contain placeholders only.

## Deployment and remaining limits

A shared service key authenticates every holder as an evaluation operator; no tenant/role authorization exists. No caller-controlled upload path is accepted. TorchScript still executes model operations and must come from a trusted operator-controlled artifact. Evaluation timeout does not kill running worker threads; deploy killable worker processes and resource limits. Rate state is per process.
