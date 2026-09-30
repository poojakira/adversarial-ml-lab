# Security Audit — 2026-09-30

## Scope
Initial pre-remediation review of the current `main` branch.

## Runtime surface
Authenticated FastAPI adversarial-evaluation service using operator-configured model/dataset artifacts.

## Verified controls
- Caller cannot choose arbitrary model/dataset paths through the API.
- API key is required with constant-time comparison.
- Evaluation parameters, concurrency, and execution timeout are bounded.
- Configured model integrity is SHA-256 pinned.
- No confirmed live API key was found in the current main branch.

## Findings to remediate/verify
1. Add request-rate limiting and request-byte limits.
2. Ensure model/dataset configured paths cannot traverse symlinks into unintended sensitive locations.
3. Confirm temporary outputs are created with restrictive permissions and deleted on all failure paths.
4. Return only generic internal errors; log detailed exceptions server-side.
5. Add structured critical alerts and health-gated deployment rollback.

## Not applicable
SQL tenant isolation, password reset, browser XSS, payments.
