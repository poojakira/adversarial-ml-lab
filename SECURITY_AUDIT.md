# Security Audit — adversarial-ml-lab

**Audit date:** 2026-09-29  
**Scope:** robustness-evaluation API, artifact loading boundary, auth, resource controls, errors, container, and CI.

## Findings captured before this remediation pass

| ID | Severity | Finding | Status |
|---|---|---|---|
| AML-001 | High | The authenticated evaluation endpoint has no request-rate limiter. | Open |
| AML-002 | Medium | No raw request-body byte limit is enforced before request parsing. | Open |
| AML-003 | Medium | Public 422 responses can include raw `FileNotFoundError`/`ValueError` text, including local artifact details. | Open |
| AML-004 | Info | Caller input cannot select arbitrary model/dataset paths; artifacts are operator configured and model SHA-256 is required. | Verified |

## Existing controls verified

- Fail-closed 32+ character API key with constant-time comparison.
- Bounded attack parameters.
- Evaluation timeout and concurrency semaphore.
- Operator-controlled model/dataset paths.
- Required model SHA-256 pin.
- Non-root container.
- Secret-hygiene CI.

## Verification plan

Add request throttling/byte caps, sanitize validation errors, and run unit/security/production workflows.
