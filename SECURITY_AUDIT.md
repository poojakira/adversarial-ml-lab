# Security Audit — adversarial-ml-lab

**Audit date:** 2026-09-29  
**Scope:** robustness-evaluation API, artifact loading boundary, auth, resource controls, errors, container, and CI.

## Findings captured before this remediation pass

| ID | Severity | Finding | Status |
|---|---|---|---|
| AML-001 | High | The authenticated evaluation endpoint now rate-limits by peer/API-key identity before evaluation work. | Fixed |
| AML-002 | Medium | HTTP middleware now enforces a bounded raw request-body size before evaluation handling. | Fixed |
| AML-003 | Medium | Artifact/configuration exceptions are now mapped to the generic public response `Evaluation configuration is invalid`. | Fixed |
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
