# Supply-Chain & CI Hardening

This lab runs adversarial attack code and produces a CI-gateable robustness
benchmark. Both the *integrity of results* and the *isolation of execution*
matter. This document is the policy behind the controls in the repo.

## 1. Signing key management (`adv_lab.eval.ci_signing`)

The evaluation report is signed with HMAC-SHA256. An HMAC signature is only
meaningful if the key is secret and managed — a hardcoded/shared key makes a
"passing" report trivially forgeable.

**Key sourcing (fail-closed).** Use `load_signing_key()`, which resolves the
master secret in this order and refuses to sign with a default:

1. `ADV_LAB_SIGNING_SECRET_FILE` — path to a file mounted from a secret store
   (GitHub Actions secret written to a file, HashiCorp Vault Agent, AWS/GCP
   Secrets Manager, or an HSM-backed tmpfs). **Preferred** — keeps the secret
   out of the process environment.
2. `ADV_LAB_SIGNING_SECRET` — the secret value directly (e.g. a GitHub Actions
   `secrets.*` env var).

Store the derivation `salt` (`ADV_LAB_SIGNING_SALT`) next to signed artifacts so
verifiers can re-derive the key.

**Rotation.** Rotate the master secret on a fixed cadence and on suspected
compromise. Bump a key version, re-sign current artifacts, and have verifiers
fetch the active version. Never reuse a revoked secret.

**Revocation.** Remove/rotate the secret in the store; artifacts signed with the
old key then fail `verify_report()` and must be re-signed by an authorized
runner.

### Preferred: keyless signing with Sigstore / cosign

Shared HMAC secrets still require distribution and rotation. For CI artifacts,
prefer **Sigstore keyless signing**, which uses short-lived, OIDC-issued
certificates and a public transparency log (Rekor) — no long-lived secret to
leak:

```yaml
# In CI (needs: permissions: id-token: write)
- uses: sigstore/cosign-installer@v3
- run: |
    cosign sign-blob --yes \
      --output-signature results/report.json.sig \
      --output-certificate results/report.json.pem \
      results/report.json
# Verify (anyone, no shared secret):
- run: |
    cosign verify-blob \
      --certificate results/report.json.pem \
      --signature results/report.json.sig \
      --certificate-identity-regexp '.*' \
      --certificate-oidc-issuer https://token.actions.githubusercontent.com \
      results/report.json
```

Keep the HMAC manifest for offline/air-gapped verification; use Sigstore for
public, transparency-logged provenance.

## 2. Sandboxing attack execution

The 20+ attack modules (`poisoning.py`, `llm.py`, `evasion.py`, …) generate
poisoned data and potentially offensive content. They must never run
unsandboxed on a CI host or workstation.

Use the provided isolation image and wrapper:

```bash
scripts/run_sandboxed.sh                                   # test suite
scripts/run_sandboxed.sh adv_lab.eval.harness --n-samples 100 --output /out/report.json
```

The wrapper enforces `--network none`, `--cap-drop=ALL`,
`--security-opt no-new-privileges`, a read-only root filesystem, and pid/memory
/cpu limits (see `Dockerfile.sandbox` and `scripts/run_sandboxed.sh`). On hosts
without Docker, an equivalent `nsjail` profile (`--disable_clone_newnet` off,
seccomp default, no new privs) achieves the same isolation.

## 3. Dependency pinning with hashes

`pyproject.toml` uses lower-bound ranges (`torch>=2.3`) for flexibility. For
reproducible, tamper-resistant installs use the pinned lock file:

```bash
# Regenerate with hashes:
pip install pip-tools
pip-compile --generate-hashes --extra dev \
    --output-file=requirements-lock.txt pyproject.toml

# Install with hash enforcement (rejects any mismatched artifact):
pip install --require-hashes -r requirements-lock.txt \
    --index-url https://download.pytorch.org/whl/cpu
```

`requirements-lock.txt` ships exact `==` pins as a baseline; replace them with
the hash-annotated `pip-compile` output to enable `--require-hashes` end to end.

## 4. CI least-privilege

`.github/workflows/ci.yml` sets a minimal top-level `permissions:` block, uses
`persist-credentials: false` on checkout, installs from the pinned lock file,
and (when `ADV_LAB_SIGNING_SECRET` is configured) signs the benchmark report.
Pin third-party actions to a full commit SHA rather than a moving tag for
strongest integrity.
