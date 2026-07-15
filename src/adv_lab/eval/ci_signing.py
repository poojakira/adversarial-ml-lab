"""CI Gate Hardening: HMAC signing, input hashing, and replay detection.

An unsigned CI gate is a broken CI gate. This module implements cryptographic
integrity controls for the evaluation pipeline:

1. HMAC-SHA256 signing of all evaluation JSON outputs, ensuring that a passing
   result cannot be forged without the signing key.
2. SHA-256 hashing of all test inputs, enabling detection of replay and
   substitution attacks against the evaluation harness.
3. Key derivation via PBKDF2 for HSM simulation (hardware root of trust).

All implementations use Python stdlib only (hashlib, hmac, json, os, base64).
No external dependencies are required.

Threat model: an attacker with write access to the CI artifact store attempts
to (a) forge a passing evaluation report, (b) replay a previously passing
evaluation against different model weights, or (c) substitute benign test
inputs for adversarial ones during evaluation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time


# Key derivation constants following NIST SP 800-132 recommendations
_PBKDF2_ITERATIONS = 600_000  # OWASP 2023 recommendation for SHA-256
_PBKDF2_HASH = "sha256"
_KEY_LENGTH = 32  # 256-bit derived key
_SALT_LENGTH = 16  # 128-bit random salt


def derive_key(
    master_secret: str,
    salt: bytes | None = None,
    iterations: int = _PBKDF2_ITERATIONS,
) -> tuple[bytes, bytes]:
    """Derive a signing key from a master secret using PBKDF2-HMAC-SHA256.

    Simulates key derivation from a hardware root of trust (HSM). The master
    secret would typically be stored in an HSM or secure enclave; this function
    performs the KDF step that produces an ephemeral signing key.

    Args:
        master_secret: The master secret (from HSM or environment variable).
        salt: Optional salt bytes. If None, generates a cryptographically
            random 128-bit salt.
        iterations: PBKDF2 iteration count. Default follows OWASP 2023
            guidance (600,000 for SHA-256).

    Returns:
        Tuple of (derived_key, salt). The salt must be stored alongside signed
        artifacts so the key can be re-derived for verification.
    """
    if salt is None:
        salt = os.urandom(_SALT_LENGTH)

    derived = hashlib.pbkdf2_hmac(
        _PBKDF2_HASH,
        master_secret.encode("utf-8"),
        salt,
        iterations,
        dklen=_KEY_LENGTH,
    )
    return derived, salt


def sign_report(report_json: str, key: bytes) -> str:
    """Sign an evaluation report JSON string with HMAC-SHA256.

    The signature covers the entire JSON payload (canonicalized). The output
    is a new JSON string containing the original payload plus the signature
    and metadata needed for verification.

    Args:
        report_json: The JSON string to sign (evaluation report).
        key: HMAC signing key (typically from derive_key()).

    Returns:
        A JSON string containing the original report under the "payload" key,
        plus "signature" (hex-encoded HMAC-SHA256) and "signed_at" timestamp.

    Raises:
        ValueError: If report_json is not valid JSON or key is empty.
    """
    if not key:
        raise ValueError("Signing key must not be empty")

    # Validate JSON
    try:
        payload = json.loads(report_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"report_json is not valid JSON: {e}") from e

    # Canonicalize: sort keys, no extra whitespace
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    # Compute HMAC-SHA256 over the canonical representation
    signature = hmac.HMAC(key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()

    signed_envelope = {
        "payload": payload,
        "signature": signature,
        "algorithm": "HMAC-SHA256",
        "signed_at": time.time(),
        "canonical_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }

    return json.dumps(signed_envelope, indent=2, sort_keys=True)


def verify_report(signed_json: str, key: bytes) -> bool:
    """Verify the HMAC-SHA256 signature of a signed evaluation report.

    Re-computes the HMAC over the canonical payload and compares it to the
    stored signature using constant-time comparison to prevent timing attacks.

    Args:
        signed_json: The signed JSON envelope (output of sign_report()).
        key: The same HMAC key used for signing.

    Returns:
        True if the signature is valid and the payload has not been tampered
        with. False otherwise.
    """
    try:
        envelope = json.loads(signed_json)
    except (json.JSONDecodeError, TypeError):
        return False

    if "payload" not in envelope or "signature" not in envelope:
        return False

    payload = envelope["payload"]
    stored_signature = envelope["signature"]

    # Re-canonicalize the payload identically to sign_report
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    # Recompute HMAC
    expected_signature = hmac.HMAC(
        key, canonical.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    # Constant-time comparison to prevent timing side-channels
    return hmac.compare_digest(expected_signature, stored_signature)


def log_input_hashes(inputs: list[bytes | str]) -> dict[str, str]:
    """Compute SHA-256 hash of every evaluation input for audit trail.

    Every evaluation run must record the cryptographic hash of each test input
    so that replay and substitution attacks on the evaluation harness are
    detectable. If an attacker swaps adversarial test inputs for benign ones,
    the hash log will not match the expected manifest.

    Args:
        inputs: List of input data items. Strings are UTF-8 encoded before
            hashing. Bytes are hashed directly.

    Returns:
        Dictionary mapping input index (as string) to hex-encoded SHA-256 hash.
        Example: {"0": "a1b2c3...", "1": "d4e5f6..."}
    """
    hashes: dict[str, str] = {}
    for idx, item in enumerate(inputs):
        if isinstance(item, str):
            data = item.encode("utf-8")
        else:
            data = item
        h = hashlib.sha256(data).hexdigest()
        hashes[str(idx)] = h
    return hashes


def detect_replay(
    current_hashes: dict[str, str],
    previous_hashes: dict[str, str],
) -> list[str]:
    """Detect replay or substitution attacks by comparing input hash logs.

    Compares the current evaluation's input hashes against a previous run's
    hashes. Any matching hashes across runs (when they should differ) indicate
    potential replay. Any mismatched hashes for the same index indicate
    potential substitution.

    Detection logic:
    - If a hash at index i differs between runs: that index was substituted.
    - If the exact same hash set appears: potential full replay attack.
    - Returns a list of anomaly descriptions.

    Args:
        current_hashes: Hash log from the current evaluation run.
        previous_hashes: Hash log from a previous (known-good) evaluation run.

    Returns:
        List of anomaly strings. Empty list means no anomalies detected.
        Each entry describes the type of anomaly and affected index.
    """
    anomalies: list[str] = []

    # Check for exact replay (all hashes identical)
    if current_hashes == previous_hashes and len(current_hashes) > 0:
        anomalies.append(
            "REPLAY_DETECTED: current input hashes are identical to previous "
            "run. This evaluation may be a replay of a cached result."
        )
        return anomalies

    # Check for per-index substitution
    all_indices = set(current_hashes.keys()) | set(previous_hashes.keys())
    for idx in sorted(all_indices, key=lambda x: int(x) if x.isdigit() else x):
        curr = current_hashes.get(idx)
        prev = previous_hashes.get(idx)

        if curr is not None and prev is not None and curr != prev:
            anomalies.append(
                f"SUBSTITUTION_DETECTED at index {idx}: hash changed from "
                f"{prev[:16]}... to {curr[:16]}..."
            )
        elif curr is None and prev is not None:
            anomalies.append(
                f"INPUT_REMOVED at index {idx}: present in previous run but "
                f"missing in current run"
            )
        elif curr is not None and prev is None:
            anomalies.append(
                f"INPUT_ADDED at index {idx}: not present in previous run but "
                f"added in current run"
            )

    return anomalies


def create_signed_manifest(
    report_json: str,
    input_hashes: dict[str, str],
    key: bytes,
) -> str:
    """Create a comprehensive signed manifest combining report and input hashes.

    This is the recommended way to produce a tamper-evident CI artifact:
    it binds the evaluation result to the exact inputs used, preventing both
    result forgery and input substitution in a single signed envelope.

    Args:
        report_json: The evaluation report JSON string.
        input_hashes: Hash log from log_input_hashes().
        key: HMAC signing key.

    Returns:
        Signed JSON string containing report, input hashes, and HMAC signature.
    """
    try:
        report_payload = json.loads(report_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"report_json is not valid JSON: {e}") from e

    manifest = {
        "report": report_payload,
        "input_hashes": input_hashes,
        "input_count": len(input_hashes),
    }

    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    signature = hmac.HMAC(
        key, manifest_json.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    signed = {
        "manifest": manifest,
        "signature": signature,
        "algorithm": "HMAC-SHA256",
        "signed_at": time.time(),
    }

    return json.dumps(signed, indent=2, sort_keys=True)
