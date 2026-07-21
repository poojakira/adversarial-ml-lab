"""CLI verifier for signed adversarial benchmark JSON artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from adv_lab.eval.ci_signing import verify_report

VALID = "VALID"
TAMPERED = "TAMPERED"
UNSIGNED = "UNSIGNED"


def classify_report(signed_json: str, key: bytes) -> str:
    """Return VALID, TAMPERED, or UNSIGNED for a benchmark artifact."""
    try:
        envelope = json.loads(signed_json)
    except json.JSONDecodeError:
        return TAMPERED
    if not isinstance(envelope, dict):
        return TAMPERED
    if "signature" not in envelope:
        return UNSIGNED
    if "payload" not in envelope:
        return TAMPERED
    return VALID if verify_report(signed_json, key) else TAMPERED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify signed benchmark results.")
    parser.add_argument("--results", required=True, help="Path to signed benchmark JSON")
    parser.add_argument(
        "--key",
        default=None,
        help="HMAC verification key. Defaults to ADV_LAB_HMAC_KEY.",
    )
    args = parser.parse_args(argv)

    key_text = args.key if args.key is not None else os.environ.get("ADV_LAB_HMAC_KEY", "")
    data = Path(args.results).read_text(encoding="utf-8")
    verdict = classify_report(data, key_text.encode("utf-8"))
    print(verdict)
    if verdict == VALID:
        return 0
    if verdict == UNSIGNED:
        return 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())