from __future__ import annotations

import json

from adv_lab.eval.benchmark_verify import TAMPERED, UNSIGNED, VALID, classify_report
from adv_lab.eval.ci_signing import sign_report


def test_classify_valid_signed_report():
    signed = sign_report(json.dumps({"passed": True}), b"secret")
    assert classify_report(signed, b"secret") == VALID


def test_classify_tampered_report():
    signed = sign_report(json.dumps({"passed": True}), b"secret")
    envelope = json.loads(signed)
    envelope["payload"]["passed"] = False
    assert classify_report(json.dumps(envelope), b"secret") == TAMPERED


def test_classify_unsigned_report():
    assert classify_report(json.dumps({"passed": True}), b"secret") == UNSIGNED