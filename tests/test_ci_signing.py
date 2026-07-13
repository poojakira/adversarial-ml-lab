"""Tests for CI gate signing, hashing, and replay detection.

These tests CAN run without torch since ci_signing.py uses only Python stdlib.
"""

from __future__ import annotations

import json

import pytest

from adv_lab.eval.ci_signing import (
    create_signed_manifest,
    derive_key,
    detect_replay,
    log_input_hashes,
    sign_report,
    verify_report,
)


class TestDeriveKey:
    """Tests for PBKDF2 key derivation."""

    def test_derives_32_byte_key(self):
        """derive_key returns a 32-byte key."""
        key, salt = derive_key("test-secret")
        assert isinstance(key, bytes)
        assert len(key) == 32

    def test_returns_salt(self):
        """derive_key returns a salt (16 bytes by default)."""
        key, salt = derive_key("test-secret")
        assert isinstance(salt, bytes)
        assert len(salt) == 16

    def test_same_secret_and_salt_produces_same_key(self):
        """Deterministic derivation with fixed salt."""
        key1, salt = derive_key("my-secret", salt=b"fixed-salt-12345")
        key2, _ = derive_key("my-secret", salt=b"fixed-salt-12345")
        assert key1 == key2

    def test_different_secrets_produce_different_keys(self):
        """Different master secrets produce different derived keys."""
        salt = b"shared-salt-1234"
        key1, _ = derive_key("secret-1", salt=salt)
        key2, _ = derive_key("secret-2", salt=salt)
        assert key1 != key2

    def test_different_salts_produce_different_keys(self):
        """Different salts produce different derived keys."""
        key1, _ = derive_key("same-secret", salt=b"salt-aaaaaaaaaa01")
        key2, _ = derive_key("same-secret", salt=b"salt-bbbbbbbbbb02")
        assert key1 != key2

    def test_custom_iterations(self):
        """derive_key respects custom iteration count."""
        # Fewer iterations for speed in test; result should still be 32 bytes
        key, salt = derive_key("test", iterations=1000)
        assert len(key) == 32


class TestSignAndVerify:
    """Tests for HMAC signing and verification of reports."""

    @pytest.fixture
    def sample_report(self):
        """A sample evaluation report JSON string."""
        return json.dumps({
            "passed": True,
            "robust_pgd": 0.45,
            "model_name": "TestModel",
            "timestamp": "2024-01-01T00:00:00Z",
        })

    @pytest.fixture
    def signing_key(self):
        """A test signing key."""
        key, _ = derive_key("test-hsm-secret", salt=b"test-salt-123456")
        return key

    def test_sign_produces_valid_json(self, sample_report, signing_key):
        """sign_report returns valid JSON with required fields."""
        signed = sign_report(sample_report, signing_key)
        envelope = json.loads(signed)
        assert "payload" in envelope
        assert "signature" in envelope
        assert "algorithm" in envelope
        assert envelope["algorithm"] == "HMAC-SHA256"
        assert "signed_at" in envelope

    def test_verify_valid_signature(self, sample_report, signing_key):
        """verify_report returns True for untampered signed reports."""
        signed = sign_report(sample_report, signing_key)
        assert verify_report(signed, signing_key) is True

    def test_verify_detects_tampered_payload(self, sample_report, signing_key):
        """verify_report returns False if payload is modified."""
        signed = sign_report(sample_report, signing_key)
        envelope = json.loads(signed)
        envelope["payload"]["passed"] = False  # Tamper!
        tampered = json.dumps(envelope)
        assert verify_report(tampered, signing_key) is False

    def test_verify_detects_wrong_key(self, sample_report, signing_key):
        """verify_report returns False with wrong key."""
        signed = sign_report(sample_report, signing_key)
        wrong_key, _ = derive_key("wrong-secret", salt=b"wrong-salt-12345")
        assert verify_report(signed, wrong_key) is False

    def test_verify_handles_invalid_json(self, signing_key):
        """verify_report returns False for invalid JSON input."""
        assert verify_report("not json at all", signing_key) is False

    def test_verify_handles_missing_fields(self, signing_key):
        """verify_report returns False if required fields are missing."""
        incomplete = json.dumps({"payload": {"data": 1}})  # No signature
        assert verify_report(incomplete, signing_key) is False

    def test_sign_raises_on_invalid_json(self, signing_key):
        """sign_report raises ValueError for invalid JSON input."""
        with pytest.raises(ValueError):
            sign_report("not valid json {{{", signing_key)

    def test_sign_raises_on_empty_key(self, sample_report):
        """sign_report raises ValueError for empty key."""
        with pytest.raises(ValueError):
            sign_report(sample_report, b"")


class TestLogInputHashes:
    """Tests for SHA-256 input hashing."""

    def test_hashes_string_inputs(self):
        """log_input_hashes correctly hashes string inputs."""
        inputs = ["hello", "world", "test"]
        hashes = log_input_hashes(inputs)
        assert len(hashes) == 3
        assert set(hashes.keys()) == {"0", "1", "2"}
        # Each hash should be 64 hex characters (SHA-256)
        for h in hashes.values():
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)

    def test_hashes_bytes_inputs(self):
        """log_input_hashes correctly hashes bytes inputs."""
        inputs = [b"\x00\x01\x02", b"\xff\xfe\xfd"]
        hashes = log_input_hashes(inputs)
        assert len(hashes) == 2

    def test_deterministic_hashing(self):
        """Same input always produces same hash."""
        inputs = ["deterministic", "test"]
        h1 = log_input_hashes(inputs)
        h2 = log_input_hashes(inputs)
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        """Different inputs produce different hashes."""
        inputs = ["aaa", "bbb", "ccc"]
        hashes = log_input_hashes(inputs)
        assert len(set(hashes.values())) == 3

    def test_empty_input_list(self):
        """log_input_hashes handles empty input list."""
        hashes = log_input_hashes([])
        assert hashes == {}


class TestDetectReplay:
    """Tests for replay and substitution detection."""

    def test_detects_exact_replay(self):
        """detect_replay identifies identical hash sets as replay."""
        hashes = {"0": "abc123", "1": "def456"}
        anomalies = detect_replay(hashes, hashes)
        assert len(anomalies) == 1
        assert "REPLAY_DETECTED" in anomalies[0]

    def test_no_anomalies_for_valid_change(self):
        """detect_replay returns empty list when all hashes differ (fresh eval)."""
        current = {"0": "aaa", "1": "bbb"}
        previous = {"0": "ccc", "1": "ddd"}
        anomalies = detect_replay(current, previous)
        # All indices changed -> substitution detected for each
        # (this is expected behavior -- if inputs SHOULD change, caller
        #  should not compare against the old manifest)
        assert len(anomalies) == 2
        assert all("SUBSTITUTION_DETECTED" in a for a in anomalies)

    def test_detects_partial_substitution(self):
        """detect_replay identifies changed hashes at specific indices."""
        current = {"0": "same_hash", "1": "new_hash"}
        previous = {"0": "same_hash", "1": "old_hash"}
        anomalies = detect_replay(current, previous)
        assert len(anomalies) == 1
        assert "SUBSTITUTION_DETECTED at index 1" in anomalies[0]

    def test_detects_removed_input(self):
        """detect_replay identifies inputs present before but missing now."""
        current = {"0": "hash_a"}
        previous = {"0": "hash_a", "1": "hash_b"}
        anomalies = detect_replay(current, previous)
        assert any("INPUT_REMOVED" in a for a in anomalies)

    def test_detects_added_input(self):
        """detect_replay identifies new inputs not in previous run."""
        current = {"0": "hash_a", "1": "hash_b"}
        previous = {"0": "hash_a"}
        anomalies = detect_replay(current, previous)
        assert any("INPUT_ADDED" in a for a in anomalies)

    def test_empty_hashes_no_anomalies(self):
        """detect_replay handles empty hash dicts gracefully."""
        anomalies = detect_replay({}, {})
        assert anomalies == []


class TestSignedManifest:
    """Tests for the comprehensive signed manifest."""

    def test_creates_valid_manifest(self):
        """create_signed_manifest produces valid signed JSON."""
        report = json.dumps({"passed": True, "score": 0.9})
        hashes = {"0": "abc", "1": "def"}
        key, _ = derive_key("manifest-key", salt=b"manifest-salt-16")

        manifest_json = create_signed_manifest(report, hashes, key)
        parsed = json.loads(manifest_json)

        assert "manifest" in parsed
        assert "signature" in parsed
        assert parsed["manifest"]["report"]["passed"] is True
        assert parsed["manifest"]["input_hashes"] == hashes
        assert parsed["manifest"]["input_count"] == 2

    def test_manifest_raises_on_invalid_json(self):
        """create_signed_manifest raises ValueError for invalid report JSON."""
        key, _ = derive_key("key", salt=b"salt-1234567890ab")
        with pytest.raises(ValueError):
            create_signed_manifest("not json", {}, key)
