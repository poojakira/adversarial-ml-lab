from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "results" / "cifar10_smallcnn_real.json"


def test_committed_cifar10_measurement_has_reproducible_provenance():
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert data["_schema"] == "adv-lab-benchmark-v1"
    assert data["_synthetic"] is False
    assert "REAL_MEASURED" in data["_status"]
    assert "seed=42" in data["_reproducibility"]
    assert data["dataset"].startswith("CIFAR-10")
    assert data["seed"] == 42
    assert re.fullmatch(r"[0-9a-f]{40}", data["commit_sha"])
    assert data["framework"]["device"] == "cpu"
    assert data["framework"]["cuda_available"] is False


def test_committed_attack_metrics_are_bounded_and_scoped():
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    results = data["results"]

    clean = results["clean_accuracy"]
    assert 0.0 <= clean <= 1.0
    assert results["clean_eval_samples"] == 10_000

    for attack in ("fgsm", "pgd_linf", "cw_l2"):
        result = results[attack]
        assert 0.0 <= result["robust_accuracy"] <= 1.0
        assert result["samples"] > 0
        assert result["robust_accuracy"] <= clean

    assert results["pgd_linf"]["epsilon_255"] == "8/255"
    assert results["pgd_linf"]["steps"] == 20
    assert results["pgd_linf"]["samples"] == 1024


def test_measurement_cannot_be_misrepresented_as_state_of_the_art():
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    boundary = data["compute_budget"].upper()
    assert "NOT SOTA" in boundary
    assert "CPU" in boundary
    assert "SUBSET" in boundary
