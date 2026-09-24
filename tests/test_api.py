from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

API_KEY = "adversarial-evaluation-api-key-at-least-32-chars"


def _client(monkeypatch, tmp_path: Path):
    model = tmp_path / "model.ts"
    dataset = tmp_path / "eval.npz"
    model.write_bytes(b"trusted-model-placeholder")
    dataset.write_bytes(b"trusted-dataset-placeholder")
    monkeypatch.setenv("ADV_API_KEY", API_KEY)
    monkeypatch.setenv("ADV_MODEL_PATH", str(model))
    monkeypatch.setenv("ADV_DATASET_PATH", str(dataset))
    monkeypatch.setenv("ADV_MODEL_SHA256", hashlib.sha256(model.read_bytes()).hexdigest())
    import adv_lab.api as api

    api = importlib.reload(api)
    return api, TestClient(api.app)


def test_health_is_public(monkeypatch, tmp_path):
    _, client = _client(monkeypatch, tmp_path)
    assert client.get("/health").status_code == 200


def test_ready_requires_secure_configuration(monkeypatch, tmp_path):
    _, client = _client(monkeypatch, tmp_path)
    assert client.get("/ready").status_code == 200


def test_evaluate_requires_api_key(monkeypatch, tmp_path):
    _, client = _client(monkeypatch, tmp_path)
    assert client.post("/evaluate", json={}).status_code == 401


def test_evaluate_uses_only_operator_configured_artifacts(monkeypatch, tmp_path):
    api, client = _client(monkeypatch, tmp_path)
    report = {
        "tool": "adversarial-ml-lab",
        "production_mode": True,
        "pass_fail": "PASS",
    }
    with patch.object(api, "benchmark_runner", return_value=report) as runner:
        response = client.post(
            "/evaluate",
            headers={"X-API-Key": API_KEY},
            json={"epsilon": 0.03, "pgd_steps": 20, "batch_size": 16},
        )

    assert response.status_code == 200
    kwargs = runner.call_args.kwargs
    assert kwargs["production"] is True
    assert kwargs["model_format"] == "torchscript"
    assert (
        kwargs["expected_model_sha256"]
        == hashlib.sha256(Path(kwargs["model_path"]).read_bytes()).hexdigest()
    )
    assert Path(kwargs["model_path"]).name == "model.ts"
    assert Path(kwargs["dataset_path"]).name == "eval.npz"


def test_request_parameters_are_bounded(monkeypatch, tmp_path):
    _, client = _client(monkeypatch, tmp_path)
    response = client.post(
        "/evaluate",
        headers={"X-API-Key": API_KEY},
        json={"epsilon": 2.0, "pgd_steps": 1000, "batch_size": 10000},
    )
    assert response.status_code == 422
