"""Authenticated service for bounded adversarial robustness evaluations.

The network boundary never accepts model or dataset paths from callers. Operators
configure one trusted, SHA-256-pinned TorchScript model and one immutable NPZ
evaluation set via environment variables; requests may only tune bounded evaluation
parameters.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from adv_lab.body_limit import RequestBodyLimit
from adv_lab.eval.benchmark_runner import PGD_ROBUST_ACC_GATE, benchmark_runner

_MAX_CONCURRENT = int(os.environ.get("ADV_MAX_CONCURRENT_EVALUATIONS", "1"))
_TIMEOUT_SECONDS = float(os.environ.get("ADV_EVALUATION_TIMEOUT_SECONDS", "300"))
_MAX_REQUEST_BYTES = int(os.environ.get("ADV_MAX_REQUEST_BYTES", "16384"))
_RATE_LIMIT_RPM = int(os.environ.get("ADV_RATE_LIMIT_RPM", "30"))
_request_log: dict[str, list[float]] = defaultdict(list)
_RATE_KEY_SECRET = os.urandom(32)
if _MAX_CONCURRENT < 1 or _MAX_CONCURRENT > 8:
    raise RuntimeError("ADV_MAX_CONCURRENT_EVALUATIONS must be between 1 and 8")
if _TIMEOUT_SECONDS <= 0 or _TIMEOUT_SECONDS > 3600:
    raise RuntimeError("ADV_EVALUATION_TIMEOUT_SECONDS must be in (0, 3600]")
if _MAX_REQUEST_BYTES < 1024 or _MAX_REQUEST_BYTES > 1024 * 1024:
    raise RuntimeError("ADV_MAX_REQUEST_BYTES must be between 1024 and 1048576")
if _RATE_LIMIT_RPM < 1 or _RATE_LIMIT_RPM > 10000:
    raise RuntimeError("ADV_RATE_LIMIT_RPM must be between 1 and 10000")

_slots = asyncio.Semaphore(_MAX_CONCURRENT)

app = FastAPI(
    title="Adversarial ML Robustness Evaluation Service",
    version="1.1.0",
    description=(
        "Run bounded FGSM/PGD robustness evaluations against operator-configured artifacts."
    ),
)


class EvaluationRequest(BaseModel):
    epsilon: float = Field(default=8 / 255, ge=0.0, le=0.5)
    pgd_steps: int = Field(default=40, ge=1, le=200)
    batch_size: int = Field(default=32, ge=1, le=512)
    min_pgd_robust_accuracy: float = Field(
        default=PGD_ROBUST_ACC_GATE,
        ge=0.0,
        le=1.0,
    )


class EvaluationResponse(BaseModel):
    evaluation_id: str
    duration_ms: float
    report: dict[str, object]


def _is_rate_limited(request: Request) -> bool:
    supplied = request.headers.get("X-API-Key", "")
    peer = request.client.host if request.client else "unknown"
    material = f"{peer}\0{supplied}".encode("utf-8")
    key = hmac.new(_RATE_KEY_SECRET, material, hashlib.sha256).hexdigest()[:32]
    now = time.time()
    cutoff = now - 60.0
    hits = [stamp for stamp in _request_log[key] if stamp > cutoff]
    if len(hits) >= _RATE_LIMIT_RPM:
        _request_log[key] = hits
        return True
    hits.append(now)
    _request_log[key] = hits
    return False


app.add_middleware(RequestBodyLimit, max_bytes=_MAX_REQUEST_BYTES)


@app.middleware("http")
async def _request_size_limit(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def _configured_api_key() -> str:
    key = os.environ.get("ADV_API_KEY", "")
    if len(key) < 32:
        raise HTTPException(
            status_code=503,
            detail="ADV_API_KEY is not securely configured",
        )
    return key


def _artifact_config() -> tuple[Path, Path, str]:
    model = Path(os.environ.get("ADV_MODEL_PATH", "")).expanduser()
    dataset = Path(os.environ.get("ADV_DATASET_PATH", "")).expanduser()
    digest = os.environ.get("ADV_MODEL_SHA256", "").strip().lower()
    if not model.is_file():
        raise HTTPException(
            status_code=503,
            detail="Configured TorchScript model is unavailable",
        )
    if not dataset.is_file():
        raise HTTPException(
            status_code=503,
            detail="Configured evaluation dataset is unavailable",
        )
    if len(digest) != 64:
        raise HTTPException(
            status_code=503,
            detail="ADV_MODEL_SHA256 is not securely configured",
        )
    try:
        int(digest, 16)
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="ADV_MODEL_SHA256 must be hexadecimal",
        ) from exc
    return model.resolve(), dataset.resolve(), digest


def _require_api_key(request: Request) -> None:
    expected = _configured_api_key()
    supplied = request.headers.get("X-API-Key", "")
    if not supplied or not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "adversarial-ml-lab"}


@app.get("/ready")
def ready() -> dict[str, object]:
    _configured_api_key()
    model, dataset, _ = _artifact_config()
    return {
        "status": "ready",
        "model": model.name,
        "dataset": dataset.name,
        "max_concurrent_evaluations": _MAX_CONCURRENT,
        "evaluation_timeout_seconds": _TIMEOUT_SECONDS,
    }


def _run_evaluation(payload: EvaluationRequest, output_path: str) -> dict[str, object]:
    model, dataset, model_sha256 = _artifact_config()
    return benchmark_runner(
        model_path=str(model),
        dataset_path=str(dataset),
        model_format="torchscript",
        expected_model_sha256=model_sha256,
        production=True,
        epsilon=payload.epsilon,
        pgd_steps=payload.pgd_steps,
        batch_size=payload.batch_size,
        min_pgd_robust_accuracy=payload.min_pgd_robust_accuracy,
        output_path=output_path,
    )


@app.post(
    "/evaluate",
    response_model=EvaluationResponse,
    dependencies=[Depends(_require_api_key)],
)
async def evaluate(payload: EvaluationRequest, request: Request) -> EvaluationResponse:
    if _is_rate_limited(request):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(
        prefix="adv-eval-", suffix=".json", delete=True
    ) as report_file:
        try:
            async with _slots:
                report = await asyncio.wait_for(
                    run_in_threadpool(_run_evaluation, payload, report_file.name),
                    timeout=_TIMEOUT_SECONDS,
                )
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Evaluation timed out") from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail="Evaluation configuration is invalid",
            ) from exc

    return EvaluationResponse(
        evaluation_id=str(uuid.uuid4()),
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
        report=report,
    )
