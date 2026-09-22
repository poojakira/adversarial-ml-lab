FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip build \
    && python -m build --wheel --outdir /wheels

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ADV_MODEL_PATH=/models/model.ts \
    ADV_DATASET_PATH=/data/eval.npz

RUN groupadd --system evaluator \
    && useradd --system --gid evaluator --create-home --home-dir /home/evaluator evaluator \
    && mkdir -p /models /data \
    && chown evaluator:evaluator /models /data

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl \
    && python -m pip install --no-cache-dir "fastapi>=0.115,<1" "uvicorn[standard]>=0.32,<1" "pydantic>=2.9,<3" \
    && rm -rf /wheels

USER evaluator
WORKDIR /work

VOLUME ["/models", "/data"]
EXPOSE 8007

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8007/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "adv_lab.api:app", "--host", "0.0.0.0", "--port", "8007", "--workers", "1", "--no-access-log"]
