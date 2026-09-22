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
    PYTHONUNBUFFERED=1

RUN groupadd --system evaluator \
    && useradd --system --gid evaluator --create-home --home-dir /home/evaluator evaluator

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

USER evaluator
WORKDIR /work

# This image is a batch admission job, not a network service. Production mode
# refuses to run without an explicit TorchScript model and NPZ evaluation set.
ENTRYPOINT ["python", "-m", "adv_lab.eval.benchmark_runner", "--production"]
