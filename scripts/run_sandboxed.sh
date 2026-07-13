#!/bin/bash
# Run adversarial attack / evaluation code inside a locked-down container.
#
# Attack modules (poisoning.py, llm.py, evasion.py, ...) must run isolated so a
# bug or malicious payload cannot touch the network, escalate privileges, or
# write outside a scratch dir. This wrapper enforces:
#   --network none        no exfiltration / no pulling remote weights
#   --cap-drop=ALL        drop all Linux capabilities
#   --security-opt no-new-privileges
#   --read-only           immutable root fs (+ tmpfs for scratch)
#   --pids-limit/--memory bound blast radius
#   non-root user (baked into Dockerfile.sandbox)
#
# Usage:
#   scripts/run_sandboxed.sh                       # runs the test suite
#   scripts/run_sandboxed.sh adv_lab.eval.harness --n-samples 100 --output /out/report.json
#
# Results written to ./sandbox-out on the host (mounted at /out).
set -euo pipefail

IMAGE="${ADV_LAB_SANDBOX_IMAGE:-adv-lab-sandbox:latest}"
OUT_DIR="${ADV_LAB_SANDBOX_OUT:-$(pwd)/sandbox-out}"
mkdir -p "$OUT_DIR"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is required. (nsjail is a lighter alternative — see docs/SUPPLY_CHAIN.md)" >&2
    exit 1
fi

# Build if the image is missing.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Building sandbox image $IMAGE ..."
    docker build -f Dockerfile.sandbox -t "$IMAGE" .
fi

exec docker run --rm \
    --network none \
    --cap-drop=ALL \
    --security-opt no-new-privileges \
    --read-only \
    --tmpfs /tmp:rw,size=256m \
    --pids-limit 256 \
    --memory 4g \
    --cpus 2 \
    -v "$OUT_DIR:/out:rw" \
    "$IMAGE" "$@"
