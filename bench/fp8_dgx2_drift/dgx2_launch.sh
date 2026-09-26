#!/usr/bin/env bash

# 2026-09-26: Starts the Qwen3.6-35B-A3B-FP8 server in a container with
# METRALE_NEMO_DUMP writing to DUMP_DIR.
#
# Owner: bench, FP8 drift investigation.
# Invariants: none beyond the types.
set -euo pipefail

DUMP_DIR="/workspace/metrale-dumps/fp8native_dgx2"
mkdir -p "${DUMP_DIR}"

sudo docker rm -f metrale-dgx2-dump 2>/dev/null || true

sudo docker run -d \
  --name metrale-dgx2-dump \
  --network host \
  --gpus all \
  --ipc=host \
  --security-opt label=disable \
  --runtime nvidia \
  -e RUST_LOG=info \
  -e METRALE_NEMO_DUMP="${DUMP_DIR}" \
  -e METRALE_DFLASH_DEBUG_DUMP_FULL=1 \
  -v "/workspace/.cache/huggingface:/root/.cache/huggingface" \
  -v "/workspace:/workspace" \
  metrale-gb10:fp8-much-better \
  serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --port 8888 \
    --bind 0.0.0.0 \
    --max-seq-len 65536 \
    --max-batch-size 4 \
    --gpu-memory-utilization 0.88 \
    --kv-cache-dtype fp8 \
    --scheduler slai \
    --enable-prefix-caching

echo "started; tail logs with: sudo docker logs -f metrale-dgx2-dump"
