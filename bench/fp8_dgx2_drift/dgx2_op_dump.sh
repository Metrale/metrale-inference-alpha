#!/usr/bin/env bash

# 2026-09-26: Starts the op-drift image with the op, GDN and NEMO dump env vars
# set, writing to DUMP_DIR. It sends no request; TOKENS_JSON is not read.
#
# Owner: bench, FP8 drift investigation.
# Invariants: none beyond the types.

set -euo pipefail

CONTAINER=metrale-op-drift
PORT=8888
DUMP_DIR=/workspace/metrale-dumps/op_drift_metrale_dgx2
TOKENS_JSON=/workspace/metrale-mtp/bench/fp8_dgx2_drift/metrale_tokens_dgx2.json

sudo docker stop "$CONTAINER" 2>/dev/null || true
sudo docker rm "$CONTAINER" 2>/dev/null || true
sudo rm -rf "$DUMP_DIR"
mkdir -p "$DUMP_DIR"

# 2026-09-26: METRALE_GDN_DUMP_LAYERS takes SSM-layer indices (the dump's layer is
# its call index modulo METRALE_GDN_DUMP_N_SSM); list all 30.
SSM_LAYERS=$(seq -s, 0 29)

# 2026-09-26: METRALE_OP_DUMP_LAYERS is left unset, which admits every layer. The
# op-dump hooks are in the qwen3_attention prefill paths, so only attention layers
# write.

sudo docker run -d \
  --name "$CONTAINER" \
  --network host \
  --gpus all \
  --ipc=host \
  -v /workspace/.cache/huggingface:/root/.cache/huggingface \
  -v "$DUMP_DIR":/dump \
  -e METRALE_OP_DUMP=/dump \
  -e METRALE_GDN_DUMP=/dump \
  -e METRALE_GDN_DUMP_LAYERS="$SSM_LAYERS" \
  -e METRALE_GDN_DUMP_N_SSM=30 \
  -e METRALE_NEMO_DUMP=/dump \
  -e RUST_LOG=info \
  -e CUDA_VISIBLE_DEVICES=0 \
  metrale-gb10:op-drift \
  serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --port "$PORT" \
    --bind 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --max-seq-len 16384 \
    --max-batch-size 1

echo "Container started: $CONTAINER"
echo "Logs: sudo docker logs -f $CONTAINER"
echo "Dumps will appear in: $DUMP_DIR"
