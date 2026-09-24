#!/usr/bin/env bash
# Phase 2b replay: feed the same 30-turn / 18920-token prompt to the RNE-image
# Metrale Engine server and capture per-layer METRALE_NEMO_DUMP at /workspace/metrale-dumps/numdrift/rne/.
#
# Prereqs:
#   - metrale-qwen container running on `metrale-gb10:fp8-dequant-rne` image with
#     `-e METRALE_NEMO_DUMP=/workspace/metrale-dumps/numdrift/rne` env var
#   - localhost:8888 reachable
#
# Output:
#   /workspace/metrale-dumps/numdrift/rne/metrale_L{0..39}.bin
#   /workspace/metrale-dumps/numdrift/rne/metrale_final_norm.bin
#   /workspace/metrale-dumps/numdrift/rne/metrale_logits.bin
#
# After this completes, run cosine_three_way_phase2b.py for the verdict.

set -euo pipefail

PROBE=/workspace/metrale-dumps/numdrift/metrale_turn11_probe.json
OUT_DIR=/workspace/metrale-dumps/numdrift/rne

if [[ ! -f "$PROBE" ]]; then
    echo "ERROR: probe prompt not found at $PROBE" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "Sending probe request (18920-token prompt) at $(date)"
curl -s -X POST http://localhost:8888/v1/chat/completions \
    -H "Content-Type: application/json" \
    --data-binary @"$PROBE" \
    -o "$OUT_DIR/response.jsonl" \
    || { echo "request failed"; exit 1; }

echo "Server responded; checking METRALE_NEMO_DUMP output at $OUT_DIR"
ls "$OUT_DIR"/metrale_L*.bin 2>&1 | wc -l
echo "If the count is 40, dump is complete."
echo "Generated tokens preview (first 200 chars of stream):"
head -c 200 "$OUT_DIR/response.jsonl"
echo
