#!/usr/bin/env bash
# 2026-09-26: Sends the saved probe request ($PROBE) to the server on localhost:8888
# and counts the per-layer dumps it wrote to $OUT_DIR. The server must run with
# METRALE_NEMO_DUMP=$OUT_DIR; it then writes metrale_L<i>.bin for each layer,
# metrale_final_norm.bin and metrale_logits.bin. cosine_three_way_phase2b.py reads the
# per-layer files.
#
# Owner: bench, FP8 dequant audit.
# Invariants: none beyond the types.

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
