#!/bin/bash

# 2026-09-26: Concurrency sweep driver: bench-metrale-concurrency.py against a
# Metrale Engine leg, then a vLLM leg on the same box, then a comparison.
#
# Owner: bench.
# Invariants: none beyond the types.
#
# A leg whose results JSON already exists is skipped, so a re-run continues where
# a killed run stopped; each leg appends a line to STATE.md. The vLLM leg tries the
# Metrale Engine checkpoint first, falls back to nvidia/Qwen3.6-27B-NVFP4, and
# records the checkpoint it served in the results JSON.
set -u
WT=/workspace/.wt-golden
CS=$WT/conc_sweep
RESULTS=$CS/results
STATE=$WT/docs/campaigns/gb10-concurrency-2026-07/STATE.md
BENCH=$WT/bench/bench-metrale-concurrency.py
METRALE_BIN=$WT/conc_sweep/met_phaseA_baseline
MODEL_METRALE=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
VLLM_IMAGE=sparkrun-eugr-vllm:latest
PORT=8888
mkdir -p "$RESULTS"

note() { # 2026-09-26: append a line to STATE.md and echo it
  echo "- $(date -u +%FT%TZ) $*" >> "$STATE"
  echo "STATE: $*"
}

teardown() { sudo docker rm -f metrale-csweep vllm-csweep >/dev/null 2>&1; sleep 3; }

wait_health() { # 2026-09-26: $1 container, $2 grep pattern, $3 max tries (5 s apart)
  for _ in $(seq 1 "$3"); do
    curl -sf -m4 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$2" && return 0
    sudo docker ps --format '{{.Names}}' | grep -q "$1" || return 1
    sleep 5
  done
  return 1
}

run_bench() { # 2026-09-26: $1 results file tag
  BENCH_PORT=$PORT BENCH_MAX_SEQ_LEN=4096 \
    BENCH_RESULTS_FILE="$RESULTS/$1.json" \
    python3 -u "$BENCH" 2>&1 | tail -30
  [ -s "$RESULTS/$1.json" ]
}

if [ -s "$RESULTS/metrale_synth.json" ]; then
  echo "SKIP metrale_synth (results exist)"
else
  teardown
  sudo docker run -d --name metrale-csweep --network host --gpus all --ipc=host \
    -e METRALE_NO_FFN_NVFP4_MMQ=1 -e METRALE_SSM_TAIL_MIDCHUNK=0 -e METRALE_MTP_CATCHUP=0 \
    -e METRALE_MTP_DRAFT_CONF=0.0 -e METRALE_MTP_GATE_FORCE=1 \
    -e METRALE_SSM_TAIL_LEASE_TTL=128 -e METRALE_BF16_TC_PREFILL=1 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
    -v "$METRALE_BIN:/usr/local/bin/met:ro" \
    metrale-gb10:followups serve "$MODEL_METRALE" \
    --host 0.0.0.0 --port $PORT --model-name "$MODEL_METRALE" \
    --max-seq-len 4096 --max-batch-size 16 --kv-cache-dtype bf16 \
    --gpu-memory-utilization 0.70 \
    --enable-prefix-caching --ssm-cache-slots 32 --ssm-checkpoint-interval 32 \
    --speculative --num-drafts 3 --mtp-quantization bf16 \
    --tool-call-parser qwen3_xml --tool-grammar off --disable-thinking >/dev/null 2>&1
  if wait_health metrale-csweep Qwen 200; then
    echo "=== LEG metrale_synth: serve up, benching ==="
    if run_bench metrale_synth; then
      note "LEG metrale_synth DONE -> results/metrale_synth.json"
    else
      note "LEG metrale_synth BENCH FAILED (no results written)"
    fi
    sudo docker logs metrale-csweep 2>&1 | grep -aic "pool exhausted" \
      | xargs -I{} echo "pool-exhausted lines: {}" | tee -a "$CS/metrale_synth.notes"
  else
    # 2026-09-26: Save the log tail before teardown removes the container.
    sudo docker logs metrale-csweep 2>&1 | tail -60 > "$CS/metrale_synth.deathlog" || true
    note "LEG metrale_synth SERVE_DIED (deathlog: conc_sweep/metrale_synth.deathlog)"
    echo "SERVE_DIED metrale_synth"
  fi
  teardown
  echo "LEG_DONE metrale_synth"
fi

if [ -s "$RESULTS/vllm_synth.json" ]; then
  echo "SKIP vllm_synth (results exist)"
else
  for VM in "$MODEL_METRALE" "nvidia/Qwen3.6-27B-NVFP4"; do
    teardown
    echo "=== LEG vllm_synth: trying checkpoint $VM ==="
    sudo docker run -d --name vllm-csweep --network host --gpus all --ipc=host \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --entrypoint bash "$VLLM_IMAGE" \
      -c "vllm serve $VM --host 0.0.0.0 --port $PORT \
          --max-model-len 32768 --gpu-memory-utilization 0.85 \
          --max-num-seqs 128" >/dev/null 2>&1
    # 2026-09-26: Up to 300 tries 5 s apart (25 min) for vLLM to come up.
    if wait_health vllm-csweep Qwen 300; then
      echo "=== vllm up on $VM, benching ==="
      if run_bench vllm_synth; then
        python3 - "$RESULTS/vllm_synth.json" "$VM" <<'PY'
import json, sys
p, m = sys.argv[1], sys.argv[2]
d = json.load(open(p)); d["served_checkpoint"] = m
json.dump(d, open(p, "w"), indent=1)
PY
        note "LEG vllm_synth DONE on $VM -> results/vllm_synth.json"
      else
        note "LEG vllm_synth BENCH FAILED on $VM"
      fi
      break
    else
      note "vllm checkpoint $VM failed to serve (trying fallback if any)"
      sudo docker logs vllm-csweep 2>&1 | tail -15 > "$CS/vllm_${VM##*/}.faillog"
    fi
  done
  teardown
  echo "LEG_DONE vllm_synth"
fi

if [ -s "$RESULTS/metrale_synth.json" ] && [ -s "$RESULTS/vllm_synth.json" ]; then
  BENCH_RESULTS_FILE="$RESULTS/metrale_synth.json" \
    python3 -u "$BENCH" --compare "$RESULTS/vllm_synth.json" \
    > "$RESULTS/compare.txt" 2>&1 || true
  note "PHASE A compare written -> results/compare.txt"
fi
note "PHASEA_DONE"
echo "PHASEA_DONE"
