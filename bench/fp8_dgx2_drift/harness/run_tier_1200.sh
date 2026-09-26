#!/usr/bin/env bash

# 2026-09-26: run_tier.sh variant with a 1200 s opencode timeout, a shorter prompt,
# a warm-up that accepts any 4 in the reply, and no aggregation at the end.
#
# Owner: bench, FP8 drift harness.
# Invariants: none beyond the types.
#
# Usage:
#   ./run_tier_1200.sh <tier-name> <N>
#     [--container <name>]    (default metrale-qwen-final)
#     [--split-dgx]           (N/2 local, the rest against --remote-api, in parallel)
#     [--remote-api <URL>]    (default http://localhost:8889/v1)
#     [--cosine-mode]         (exec ../cosine_run.py instead)
#     [--skip-warmup]         (skip the "What is 2+2?" check)
#     [--bail]                (exit 5 after the first failed run)
#
# Outputs: runs/run_<tier>_<i>.json per run.

set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <tier-name> <N> [--container <name>] [--split-dgx] [--remote-api URL] [--cosine-mode] [--skip-warmup]" >&2
  exit 2
fi

TIER="$1"
N="$2"
shift 2

CONTAINER="metrale-qwen-final"
SPLIT_DGX=0
REMOTE_API="http://localhost:8889/v1"
COSINE_MODE=0
SKIP_WARMUP=0
BAIL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --container) CONTAINER="$2"; shift 2 ;;
    --split-dgx) SPLIT_DGX=1; shift ;;
    --remote-api) REMOTE_API="$2"; shift 2 ;;
    --cosine-mode) COSINE_MODE=1; shift ;;
    --skip-warmup) SKIP_WARMUP=1; shift ;;
    --bail) BAIL=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

HARNESS_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="${HARNESS_DIR}/runs"
mkdir -p "${RUNS_DIR}"

LOCAL_API="http://localhost:8888/v1"

# 2026-09-26: --cosine-mode replaces this process with ../cosine_run.py.
if [[ "${COSINE_MODE}" == "1" ]]; then
  echo "=== cosine-mode: running cosine_run.py (per-layer drift diagnostic) ===" >&2
  exec python3 "${HARNESS_DIR}/../cosine_run.py"
fi

# 2026-09-26: Exits 4 unless the reply contains a 4 anywhere, so "42" also passes.
warmup_endpoint() {
  local api="$1"
  local label="$2"
  echo "[warmup] ${label} ${api} ..." >&2
  local body
  body=$(curl -sS -m 60 "${api}/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3.6-35B-A3B-FP8","messages":[{"role":"user","content":"What is 2+2? Respond with just the number."}],"max_tokens":80,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' 2>&1)
  # 2026-09-26: Check content and reasoning_content together.
  local merged
  merged=$(echo "${body}" | python3 -c "import sys, json; d = json.loads(sys.stdin.read()); m = d.get('choices',[{}])[0].get('message',{}); print((m.get('content','') or '') + ' ' + (m.get('reasoning_content','') or ''))" 2>/dev/null)
  if [[ -z "${merged}" ]]; then
    echo "[warmup] FATAL: ${label} returned no parseable response" >&2
    echo "[warmup] raw body (first 400 chars): ${body:0:400}" >&2
    exit 4
  fi
  if ! echo "${merged}" | grep -q '4'; then
    echo "[warmup] FATAL: ${label} did not emit '4' for '2+2' — catastrophic regression, halting" >&2
    echo "[warmup] response excerpt: ${merged:0:300}" >&2
    exit 4
  fi
  echo "[warmup] ${label} OK" >&2
}

if [[ "${SKIP_WARMUP}" == "0" ]]; then
  if ! sudo docker ps --filter "name=${CONTAINER}" --format '{{.Names}}' | grep -q "${CONTAINER}"; then
    echo "FATAL: container '${CONTAINER}' is not running" >&2
    exit 3
  fi
  if ! curl -sS -m 5 "${LOCAL_API}/models" >/dev/null 2>&1; then
    echo "FATAL: metrale /v1/models not responding on localhost:8888" >&2
    exit 3
  fi
  warmup_endpoint "${LOCAL_API}" "local-metrale"

  if [[ "${SPLIT_DGX}" == "1" ]]; then
    if ! curl -sS -m 5 "${REMOTE_API}/models" >/dev/null 2>&1; then
      echo "FATAL: remote vLLM/metrale /v1/models not responding at ${REMOTE_API}" >&2
      echo "       (expected an SSH tunnel: ssh -L 8889:localhost:8888 claude@10.10.10.2)" >&2
      exit 3
    fi
    warmup_endpoint "${REMOTE_API}" "remote-dgx2"
  fi
fi

# 2026-09-26: The prompt is the same for every run: the run directory reaches
# opencode through --dir, not the prompt text.
PROMPT='Please create a pure rust Axum project here in the current working directory. Just have a ping/pong endpoint. The server MUST bind to the port from the METRALE_HARNESS_PORT env var (default 3001) — use `let port: u16 = std::env::var("METRALE_HARNESS_PORT").unwrap_or_else(|_| "3001".to_string()).parse().unwrap();` then bind to `0.0.0.0:port`. Add tests, run them and prove all tests pass, then run the server and use curl to prove it works. Finally, tear down the server.'

run_one() {
  local i="$1"
  local api="$2"
  local extra_env="$3"
  local label="$4"

  local TARGET="/tmp/harness-${TIER}-r${i}"
  local OC_JSON="/tmp/harness-${TIER}-r${i}.json"
  local OC_ERR="/tmp/harness-${TIER}-r${i}.err"
  local METRALE_LOG="/tmp/harness-${TIER}-r${i}.metrale.log"
  local OUT_JSON="${RUNS_DIR}/run_${TIER}_${i}.json"

  rm -rf "${TARGET}" "${OC_JSON}" "${OC_ERR}" "${METRALE_LOG}"
  : > "${METRALE_LOG}"  # 2026-09-26: filled below for local runs only
  mkdir -p "${TARGET}"

  echo "--- run ${i}/${N} [${label}] target=${TARGET} ---" >&2

  local START_TS END_TS START_TS_INT
  START_TS=$(date +%s.%N)
  # 2026-09-26: opencode is killed after 1200 s. METRALE_HARNESS_PORT=3001 goes to
  # the agent, which the prompt tells to bind it, and to score_run.py, whose
  # webserver test binds a free port of its own instead.
  METRALE_HARNESS_PORT=3001 \
  ${extra_env} \
    timeout 1200 opencode run --dangerously-skip-permissions --dir "${TARGET}" --format json \
    "${PROMPT}" > "${OC_JSON}" 2> "${OC_ERR}" || true
  END_TS=$(date +%s.%N)

  # 2026-09-26: The server log since this run's start (local runs only).
  if [[ "${label}" == "local" ]]; then
    START_TS_INT=${START_TS%.*}
    sudo docker logs "${CONTAINER}" --since "${START_TS_INT}" 2>&1 > "${METRALE_LOG}" || true
  fi

  METRALE_HARNESS_PORT=3001 \
    python3 "${HARNESS_DIR}/score_run.py" \
    --tier "${TIER}" \
    --run "${i}" \
    --target "${TARGET}" \
    --opencode-json "${OC_JSON}" \
    --opencode-stderr "${OC_ERR}" \
    --metrale-log-window "${METRALE_LOG}" \
    --probe-start-ts "${START_TS}" \
    --probe-end-ts "${END_TS}" \
    --webserver-port 3001 \
    --out "${OUT_JSON}"

  local files_count cargo_ok drift_lean drift_empty drift_pathdrift wall webserver_ok
  files_count=$(jq -r '.filesystem.files_count' "${OUT_JSON}")
  cargo_ok=$(jq -r '.cargo.cargo_toml_valid' "${OUT_JSON}")
  drift_lean=$(jq -r '.drift.write_content_starts_with_lean' "${OUT_JSON}")
  drift_empty=$(jq -r '.drift.write_empty_path' "${OUT_JSON}")
  drift_pathdrift=$(jq -r '.drift.write_path_drift_from_target' "${OUT_JSON}")
  wall=$(jq -r '.wall_time_s' "${OUT_JSON}")
  webserver_ok=$(jq -r '.webserver.webserver_ok // false' "${OUT_JSON}")
  echo "    files=${files_count} cargo_valid=${cargo_ok} webserver_ok=${webserver_ok} lean=${drift_lean} empty_path=${drift_empty} path_drift=${drift_pathdrift} wall=${wall}s" >&2

  # 2026-09-26: --bail: exit 5 on the first run whose cargo_valid or webserver_ok is not true.
  if [[ "${BAIL}" == "1" ]] && { [[ "${cargo_ok}" != "true" ]] || [[ "${webserver_ok}" != "true" ]]; }; then
    echo "[bail] run ${i} failed cargo_valid=${cargo_ok} webserver_ok=${webserver_ok} — exiting early (--bail)" >&2
    exit 5
  fi
}

echo "=== tier=${TIER} runs=${N} container=${CONTAINER} split_dgx=${SPLIT_DGX} ===" >&2
echo "harness: ${HARNESS_DIR}" >&2

if [[ "${SPLIT_DGX}" == "1" ]]; then
  # 2026-09-26: N/2 (rounded down) runs locally and the rest remotely, in parallel.
  local_count=$(( N / 2 ))
  remote_count=$(( N - local_count ))
  echo "split: local=${local_count} remote=${remote_count} (remote=${REMOTE_API})" >&2

  (
    for i in $(seq 1 "${local_count}"); do
      run_one "${i}" "${LOCAL_API}" "" "local"
    done
  ) &
  LOCAL_PID=$!

  (
    for i in $(seq $((local_count + 1)) "${N}"); do
      run_one "${i}" "${REMOTE_API}" "XDG_CONFIG_HOME=${XDG_CONFIG_HOME_OVERRIDE:-/tmp/oc-tunnel-config}" "remote"
    done
  ) &
  REMOTE_PID=$!

  wait "${LOCAL_PID}" "${REMOTE_PID}"
else
  for i in $(seq 1 "${N}"); do
    run_one "${i}" "${LOCAL_API}" "" "local"
  done
fi

echo "=== tier ${TIER} complete (N=${N}). Run aggregate.py next. ===" >&2
