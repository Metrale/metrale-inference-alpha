#!/usr/bin/env bash

# 2026-09-26: Sequential opencode runs (agent "harness") with the target path written
# into the prompt and the cwd outside it, each scored with score_run.py; no
# aggregation at the end.
#
# Owner: bench, FP8 drift harness.
# Invariants: none beyond the types.
#
# Usage: ./run_tier_steps.sh <tier-name> <N> [--container <name>]
# Outputs: runs/run_<tier>_<i>.json per run.

set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <tier-name> <N> [--container <name>]" >&2
  exit 2
fi

TIER="$1"
N="$2"
shift 2

CONTAINER="metrale-qwen-final"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --container) CONTAINER="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

HARNESS_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="${HARNESS_DIR}/runs"
mkdir -p "${RUNS_DIR}"

# 2026-09-26: Exit 3 unless the container is running and /v1/models answers.
if ! sudo docker ps --filter "name=${CONTAINER}" --format '{{.Names}}' | grep -q "${CONTAINER}"; then
  echo "FATAL: container '${CONTAINER}' is not running" >&2
  exit 3
fi
if ! curl -sS -m 5 http://localhost:8888/v1/models >/dev/null 2>&1; then
  echo "FATAL: metrale /v1/models not responding on localhost:8888" >&2
  exit 3
fi

PROMPT_TEMPLATE='Please create a pure rust Axum project inside __TARGET__. Just have a ping/pong endpoint. Add tests, run them and prove all tests pass, then run the server and use curl to prove it works. Finally, tear down the server.'

echo "=== tier=${TIER} runs=${N} container=${CONTAINER} ===" >&2
echo "harness: ${HARNESS_DIR}" >&2

for i in $(seq 1 "${N}"); do
  TARGET="/tmp/harness-${TIER}-r${i}"
  OC_JSON="/tmp/harness-${TIER}-r${i}.json"
  OC_ERR="/tmp/harness-${TIER}-r${i}.err"
  METRALE_LOG="/tmp/harness-${TIER}-r${i}.metrale.log"
  OUT_JSON="${RUNS_DIR}/run_${TIER}_${i}.json"

  rm -rf "${TARGET}" "${OC_JSON}" "${OC_ERR}" "${METRALE_LOG}"
  mkdir -p "/tmp/harness-${TIER}-r${i}-cwd"
  cd "/tmp/harness-${TIER}-r${i}-cwd"

  PROMPT="${PROMPT_TEMPLATE//__TARGET__/${TARGET}}"

  echo "--- run ${i}/${N} target=${TARGET} ---" >&2

  START_TS=$(date +%s.%N)
  # 2026-09-26: opencode is killed after 360 s.
  timeout 360 opencode run --dangerously-skip-permissions --agent harness --format json \
    "${PROMPT}" > "${OC_JSON}" 2> "${OC_ERR}" || true
  END_TS=$(date +%s.%N)

  # 2026-09-26: The server log since this run's start, in whole seconds.
  START_TS_INT=${START_TS%.*}
  sudo docker logs "${CONTAINER}" --since "${START_TS_INT}" 2>&1 > "${METRALE_LOG}" || true

  python3 "${HARNESS_DIR}/score_run.py" \
    --tier "${TIER}" \
    --run "${i}" \
    --target "${TARGET}" \
    --opencode-json "${OC_JSON}" \
    --opencode-stderr "${OC_ERR}" \
    --metrale-log-window "${METRALE_LOG}" \
    --probe-start-ts "${START_TS}" \
    --probe-end-ts "${END_TS}" \
    --out "${OUT_JSON}"

  files_count=$(jq -r '.filesystem.files_count' "${OUT_JSON}")
  cargo_ok=$(jq -r '.cargo.cargo_toml_valid' "${OUT_JSON}")
  drift_lean=$(jq -r '.drift.write_content_starts_with_lean' "${OUT_JSON}")
  drift_empty=$(jq -r '.drift.write_empty_path' "${OUT_JSON}")
  drift_pathdrift=$(jq -r '.drift.write_path_drift_from_target' "${OUT_JSON}")
  wall=$(jq -r '.wall_time_s' "${OUT_JSON}")
  echo "    files=${files_count} cargo_valid=${cargo_ok} lean=${drift_lean} empty_path=${drift_empty} path_drift=${drift_pathdrift} wall=${wall}s" >&2
done

echo "=== tier ${TIER} complete (N=${N}). Run aggregate.py next. ===" >&2
