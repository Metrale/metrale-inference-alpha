#!/usr/bin/env bash

# 2026-09-26: Runs N agent sessions against the local server (or a remote one
# through a tunnel), scores each with score_run.py, then aggregates the tier.
#
# Owner: bench, FP8 drift harness.
# Invariants: none beyond the types.
#
# Usage:
#   ./run_tier.sh <tier-name> <N>
#     [--container <name>]    (default metrale-qwen-final)
#     [--split-dgx]           (N/2 local, the rest against --remote-api, in parallel)
#     [--remote-api <URL>]    (default http://localhost:8889/v1)
#     [--remote-only]         (all N runs against --remote-api)
#     [--cosine-mode]         (exec ../cosine_run.py instead)
#     [--skip-warmup]         (skip the "What is 2+2?" check)
#     [--bail]                (exit 5 after the first failed run)
#     [--claude-code]         (drive the claude CLI instead of opencode)
#     [--prompt-file PATH|-]  (read the prompt from a file, or stdin)
#
# Outputs: runs/run_<tier>_<i>.json per run; aggregate.py then writes
# reports/<tier>.csv and reports/<tier>.md.
# Warm-up: unless skipped, the local endpoint (and the remote one under --split-dgx)
# must answer "What is 2+2?" with a standalone 4 before any run, or the script exits 4.


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
REMOTE_ONLY=0
BAIL=0
# 2026-09-26: --claude-code: drive the `claude` CLI as user claude (sudo -u claude env
# ANTHROPIC_BASE_URL=... claude -p ...) instead of opencode. The permission mode is
# CC_PERMISSION_MODE, default plan.

CLAUDE_CODE=0
# 2026-09-26: --prompt-file PATH: read the prompt from a file instead of the built-in
# PROMPT; `--prompt-file -` reads stdin.

PROMPT_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --container) CONTAINER="$2"; shift 2 ;;
    --split-dgx) SPLIT_DGX=1; shift ;;
    --remote-api) REMOTE_API="$2"; shift 2 ;;
    --cosine-mode) COSINE_MODE=1; shift ;;
    --skip-warmup) SKIP_WARMUP=1; shift ;;
    --remote-only) REMOTE_ONLY=1; shift ;;
    --bail) BAIL=1; shift ;;
    --claude-code) CLAUDE_CODE=1; shift ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

HARNESS_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="${HARNESS_DIR}/runs"
mkdir -p "${RUNS_DIR}"

LOCAL_API="http://localhost:8888/v1"

# 2026-09-26: The agent's own cargo commands build in the shared warm target dir
# that score_run.py's webserver build uses: METRALE_WARM_TARGET_DIR, with the same
# default as warm_cargo_cache.sh and score_run.py's _warm_target_dir()
# (crates/bench agentic warm_tests.rs checks that the three agree).
#
# This script does not warm the dir; warm_cargo_cache.sh does, for both the debug
# profile (`cargo test`, `cargo run`) and the release profile
# (`cargo build --release`).

















METRALE_WARM_TARGET_DIR="${METRALE_WARM_TARGET_DIR:-${HOME}/.cargo/metrale-warm-target}"
# 2026-09-26: Exported for the agent's cargo commands.
#
# Network stays on (CARGO_NET_OFFLINE is not set): a generation that pins a dep
# version outside the pre-warmed set must still resolve.


export CARGO_TARGET_DIR="${METRALE_WARM_TARGET_DIR}"

# 2026-09-26: apply_output_cap writes METRALE_OPENCODE_OUTPUT_CAP (default 8192) as
# limit.output of every model in the opencode config, and rewrites the file only
# when a value differs.











METRALE_OPENCODE_OUTPUT_CAP="${METRALE_OPENCODE_OUTPUT_CAP:-8192}"
apply_output_cap() {
  local cfg="${1:-${HOME}/.config/opencode/opencode.json}"
  [[ -f "${cfg}" ]] || { echo "[output-cap] no opencode config at ${cfg}; skipping" >&2; return 0; }
  METRALE_OPENCODE_OUTPUT_CAP="${METRALE_OPENCODE_OUTPUT_CAP}" python3 - "${cfg}" <<'PY' >&2 || true
import json, os, sys, tempfile
cfg = sys.argv[1]
cap = int(os.environ["METRALE_OPENCODE_OUTPUT_CAP"])
try:
    d = json.load(open(cfg))
except Exception as e:
    print(f"[output-cap] cannot parse {cfg}: {e}"); sys.exit(0)
changed = False
for prov in (d.get("provider") or {}).values():
    for mdl in (prov.get("models") or {}).values():
        lim = mdl.setdefault("limit", {})
        if lim.get("output") != cap:
            lim["output"] = cap
            changed = True
if changed:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(cfg) or ".")
    os.write(fd, (json.dumps(d, indent=2) + "\n").encode()); os.close(fd)
    os.replace(tmp, cfg)
    print(f"[output-cap] set limit.output={cap} in {cfg}")
else:
    print(f"[output-cap] limit.output already {cap} in {cfg}")
PY
}
apply_output_cap "${HOME}/.config/opencode/opencode.json"

# 2026-09-26: --cosine-mode replaces this process with ../cosine_run.py.
if [[ "${COSINE_MODE}" == "1" ]]; then
  echo "=== cosine-mode: running cosine_run.py (per-layer drift diagnostic) ===" >&2
  exec python3 "${HARNESS_DIR}/../cosine_run.py"
fi

# 2026-09-26: Exits 4 unless the reply contains a standalone 4.
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
  if ! echo "${merged}" | grep -Eq '(^|[^[:alnum:]_])4([^[:alnum:]_]|$)'; then
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

# 2026-09-26: The prompt is the same for every run: the run directory reaches the
# agent through --dir (or as its cwd), never through the prompt text.




PROMPT='Please create a pure rust Axum project here in the current working directory. Just have a ping/pong endpoint. The server MUST bind to the port from the METRALE_HARNESS_PORT env var (default 3001) — use `let port: u16 = std::env::var("METRALE_HARNESS_PORT").unwrap_or_else(|_| "3001".to_string()).parse().unwrap();` then bind to `0.0.0.0:port`. Add tests, run them and prove all tests pass, then run the server and use curl to prove it works. Whenever you run the server or any long-lived process in the background, always start it detached with its output redirected to a file (for example `setsid cargo run > /tmp/server.log 2>&1 &`) so your shell never blocks waiting on it, and wrap any command that might hang, such as curl checks or process kills, in a short `timeout 15`. Finally, tear down the server by killing whatever is listening on its port rather than guessing the process name, always wrapped in a short timeout so it can never stall your shell, for example `timeout 5 fuser -k ${METRALE_HARNESS_PORT:-3001}/tcp 2>/dev/null || true`.'



if [[ -n "${PROMPT_FILE}" ]]; then
  if [[ "${PROMPT_FILE}" == "-" ]]; then
    PROMPT="$(cat)"
  else
    [[ -r "${PROMPT_FILE}" ]] || { echo "FATAL: --prompt-file '${PROMPT_FILE}' not readable" >&2; exit 2; }
    PROMPT="$(cat "${PROMPT_FILE}")"
  fi
  [[ -n "${PROMPT}" ]] || { echo "FATAL: prompt from '${PROMPT_FILE}' is empty" >&2; exit 2; }
fi


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
  # 2026-09-26: Created before the agent starts; the --claude-code branch cd's into it.

  mkdir -p "${TARGET}"

  echo "--- run ${i}/${N} [${label}] target=${TARGET} ---" >&2

  local START_TS END_TS START_TS_INT
  START_TS=$(date +%s.%N)
  # 2026-09-26: Each agent session is killed after OC_TIMEOUT seconds (default 360).
  # METRALE_HARNESS_PORT (HPORT, default 3001) goes to the agent, which the prompt
  # tells to bind it, and to score_run.py, whose webserver test binds a free port
  # of its own instead (_free_port).




  if [[ "${CLAUDE_CODE}" == "1" ]]; then
    # 2026-09-26: Claude Code as user claude, with ANTHROPIC_BASE_URL at the local
    # server and the run directory as cwd; the prompt goes in on stdin. `timeout`
    # runs inside sudo so it is claude's direct parent; -k 10 sends SIGKILL 10 s
    # after the SIGTERM.








    ( cd "${TARGET}" && \
      printf '%s' "${PROMPT}" | sudo -n -u claude \
        timeout -k 10 "${OC_TIMEOUT:-360}" \
        env \
          ANTHROPIC_BASE_URL=http://localhost:8888 \
          ANTHROPIC_AUTH_TOKEN=dummy \
          METRALE_HARNESS_PORT=${HPORT:-3001} \
          /workspace/.local/bin/claude -p \
            --output-format stream-json --verbose \
            --permission-mode "${CC_PERMISSION_MODE:-plan}" \
        ) > "${OC_JSON}" 2> "${OC_ERR}" || true
  else
    # 2026-09-26: opencode, with the run directory as --dir. A session with no
    # tool_use event and no file in the run directory is run again, up to
    # OC_EMPTY_RETRIES times (default 2), then scored as it is.






    local _attempt=0
    local _max_empty="${OC_EMPTY_RETRIES:-2}"
    while :; do
      rm -rf "${TARGET}"; mkdir -p "${TARGET}"
      METRALE_HARNESS_PORT=${HPORT:-3001} \
        METRALE_AGENT_SHELL=1 \
        env ${extra_env} \
        timeout "${OC_TIMEOUT:-360}" opencode run --dangerously-skip-permissions --dir "${TARGET}" --format json \
        "${PROMPT}" > "${OC_JSON}" 2> "${OC_ERR}" || true
      # 2026-09-26: `grep -c` prints 0 (and exits 1) on no match; tr and the default
      # keep _tool_uses a single integer for the -gt test.

      local _tool_uses _real_files
      _tool_uses=$(grep -c '"type":"tool_use"' "${OC_JSON}" 2>/dev/null | tr -d '[:space:]')
      _tool_uses="${_tool_uses:-0}"
      _real_files=$(find "${TARGET}" -type f -not -path '*/.git/*' 2>/dev/null | wc -l | tr -d '[:space:]')
      _real_files="${_real_files:-0}"
      if [[ "${_tool_uses}" -gt 0 || "${_real_files}" -gt 0 ]]; then
        break
      fi
      _attempt=$(( _attempt + 1 ))
      if [[ "${_attempt}" -gt "${_max_empty}" ]]; then
        echo "    [empty-session] run ${i}: still empty after ${_max_empty} retries — scoring as-is" >&2
        break
      fi
      echo "    [empty-session] run ${i}: 0 tool_calls + 0 files (transient opencode glitch) — retry ${_attempt}/${_max_empty}" >&2
    done
  fi
  END_TS=$(date +%s.%N)

  # 2026-09-26: Kill every process whose cwd is inside this run's directory, such as
  # a server the agent started with `cargo run &`: after the timeout's SIGTERM it is
  # reparented and keeps its port. Matching on cwd leaves every other process
  # alone. There is no sudo, so this reaches only this user's processes; under
  # --claude-code the agent's processes belong to user claude.



  if [[ -n "${TARGET}" && -d "${TARGET}" ]]; then
    _tdir_real=$(readlink -f "${TARGET}" 2>/dev/null || echo "${TARGET}")
    for _pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
      _cwd=$(readlink -f "/proc/${_pid}/cwd" 2>/dev/null) || continue
      case "${_cwd}" in
        "${_tdir_real}"|"${_tdir_real}"/*) kill -9 "${_pid}" 2>/dev/null || true ;;
      esac
    done
  fi

  # 2026-09-26: The server log since this run's start (local runs only).
  if [[ "${label}" == "local" ]]; then
    START_TS_INT=${START_TS%.*}
    sudo docker logs "${CONTAINER}" --since "${START_TS_INT}" 2>&1 > "${METRALE_LOG}" || true
  fi

  METRALE_HARNESS_PORT=${HPORT:-3001} \
    python3 "${HARNESS_DIR}/score_run.py" \
    --tier "${TIER}" \
    --run "${i}" \
    --target "${TARGET}" \
    --opencode-json "${OC_JSON}" \
    --opencode-stderr "${OC_ERR}" \
    --metrale-log-window "${METRALE_LOG}" \
    --probe-start-ts "${START_TS}" \
    --probe-end-ts "${END_TS}" \
    --webserver-port ${HPORT:-3001} \
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

  # 2026-09-26: Under --claude-code, also report the longest run of identical lines
  # in the assistant text and the count of server log lines in this run's window
  # that match loop/repeat/watchdog-style words.

  if [[ "${CLAUDE_CODE}" == "1" ]]; then
    local cc_rep cc_wd
    cc_rep=$(python3 - "${OC_JSON}" <<'PY' 2>/dev/null || echo "parse_error"
import json, sys, re
txt = []
for ln in open(sys.argv[1], errors="replace"):
    ln = ln.strip()
    if not ln:
        continue
    try:
        e = json.loads(ln)
    except Exception:
        continue
    # 2026-09-26: Text blocks of each event's `message.content`, plus any `result` string.
    msg = e.get("message") if isinstance(e, dict) else None
    if isinstance(msg, dict):
        for blk in msg.get("content", []) or []:
            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                txt.append(blk["text"])
    if isinstance(e, dict) and isinstance(e.get("result"), str):
        txt.append(e["result"])
blob = "\n".join(txt)
lines = [l.strip() for l in blob.splitlines() if len(l.strip()) > 12]
# 2026-09-26: Longest run of one line (over 12 characters) repeated back to back.
best = 1; cur = 1
for a, b in zip(lines, lines[1:]):
    cur = cur + 1 if a == b else 1
    best = max(best, cur)
# 2026-09-26: Lines that repeat an earlier line anywhere in the text.
dup = len(lines) - len(set(lines))
print(f"chars={len(blob)} lines={len(lines)} max_consecutive_repeat={best} dup_lines={dup}")
PY
)
    cc_wd=$(grep -ciE 'loop|repeat|watchdog|stuck|NoSsmSnapshot|fuzzy|simhash|attractor|degener' "${METRALE_LOG}" 2>/dev/null || echo 0)
    echo "    [claude-code] ${cc_rep} | metrale_watchdog_hits=${cc_wd}  (raw: ${OC_JSON}, metrale-log: ${METRALE_LOG})" >&2
  fi

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
elif [[ "${REMOTE_ONLY}" == "1" ]]; then
  echo "remote-only: all ${N} runs against ${REMOTE_API}" >&2
  for i in $(seq 1 "${N}"); do
    run_one "${i}" "${REMOTE_API}" "XDG_CONFIG_HOME=${XDG_CONFIG_HOME_OVERRIDE:-/tmp/oc-tunnel-config}" "remote"
  done
else
  for i in $(seq 1 "${N}"); do
    run_one "${i}" "${LOCAL_API}" "" "local"
  done
fi

echo "=== tier ${TIER} complete (N=${N}). Aggregating... ===" >&2
# 2026-09-26: Exit with aggregate.py's status: cargo plus webserver failures,
# capped at 255.
python3 "${HARNESS_DIR}/aggregate.py" --tier "${TIER}" >&2
agg_rc=$?
echo "=== tier ${TIER}: exit code ${agg_rc} (total cargo+webserver failures; 0 = all green) ===" >&2
exit "${agg_rc}"
