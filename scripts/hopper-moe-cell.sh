#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# One frozen same-host internal rehearsal cell. Requires qualified idle server.
set -euo pipefail
engine=${1:?engine atlas or vllm}
c=${2:?concurrency 1/8/32/128}
rep=${3:?repetition 1/2/3}
case "$engine" in atlas) port=8888 ;; vllm) port=8000 ;; *) exit 2 ;; esac
case "$c" in 1|8|32|128) ;; *) exit 2 ;; esac
case "$rep" in 1|2|3) ;; *) exit 2 ;; esac
root=/home/ubuntu
campaign=${HOPPER_RUN_ID:?Set a unique comparison run identifier}
[[ "$campaign" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
receipts=$root/hopper-receipts/cells/$campaign/$engine
mkdir -p "$receipts"
name=$engine-c$c-r$rep
test ! -e "$receipts/$name.json"
test ! -e "$receipts/$name.log"
n=$((4*c))
if [ "$n" -lt 16 ]; then n=16; fi
image=${HOPPER_CLIENT_IMAGE:?Set frozen CPU client image ID including vllm bench extras}
command=(docker run --rm --name "hopper-client-$name" --network host
  --cpuset-cpus 0,1 --entrypoint vllm
  -v "$root/models/qwen36:/model:ro"
  -v "$root/hopper-corpus/frozen-v1:/corpus:ro"
  -v "$receipts:/results" "$image" bench serve
  --backend openai --base-url "http://127.0.0.1:$port"
  --endpoint /v1/completions --model Qwen/Qwen3.6-35B-A3B-FP8
  --tokenizer /model --dataset-name custom --dataset-path "/corpus/c$c-r$rep.jsonl"
  --skip-chat-template --disable-shuffle --custom-output-len 1024
  --num-prompts "$n" --max-concurrency "$c" --request-rate inf
  --seed 42 --temperature 0
  --extra-body '{"seed":42,"min_tokens":1024,"presence_penalty":0,"frequency_penalty":0,"repetition_penalty":1}'
  --num-warmups 0 --ready-check-timeout-sec 0
  --percentile-metrics ttft,itl,tpot,e2el --metric-percentiles 50,90
  --save-result --save-detailed --result-dir /results --result-filename "$name.json")
printf '%q ' "${command[@]}" > "$receipts/$name.command.sh"
printf '\n' >> "$receipts/$name.command.sh"
date -u +%FT%TZ > "$receipts/$name.start-utc.txt"
nvidia-smi -q > "$receipts/$name.gpu-before.txt"
set +e
"${command[@]}" > "$receipts/$name.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$receipts/$name.exit-code.txt"
date -u +%FT%TZ > "$receipts/$name.end-utc.txt"
nvidia-smi -q > "$receipts/$name.gpu-after.txt"
cat "$receipts/$name.log"
if [ "$result" -eq 0 ]; then
  python3 "$(dirname "$0")/hopper-moe-validate.py" "$receipts/$name.json" \
    --requests "$n" --concurrency "$c" > "$receipts/$name.validation.json"
fi
exit "$result"
