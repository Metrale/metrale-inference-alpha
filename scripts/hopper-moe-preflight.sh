#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Read-only host admission; receipts stay outside the source checkout.
set -euo pipefail
out=${1:?Usage: hopper-moe-preflight.sh RECEIPT_DIRECTORY}
mkdir -p "$out"
date -u +%FT%TZ > "$out/utc.txt"
nvidia-smi -q > "$out/nvidia-smi-q.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,driver_version --format=csv > "$out/gpus.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$out/processes.csv"
df -h > "$out/disks.txt"
free -h > "$out/ram.txt"
lscpu > "$out/cpu.txt"
for program in docker nvcc cargo python3; do
  command -v "$program" >> "$out/tools.txt" || true
done
if command -v docker >/dev/null; then
  docker version > "$out/docker.txt" 2>&1 || true
fi
cat "$out/gpus.csv" "$out/processes.csv" "$out/disks.txt"
printf 'Inspect these receipts before starting GPU workloads: %s\n' "$out"
