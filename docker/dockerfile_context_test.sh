#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Every Dockerfile under docker/ that compiles metrale-server must COPY the
# repo-root inputs the release build reads. `metrale-bench` (a dependency of
# metrale-server) include_str!s tests/fixtures/bench_prompt_*.txt, so an image
# without tests/fixtures fails at compile time with "couldn't read", after the
# CUDA kernels have already built. No Docker daemon, no GPU: a structural check
# over the checked-in files.
#
# Usage: bash docker/dockerfile_context_test.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REQUIRED=(Cargo.toml Cargo.lock crates/ vendor/ kernels/ tests/fixtures/)

instructions() { sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$1"; }

# `COPY tests/ tests/` also satisfies `tests/fixtures/`.
copies() {
  local f="$1" want="$2"
  instructions "$f" | grep -E '^[[:space:]]*COPY[[:space:]]' | grep -v -- '--from=' \
    | awk '{ for (i = 2; i < NF; i++) print $i }' \
    | while read -r src; do
        case "$want" in "$src"|"${src%/}/"*) echo hit; break ;; esac
      done | grep -q hit
}

checked=0
failed=0
while IFS= read -r f; do
  instructions "$f" | grep -qE 'cargo build .*-p metrale-server' || continue
  checked=$((checked + 1))
  for want in "${REQUIRED[@]}"; do
    if ! copies "$f" "$want"; then
      echo "FAIL ${f#"$ROOT"/}: no COPY of $want" >&2
      failed=$((failed + 1))
    fi
  done
done < <(find "$ROOT/docker" -type f -name 'Dockerfile*' ! -name '*.dockerignore' | sort)

[ "$checked" -gt 0 ] || { echo "FAIL: no Dockerfile builds metrale-server" >&2; exit 1; }
[ "$failed" -eq 0 ] || exit 1
echo "ok: $checked Dockerfiles copy ${REQUIRED[*]}"
