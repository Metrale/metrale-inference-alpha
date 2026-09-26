#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Enforce the 500-LoC cap on crates/**/*.rs.
#
# Extracted from file-size-cap.yml so the standalone job and the batched
# `cheap checks` job run THE SAME CODE. main's body verbatim, INCLUDING the
# "scanned nothing" assertion: `find crates ...` on a missing directory feeds
# the loop nothing, the violation count stays 0, and the cap goes green
# unenforced repo-wide. The first extraction dropped that line and the
# certification self-test caught it.
set -euo pipefail
set -euo pipefail
# Cap is 500 LoC per Rust source file in crates/.
# Allow-list: previously held the last two unsplittable
# monoliths (`chat.rs` 1121 LoC and `chat_stream.rs` 1484 LoC).
# Wave 4g (2026-05-03) lifted both via the `Ctx`+`State`
# struct pattern.
#
# The five entries below are launch-day carry-overs: each is
# marginally over 500 LoC (505-528) and waiting on a clean
# split. Adding to this list requires a rationale comment AND
# a tracking issue.
allow_list=(
  # 2026-09-04 GDN batched-verify line (#844): 470 -> 513 LoC. NOT split
  # because the file is ONE function -- `impl TransformerModel` opens at line
  # 38 and `decode_verify_graphed_dispatch` runs to the end, so there is no
  # piecewise seam to cut on. The other two files this line pushed over the cap
  # WERE split in this commit (qwen3_ssm/mod.rs 506 -> 86 by moving the struct
  # declaration, trait_layer.rs 501 -> 485 by moving its one free fn), so this
  # entry is the residue, not the habit. Tracked: #872
  # 2026-08-15 concurrency cycle (#525): grew past 500 in this stack's
  # instrumentation work. Reduce when feasible.
  # 2026-08-15 stack merge (#516 video + #514 batch6 + reasoning-effort):
  # these crossed 500 in the merged stacks; splitting them mid-campaign
  # was judged riskier than shipping. Reduce when feasible.
  # 524 LoC — single-file FFI surface for the NCCL backend.
  # Splitting requires moving the unsafe-impl Send/Sync block
  # and ~30 cudarc/cuda-driver C-binding wrappers into a
  # `nccl_ffi/` submodule; deferred until after launch.
  "crates/comm/src/nccl_backend.rs"
  # ── PR: LoRA-on-HSS-tier carry-over ──
  # ── PR #90 (fix/in-think-tool-call-leak) carry-overs ──
  # The overnight coherence-debugging work grew these past 500 LoC
  # (19 newly over-cap + impl_a2/compile_tools inherited from main).
  # Allow-listed to unblock the PR; clean splits tracked as follow-up
  # (per-file Ctx/State or sub-module extraction, Metrale Engine idiom).
  # 944 LoC — NO SEAM: `forward_prefill_fp8` is one 922-line fn.
  # 754 LoC — NO SEAM: `MoeLayer::forward` is one 690-line fn.
  # 521 LoC — parallel-tool-call fixture suite added by #231. Split
  # into per-format (hermes/qwen3_coder) sibling test files. #244.
  # ── PR #169 (perf/qwen36-dense-ffn-tc-prefill) carry-over,
  # grown by PR #223 (perf/qwen36-27b-prefill-nvfp4) ──
  # 3131 LoC — SEAM NEEDS A VISIBILITY RE-SPELLING. The pub methods could
  # move to children, but the private helpers they share (`apply_lora_*`,
  # `ensure_*_weight`, `w4a16_prefill_gemm`) are called from several of them
  # and would have to stay here with the 270-line struct: about 720 lines.
  # Moving it needs `pub(super)` / `pub(in ..)` spelled for the new depth
  # (same scope); the size-cap pass is moves only, so that is an owner call.
  # `forward_prefill_inner` is one 1064-line fn and stays over the cap either way.
  # ── PR #223 (perf/qwen36-27b-prefill-nvfp4) carry-over ──
  # ── PR #202 (perf/vision-vit-perf) carry-over ──
  # ── PR #186 (DeepSeek-V4-Flash port) carry-over ──
  # 12 files the V4 port lands over the cap. Allow-listed to ship DS4F
  # to users (today it's runnable only by building the #186 branch);
  # consistent with the existing big-but-unsplittable entries above
  # (build.rs 891, metal_backend.rs 733). The shared-file growth
  # (prefill_inner/decode_inner — V4 added CSA/HCA + MLA decode paths)
  # and the V4-new kernel/loader/MTP-head files read top-to-bottom like
  # the other compute-heavy entries. Clean split tracked in #219.
  # 977 LoC — NO SEAM: `prefill_inner` is one 510-line fn, and the
  # private `prefill_inner_hc` it calls cannot move away from it without
  # widening its visibility.
  # 793 LoC — SEAM NEEDS A VISIBILITY RE-SPELLING: the pub(super)
  # `decode_inner` (421 lines) calls the private `decode_inner_hc` (354).
  # Moving it needs `pub(super)` / `pub(in ..)` spelled for the new depth
  # (same scope); the size-cap pass is moves only, so that is an owner call.
  # 585 LoC — SEAM NEEDS A VISIBILITY RE-SPELLING: the pub(super)
  # `ms_mla_decode` calls the private 361-line `ms_mla_decode_one`.
  # Moving it needs `pub(super)` / `pub(in ..)` spelled for the new depth
  # (same scope); the size-cap pass is moves only, so that is an owner call.
  # 702 LoC — `ModelConfig`. NO SEAM: the file is the one 695-line
  # struct (148 `pub` fields, one per parsed config.json dimension), and a
  # struct definition cannot span files. It was moved out of `lib.rs` so
  # that everything else in the crate root fits the cap.
  # ── #203 (Holo/Ornith GB10 enablement) carry-over ──
  # ── #214 (batched decode engine) carry-over ──
  # ── PR #229 (perf/qwen36-27b-prefill-nvfp4) carry-overs ──
  # Three files inherited over-cap from main unchanged (grew on
  # main between this branch's base and its main-merge; would fail
  # on any fresh PR): 632 / 521 / 533 LoC. Clean splits tracked
  # as follow-up.
  # 1357 LoC — NO SEAM: `decode_batched_inner` is one 1242-line fn.
  # 503 LoC — batched-recurrent GDN decode; marginally over (added the
  # per-seq SSM out_proj all-reduce for HeadParallel). Back under 500
  # once the batched/graph paths consolidate. Follow-up.
  # -- #300/#301 (mixed-precision NVFP4 weights) carry-over --
  # -- K=gamma DFlash base engine carry-overs (this PR) --
  # Drafter block forward + paged attention + weight assembly are
  # compute-heavy launch sequences that read top-to-bottom (same
  # rationale as the qwen3_attention compute-heavy entries above);
  # trait_impl/mod.rs grew with the ctx-append dispatch family.
  # Splits tracked as follow-up per the #90/#186 precedent.
  # -- PR #373 (prefix-cache refcount fixes) carry-over --
  # 505 LoC — paged KV block pool. Sat at 493 (under cap); the two
  # refcount fixes push it 12 over: `return_evicted_block` must release
  # only the prefix cache's own ref (force-zeroing freed blocks a live
  # sequence still held → aliased KV pointer → CUDA-700), and `dec_ref`
  # is now saturating because `debug_assert` is compiled out in release
  # with no `overflow-checks`, so a 0-ref decrement wrapped to u32::MAX
  # and pinned the block forever. Both need their "why" inline — the
  # bugs are invisible from the code alone. Comments already trimmed
  # 520 → 505. Single `impl PagedKvCache` (alloc/free/refcount/block IO)
  # whose ordering is read top-to-bottom, same rationale as the other
  # compute/state-heavy entries above. Clean split (block IO → sibling)
  # tracked as follow-up.
  # ── Wave-55 integration (perf/enterprise-concurrency-v2 -> main) ──
  # Thirteen of the fifteen below were ALREADY over the cap on the
  # branch tip before this merge; two (`model/types.rs`,
  # `scheduler/mtp_step.rs`) crossed it in the merge itself, from the
  # union of main's carried-context refactor with the campaign's
  # multi-sequence work. All fifteen are the enterprise-concurrency
  # campaign's compute/dispatch surfaces, where the read order IS the
  # launch order; splitting them mid-campaign is exactly the change
  # that would have gone unmeasured. Split tracked as follow-up.
  #
  # 647 LoC — R-row batched verify forward + graph capture.
  # 629 LoC — NO SEAM: `decode_ms_ssm_recurrent` is one 613-line fn.
  # -- PR #469 (SSM State Poisoning Gate) --
  # 522 LoC — gate coverage table. Crossed the cap when the sixth
  # mandatory gate (ssm-state-poisoning-gate) added SSM_POISON_EXCLUDES
  # plus its own exclusion entries in the four sibling EXCLUDES
  # arrays. Splitting it here is NOT free: coverage.rs is a
  # BOUNDARY_FILE, so any edit to it (including a pure split)
  # invalidates every standing gate record and forces a full GPU
  # re-cycle — which is exactly why the entry goes here instead of
  # a split commit in this PR. A clean split (per-gate EXCLUDES
  # arrays -> sibling module) is tracked as follow-up; this entry
  # lands with the PR that tipped it.
  "crates/bench/src/gate/coverage.rs"
  # -- PR #464 (batch4 stack: #400/#401/#402/#350/#403) --
  # Three files the restack tipped marginally over, each of which sat
  # JUST under the cap on main (498/489/499 -> 509/501/501). The
  # overage is a mechanical consequence of merging five upstream
  # branches, not new monolith growth, and the content belongs to the
  # upstream PRs rather than to this restack -- same footing as the
  # #90/#229/#300 carry-over entries above. Clean splits tracked as
  # follow-up with the upstream authors.
  # 503 LoC — NO SEAM: `run_routed_grouped_gemm` is one 489-line fn;
  # with its imports and `impl` header no file can hold it in 500.
  # 504 LoC — SEAM NEEDS A VISIBILITY RE-SPELLING: the pub(super)
  # `prefill_attention_paged_qkv` calls the private 360-line
  # `prefill_one_proj`, and neither can move to a child as it is.
  # Moving it needs `pub(super)` / `pub(in ..)` spelled for the new depth
  # (same scope); the size-cap pass is moves only, so that is an owner call.

  # ── GLM-5.3-Flash (`glm5_next`) native port — 2026-09-06 ──
  #
  # SHARED RATIONALE for the 25 entries below. Every one of them is on
  # the execution path of `metrale-glm53:integ1`, the image that passed
  # the ten-phase GB10 qualification of integration candidate
  # 4d0fef93 on 2026-09-06: six-probe byte-identity against the sealed
  # reference (6/6 EXACT), a bracketed control-candidate-control A/B
  # inside +0.325 % TTFT and -0.29 % decode, 222-sequence Unit2
  # lifecycle with one live/live_mb value, 7/7 tool-and-grammar arms,
  # 30/30 auto smoke, and bounded C=2/C=3. That evidence is bound to
  # the TREE, not to the behaviour in the abstract: a split moves code
  # between compilation units, changes inlining and therefore
  # instruction selection, and the byte-identity gate is exactly the
  # thing that would have to be re-earned on a GPU to know it still
  # holds. Splitting during the upstream re-cut would spend that
  # evidence to satisfy a line count, on a port that has not yet been
  # reviewed once.
  #
  # These are therefore carry-overs on the same terms as the #516/#514
  # and launch-day entries above, not a new class of exception. They
  # are tracked as ONE follow-up — "GLM-5.3 upstream re-cut: LoC split
  # follow-up" — to be opened against this PR; the test-only half of
  # the same debt was NOT deferred and is already paid: nine
  # microtest examples, `glm5next_kda_ref/tests.rs` and
  # `scheduler/lifecycle_tests.rs` were split in this same commit,
  # which is where splitting is mechanical and risks nothing.
  #
  # 1417 LoC — the DSA block (NoPE MLA over indexer-selected tokens).
  # The largest single file in the port and the one that is least
  # safe to touch: it holds the prefill selector, the batched-row
  # selector promoted at d8c6743a, the decode attend path and the
  # CUDA-graph capture geometry, all of which are read against each
  # other because the graph bakes per-sequence pointers (A56). A55
  # (an indexer read 5120 B past `q_resid`) and A58 (the drafter
  # handing DSA a whole block-table pool) were both found by reading
  # this file top to bottom.
  #
  # ── Shared-runtime files this port GREW past the cap ──
  # Each already sat just under 500 on main and crossed it here. The
  # additions are GLM arms in existing dispatch/accounting code, so a
  # split would be re-cutting an upstream file to hold one model's
  # arm — and each is on the GPU-qualified path described above.
  #
  # 592 LoC (was 472) — vocab-parallel LM head and the decode
  # dispatch band. The band's upper edge decides which kernel a decode
  # width lands on, i.e. which bits it produces; this is the file that
  # has to be read against `ops/gemm_quant.rs`.
  # 614 LoC — CUDA backend GPU impl. NO SEAM: the body is one 498-line
  # `impl GpuBackend for MetraleCudaBackend`, and a trait impl cannot
  # span files, so the impl plus its imports is over 500 on its own.
  # 541 LoC (was 484) — model-dependent serve runtime phase; grew by
  # the multi-EOS stop-token set and the GLM capability gates.
)

# ★ The scanned tree must EXIST and must contain Rust. `find crates`
# on a missing directory writes to stderr and yields nothing; the
# loop below then runs zero times, `violations` stays 0, and this
# required check prints its tick and exits 0 -- the cap silently
# unenforced for the whole repo. `set -euo pipefail` does not help:
# the failure is inside a process substitution, whose status is not
# the command's. Same shape as the render-thread gate's missing-tree
# bug. Assert the input before trusting an empty result.
# Counted inside the `-d` guard on purpose: under `set -euo pipefail`
# a bare `find crates | wc -l` on a missing directory kills the step
# at the assignment, which is a non-zero exit with no explanation of
# why -- right verdict, useless log.
scanned=0
if [[ -d crates ]]; then
  scanned=$(find crates -name '*.rs' -not -name '*.bak' -not -path '*/target/*' | wc -l)
fi
if [[ $scanned -eq 0 ]]; then
  echo "::error::found no .rs files under crates/, so the 500-line cap scanned nothing."
  echo "Either crates/ moved or the find expression stopped matching. An unscanned"
  echo "tree is an unenforced cap, and this check would have gone green without"
  echo "this line."
  exit 1
fi
echo "scanning $scanned Rust source file(s) under crates/"

violations=0
while IFS= read -r -d '' file; do
  lines=$(wc -l < "$file")
  if [[ $lines -gt 500 ]]; then
    # Strip leading "./" if present
    normalized="${file#./}"
    # Check allow-list
    allowed=0
    for entry in "${allow_list[@]}"; do
      if [[ "$normalized" == "$entry" ]]; then
        allowed=1
        break
      fi
    done
    if [[ $allowed -eq 1 ]]; then
      echo "::warning file=$normalized::Allow-listed: $lines LoC (>500). Reduce when feasible."
    else
      echo "::error file=$normalized::$lines LoC exceeds 500-line cap. Split into sub-modules per Metrale Engine idiom (see crates/model-layers/src/layers/qwen3_attention/ for compute-heavy template, crates/model-arch/src/weight_loader/ for variant-dispatch template)."
      violations=$((violations + 1))
    fi
  fi
done < <(find crates -name '*.rs' -not -name '*.bak' -not -path '*/target/*' -print0)

if [[ $violations -gt 0 ]]; then
  echo ""
  echo "❌ $violations file(s) exceed the 500-line cap. Split them per Metrale Engine idiom."
  echo ""
  echo "Quick fix: extract a cohesive cluster (impl block, related fns, tests)"
  echo "to a sibling file:"
  echo "  1. cp foo.rs foo.rs.bak"
  echo "  2. mkdir foo/  (Rust 2018+ allows foo.rs + foo/ submodules)"
  echo "  3. Move the cluster to foo/<cluster>.rs with pub(super) on cross-file private items"
  echo "  4. Add 'mod <cluster>;' + 'pub use <cluster>::*' as needed in foo.rs"
  echo "  5. cargo check -p <crate> must pass"
  exit 1
fi

echo "✓ All .rs files in crates/ are ≤500 LoC (or in the allow-list)."
