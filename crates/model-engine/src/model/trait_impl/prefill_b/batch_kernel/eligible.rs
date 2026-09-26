// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Admission rules for the kernel-batched prefill in `batch_kernel.rs`.
//!
//! Holds `kernel_batched_eligible` and its pure core
//! `check_kernel_batched_eligible` (tested in `batch_kernel_tests.rs`), the
//! prefix-reservation predicates, the switches they read, and the
//! `codispatch_btcheck` diagnostic.
//!
//! Owner: model-engine.
//! Invariants:
//! - `check_kernel_batched_eligible`, `cache_batch_matches_compatible` and
//!   `batched_reserve_hybrid_ssm_ok` read only their arguments.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use metrale_config::ModelConfig;

use super::super::super::super::types::TransformerModel;
use crate::traits::PrefillSlice;
use metrale_telemetry::prefix_cache::PrefixMatch;

/// 2026-09-25: Whether the model uses MLA attention (`kv_lora_rank > 0`, the
/// same test that sets `AttentionType::Mla`), which the batched path rejects.
/// Pinned by `mistral_config_is_rejected_as_mla` in `batch_kernel_tests.rs`.
pub(in crate::model) fn config_is_mla(config: &ModelConfig) -> bool {
    config.kv_lora_rank > 0
}

/// 2026-09-25: Whether chunk-0 streams may use the batched prefill path:
/// `layers::ops::prefill_batched_first_chunk_enabled`, which is
/// `--prefill-codispatch` (else `METRALE_PREFILL_CODISPATCH`) or
/// `METRALE_Q12_BATCHED_FIRST_CHUNK`.
pub(super) fn first_chunk_batched_enabled() -> bool {
    metrale_model_layers::layers::ops::prefill_batched_first_chunk_enabled()
}

impl TransformerModel {
    /// 2026-09-25: True when the kernel-batched path may take these streams.
    /// The dispatcher runs the per-stream path when it is false.
    pub(in crate::model) fn kernel_batched_eligible(&self, streams: &[PrefillSlice<'_>]) -> bool {
        // 2026-09-25: MoE models are admitted: the routed-MoE prefill works on
        // the flat token list, and nothing under layers/moe reads `chunk_len`,
        // `batch_size` or `stream_idx`.
        let varlen = varlen_prefill_enabled();
        // 2026-09-25: Effective staged length per stream. `peek_matched_tokens`
        // only reads the radix tree: no refs, no LRU update, no hit or miss
        // counted.
        let bs = self.kv_cache.lock().block_size();
        let effective_arena = effective_arena_charge_enabled();
        let eff = |s: &PrefillSlice<'_>| -> usize {
            if !effective_arena {
                return s.chunk_len;
            }
            let matched =
                self.prefix_cache
                    .peek_matched_tokens(s.prompt_tokens, bs, s.seq.adapter_id);
            let skip = matched.saturating_sub(s.chunk_start).min(s.chunk_len);
            // 2026-09-25: A fully cached last chunk still stages one token, the
            // row the LM head reads. A fully cached middle chunk stages nothing
            // and reports 0, which makes `check_kernel_batched_eligible` reject
            // the batch.
            match s.chunk_len - skip {
                0 if s.is_last_chunk => 1,
                n => n,
            }
        };
        check_kernel_batched_eligible(
            streams
                .iter()
                .map(|s| (s.chunk_len, eff(s), s.chunk_start, s.is_last_chunk)),
            streams.len(),
            self.buffers.max_batch_tokens(),
            config_is_mla(&self.config),
            self.config.head_dim,
            self.buffers.scratch_bytes(),
            self.config.num_experts_per_tok,
            self.config.mrope_interleaved,
            // 2026-09-25: Chunk 0 is admitted with codispatch or varlen on.
            metrale_model_layers::layers::ops::prefill_batched_first_chunk_enabled() || varlen,
            varlen,
        )
    }
}

/// 2026-09-25: Whether reserved prefix matches can share one stacked forward.
///
/// Narrower than the single-stream prefix-cache path: every match must have
/// the same matched length and block count, below `chunk_len`, with no disk
/// blocks and no SSM snapshot (resident or tiered). An empty list is rejected.
/// The caller owns the reservation and its release.
pub(in crate::model) fn cache_batch_matches_compatible(
    matches: &[PrefixMatch],
    chunk_len: usize,
) -> bool {
    let Some(first) = matches.first() else {
        return false;
    };
    let matched = first.matched_tokens;
    // 2026-09-25: A match covering the whole chunk is left to the per-stream
    // path, which has the single-token logits and early-return cases for it.
    if matched >= chunk_len {
        return false;
    }
    matches.iter().all(|m| {
        m.matched_tokens == matched
            && m.matched_blocks.len() == first.matched_blocks.len()
            && m.matched_disk_block_ids.is_empty()
            && m.ssm_snapshot.is_none()
            && m.ssm_snapshot_tokens == 0
            && m.ssm_snapshot_tier_key.is_none()
            && m.ssm_snapshot_tier_tokens == 0
    })
}

/// 2026-09-25: A model with SSM layers may use the batched prefix reservation
/// only when every reservation is cold (`matched_tokens == 0`); a warm match
/// on such a model makes the reservation return `None`, and the batch runs
/// per-stream. Attention-only models are not restricted here.
pub(in crate::model) fn batched_reserve_hybrid_ssm_ok(
    matches: &[PrefixMatch],
    hybrid_ssm: bool,
) -> bool {
    !hybrid_ssm || matches.iter().all(|m| m.matched_tokens == 0)
}

impl TransformerModel {
    /// 2026-09-25: Diagnostic: warn when two streams of the batch share a KV
    /// block or an SSM slot, and log each stream's slot and prompt shape. Runs
    /// only with `METRALE_CODISPATCH_BTCHECK=1`.
    pub(super) fn codispatch_btcheck(&self, streams: &[PrefillSlice<'_>], n: usize) {
        if std::env::var("METRALE_CODISPATCH_BTCHECK").ok().as_deref() != Some("1") {
            return;
        }
        let mut owner: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut slot_owner: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut dump: Vec<(usize, usize, Option<usize>, usize, u32)> = Vec::new();
        for (b, slice) in streams.iter().enumerate() {
            let bt = slice.seq.block_table.clone();
            let slot = slice.seq.slot_idx;
            // 2026-09-25: The slot owned through the `ssm_slot` guard, plus the
            // prompt length and first token to tell the prompts apart.
            let guard_slot = slice.seq.ssm_slot.as_ref().and_then(|g| g.idx());
            let ptoks = slice.prompt_tokens.len();
            let tok0 = slice.prompt_tokens.first().copied().unwrap_or(0);
            if let Some(gs) = guard_slot {
                if let Some(&prev) = slot_owner.get(&gs) {
                    tracing::warn!(
                        "METRALE_GUARDSHARE n={n}: GUARD slot {gs} SHARED by stream {prev} and {b}"
                    );
                } else {
                    slot_owner.insert(gs, b);
                }
            }
            for &blk in &bt {
                if let Some(&prev) = owner.get(&blk) {
                    tracing::warn!(
                        "METRALE_BTSHARE n={n}: KV block {blk} SHARED by stream {prev} and {b}"
                    );
                } else {
                    owner.insert(blk, b);
                }
            }
            dump.push((b, slot, guard_slot, ptoks, tok0));
        }
        tracing::warn!("METRALE_BTDUMP n={n} (stream,slot_idx,guard_slot,ptoks,tok0): {dump:?}");
    }
}

/// 2026-09-25: Whether varlen batched prefill is on (`--prefill-varlen-batch`,
/// else `METRALE_PREFILL_VARLEN`); it admits streams of different chunk
/// lengths into one forward.
pub(in crate::model) fn varlen_prefill_enabled() -> bool {
    // 2026-09-25: The same predicate the batched-attention layer reads.
    metrale_model_layers::layers::ops::prefill_varlen_enabled()
}

/// 2026-09-25: The pure admission rules behind
/// [`TransformerModel::kernel_batched_eligible`]. Each stream is
/// `(chunk_len, eff_len, chunk_start, is_last_chunk)`, where `eff_len` is the
/// number of tokens the stream will stage.
#[allow(clippy::too_many_arguments)]
pub(in crate::model) fn check_kernel_batched_eligible<I>(
    streams: I,
    n: usize,
    arena_cap: usize,
    is_mla: bool,
    head_dim: usize,
    scratch_cap: usize,
    top_k: usize,
    mrope: bool,
    allow_chunk_zero: bool,
    varlen: bool,
) -> bool
where
    I: IntoIterator<Item = (usize, usize, usize, bool)>,
{
    if n < 2 {
        return false;
    }
    if is_mla {
        return false;
    }
    if head_dim > 256 {
        return false;
    }
    let mut first: Option<(usize, usize, bool)> = None;
    let mut total = 0usize;
    let mut max_chunk_len = 0usize;
    for (chunk_len, eff_len, chunk_start, is_last) in streams {
        // 2026-09-25: `chunk_start` and `is_last_chunk` must match across
        // streams: the batch shares one `effective_seq_len_start`, and the
        // finalize step takes `is_last_chunk` from the batch as a whole.
        // `chunk_len` must match too unless varlen is on.
        match first {
            None => first = Some((chunk_len, chunk_start, is_last)),
            Some((cl, cs, il)) => {
                if (!varlen && chunk_len != cl) || chunk_start != cs || is_last != il {
                    return false;
                }
            }
        }
        // 2026-09-25: The arena is charged by `eff_len`, the tokens a stream
        // will stage: the packed layout advances by `proc_count`, which a
        // prefix hit shrinks to the uncached suffix. The caller passes
        // `eff_len == chunk_len` unless `METRALE_Q12_EFFECTIVE_ARENA=1` probed a
        // hit. A zero-token stream has no segment in the packed layout, so the
        // batch is rejected and runs per-stream, where `proc_range` returns
        // early for a fully cached middle chunk.
        if eff_len == 0 {
            return false;
        }
        total += eff_len;
        max_chunk_len = max_chunk_len.max(eff_len);
    }
    let Some((_chunk_len, chunk_start, _)) = first else {
        return false;
    };
    // 2026-09-25: Chunk 0 is admitted only when `allow_chunk_zero` is set.
    if chunk_start == 0 && !allow_chunk_zero {
        return false;
    }
    if total > arena_cap {
        return false;
    }
    // 2026-09-25: The kernel-batched staging footprint must fit in scratch.
    // Checked here, before any stream is mutated, so a rejection runs
    // per-stream from a clean state.
    let scratch_needed = if varlen {
        metrale_gpu_runtime::buffers::q12_batched_scratch_bytes_varlen(
            n,
            total,
            max_chunk_len,
            top_k,
            mrope,
        )
    } else {
        metrale_gpu_runtime::buffers::q12_batched_scratch_bytes(n, max_chunk_len, top_k, mrope)
    };
    scratch_needed <= scratch_cap
}

/// 2026-09-25: `METRALE_Q12_EFFECTIVE_ARENA=1` charges the batched-prefill arena
/// by the tokens each stream will stage (chunk minus the probed cached prefix)
/// instead of the raw chunk length; off otherwise.
///
/// The staging loop in `batch_kernel.rs` checks each stream's real packed
/// offset against the arena and returns an error if a match shrank after the
/// probe.
pub(in crate::model) fn effective_arena_charge_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_Q12_EFFECTIVE_ARENA").as_deref() == Ok("1"))
}
