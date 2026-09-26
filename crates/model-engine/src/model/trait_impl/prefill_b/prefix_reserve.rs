// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched prefill prefix-cache admission: reserve every stream's prefix
//! match up front, so a kernel-batched wave is admitted or declined as a whole, and
//! release what was reserved when it is declined.
//!
//! Owner: model-engine prefill (batched).
//! Invariants: none beyond the types.
#![allow(unused_imports, dead_code, clippy::too_many_arguments)]
use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_telemetry::prefix_cache::PrefixMatch;

use super::super::super::types::TransformerModel;
use crate::traits::{PrefillSlice, SequenceState};

impl TransformerModel {
    /// 2026-09-25: Look up every stream's prefix match before the batched path changes
    /// any sequence. `Some(vec![])` when the prefix cache is inactive or
    /// the wave does not start at chunk 0; `None` for an empty `streams`, or when the
    /// wave is declined, after releasing the matches taken so far. A release decrements
    /// no more than each lookup's `matched_tokens` (`release_matched`).
    pub(in crate::model) fn prefill_b_reserve_batched_prefix_matches(
        &self,
        streams: &[PrefillSlice<'_>],
        block_size: usize,
    ) -> Option<Vec<PrefixMatch>> {
        if !self.prefix_cache.is_active() || streams.first()?.chunk_start != 0 {
            return Some(Vec::new());
        }
        // 2026-09-25: A multi-rank world agrees on the match through `ep_min_u32` in
        // `prefix_lookup`, which this reservation does not run, so it declines.
        if self.multi_rank_protocol_active() {
            tracing::info!(
                target: "metrale::q12",
                "batched prefix reservation declined: multi-rank world needs \
                 the EP min-reduction — falling back to per-stream"
            );
            return None;
        }

        let mut matches = Vec::with_capacity(streams.len());
        for slice in streams {
            let seq = &*slice.seq;
            if self.tokens_have_vision_pad(slice.prompt_tokens)
                || seq.collect_prompt_logprobs.is_some()
            {
                tracing::info!(
                    target: "metrale::q12",
                    "batched prefix reservation declined: vision pads or \
                     prompt-logprob collection — falling back to per-stream"
                );
                self.release_batched_prefix_reservations(streams, &matches, block_size);
                return None;
            }
            matches.push(self.prefix_cache.lookup(
                slice.prompt_tokens,
                block_size,
                seq.session_hash,
                seq.adapter_id,
            ));
        }

        // 2026-09-25: Hybrid-SSM models: any warm match declines, and the per-stream path
        // does the snapshot restore. An all-cold reservation acquires no blocks and
        // restores nothing, so it is admitted (`batched_reserve_hybrid_ssm_ok`).
        if !super::batch_kernel::batched_reserve_hybrid_ssm_ok(
            &matches,
            self.config.num_ssm_layers() != 0,
        ) {
            tracing::info!(
                target: "metrale::q12",
                "batched prefix reservation declined: hybrid-SSM model with a \
                 warm prefix match — falling back to per-stream"
            );
            self.release_batched_prefix_reservations(streams, &matches, block_size);
            return None;
        }

        if !super::batch_kernel::cache_batch_matches_compatible(&matches, streams[0].chunk_len) {
            tracing::info!(
                target: "metrale::q12",
                "batched prefix reservation declined: prefix matches not \
                 batch-compatible — falling back to per-stream"
            );
            self.release_batched_prefix_reservations(streams, &matches, block_size);
            return None;
        }
        Some(matches)
    }

    fn release_batched_prefix_reservations(
        &self,
        streams: &[PrefillSlice<'_>],
        matches: &[PrefixMatch],
        block_size: usize,
    ) {
        for (slice, prefix_match) in streams.iter().zip(matches) {
            if prefix_match.matched_tokens > 0 {
                self.prefix_cache.release_matched(
                    slice.prompt_tokens,
                    block_size,
                    prefix_match.matched_tokens,
                    slice.seq.adapter_id,
                );
            }
        }
    }
}
