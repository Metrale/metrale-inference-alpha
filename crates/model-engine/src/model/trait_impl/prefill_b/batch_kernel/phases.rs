// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Phases of `prefill_batch_chunk_kernel_batched`, each called once from it: the
//! KV allocation pre-flight, the shared layer loop and the per-stream finalize.
//!
//! Owner: model-engine.
//! Invariants:
//! - `kv_preflight` returns `Flow::Return` before any sequence is changed.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::super::types::TransformerModel;
use super::KernelBatchResult;
use crate::traits::{PrefillSlice, SequenceState};
use metrale_model_layers::layer::{BatchedAttnMetadata, ForwardContext, GdnPrefillBuffers};

/// 2026-09-26: One stream's setup result, read by the layer loop and the finalize step.
pub(super) struct PerStreamMeta {
    pub(super) chunk_start: usize,
    pub(super) proc_start: usize,
    pub(super) proc_count: usize,
    pub(super) effective_seq_len_start: usize,
    pub(super) kv_write_start_eff: usize,
    pub(super) block_table_dev: DevicePtr,
    pub(super) seq_len_dev: DevicePtr,
    pub(super) num_blocks: usize,
    // 2026-09-25: Σ proc_count of earlier streams: this stream's hidden-row
    // offset, read by the finalize step.
    pub(super) proc_off: usize,
}

/// 2026-09-26: Whether the caller goes on (`Proceed`) or returns the carried result.
pub(super) enum Flow {
    Proceed,
    Return(KernelBatchResult),
}

impl TransformerModel {
    /// 2026-09-26: The allocation pre-flight: evict prefix-cache blocks until the batch's
    /// block need fits, or return `NotAdmitted` when it cannot.
    pub(super) fn kv_preflight(
        &self,
        streams: &[PrefillSlice<'_>],
        kv_cache: &mut PagedKvCache,
    ) -> Flow {
        let bs = kv_cache.block_size();
        let mut needed = 0usize;
        for s in streams.iter() {
            let through = (s.chunk_start + s.chunk_len).div_ceil(bs);
            // 2026-09-25: Blocks this stream already has, plus the ones its
            // prefix match will hand it (reused, not allocated).
            let matched_blocks =
                self.prefix_cache
                    .peek_matched_tokens(s.prompt_tokens, bs, s.seq.adapter_id)
                    / bs;
            let have = s.seq.block_table.len() + matched_blocks;
            needed += through.saturating_sub(have);
        }
        // 2026-09-25: Evicting prefix-cache blocks changes no sequence. It
        // loops because an eviction frees nothing for a block a live
        // sequence still holds.
        while kv_cache.num_free_blocks() < needed {
            let short = needed - kv_cache.num_free_blocks();
            let evicted = self.prefix_cache.evict(short);
            if evicted.is_empty() {
                break;
            }
            super::super::super::super::block_mgmt::apply_evicted_blocks(evicted, kv_cache);
        }
        let free = kv_cache.num_free_blocks();
        if free < needed {
            tracing::debug!(
                target: "metrale::q12",
                n = streams.len(),
                needed,
                free,
                "Q12 kernel-batched declined: cohort needs more KV than is \
                 reclaimable — falling back to per-stream so one exhaustion \
                 cannot fail the whole batch"
            );
            return Flow::Return(KernelBatchResult::NotAdmitted);
        }
        Flow::Proceed
    }

    /// 2026-09-26: One pass over the layers for all streams at their packed offsets.
    pub(super) fn run_batched_layers(
        &self,
        streams: &mut [PrefillSlice<'_>],
        per_stream: &[PerStreamMeta],
        hidden_base: DevicePtr,
        _residual_base: DevicePtr,
        kv_cache: &mut PagedKvCache,
        kv_write_starts: &[usize],
        seq_lens_start: usize,
        meta: &BatchedAttnMetadata,
        gdn_bufs: &GdnPrefillBuffers,
        h_state_ptrs_off: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut seqs_vec: Vec<&mut SequenceState> =
                streams.iter_mut().map(|s| &mut *s.seq).collect();

            if layer.is_ssm_layer() {
                let proc_starts: Vec<usize> = per_stream.iter().map(|m| m.proc_start).collect();
                self.prefill_ssm_batched_layer(
                    layer.as_ref(),
                    layer_idx,
                    hidden_base,
                    _residual_base,
                    &mut seqs_vec,
                    kv_cache,
                    &proc_starts,
                    meta,
                    gdn_bufs,
                    h_state_ptrs_off,
                    ctx,
                    stream,
                )?;
            } else {
                self.prefill_attn_batched_layer(
                    layer.as_ref(),
                    layer_idx,
                    hidden_base,
                    _residual_base,
                    &mut seqs_vec,
                    kv_cache,
                    kv_write_starts,
                    seq_lens_start,
                    meta,
                    ctx,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// 2026-09-26: Append each stream's chunk to its sequence, then finalize it (last chunk)
    /// or save its checkpoint. Returns one logits pointer per stream, NULL when not last.
    pub(super) fn finalize_batched_streams(
        &self,
        streams: &mut [PrefillSlice<'_>],
        per_stream: &[PerStreamMeta],
        kv_cache: &mut PagedKvCache,
        is_last_chunk: bool,
        row_base: usize,
        n: usize,
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        let mut logits_out: Vec<DevicePtr> = Vec::with_capacity(n);
        for (b, slice) in streams.iter_mut().enumerate() {
            let tokens = slice.prompt_tokens;
            let chunk_start = slice.chunk_start;
            let cl = slice.chunk_len;
            let seq = &mut *slice.seq;
            let m = &per_stream[b];

            seq.tokens
                .extend_from_slice(&tokens[chunk_start..chunk_start + cl]);
            seq.seq_len = chunk_start + cl;

            let logits = if is_last_chunk {
                self.prefill_b_finalize_last_at(
                    tokens,
                    seq,
                    kv_cache,
                    chunk_start,
                    cl,
                    m.proc_count,
                    // 2026-09-25: The stream's packed hidden offset; finalize
                    // reads its last token at `proc_off + proc_count - 1`.
                    m.proc_off,
                    // 2026-09-25: Shifted clear of the decode lanes in a mixed
                    // step; `b` alone would land on decode lane `b`'s logits row.
                    row_base + b,
                    stream,
                )?
            } else {
                self.prefill_b_save_checkpoint(tokens, seq, kv_cache, chunk_start, cl, stream)?;
                DevicePtr::NULL
            };
            logits_out.push(logits);
        }
        Ok(logits_out)
    }
}
