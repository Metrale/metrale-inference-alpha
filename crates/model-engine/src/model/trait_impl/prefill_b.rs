// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Chunked prefill, `prefill_chunk_dispatch`: one chunk of a prompt through the step modules under `prefill_b/`.
//!
//! Steps, in order: `embed_chunk` (embedding and vision-pad overlay),
//! `prefix_lookup` (prefix cache, EP agreement, Marconi restore), `proc_range`
//! (processing range; may return early), `upload_meta` and `upload_paged`
//! (metadata upload), `forward_layers`, then `finalize_last` on the last chunk
//! or `save_checkpoint` on any other. Once taken, the `kv_cache` lock is held
//! for the rest of the chunk and passed to each step as `&mut`. The multi-stream
//! path is in `batch.rs` and `batch_kernel.rs`.
//!
//! Owner: model-engine.
//! Invariants:
//! - A chunk that returns `Ok`, including a fully cached one, appends its tokens
//!   to `seq.tokens` and sets `seq.seq_len` to `chunk_start + chunk_len`.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::traits::{Model, SequenceState};

mod batch;
mod batch_kernel;
#[cfg(test)]
mod batch_kernel_tests;
mod batched_layer;
mod embed_chunk;
mod exact_leaf;
mod finalize_last;
mod forward_layers;
mod h_state_ptrs;
mod midchunk_capture;
mod prefix_lookup;
mod prefix_reserve;
mod proc_range;
mod prompt_logprobs;
mod save_checkpoint;
mod snap_agree;
#[cfg(test)]
mod snap_agree_tests;
mod stage_batched;
mod upload_meta;
mod upload_paged;

impl TransformerModel {
    pub(super) fn prefill_chunk_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        let total = tokens.len();
        assert!(
            chunk_start + chunk_len <= total,
            "chunk_start({chunk_start}) + chunk_len({chunk_len}) > total({total})"
        );

        // 2026-09-25: Tail-checkpoint split. A later turn's prefix match is
        // block-aligned, and a snapshot deeper than the match cannot be
        // restored. On the last chunk of an SSM model with snapshots and prefix
        // caching on, the prefill is split once at `cut`, one block below the
        // last block boundary under `total`, so `prefill_b_save_checkpoint`
        // saves a snapshot there (save_checkpoint.rs treats `cut` as a prompt
        // tail, whatever `--ssm-checkpoint-interval` is). The split does not
        // depend on radix contents, so a prompt is processed in the same passes
        // cold and warm, and on every rank. `METRALE_NO_TAIL_SPLIT=1` turns the
        // split off. Prompts with vision pads are not split.
        if is_last_chunk
            && self.config.num_ssm_layers() > 0
            && self.ssm_snapshots.is_enabled()
            && self.prefix_cache.is_active()
            && !self.tokens_have_vision_pad(tokens)
        {
            let bs = self.kv_cache.lock().block_size();
            // 2026-09-25: One block below the last block boundary strictly under
            // `total`.
            let cut = ((total.saturating_sub(1) / bs) * bs).saturating_sub(bs);
            let split_disabled = std::env::var("METRALE_NO_TAIL_SPLIT").as_deref() == Ok("1");
            if !split_disabled && cut > chunk_start && cut < total {
                self.prefill_chunk_dispatch(
                    tokens,
                    seq,
                    chunk_start,
                    cut - chunk_start,
                    false,
                    stream,
                )?;
                return self.prefill_chunk_dispatch(tokens, seq, cut, total - cut, true, stream);
            }
        }

        let arena_cap = self.buffers.max_batch_tokens();
        if chunk_len > arena_cap {
            anyhow::bail!(
                "Prefill chunk ({chunk_len} tokens) exceeds buffer arena capacity ({arena_cap} tokens). \
                 Reduce --max-prefill-tokens or prompt length."
            );
        }

        let stream = if self.multi_rank_protocol_active() {
            self.gpu.default_stream()
        } else {
            stream
        };

        // 2026-09-25: With `comm` set, every buffer is zeroed on every chunk.
        // Otherwise only the first chunk zeroes, and only the prefill
        // essentials; later chunks rely on the embedding and the layer forward
        // writing each buffer before it is read.
        if self.comm.is_some() {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        } else if chunk_start == 0 {
            self.buffers
                .zero_prefill_essentials(self.gpu.as_ref(), stream)?;
        }

        let mut kv_cache = self.kv_cache.lock();

        // 2026-09-25: Embed the chunk and overlay vision-pad positions.
        self.prefill_b_embed_chunk(tokens, chunk_start, chunk_len, stream)?;

        // 2026-09-25: Prefix-cache lookup, EP agreement and Marconi snapshot restore.
        let (kv_write_start, marconi_skip) = self.prefill_b_prefix_lookup(
            tokens,
            seq,
            chunk_start,
            total,
            &mut kv_cache,
            stream,
            None,
        )?;

        if std::env::var("METRALE_SSM_SAVE_DUMP").is_ok() {
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                &format!("chunk_entry start={chunk_start} len={chunk_len} kvws={kv_write_start}"),
            );
        }

        let bs = kv_cache.block_size();
        let end_pos = chunk_start + chunk_len;
        let blocks_needed = (end_pos - 1) / bs + 1;
        super::super::block_mgmt::ensure_blocks_through_prefill(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
            self.levers.kv_poison,
        )?;

        // 2026-09-25: Processing range for this chunk; a fully cached chunk
        // returns early.
        let (proc_start, proc_count, effective_seq_len_start) = match self.prefill_b_proc_range(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            is_last_chunk,
            kv_write_start,
            marconi_skip,
            // 2026-09-25: A single stream's hidden rows start at the buffer base.
            self.buffers.hidden_states(),
            stream,
        )? {
            proc_range::ProcRange::Compute {
                proc_start,
                proc_count,
                effective_seq_len_start,
            } => (proc_start, proc_count, effective_seq_len_start),
            proc_range::ProcRange::EarlyReturn(ptr) => {
                // 2026-09-25: A fully cached chunk still records its tokens:
                // decode-checkpoint registration and the radix insert in
                // `cache_sequence` read `seq.tokens`, paired with the full block
                // table.
                seq.tokens
                    .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
                seq.seq_len = chunk_start + chunk_len;
                seq.last_decode_ckpt_block = seq.tokens.len() / bs;
                return Ok(ptr);
            }
        };

        // 2026-09-25: Upload positions (MRoPE when enabled) and slot metadata.
        let upload_meta::MetaLayout {
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
        } = self.prefill_b_upload_meta(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            proc_start,
            proc_count,
            effective_seq_len_start,
            &kv_cache,
            stream,
        )?;

        // 2026-09-25: Paged metadata (block table and `seq_len`).
        if needs_paged {
            self.prefill_b_upload_paged(
                seq,
                total,
                proc_start,
                proc_count,
                meta_base,
                slot_offset,
                &kv_cache,
                stream,
            )?;
        }

        self.gpu.synchronize(stream)?;

        // 2026-09-25: Mid-chunk tail SSM capture is planned before the forward
        // pass, which uses the plan. `None` (flag off, or the pass does not span
        // `tb`, among other cases) means no capture.
        let midcap_plan = self.prepare_midchunk_capture(
            tokens,
            seq,
            &mut kv_cache,
            proc_start,
            proc_count,
            stream,
        );

        // 2026-09-25: Forward through all layers.
        self.prefill_b_forward_layers(
            seq,
            &mut kv_cache,
            chunk_start,
            chunk_len,
            is_last_chunk,
            proc_count,
            effective_seq_len_start,
            kv_write_start,
            marconi_skip,
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
            midcap_plan.as_ref(),
            stream,
        )?;

        // 2026-09-25: Register the captured slots once the pass has written the
        // `tb` state into them.
        if let Some(plan) = midcap_plan.as_ref() {
            self.finalize_midchunk_capture(tokens, seq, plan);
        }

        // 2026-09-25: Append this chunk's tokens; the early-return arm above
        // appends them itself.
        seq.tokens
            .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
        seq.seq_len = chunk_start + chunk_len;
        // 2026-09-25: Prime the decode-checkpoint gate; after the last chunk it
        // holds the prompt's full-block count (see prefill_a.rs).
        seq.last_decode_ckpt_block = seq.tokens.len() / bs;

        // 2026-09-25: Prompt logprobs are projected while this chunk's hidden
        // rows are live, before `prefill_b_finalize_last` overwrites the norm
        // output and logits. A no-op unless `seq.collect_prompt_logprobs` is
        // set.
        self.collect_prompt_logprobs_chunk(
            tokens,
            seq,
            chunk_start,
            proc_start,
            proc_count,
            stream,
        )?;

        if is_last_chunk {
            // 2026-09-25: Final norm, LM head, prefix-cache insert and snapshot save.
            self.prefill_b_finalize_last(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                proc_count,
                stream,
            )
        } else {
            // 2026-09-25: Intermediate Marconi checkpoint.
            self.prefill_b_save_checkpoint(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                stream,
            )?;
            Ok(DevicePtr::NULL)
        }
    }
}
