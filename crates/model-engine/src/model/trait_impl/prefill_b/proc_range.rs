// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The processing range of a prefill chunk after the prefix-cache skip,
//! with the re-embed of the tokens that are actually processed.
//!
//! Owner: model-engine prefill.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;
use metrale_model_layers::layers::ops;

pub(in crate::model) enum ProcRange {
    /// 2026-09-25: Process `proc_count` tokens starting at `proc_start`.
    Compute {
        proc_start: usize,
        proc_count: usize,
        effective_seq_len_start: usize,
    },
    /// 2026-09-25: The whole chunk is cached and it is not the last chunk.
    /// `prefill_chunk_dispatch` and `batch.rs` append the chunk's tokens and return this
    /// pointer; the kernel-batched path treats it as an error.
    EarlyReturn(DevicePtr),
}

impl TransformerModel {
    pub(in crate::model) fn prefill_b_proc_range(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        kv_write_start: usize,
        marconi_skip: bool,
        hidden_dst: DevicePtr,
        stream: u64,
    ) -> Result<ProcRange> {
        let h = self.config.hidden_size;
        // 2026-09-25: Re-embeds go to the caller's `hidden_dst`: the stream's own offset
        // in the kernel-batched path, `buffers.hidden_states()` otherwise.
        let hidden = hidden_dst;

        // 2026-09-25: `seq.kv_valid_tokens` tracks the prefix from token 0 whose paged
        // K/V is known to be written, for the insert-side cap in `finalize_last` and
        // `save_checkpoint`:
        //   - chunk 0 resets it to the reused match (`kv_write_start` when
        //     `marconi_skip`, else 0);
        //   - a pass over `[start, start + count)` raises it to at least `start + count`;
        //   - the single-token last-chunk re-run leaves it unchanged.
        if chunk_start == 0 {
            seq.kv_valid_tokens = if marconi_skip { kv_write_start } else { 0 };
        }

        if marconi_skip && kv_write_start > chunk_start {
            let skip_in_chunk = (kv_write_start - chunk_start).min(chunk_len);
            if skip_in_chunk >= chunk_len {
                // 2026-09-25: The whole chunk is cached. The caller, not this function,
                // appends the chunk's tokens.
                seq.seq_len = chunk_start + chunk_len;
                if is_last_chunk {
                    // 2026-09-25: The last chunk still needs logits: re-embed only the last
                    // token into row 0 and process it.
                    let last_tok = tokens[chunk_start + chunk_len - 1];
                    // 2026-09-25: SAFETY: 4 bytes, `size_of::<u32>()`, over the initialised
                    // local `last_tok`.
                    let last_tok_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(&last_tok as *const u32 as *const u8, 4)
                    };
                    let token_id_dev = self.buffers.scratch();
                    self.gpu
                        .copy_h2d_async(last_tok_bytes, token_id_dev, stream)?;
                    if self.has_ngram_embedding() {
                        let last = chunk_start + chunk_len;
                        let cs = last.saturating_sub(self.ngram_lookbehind() + 1);
                        self.embed_tokens_fused(&tokens[cs..last], 1, hidden, stream)?;
                    } else {
                        ops::batched_embed(
                            self.gpu.as_ref(),
                            self.batched_embed_kernel,
                            token_id_dev,
                            self.embed_tokens.weight,
                            hidden,
                            1,
                            h as u32,
                            stream,
                        )?;
                    }
                    self.scale_embeddings(hidden, 1usize, stream)?;
                    Ok(ProcRange::Compute {
                        proc_start: chunk_start + chunk_len - 1,
                        proc_count: 1,
                        effective_seq_len_start: chunk_start + chunk_len - 1,
                    })
                } else {
                    Ok(ProcRange::EarlyReturn(DevicePtr::NULL))
                }
            } else {
                // 2026-09-25: Part of the chunk is cached: re-embed only the uncached tail.
                let uncached_start = chunk_start + skip_in_chunk;
                let uncached_count = chunk_len - skip_in_chunk;
                let uncached_tokens = &tokens[uncached_start..uncached_start + uncached_count];
                // 2026-09-25: SAFETY: `uncached_tokens` has length `uncached_count` (an
                // out-of-range slice panics above), so the `uncached_count * 4` bytes are
                // inside a live `&[u32]`, and `u8` has no alignment requirement.
                let token_ids_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        uncached_tokens.as_ptr() as *const u8,
                        uncached_count * 4,
                    )
                };
                let token_ids_dev = self.buffers.scratch();
                self.gpu
                    .copy_h2d_async(token_ids_bytes, token_ids_dev, stream)?;
                if self.has_ngram_embedding() {
                    let cs = uncached_start.saturating_sub(self.ngram_lookbehind());
                    self.embed_tokens_fused(
                        &tokens[cs..uncached_start + uncached_count],
                        uncached_count,
                        hidden,
                        stream,
                    )?;
                } else {
                    ops::batched_embed(
                        self.gpu.as_ref(),
                        self.batched_embed_kernel,
                        token_ids_dev,
                        self.embed_tokens.weight,
                        hidden,
                        uncached_count as u32,
                        h as u32,
                        stream,
                    )?;
                }
                self.scale_embeddings(hidden, uncached_count, stream)?;
                seq.kv_valid_tokens = seq.kv_valid_tokens.max(uncached_start + uncached_count);
                Ok(ProcRange::Compute {
                    proc_start: uncached_start,
                    proc_count: uncached_count,
                    effective_seq_len_start: uncached_start,
                })
            }
        } else {
            seq.kv_valid_tokens = seq.kv_valid_tokens.max(chunk_start + chunk_len);
            Ok(ProcRange::Compute {
                proc_start: chunk_start,
                proc_count: chunk_len,
                effective_seq_len_start: chunk_start,
            })
        }
    }
}
