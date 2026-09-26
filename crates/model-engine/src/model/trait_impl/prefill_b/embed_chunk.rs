// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Chunked-prefill embed: gather the chunk's token embeddings into a
//! hidden buffer, then splice vision encoder rows over pad tokens.
//!
//! Owner: model-engine prefill.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;

use super::super::super::types::TransformerModel;
use metrale_model_layers::layers::ops;

impl TransformerModel {
    pub(super) fn prefill_b_embed_chunk(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        stream: u64,
    ) -> Result<()> {
        let hidden = self.buffers.hidden_states();
        self.prefill_b_embed_chunk_at(tokens, chunk_start, chunk_len, hidden, stream)
    }

    /// 2026-09-25: Embed `tokens[chunk_start..chunk_start + chunk_len]` into rows
    /// `0..chunk_len` of `hidden_dst`, then apply the overlay-adapter rows, the
    /// embedding scale and the vision-pad splice. Callers: `prefill_b_embed_chunk`
    /// (the arena's `hidden_states()`) and the batched prefill in `batch_kernel.rs`
    /// (each stream's own `hidden_dst`).
    pub(in crate::model) fn prefill_b_embed_chunk_at(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        hidden_dst: metrale_gpu_runtime::gpu::DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let elem_bytes = 2usize;

        // 2026-09-25: Token embedding: `chunk_len` rows of `hidden_size` at `hidden_dst`.
        {
            let chunk_tokens = &tokens[chunk_start..chunk_start + chunk_len];
            // 2026-09-25: SAFETY: `chunk_tokens` has length `chunk_len` (an out-of-range
            // chunk panics in the slice index above), so the `chunk_len * 4` bytes are
            // inside a live `&[u32]`, and `u8` has no alignment requirement.
            let token_ids_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(chunk_tokens.as_ptr() as *const u8, chunk_len * 4)
            };
            let token_ids_dev = self.buffers.scratch();
            self.gpu
                .copy_h2d_async(token_ids_bytes, token_ids_dev, stream)?;
            // 2026-09-25: The ids are also staged into `token_ids()`, because the start of
            // `scratch()` is reused by MoE routing (`upload_meta.rs` reserves it). The layer
            // forward passes `token_ids()` to DeepSeek-V4 hash-MoE, which reads
            // `tid2eid[token_id]` for each row in chunk order.
            self.gpu
                .copy_h2d_async(token_ids_bytes, self.buffers.token_ids(), stream)?;
            if self.has_ngram_embedding() {
                // 2026-09-25: n-gram hashes read up to `ngram_lookbehind()` tokens
                // before the chunk, so those earlier prompt tokens are passed as well.
                let cs = chunk_start.saturating_sub(self.ngram_lookbehind());
                self.embed_tokens_fused(
                    &tokens[cs..chunk_start + chunk_len],
                    chunk_len,
                    hidden_dst,
                    stream,
                )?;
            } else {
                ops::batched_embed(
                    self.gpu.as_ref(),
                    self.batched_embed_kernel,
                    token_ids_dev,
                    self.embed_tokens.weight,
                    hidden_dst,
                    chunk_len as u32,
                    h as u32,
                    stream,
                )?;
            }
            if std::env::var("METRALE_DUMP_EMBED").ok().as_deref() == Some("1") {
                self.gpu.synchronize(stream)?;
                let offset = (chunk_len - 1) * h * 2;
                let mut buf = vec![0u8; h * 2];
                let _ = self.gpu.copy_d2h(hidden_dst.offset(offset), &mut buf);
                let v: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                tracing::info!(
                    "METRALE_EMBED post-batched_embed (chunk_start={}, last_tok_id={}): |x|={:.4} first5={:?}",
                    chunk_start,
                    tokens[chunk_start + chunk_len - 1],
                    n,
                    &v[..5]
                );
            }
            // 2026-09-25: Overlay-adapter rows replace overridden vocab ids after the
            // gather and before `scale_embeddings`, so an override row is scaled like a
            // gathered one. `token_ids()` holds this chunk's ids (staged above); a NULL
            // `seq_slot` selects the uniform active route. No-op when no overlay is installed.
            self.apply_embed_overlay(
                self.buffers.token_ids(),
                metrale_gpu_runtime::gpu::DevicePtr(0),
                hidden_dst,
                chunk_len as u32,
                stream,
            )?;
            self.scale_embeddings(hidden_dst, chunk_len, stream)?;
            if std::env::var("METRALE_DUMP_EMBED").ok().as_deref() == Some("1") {
                self.gpu.synchronize(stream)?;
                let offset = (chunk_len - 1) * h * 2;
                let mut buf = vec![0u8; h * 2];
                let _ = self.gpu.copy_d2h(hidden_dst.offset(offset), &mut buf);
                let v: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                tracing::info!(
                    "METRALE_EMBED post-scale_embeddings: |x|={:.4} first5={:?}",
                    n,
                    &v[..5]
                );
            }
        }

        // 2026-09-25: Vision splice: each image-pad or video-pad position in the chunk gets
        // the next merged row of the encoder's packed `buf_out` (filled by
        // `prepare_vision_embed`), starting at this request's `vision_row_base`.
        {
            let pending = *self.vision_embed_patches.lock();
            if pending > 0
                && let Some(ve) = &self.vision_encoder
            {
                let chunk_tokens = &tokens[chunk_start..chunk_start + chunk_len];
                let (image_pad, video_pad) = self.vision_pad_ids();
                let row_base = *self.vision_row_base.lock();
                let mut img_idx = 0usize;
                for (i, &tok) in chunk_tokens.iter().enumerate() {
                    if tok == image_pad || tok == video_pad {
                        let src = ve.out_row(row_base + img_idx);
                        let dst = hidden_dst.offset(i * h * elem_bytes);
                        self.gpu
                            .copy_d2d_async(src, dst, ve.out_hidden_size() * 2, stream)?;
                        img_idx += 1;
                    }
                }
            }
        }

        Ok(())
    }
}
