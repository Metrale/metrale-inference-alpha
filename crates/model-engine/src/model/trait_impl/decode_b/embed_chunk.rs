// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The fused mixed forward's prefill-chunk embed: upload the chunk's token ids and
//! embed them into the rows above the padded decode rows.
//!
//! Owner: model-engine (decode).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layers::ops;

use super::super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-09-26: Embed `prefill_tokens[prefill_chunk_start..prefill_chunk_start + n_prefill]`
    /// into `prefill_hidden` with one batched embed (or the fused n-gram embed), then scale.
    pub(super) fn mixed_embed_prefill_chunk(
        &self,
        prefill_tokens: &[u32],
        prefill_chunk_start: usize,
        n_prefill: usize,
        prefill_hidden: DevicePtr,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        let chunk_tokens = &prefill_tokens[prefill_chunk_start..prefill_chunk_start + n_prefill];
        // 2026-09-25: SAFETY: `chunk_tokens` is sliced above with an end bound of
        // `prefill_chunk_start + n_prefill`, so its length is `n_prefill` (an
        // out-of-range chunk panics in that slice first), and the byte length
        // is `n_prefill * 4` over a live `&[u32]`.
        let token_ids_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(chunk_tokens.as_ptr() as *const u8, n_prefill * 4)
        };
        // 2026-09-25: `norm_output` stages the token ids; the first layer overwrites it.
        let token_ids_dev = self.buffers.norm_output();
        self.gpu
            .copy_h2d_async(token_ids_bytes, token_ids_dev, stream)?;
        if self.has_ngram_embedding() {
            // 2026-09-25: The n-gram hashes read behind the chunk, so pass the
            // earlier tokens of this prefill too.
            let cs = prefill_chunk_start.saturating_sub(self.ngram_lookbehind());
            self.embed_tokens_fused(
                &prefill_tokens[cs..prefill_chunk_start + n_prefill],
                n_prefill,
                prefill_hidden,
                stream,
            )?;
        } else {
            ops::batched_embed(
                self.gpu.as_ref(),
                self.batched_embed_kernel,
                token_ids_dev,
                self.embed_tokens.weight,
                prefill_hidden,
                n_prefill as u32,
                h as u32,
                stream,
            )?;
        }
        self.scale_embeddings(prefill_hidden, n_prefill, stream)?;
        Ok(())
    }
}
