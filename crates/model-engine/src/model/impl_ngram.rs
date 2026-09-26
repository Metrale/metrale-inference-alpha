// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Multi-token input embedding: the fused n-gram embedding when the model has
//! one, else a plain `embed_tokens` gather.
//!
//! An n-gram embedding adds hashed lookups keyed on each token's predecessors, so
//! embedding needs the sequence context, not only the ids being embedded.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::TransformerModel;
use metrale_model_layers::layers::ops;

impl TransformerModel {
    /// 2026-09-25: Embed `seq_len` tokens into `out` (`[seq_len, hidden]` BF16).
    ///
    /// `ctx_tokens` must end with the tokens being embedded and may be
    /// preceded by up to `ngram_lookbehind()` earlier tokens of the same
    /// sequence, which the n-gram hash reads. A missing predecessor reads as
    /// id 0 (as after an EOS), so passing only the new tokens is correct only
    /// at the start of a sequence. Without an n-gram embedding the last
    /// `seq_len` ids are gathered from `embed_tokens`.
    pub(crate) fn embed_tokens_fused(
        &self,
        ctx_tokens: &[u32],
        seq_len: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if self.gpu.op_cache().first_n("diag:ngram_embed_arm", 3) {
            tracing::info!(
                "embed_tokens_fused: ngram={} ctx_len={} seq_len={}",
                self.ngram_embed.is_some(),
                ctx_tokens.len(),
                seq_len
            );
        }
        if let Some(ngram) = self.ngram_embed.as_ref() {
            let mut ng = ngram
                .lock()
                .map_err(|_| anyhow::anyhow!("ngram embedding mutex poisoned"))?;
            return ng.embed(ctx_tokens, seq_len, out, self.gpu.as_ref(), stream);
        }
        let ids = &ctx_tokens[ctx_tokens.len() - seq_len..];
        let bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        let ids_dev = self.buffers.scratch();
        self.gpu.copy_h2d_async(&bytes, ids_dev, stream)?;
        ops::batched_embed(
            self.gpu.as_ref(),
            self.batched_embed_kernel,
            ids_dev,
            self.embed_tokens.weight,
            out,
            seq_len as u32,
            self.config.hidden_size as u32,
            stream,
        )
    }

    /// 2026-09-25: How many earlier tokens the n-gram hash reads
    /// (`neighbor_num - 1`). Zero when this model has no n-gram embedding, so
    /// callers can build the context slice unconditionally.
    pub(crate) fn ngram_lookbehind(&self) -> usize {
        match self.ngram_embed.as_ref() {
            Some(ng) => ng
                .lock()
                .map(|g| g.dims.neighbor_num.saturating_sub(1))
                .unwrap_or(0),
            None => 0,
        }
    }
}
