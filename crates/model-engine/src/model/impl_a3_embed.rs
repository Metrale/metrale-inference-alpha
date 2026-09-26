// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-token embedding for the decode paths.
//!
//! Owner: model-engine.
//! Invariants:
//! - `embed` returns an error on a model with an n-gram embedding, whose
//!   hashed lookups depend on the token's predecessors; callers holding the
//!   sequence use `embed_ctx`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::TransformerModel;

impl TransformerModel {
    pub(super) fn embed(&self, token: u32, output: DevicePtr, stream: u64) -> Result<()> {
        // 2026-09-25: Without its predecessors, an n-gram model's embedding
        // would miss every hashed lookup and still look plausible, so refuse.
        anyhow::ensure!(
            self.ngram_embed.is_none(),
            "embed(): this model fuses n-gram lookups into its input embedding,              so embedding a token without its preceding context would silently              drop 12/13 of the signal. This code path has not been given the              sequence's tokens yet — use embed_ctx()."
        );
        self.embed_row(token, output, stream)
    }

    /// 2026-09-25: The plain single-row gather, with no n-gram guard. Shared by
    /// `embed` and by `embed_ctx`'s non-n-gram arm.
    fn embed_row(&self, token: u32, output: DevicePtr, stream: u64) -> Result<()> {
        let h = self.config.hidden_size;
        let row_bytes = h * 2;
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        self.gpu.copy_d2d_async(src, output, row_bytes, stream)?;
        // 2026-09-25: A no-op unless the config sets `embed_scale` (the gemma4
        // parser sets sqrt(hidden_size)).
        self.scale_embeddings(output, 1, stream)
    }

    /// 2026-09-25: Embed one token preceded by `history` in its sequence.
    /// `history` must not already include `token`.
    pub(super) fn embed_ctx(
        &self,
        history: &[u32],
        token: u32,
        output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if self.ngram_embed.is_none() {
            return self.embed_row(token, output, stream);
        }
        let lb = self.ngram_lookbehind();
        let tail = &history[history.len().saturating_sub(lb)..];
        let mut ctx = Vec::with_capacity(tail.len() + 1);
        ctx.extend_from_slice(tail);
        ctx.push(token);
        self.embed_tokens_fused(&ctx, 1, output, stream)?;
        self.scale_embeddings(output, 1, stream)
    }
}
