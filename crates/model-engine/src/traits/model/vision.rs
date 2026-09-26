// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelVision`, one of the supertraits `Model` is made of. Its methods, default
//! bodies and docs are the ones `Model` declared before the split.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use anyhow::Result;

/// 2026-09-26: Vision-encoder input for prompts with images.
pub trait ModelVision {
    /// 2026-09-25: Encode images with the vision encoder and keep the embeddings for the next
    /// prefill. Call it before `prefill_chunk` for a prompt with image pad tokens. Default:
    /// `Ok(())`.
    fn prepare_vision_embed(&self, _images: &[metrale_model_layers::VisionItem]) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Give every rank the same merged vision rows for this prompt. Otherwise only
    /// rank 0 has the encoder's rows, the other ranks keep the pad-token embedding, and the
    /// per-layer all-reduce mixes the two. The head calls it after the prompt broadcast and the
    /// worker at the matching point, so the collective pairs. Default: `Ok(())`.
    fn ep_sync_vision_embeds(&self, _tokens: &[u32]) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Encode the images of several requests in one encoder pass. Returns, per
    /// request in order, `(patch_row_offset, grid_index_offset, num_images, patch_row_count)`
    /// in the shared output buffer. Default: an empty `Vec`.
    fn prepare_vision_embed_batched(
        &self,
        _per_request: &[Vec<metrale_model_layers::VisionItem>],
    ) -> Result<Vec<(usize, usize, usize, usize)>> {
        Ok(Vec::new())
    }

    /// 2026-09-25: Where the next `prefill_chunk` reads its vision rows in the batched encoder
    /// output: row offset, grid index offset and image count; `(0, 0, 0)` resets. Default:
    /// no-op.
    fn set_vision_slice_base(&self, _row_base: usize, _grid_base: usize, _owned_images: usize) {}

    /// 2026-09-25: Whether `tokens` holds a vision pad token, whose KV came from image
    /// embeddings that re-prefilling the tokens cannot reproduce. Decode preemption then does
    /// not pick the sequence for requeue-and-re-prefill unless it can be spilled. Default
    /// `false`.
    fn tokens_contain_vision_pad(&self, _tokens: &[u32]) -> bool {
        false
    }
}
