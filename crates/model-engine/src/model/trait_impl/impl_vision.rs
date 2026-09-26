// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `impl ModelVision for TransformerModel`, mostly delegating to `<method>_dispatch`
//! helpers in the sibling modules.
//!
//! Owner: model-engine.
//! Invariants: the ones in `trait_impl/mod.rs`.

use anyhow::Result;

use crate::model::types::TransformerModel;
use crate::traits::ModelVision;

impl ModelVision for TransformerModel {
    fn prepare_vision_embed(&self, images: &[metrale_model_layers::VisionItem]) -> Result<()> {
        self.prepare_vision_embed_dispatch(images)
    }

    fn prepare_vision_embed_batched(
        &self,
        per_request: &[Vec<metrale_model_layers::VisionItem>],
    ) -> Result<Vec<(usize, usize, usize, usize)>> {
        self.prepare_vision_embed_batched_dispatch(per_request)
    }

    fn ep_sync_vision_embeds(&self, tokens: &[u32]) -> Result<()> {
        TransformerModel::ep_sync_vision_embeds(self, tokens)
    }

    fn set_vision_slice_base(&self, row_base: usize, grid_base: usize, owned_images: usize) {
        *self.vision_row_base.lock() = row_base;
        *self.vision_grid_base.lock() = grid_base;
        *self.vision_owned_images.lock() = owned_images;
    }

    fn tokens_contain_vision_pad(&self, tokens: &[u32]) -> bool {
        self.tokens_have_vision_pad(tokens)
    }
}
