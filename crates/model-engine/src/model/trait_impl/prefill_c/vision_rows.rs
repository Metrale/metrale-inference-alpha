// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `prefill_twophase`'s vision rows: the image/video pad positions of the embedded
//! prompt take the vision encoder's output rows.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-09-26: When vision patches are pending, copy the encoder's rows over the pad
    /// positions of `hidden` (row stride `h * fp32` bytes), in token order.
    pub(super) fn twophase_vision_rows(
        &self,
        tokens: &[u32],
        hidden: DevicePtr,
        h: usize,
        fp32: usize,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Overwrite the image/video pad positions with the vision encoder's
        // output rows, in order.
        let pending = *self.vision_embed_patches.lock();
        if pending > 0
            && let Some(ve) = &self.vision_encoder
        {
            let (image_pad, video_pad) = self.vision_pad_ids();
            let mut img_idx = 0usize;
            for (i, &tok) in tokens.iter().enumerate() {
                if tok == image_pad || tok == video_pad {
                    let src = ve.out_row(img_idx);
                    let dst = hidden.offset(i * h * fp32);
                    self.gpu
                        .copy_d2d_async(src, dst, ve.out_hidden_size() * 2, stream)?;
                    img_idx += 1;
                }
            }
        }
        Ok(())
    }
}
