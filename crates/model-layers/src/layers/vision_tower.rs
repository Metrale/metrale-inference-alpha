// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The vision tower a built model holds, whichever family bound it.
//!
//! Owner: model-layers (vision).
//! Invariants: none beyond the types.
//!
//! Two towers exist: the Qwen3-VL-shaped [`VisionEncoder`] and GLM-5.3's
//! [`GlmVit`]. The engine uses them only through this enum:
//!
//! * `forward_batched(images) -> per-image (post_h, post_w, merged_p)`,
//! * a packed `[Σmerged_p, out_hidden_size]` BF16 output, image-ordered, read
//!   through `out_row`,
//! * `out_hidden_size`,
//! * `ensure_scratch`, for a tensor-parallel worker that receives rows instead
//!   of encoding.
//!
//! An enum rather than a trait object, so adding a tower means adding a
//! variant to every match here.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::glm_vit::GlmVit;
use super::vision_encoder::VisionEncoder;

pub enum VisionTower {
    /// 2026-09-25: Built by `weight_loader::qwen3_vl` and `weight_loader::qwen35`
    /// (which the qwen35_dense and qwen4_exp loaders call).
    Qwen(Box<VisionEncoder>),
    /// 2026-09-25: GLM-5.3-Flash (`glm5_next`).
    Glm(Box<GlmVit>),
}

impl VisionTower {
    pub fn qwen(encoder: VisionEncoder) -> Self {
        Self::Qwen(Box::new(encoder))
    }

    pub fn glm(encoder: GlmVit) -> Self {
        Self::Glm(Box::new(encoder))
    }

    /// 2026-09-25: Encode a batch of `(pixels, grid_h, grid_w)` temporal groups. Returns
    /// `(post_merge_h, post_merge_w, merged_patches)` per group, in order.
    pub fn forward_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        match self {
            Self::Qwen(e) => e.forward_batched(images, gpu, stream),
            Self::Glm(e) => e.forward_batched(images, gpu, stream),
        }
    }

    /// 2026-09-25: Width of one merged embedding row.
    pub fn out_hidden_size(&self) -> usize {
        match self {
            Self::Qwen(e) => e.out_hidden_size,
            Self::Glm(e) => e.out_hidden_size,
        }
    }

    /// 2026-09-25: Allocate the encoder's device scratch if it is not allocated yet.
    ///
    /// Normally the first image does this. A tensor-parallel worker receives
    /// the merged rows by broadcast instead of encoding, so it calls this
    /// before `out_row` names a destination.
    pub fn ensure_scratch(&self, gpu: &dyn GpuBackend) -> Result<()> {
        match self {
            Self::Qwen(e) => e.scratch_init(gpu),
            Self::Glm(e) => e.scratch_init(gpu),
        }
    }

    /// 2026-09-25: Device pointer to merged row `row` of the packed output. The
    /// Qwen arm panics unless `forward_batched` or `ensure_scratch` has
    /// allocated the scratch.
    pub fn out_row(&self, row: usize) -> DevicePtr {
        match self {
            Self::Qwen(e) => e.scratch().buf_out.offset(row * e.out_hidden_size * 2),
            Self::Glm(e) => e.out_row(row),
        }
    }
}
