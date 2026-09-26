// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The unit of vision input handed to the model.
//!
//! Owner: model-layers (vision input).
//! Invariants: none beyond the types.

/// 2026-09-25: One vision item: a still image, or a video, ready for the encoder.
///
/// `groups` holds the temporal groups: one for a still image (`image`), and
/// `grid_t` for a video, each group a full `grid_h × grid_w` patch plane built
/// from `temporal_patch_size` consecutive frames. All groups share the item's
/// one grid, so the encoder takes each group as an image of that grid.
///
/// The group count lives in the item rather than in a parallel vector, so the
/// pad run (`pad_count`) and the encoder grids both read it from `t_len`.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionItem {
    /// 2026-09-25: Per temporal group: `[grid_h * grid_w, C * temporal_patch_size * patch^2]`.
    pub groups: Vec<Vec<f32>>,
    /// 2026-09-25: Pre-merge patch grid, shared by all of this item's groups.
    pub grid_h: usize,
    pub grid_w: usize,
}

impl VisionItem {
    /// 2026-09-25: A still image: one temporal group.
    pub fn image(pixels: Vec<f32>, grid_h: usize, grid_w: usize) -> Self {
        Self {
            groups: vec![pixels],
            grid_h,
            grid_w,
        }
    }

    /// 2026-09-25: Temporal extent, in groups; at least 1.
    pub fn t_len(&self) -> usize {
        self.groups.len().max(1)
    }

    /// 2026-09-25: Merged tokens this item occupies in the prompt: `t_len` times
    /// the spatially merged grid. The chat request builder uses it as the
    /// length of the item's pad run.
    pub fn pad_count(&self, spatial_merge_size: usize) -> usize {
        let sms = spatial_merge_size.max(1);
        self.t_len() * (self.grid_h / sms) * (self.grid_w / sms)
    }
}
