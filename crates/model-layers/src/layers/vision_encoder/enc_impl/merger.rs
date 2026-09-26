// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `apply_merger`: per-patch norm → spatial merge → fc1 → GELU → fc2.
//!
//! Owner: model-layers (vision).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{MergerLayer, VisionEncoder};

impl VisionEncoder {
    /// 2026-09-25: Apply one DeepStack or final merger: per-patch norm → spatial
    /// merge → fc1 → GELU → fc2, writing `merged_p` rows to `out_slice`.
    ///
    /// The LayerNorm runs in place on `hidden_src`, per patch, with
    /// `hidden_size`-wide `norm_w`/`norm_b`, before the spatial merge. A
    /// DeepStack merger, which runs between blocks, must be given a copy of
    /// `buf_h1` (`forward_batched` passes `buf_h2`); the final merger runs after
    /// the last block and may normalise `buf_h1` itself.
    pub(super) fn apply_merger(
        &self,
        m: &MergerLayer,
        p: usize,
        grid_h: usize,
        grid_w: usize,
        hidden_src: DevicePtr,
        out_slice: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let ms = self.spatial_merge_size as u32;
        let hidden = self.hidden_size as u32;
        let merged_in = ms * ms * self.hidden_size as u32;
        let out_h_size = self.out_hidden_size as u32;
        let merged_p = (p / (self.spatial_merge_size * self.spatial_merge_size)) as u32;

        // 2026-09-25: 1. Norm each of the p patches in place on hidden_src.
        KernelLaunch::new(gpu, self.k_norm)
            .grid([p as u32, 1, 1])
            .block([hidden.min(1024), 1, 1])
            .arg_ptr(hidden_src)
            .arg_ptr(m.norm_w)
            .arg_ptr(m.norm_b)
            .arg_u32(p as u32)
            .arg_u32(hidden)
            .arg_f32(1e-6)
            .launch(stream)?;
        // 2026-09-25: 2. Spatial merge: hidden_src[p, hidden] → buf_merge_in[merged_p, merged_in].
        KernelLaunch::new(gpu, self.k_merge)
            .grid([merged_p, 1, 1])
            .block([merged_in.min(1024), 1, 1])
            .arg_ptr(hidden_src)
            .arg_ptr(self.scratch().buf_merge_in)
            .arg_u32(grid_h as u32)
            .arg_u32(grid_w as u32)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(ms)
            .launch(stream)?;
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_merge_in,
            m.fc1_w,
            m.fc1_b,
            self.scratch().buf_merge_fc1,
            merged_p,
            merged_in,
            merged_in,
            stream,
        )?;
        KernelLaunch::new(gpu, self.k_gelu)
            .grid([div_ceil(merged_p * merged_in, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.scratch().buf_merge_fc1)
            .arg_u32(merged_p * merged_in)
            .launch(stream)?;
        self.vit_gemm_bias(
            gpu,
            self.scratch().buf_merge_fc1,
            m.fc2_w,
            m.fc2_b,
            out_slice,
            merged_p,
            out_h_size,
            merged_in,
            stream,
        )
    }
}
