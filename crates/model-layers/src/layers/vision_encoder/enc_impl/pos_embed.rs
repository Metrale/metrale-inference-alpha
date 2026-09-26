// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Positional embedding helpers: bilinear interpolation of the learned
//! pos_embed grid, and the per-patch 2D rotary cos/sin tables.
//!
//! Owner: model-layers (vision).
//! Invariants:
//! - `resample_pos_embed_into` refuses an image of more than `p_max` patches
//!   before it uploads anything.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::VisionEncoder;
use super::f32_to_bf16_bits;

impl VisionEncoder {
    /// 2026-09-25: Resample into the start of `buf_pos_resampled` (single-image path).
    pub(super) fn resample_pos_embed(
        &self,
        grid_h: usize,
        grid_w: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.resample_pos_embed_into(
            grid_h,
            grid_w,
            self.scratch().buf_pos_resampled,
            gpu,
            stream,
        )
    }

    /// 2026-09-25: Bilinearly interpolate the learned pos_embed grid
    /// `[num_grid_per_side × num_grid_per_side, hidden_size]` to
    /// `[grid_h × grid_w, hidden_size]` in row-major order, convert to BF16 and
    /// upload to `dst`. The index and weight formulas are those of
    /// `fast_pos_embed_interpolate` in transformers' `modeling_qwen3_vl.py`.
    /// The batched path passes this image's row slice of `buf_pos_resampled`.
    pub(super) fn resample_pos_embed_into(
        &self,
        grid_h: usize,
        grid_w: usize,
        dst: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size;
        let n = self.num_grid_per_side;
        let p = grid_h * grid_w;

        // 2026-09-25: Capacity guard. This upload runs before the pixel upload that
        // `check_pixel_len` guards, so it is the first write an oversized
        // image reaches.
        anyhow::ensure!(
            p <= self.p_max,
            "vision: image is {grid_h}x{grid_w} patches ({p} total) but this encoder holds \
             {} — raise the area bound (--vision-max-pixels) only if the encoder was built \
             for it, since its buffers are sized from that same bound and the ViT score \
             matrix is O(patches^2)",
            self.p_max
        );

        let mut out_bf16 = vec![0u16; p * h];

        // 2026-09-25: Sample positions as torch.linspace(0, n-1, grid_dim), so the
        // endpoints hit the grid corners exactly.
        let denom_h = (grid_h.max(2) - 1) as f32;
        let denom_w = (grid_w.max(2) - 1) as f32;
        for gh in 0..grid_h {
            let fy = if grid_h <= 1 {
                0.0
            } else {
                gh as f32 * (n - 1) as f32 / denom_h
            };
            let y_f = fy.floor() as i32;
            let y_c = (y_f + 1).min(n as i32 - 1);
            let dy = fy - y_f as f32;
            let y_f_u = y_f.clamp(0, n as i32 - 1) as usize;
            let y_c_u = y_c.clamp(0, n as i32 - 1) as usize;
            for gw in 0..grid_w {
                let fx = if grid_w <= 1 {
                    0.0
                } else {
                    gw as f32 * (n - 1) as f32 / denom_w
                };
                let x_f = fx.floor() as i32;
                let x_c = (x_f + 1).min(n as i32 - 1);
                let dx = fx - x_f as f32;
                let x_f_u = x_f.clamp(0, n as i32 - 1) as usize;
                let x_c_u = x_c.clamp(0, n as i32 - 1) as usize;

                let w00 = (1.0 - dy) * (1.0 - dx);
                let w01 = (1.0 - dy) * dx;
                let w10 = dy * (1.0 - dx);
                let w11 = dy * dx;

                let i00 = (y_f_u * n + x_f_u) * h;
                let i01 = (y_f_u * n + x_c_u) * h;
                let i10 = (y_c_u * n + x_f_u) * h;
                let i11 = (y_c_u * n + x_c_u) * h;
                let out_off = (gh * grid_w + gw) * h;

                for k in 0..h {
                    let v = w00 * self.pos_embed_host_f32[i00 + k]
                        + w01 * self.pos_embed_host_f32[i01 + k]
                        + w10 * self.pos_embed_host_f32[i10 + k]
                        + w11 * self.pos_embed_host_f32[i11 + k];
                    out_bf16[out_off + k] = f32_to_bf16_bits(v);
                }
            }
        }
        // 2026-09-25: SAFETY: `out_bf16` is a live `vec![0u16; p * h]` and the
        // byte length is derived from that same Vec's `len()`, so the view
        // cannot leave the allocation. Every element was zero-initialised by
        // the `vec!` before the interpolation loop wrote it, so no byte is
        // uninitialised; `u16` has no invalid bit patterns and `u8` has
        // alignment 1. The view is read-only and dies before `out_bf16`.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(out_bf16.as_ptr() as *const u8, out_bf16.len() * 2)
        };
        gpu.copy_h2d_async(bytes, dst, stream)
    }

    /// 2026-09-25: Build rope into the start of `buf_rope_cos`/`buf_rope_sin`.
    pub(super) fn build_rope_cossin(
        &self,
        grid_h: usize,
        grid_w: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.build_rope_cossin_into(
            grid_h,
            grid_w,
            self.scratch().buf_rope_cos,
            self.scratch().buf_rope_sin,
            gpu,
            stream,
        )
    }

    /// 2026-09-25: Build per-patch 2D rotary cos/sin in row-major patch order. For
    /// the patch at (row, col) the cos/sin row is
    /// `[row_freq; col_freq; row_freq; col_freq]` of length head_dim, where
    /// `row_freq[k] = cos/sin(row * inv_freq[k])` and
    /// `col_freq[k] = cos/sin(col * inv_freq[k])`. Uploaded as BF16.
    pub(super) fn build_rope_cossin_into(
        &self,
        grid_h: usize,
        grid_w: usize,
        cos_dst: DevicePtr,
        sin_dst: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let p = grid_h * grid_w;
        let hd = self.head_dim;
        let half = hd / 2;
        let inv_n = self.rope_inv_freq.len();
        debug_assert_eq!(inv_n * 2, half);

        let mut cos_bf16 = vec![0u16; p * hd];
        let mut sin_bf16 = vec![0u16; p * hd];

        // 2026-09-25: METRALE_VISION_ROPE=0 uploads cos=1, sin=0, so the rotation
        // is the identity.
        let rope_on = std::env::var("METRALE_VISION_ROPE")
            .map(|v| v != "0")
            .unwrap_or(true);
        let one_bf16 = f32_to_bf16_bits(1.0);
        let zero_bf16 = f32_to_bf16_bits(0.0);
        for gh in 0..grid_h {
            for gw in 0..grid_w {
                let p_idx = gh * grid_w + gw;
                let row = gh as f32;
                let col = gw as f32;
                let off = p_idx * hd;
                for k in 0..inv_n {
                    if !rope_on {
                        for d in [k, inv_n + k, half + k, half + inv_n + k] {
                            cos_bf16[off + d] = one_bf16;
                            sin_bf16[off + d] = zero_bf16;
                        }
                        continue;
                    }
                    let rf = row * self.rope_inv_freq[k];
                    let cf = col * self.rope_inv_freq[k];
                    let (rs, rc) = (rf.sin(), rf.cos());
                    let (cs, cc) = (cf.sin(), cf.cos());
                    // 2026-09-25: First half: `0..inv_n` row freq, `inv_n..2*inv_n` col freq.
                    cos_bf16[off + k] = f32_to_bf16_bits(rc);
                    sin_bf16[off + k] = f32_to_bf16_bits(rs);
                    cos_bf16[off + inv_n + k] = f32_to_bf16_bits(cc);
                    sin_bf16[off + inv_n + k] = f32_to_bf16_bits(cs);
                    // 2026-09-25: The second half duplicates the first.
                    cos_bf16[off + half + k] = f32_to_bf16_bits(rc);
                    sin_bf16[off + half + k] = f32_to_bf16_bits(rs);
                    cos_bf16[off + half + inv_n + k] = f32_to_bf16_bits(cc);
                    sin_bf16[off + half + inv_n + k] = f32_to_bf16_bits(cs);
                }
            }
        }
        // 2026-09-25: SAFETY (both): `cos_bf16`/`sin_bf16` are live `vec![0u16; p * hd]`
        // and each byte length is derived from that same Vec's `len()`, so
        // neither view leaves its allocation. The `vec!` zeroed
        // every element, so no byte is uninitialised even where the rope loop
        // skips one; `u16` has no invalid bit patterns and `u8` has alignment
        // 1. Both views are read-only and die before their Vecs.
        let cos_b: &[u8] = unsafe {
            std::slice::from_raw_parts(cos_bf16.as_ptr() as *const u8, cos_bf16.len() * 2)
        };
        let sin_b: &[u8] = unsafe {
            std::slice::from_raw_parts(sin_bf16.as_ptr() as *const u8, sin_bf16.len() * 2)
        };
        gpu.copy_h2d_async(cos_b, cos_dst, stream)?;
        gpu.copy_h2d_async(sin_b, sin_dst, stream)
    }
}
