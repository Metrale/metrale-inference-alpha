// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The post-block tail: 2×2 strided-conv downsample, then the merger MLP.
//!
//! Owner: model-layers (vision).
//! Invariants: none beyond the types.
//!
//! In order: `glm_vit_im2col_2x2` turns each run of four tokens into one
//! `[4 * hidden]` row; the 2×2 stride-2 downsample conv runs as a GEMM with bias
//! to `out_hidden_size`; `merger.proj` is a GEMM without bias; then a LayerNorm
//! with weight and bias (not an RMSNorm), an exact erf GELU, and a SwiGLU MLP
//! through `projection_intermediate_size` whose gate is clamped above at
//! `swiglu_limit` and whose up is clamped to `±swiglu_limit`
//! (`glm_vit_swiglu_clamp`).
//!
//! Grouping four consecutive tokens is one 2×2 block only because the token
//! stream is block-major (`rope::block_major_position_ids`).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::GlmVit;

/// 2026-09-25: Epsilon of the merger's LayerNorm. It is separate from the
/// config's `rms_norm_eps`, which only the RMSNorms read.
const LAYERNORM_EPS: f32 = 1e-5;

impl GlmVit {
    /// 2026-09-25: Consume `merged_p * 4` rows of `buf_h1` and write `merged_p` rows of
    /// `out` (`[merged_p, out_hidden_size]`).
    pub(super) fn downsample_and_merge(
        &self,
        merged_p: usize,
        out: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let s = self.scratch();
        let mp = merged_p as u32;
        let hidden = self.hidden_size as u32;
        let out_h = self.out_hidden_size as u32;
        let pi = self.projection_intermediate_size as u32;
        // 2026-09-25: The conv's K: one 2×2 block of `hidden` channels, in
        // `(in_channel, kh, kw)` order.
        let conv_k = hidden * 4;

        // 2026-09-25: im2col: `buf_h1 [4*mp, hidden]` to `buf_merge_a [mp, 4*hidden]`.
        let n_elems = mp * conv_k;
        KernelLaunch::new(gpu, self.k_im2col)
            .grid([div_ceil(n_elems, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_h1)
            .arg_ptr(s.buf_merge_a)
            .arg_u32(mp)
            .arg_u32(hidden)
            .launch(stream)?;

        // 2026-09-25: The downsample conv as a GEMM against the flattened weight, with bias.
        self.gemm(
            gpu,
            s.buf_merge_a,
            self.merger.downsample_w,
            Some(self.merger.downsample_b),
            s.buf_merge_b,
            mp,
            out_h,
            conv_k,
            stream,
        )?;

        // 2026-09-25: `merger.proj`, without bias, like the gate/up/down GEMMs below.
        self.gemm(
            gpu,
            s.buf_merge_b,
            self.merger.proj_w,
            None,
            s.buf_merge_a,
            mp,
            out_h,
            out_h,
            stream,
        )?;
        self.layernorm(
            gpu,
            s.buf_merge_a,
            self.merger.norm_w,
            self.merger.norm_b,
            mp,
            out_h,
            LAYERNORM_EPS,
            stream,
        )?;
        // 2026-09-25: Exact erf GELU (`glm_vit_gelu_erf`).
        KernelLaunch::new(gpu, self.k_gelu)
            .grid([div_ceil(mp * out_h, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.buf_merge_a)
            .arg_u32(mp * out_h)
            .launch(stream)?;

        self.gemm(
            gpu,
            s.buf_merge_a,
            self.merger.gate_w,
            None,
            s.buf_merge_g,
            mp,
            pi,
            out_h,
            stream,
        )?;
        self.gemm(
            gpu,
            s.buf_merge_a,
            self.merger.up_w,
            None,
            s.buf_merge_u,
            mp,
            pi,
            out_h,
            stream,
        )?;
        self.swiglu(
            gpu,
            s.buf_merge_g,
            s.buf_merge_u,
            s.buf_merge_g,
            mp * pi,
            stream,
        )?;
        self.gemm(
            gpu,
            s.buf_merge_g,
            self.merger.down_w,
            None,
            out,
            mp,
            out_h,
            pi,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    /// 2026-09-25: CPU copy of `glm_vit_im2col_2x2`'s index arithmetic:
    /// `dst[b, c*4 + t] = src[b*4 + t, c]`.
    fn im2col_ref(src: &[f32], merged_p: usize, c_dim: usize) -> Vec<f32> {
        let mut dst = vec![0.0f32; merged_p * 4 * c_dim];
        for b in 0..merged_p {
            for t in 0..4 {
                for c in 0..c_dim {
                    dst[b * 4 * c_dim + c * 4 + t] = src[(b * 4 + t) * c_dim + c];
                }
            }
        }
        dst
    }

    /// 2026-09-25: A 2×2 block's four tokens land contiguously within each
    /// channel's slot, in `(kh, kw)` order, matching the conv weight's
    /// `(in_c, kh, kw)` flattening.
    #[test]
    fn im2col_groups_by_channel_then_kernel_position() {
        let c_dim = 3;
        // 2026-09-25: Token t of block 0 carries the value 10*t + c.
        let src: Vec<f32> = (0..4)
            .flat_map(|t| (0..c_dim).map(move |c| (10 * t + c) as f32))
            .collect();
        let dst = im2col_ref(&src, 1, c_dim);
        assert_eq!(&dst[0..4], &[0.0, 10.0, 20.0, 30.0]);
        assert_eq!(&dst[4..8], &[1.0, 11.0, 21.0, 31.0]);
        assert_eq!(&dst[8..12], &[2.0, 12.0, 22.0, 32.0]);
    }

    /// 2026-09-25: Every input element appears exactly once: with kernel equal
    /// to stride, the im2col is a permutation.
    #[test]
    fn im2col_is_a_permutation() {
        let (mp, c_dim) = (5usize, 7usize);
        let src: Vec<f32> = (0..mp * 4 * c_dim).map(|i| i as f32).collect();
        let mut dst = im2col_ref(&src, mp, c_dim);
        assert_eq!(dst.len(), src.len());
        dst.sort_by(f32::total_cmp);
        assert_eq!(dst, src);
    }
}
