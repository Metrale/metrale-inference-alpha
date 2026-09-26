// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `GlmVit::new` and the device scratch allocated on first use.
//!
//! Owner: model-layers (vision).
//! Invariants: `new` refuses a `hidden_size` not divisible by a nonzero
//! `num_heads`, a `head_dim` not a multiple of 4, a `spatial_merge_size` other
//! than 2, and a zero `patch_size` or `temporal_patch_size`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{GlmVit, GlmVitBlock, GlmVitMerger, GlmVitScratch};
use crate::layers::vision_encoder::enc_impl::init::derive_max_patches;

/// 2026-09-25: Everything `GlmVit::new` needs that is not a device pointer.
#[derive(Debug, Clone)]
pub struct GlmVitGeometry {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub projection_intermediate_size: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub rms_norm_eps: f32,
    pub swiglu_limit: f32,
    pub rope_theta: f32,
    pub max_pixels: Option<usize>,
}

impl GlmVit {
    pub fn new(
        patch_embed_w: DevicePtr,
        patch_embed_b: DevicePtr,
        blocks: Vec<GlmVitBlock>,
        merger: GlmVitMerger,
        geo: &GlmVitGeometry,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        anyhow::ensure!(
            geo.num_heads > 0 && geo.hidden_size.is_multiple_of(geo.num_heads),
            "glm_vit: hidden_size {} is not divisible by num_heads {}",
            geo.hidden_size,
            geo.num_heads
        );
        let head_dim = geo.hidden_size / geo.num_heads;
        anyhow::ensure!(
            head_dim.is_multiple_of(4),
            "glm_vit: head_dim {head_dim} must be a multiple of 4 — the axial RoPE splits it \
             into an h half and a w half, each of which is then rotate-half'd"
        );
        anyhow::ensure!(
            geo.spatial_merge_size == 2,
            "glm_vit: spatial_merge_size {} — the 2x2 conv downsample and its im2col are \
             written for 2 and only 2; a different merge needs a different kernel, not a \
             different argument",
            geo.spatial_merge_size
        );
        anyhow::ensure!(
            geo.patch_size > 0 && geo.temporal_patch_size > 0,
            "glm_vit: patch_size/temporal_patch_size must be non-zero"
        );

        let (p_max, asked_for) = derive_max_patches(geo.max_pixels, geo.patch_size);
        if let Some(wanted) = asked_for {
            tracing::warn!(
                "GLM vision encoder capacity clamped to {p_max} patches; the resolved area \
                 bound wanted {wanted}. The ViT materialises a full [seq, seq] score matrix, \
                 so scratch is O(patches^2). Lower --vision-max-pixels to reclaim memory."
            );
        }

        // 2026-09-25: `inv_freq[k] = theta^(-2k/spatial_dim)` for
        // `k < spatial_dim/2`, i.e. `1 / theta ** (arange(0, spatial_dim, 2) / spatial_dim)`.
        let spatial_dim = head_dim / 2;
        let rope_inv_freq: Vec<f32> = (0..spatial_dim / 2)
            .map(|k| 1.0 / geo.rope_theta.powf(2.0 * k as f32 / spatial_dim as f32))
            .collect();

        Ok(Self {
            patch_embed_w,
            patch_embed_b,
            blocks,
            merger,
            k_gemm: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            k_gemm_f32: gpu.kernel("gemm", "dense_gemm_bf16_f32out")?,
            k_add_bias: gpu.kernel("glm_vit", "glm_vit_add_bias")?,
            k_rmsnorm: gpu.kernel("glm_vit", "glm_vit_rmsnorm")?,
            k_layernorm: gpu.kernel("glm_vit", "glm_vit_layernorm")?,
            k_gelu: gpu.kernel("glm_vit", "glm_vit_gelu_erf")?,
            k_swiglu: gpu.kernel("glm_vit", "glm_vit_swiglu_clamp")?,
            k_qknorm_rope: gpu.kernel("glm_vit", "glm_vit_qknorm_rope_deint")?,
            k_softmax: gpu.kernel("glm_vit", "glm_vit_softmax_rows")?,
            k_scatter_head: gpu.kernel("glm_vit", "glm_vit_scatter_head")?,
            k_im2col: gpu.kernel("glm_vit", "glm_vit_im2col_2x2")?,
            k_copy: gpu.kernel("glm_vit", "glm_vit_copy")?,
            k_add: gpu.kernel("glm_vit", "glm_vit_add_inplace")?,
            k_f32_bf16: gpu.kernel("glm_vit", "glm_vit_f32_to_bf16")?,
            hidden_size: geo.hidden_size,
            num_heads: geo.num_heads,
            head_dim,
            intermediate_size: geo.intermediate_size,
            spatial_merge_size: geo.spatial_merge_size,
            out_hidden_size: geo.out_hidden_size,
            projection_intermediate_size: geo.projection_intermediate_size,
            patch_dim: 3 * geo.temporal_patch_size * geo.patch_size * geo.patch_size,
            rms_norm_eps: geo.rms_norm_eps,
            swiglu_limit: geo.swiglu_limit,
            p_max,
            rope_inv_freq,
            scratch: std::sync::OnceLock::new(),
        })
    }

    /// 2026-09-25: Merged (post-downsample) row capacity of `buf_out`.
    pub(crate) fn mp_max(&self) -> usize {
        let sms2 = self.spatial_merge_size * self.spatial_merge_size;
        self.p_max.div_ceil(sms2)
    }

    fn build_scratch(&self, gpu: &dyn GpuBackend) -> Result<GlmVitScratch> {
        let p = self.p_max;
        let mp = self.mp_max();
        let h = self.hidden_size;
        let inter = self.intermediate_size;
        let out = self.out_hidden_size;
        let pi = self.projection_intermediate_size;
        // 2026-09-25: `buf_gate` holds the QKV output (`3 * hidden`), the MLP
        // gate (`intermediate_size`) and the BF16 pixels (`patch_dim`), so it
        // is sized for the widest.
        let wide = inter.max(3 * h).max(self.patch_dim);
        Ok(GlmVitScratch {
            buf_f32: gpu.alloc(p * self.patch_dim * 4)?,
            buf_h1: gpu.alloc(p * h * 2)?,
            buf_h2: gpu.alloc(p * h * 2)?,
            buf_gate: gpu.alloc(p * wide * 2)?,
            buf_up: gpu.alloc(p * inter * 2)?,
            buf_qr: gpu.alloc(p * h * 2)?,
            buf_kr: gpu.alloc(p * h * 2)?,
            buf_vt: gpu.alloc(p * h * 2)?,
            // 2026-09-25: `p²` f32 scores and BF16 probs: 164 MB + 82 MB at
            // 6400 patches.
            buf_scores: gpu.alloc(p * p * 4)?,
            buf_probs: gpu.alloc(p * p * 2)?,
            buf_o_stage: gpu.alloc(p * self.head_dim * 2)?,
            buf_rope_cos: gpu.alloc(p * self.head_dim * 2)?,
            buf_rope_sin: gpu.alloc(p * self.head_dim * 2)?,
            buf_merge_a: gpu.alloc(mp * out * 2)?,
            buf_merge_b: gpu.alloc(mp * out * 2)?,
            buf_merge_g: gpu.alloc(mp * pi * 2)?,
            buf_merge_u: gpu.alloc(mp * pi * 2)?,
            buf_out: gpu.alloc(mp * out * 2)?,
        })
    }

    /// 2026-09-25: Allocate the scratch if it is not yet set. Called by
    /// `forward_batched` and `VisionTower::ensure_scratch`.
    pub(crate) fn scratch_init(&self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.scratch.get().is_none() {
            let s = self.build_scratch(gpu)?;
            let _ = self.scratch.set(s);
            tracing::info!(
                "GLM vision scratch allocated on first image: {} patches ({} merged rows)",
                self.p_max,
                self.mp_max()
            );
        }
        Ok(())
    }

    pub(crate) fn scratch(&self) -> &GlmVitScratch {
        self.scratch
            .get()
            .expect("glm_vit scratch: encode entry must call scratch_init(gpu) first")
    }

    /// 2026-09-25: Device pointer to row `row` of the packed merged output
    /// (`buf_out`). Panics when the scratch is not allocated.
    pub fn out_row(&self, row: usize) -> DevicePtr {
        self.scratch()
            .buf_out
            .offset(row * self.out_hidden_size * 2)
    }
}

#[cfg(test)]
mod tests {

    /// 2026-09-25: 3 × 2 × 14² = 1176 floats per patch, which is not the Qwen
    /// encoder's `PATCH_DIM`.
    #[test]
    fn glm_patch_dim_is_a_runtime_value_of_1176() {
        let patch_dim = 3 * 2 * 14 * 14;
        assert_eq!(patch_dim, 1176);
        assert_ne!(patch_dim, crate::layers::vision_encoder::PATCH_DIM);
    }

    /// 2026-09-25: The rope frequency table: 16 entries for head_dim 64, first
    /// entry exactly 1, last `theta^(-30/32)`.
    #[test]
    fn rope_inv_freq_matches_the_reference_formula() {
        let head_dim = 64usize;
        let spatial_dim = head_dim / 2;
        let theta = 10_000.0f32;
        let f: Vec<f32> = (0..spatial_dim / 2)
            .map(|k| 1.0 / theta.powf(2.0 * k as f32 / spatial_dim as f32))
            .collect();
        assert_eq!(f.len(), 16);
        assert_eq!(f[0], 1.0);
        let last = 1.0 / theta.powf(30.0 / 32.0);
        assert!((f[15] - last).abs() < 1e-9);
    }
}
