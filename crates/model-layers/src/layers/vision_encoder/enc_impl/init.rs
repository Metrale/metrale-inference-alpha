// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `VisionEncoder::new`, the patch capacity, and the lazily allocated scratch.
//!
//! Owner: model-layers (vision).
//! Invariants:
//! - `derive_max_patches` never returns 0 and never more than `CEILING_MAX_PATCHES`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::{MergerLayer, PATCH_DIM, ViTBlock, VisionEncoder, VisionScratch};

/// 2026-09-25: Encoder capacity, in patches, when no area bound is given.
/// 6400 = 80×80, i.e. 1280×1280 at patch 16.
pub const FALLBACK_MAX_PATCHES: usize = 6400;

/// 2026-09-25: Ceiling on the derived capacity, in patches. 16384 = 128×128, i.e.
/// 2048×2048 at patch 16.
///
/// The ViT attention scratch holds a full `[p_max, p_max]` f32 score matrix
/// and a BF16 probability matrix (`build_scratch`), so the allocation grows
/// with the square of the capacity. Qwen3.8-27B's preprocessor config declares
/// `size.longest_edge = 16777216` (4096²), which would be 65536 patches.
pub const CEILING_MAX_PATCHES: usize = 16384;

/// 2026-09-25: Patches the encoder must hold to serve an area bound, clamped to
/// `CEILING_MAX_PATCHES`.
///
/// `max_pixels` is an area, so patches = area / patch² (at least 1; a
/// `patch_size` of 0 counts as 1). `None` or 0 gives `FALLBACK_MAX_PATCHES`.
/// The second value is `Some(wanted)` when the ceiling clamped the result, so
/// the caller can log it.
pub fn derive_max_patches(max_pixels: Option<usize>, patch_size: usize) -> (usize, Option<usize>) {
    let Some(area) = max_pixels.filter(|&a| a > 0) else {
        return (FALLBACK_MAX_PATCHES, None);
    };
    let per_patch = patch_size.max(1) * patch_size.max(1);
    let wanted = (area / per_patch).max(1);
    if wanted > CEILING_MAX_PATCHES {
        (CEILING_MAX_PATCHES, Some(wanted))
    } else {
        (wanted.max(FALLBACK_MAX_PATCHES.min(wanted)), None)
    }
}

impl VisionEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        patch_embed_w: DevicePtr,
        patch_embed_b: DevicePtr,
        pos_embed: DevicePtr,
        num_position_embeddings: usize,
        blocks: Vec<ViTBlock>,
        deepstack: Vec<MergerLayer>,
        deepstack_indexes: Vec<usize>,
        merger: MergerLayer,
        hidden_size: usize,
        num_heads: usize,
        spatial_merge_size: usize,
        out_hidden_size: usize,
        intermediate_size: usize,
        patch_size: usize,
        max_pixels: Option<usize>,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let (p_max, asked_for) = derive_max_patches(max_pixels, patch_size);
        match asked_for {
            Some(wanted) => tracing::warn!(
                "Vision encoder capacity {p_max} patches ({}x{} px) — the resolved area bound \
                 wanted {wanted} patches, clamped: the ViT score matrix is O(patches^2) and \
                 {wanted} would need ~{:.1} GB of scratch. Lower --vision-max-pixels to reclaim \
                 memory, or raise CEILING_MAX_PATCHES only alongside tiled ViT attention.",
                (p_max as f64).sqrt() as usize * patch_size,
                (p_max as f64).sqrt() as usize * patch_size,
                ((wanted * wanted * 6) as f64) / (1024.0 * 1024.0 * 1024.0),
            ),
            None => tracing::info!(
                "Vision encoder capacity {p_max} patches ({}x{} px){}",
                (p_max as f64).sqrt() as usize * patch_size,
                (p_max as f64).sqrt() as usize * patch_size,
                if max_pixels.is_some() {
                    " from the resolved area bound"
                } else {
                    " (no bound declared — historical fallback)"
                }
            ),
        }

        // 2026-09-25: The side of the square pos_embed grid; a non-square count is
        // refused.
        let num_grid_per_side = (num_position_embeddings as f64).sqrt().round() as usize;
        anyhow::ensure!(
            num_grid_per_side * num_grid_per_side == num_position_embeddings,
            "non-square pos_embed: {num_position_embeddings} is not a perfect square"
        );

        // 2026-09-25: Keep a host f32 copy of pos_embed for the per-image bilinear
        // interpolation in `resample_pos_embed_into`.
        let pos_n = num_position_embeddings * hidden_size;
        let mut pe_bytes = vec![0u8; pos_n * 2];
        gpu.copy_d2h(pos_embed, &mut pe_bytes)?;
        let pos_embed_host_f32: Vec<f32> = pe_bytes
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect();

        // 2026-09-25: Vision RoPE inverse frequencies: `rope_dim = head_dim / 2`,
        // `inv_freq[k] = theta^(-2k/rope_dim)` for k in [0, rope_dim/2), with
        // theta fixed at 10000.
        let rope_dim = head_dim / 2;
        let rope_half = rope_dim / 2;
        let theta: f32 = 10_000.0;
        let rope_inv_freq: Vec<f32> = (0..rope_half)
            .map(|k| 1.0 / theta.powf(2.0 * k as f32 / rope_dim as f32))
            .collect();

        Ok(Self {
            patch_embed_w,
            patch_embed_b,
            pos_embed,
            blocks,
            deepstack,
            deepstack_indexes,
            merger,
            k_gemm: gpu.kernel("vision_encoder", "vision_gemm_bias")?,
            // 2026-09-25: Tensor-core pipelined matmul + a row-broadcast bias add.
            // Optional: when either is null, `vit_gemm_bias` uses `vision_gemm_bias`.
            k_gemm_pipelined: crate::layers::try_kernel(gpu, "gemm", "dense_gemm_bf16_pipelined"),
            k_add_bias: crate::layers::try_kernel(gpu, "vision_encoder", "vision_add_bias"),
            k_norm: gpu.kernel("vision_encoder", "vision_layer_norm")?,
            k_add: gpu.kernel("vision_encoder", "vision_add_inplace")?,
            k_gelu: gpu.kernel("vision_encoder", "vision_gelu")?,
            // 2026-09-25: Warp-per-query ViT attention, required: both
            // `vision_encoder.cu` sources (qwen3-vl-30b-a3b, qwen3.6-35b-a3b)
            // define it, and it is the fallback for the GEMM-based path.
            k_attn: gpu.kernel("vision_encoder", "vision_attention_rope")?,
            // 2026-09-25: GEMM-based ViT attention kernels, optional: only
            // `qwen3.6-35b-a3b/nvfp4/vision_encoder.cu` defines them. Trees built
            // from the qwen3-vl-30b-a3b source leave them null, and `vit_block`
            // then uses `k_attn`.
            k_rope_deint: crate::layers::try_kernel(gpu, "vision_encoder", "vit_rope_deinterleave"),
            k_softmax: crate::layers::try_kernel(gpu, "vision_encoder", "vit_softmax_rows"),
            k_scatter_head: crate::layers::try_kernel(gpu, "vision_encoder", "vit_scatter_head"),
            // 2026-09-25: f32-out dense GEMM for the raw QKᵀ scores of the GEMM-based
            // attention; optional, like the kernels above.
            k_gemm_f32: crate::layers::try_kernel(gpu, "gemm", "dense_gemm_bf16_f32out"),
            k_merge: gpu.kernel("vision_encoder", "vision_spatial_merge")?,
            k_f32_bf16: gpu.kernel("vision_encoder", "vision_f32_to_bf16")?,
            k_copy: gpu.kernel("vision_encoder", "vision_bf16_copy")?,
            hidden_size,
            num_heads,
            head_dim,
            spatial_merge_size,
            out_hidden_size,
            intermediate_size,
            p_max,
            num_grid_per_side,
            scratch: std::sync::OnceLock::new(),
            pos_embed_host_f32,
            rope_inv_freq,
        })
    }
}

impl VisionEncoder {
    /// 2026-09-25: Allocate the ViT scratch group. Only `scratch_init` calls it.
    fn build_scratch(&self, gpu: &dyn GpuBackend) -> Result<VisionScratch> {
        let p_max = self.p_max;
        let hidden_size = self.hidden_size;
        let intermediate_size = self.intermediate_size;
        let out_hidden_size = self.out_hidden_size;
        let merger_in_dim = self.spatial_merge_size * self.spatial_merge_size * hidden_size;
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let buf_f32 = gpu.alloc(p_max * PATCH_DIM * 4)?;
        let buf_h1 = gpu.alloc(p_max * hidden_size * 2)?;
        let buf_h2 = gpu.alloc(p_max * hidden_size * 2)?;
        let buf_wide = gpu.alloc(p_max * intermediate_size * 2)?;
        let buf_merge_in = gpu.alloc((p_max / 4) * merger_in_dim * 2)?;
        let buf_merge_fc1 = gpu.alloc((p_max / 4) * merger_in_dim * 2)?;
        let buf_out = gpu.alloc(p_max * out_hidden_size * 2)?;
        let buf_pos_resampled = gpu.alloc(p_max * hidden_size * 2)?;
        let buf_rope_cos = gpu.alloc(p_max * head_dim * 2)?;
        let buf_rope_sin = gpu.alloc(p_max * head_dim * 2)?;

        // 2026-09-25: GEMM-based ViT attention scratch, sized to p_max so any image
        // the encoder admits fits: head-contiguous Qr/Kr `[H, p_max, D]` and Vt
        // `[H, D, p_max]` BF16, the `[p_max, p_max]` f32 scores and BF16 probs
        // (reused by every head), and a `[p_max, D]` BF16 per-head output stage.
        let attn_max = p_max;
        let qkv_head_elems = p_max * num_heads * head_dim;
        let buf_qr = gpu.alloc(qkv_head_elems * 2)?;
        let buf_kr = gpu.alloc(qkv_head_elems * 2)?;
        let buf_vt = gpu.alloc(qkv_head_elems * 2)?;
        let buf_scores = gpu.alloc(attn_max * attn_max * 4)?;
        let buf_probs = gpu.alloc(attn_max * attn_max * 2)?;
        let buf_o_stage = gpu.alloc(p_max * head_dim * 2)?;
        Ok(VisionScratch {
            buf_f32,
            buf_h1,
            buf_h2,
            buf_wide,
            buf_merge_in,
            buf_merge_fc1,
            buf_out,
            buf_pos_resampled,
            buf_rope_cos,
            buf_rope_sin,
            buf_qr,
            buf_kr,
            buf_vt,
            buf_scores,
            buf_probs,
            buf_o_stage,
        })
    }

    /// 2026-09-25: Allocate the ViT scratch if it is not set yet.
    ///
    /// Racing callers may each build a group; `OnceLock::set` keeps one.
    pub(crate) fn scratch_init(&self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.scratch.get().is_none() {
            let s = self.build_scratch(gpu)?;
            // 2026-09-25: If a racing caller set it first, this group's buffers are
            // not freed here.
            let _ = self.scratch.set(s);
            tracing::info!(
                "Vision scratch allocated on first image: {} patches",
                self.p_max
            );
        }
        Ok(())
    }

    /// 2026-09-25: Scratch accessor for the encode path. Panics if `scratch_init` has
    /// not run: that is a wiring bug, and this accessor cannot return an error.
    pub(crate) fn scratch(&self) -> &VisionScratch {
        self.scratch
            .get()
            .expect("vision scratch: encode entry must call scratch_init(gpu) first")
    }
}

#[cfg(test)]
mod derive_tests {
    use super::*;

    /// 2026-09-25: Qwen3.8-27B's `size.longest_edge` (4096²), at patch 16.
    const Q38_BOUND: usize = 16_777_216;

    #[test]
    fn no_bound_keeps_the_historical_capacity() {
        // 2026-09-25: No bound, or a zero bound, gives the fallback capacity.
        assert_eq!(derive_max_patches(None, 16), (FALLBACK_MAX_PATCHES, None));
        assert_eq!(
            derive_max_patches(Some(0), 16),
            (FALLBACK_MAX_PATCHES, None)
        );
    }

    #[test]
    fn a_declared_bound_over_the_ceiling_is_clamped_and_reported() {
        // 2026-09-25: Qwen3.8-27B's bound asks for 65536 patches; it gets the
        // ceiling, and the amount it asked for comes back.
        let (got, asked) = derive_max_patches(Some(Q38_BOUND), 16);
        assert_eq!(got, CEILING_MAX_PATCHES);
        assert_eq!(
            asked,
            Some(65_536),
            "the caller must be able to report the ask"
        );
    }

    #[test]
    fn a_low_operator_bound_shrinks_the_allocation() {
        // 2026-09-25: A bound below the fallback allocates less than the fallback.
        let (got, asked) = derive_max_patches(Some(512 * 512), 16);
        assert_eq!(asked, None, "under the ceiling, nothing was clamped");
        assert_eq!(got, 1024, "512x512 at patch 16 is 32x32 = 1024 patches");
        assert!(
            got < FALLBACK_MAX_PATCHES,
            "a low bound must allocate LESS than the historical default"
        );
    }

    #[test]
    fn capacity_tracks_patch_size() {
        // 2026-09-25: patches = area / patch^2, so a finer grid needs more rows for
        // the same pixel area.
        let (at16, _) = derive_max_patches(Some(1024 * 1024), 16);
        let (at14, _) = derive_max_patches(Some(1024 * 1024), 14);
        assert!(
            at14 > at16,
            "finer patches need more rows: {at14} vs {at16}"
        );
    }

    #[test]
    fn the_ceiling_matches_the_measured_affordable_rung() {
        // 2026-09-25: Pins the ceiling at 16384 patches, a 2048x2048 image at patch 16.
        assert_eq!(CEILING_MAX_PATCHES, 16_384);
        let side = (CEILING_MAX_PATCHES as f64).sqrt() as usize * 16;
        assert_eq!(side, 2048, "the ceiling should be a clean square image");
    }

    #[test]
    fn degenerate_inputs_do_not_produce_a_zero_allocation() {
        // 2026-09-25: The capacity is never 0, even for a 1-pixel area or patch size 0.
        assert_eq!(derive_max_patches(Some(1), 16), (1, None));
        assert_eq!(derive_max_patches(Some(1024), 0), (1024, None));
    }
}
