// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GLM-5.3-Flash vision tower: ViT blocks with per-head
//! QK-RMSNorm, 2D axial RoPE and clamped SwiGLU, then a 2×2 strided-conv
//! downsample and a SwiGLU merger.
//!
//! A separate type from [`super::VisionEncoder`] (the Qwen3-VL tower), with
//! its own kernel module (`glm_vit`); both use the `gemm` module's
//! `dense_gemm_bf16_pipelined` and `dense_gemm_bf16_f32out`. Callers reach
//! either through [`super::VisionTower`], which gives both the same
//! `forward_batched` result and `out_row` contract.
//!
//! Owner: model-layers (vision).
//! Invariants:
//! - Assumes every weight pointer is a BF16 tensor. `GlmVit::new` does not
//!   check; the loader (`load_glm5_next_vision`) refuses other dtypes.
//! - The scratch is set at most once, for `p_max` patch rows.

use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

/// 2026-09-25: One ViT block's BF16 weights.
pub struct GlmVitBlock {
    pub norm1_w: DevicePtr,
    pub qkv_w: DevicePtr,
    pub qkv_b: DevicePtr,
    /// 2026-09-25: Per-head RMSNorm weight for Q, `[head_dim]`. The kernel applies
    /// it after the QKV split and before the rotary transform.
    pub q_norm_w: DevicePtr,
    pub k_norm_w: DevicePtr,
    pub proj_w: DevicePtr,
    pub proj_b: DevicePtr,
    pub norm2_w: DevicePtr,
    pub gate_w: DevicePtr,
    pub gate_b: DevicePtr,
    pub up_w: DevicePtr,
    pub up_b: DevicePtr,
    pub down_w: DevicePtr,
    pub down_b: DevicePtr,
}

/// 2026-09-25: Everything after the last block: the post-norm, the 2×2 conv
/// downsample and the merger MLP whose output fills `buf_out`.
pub struct GlmVitMerger {
    /// 2026-09-25: Weight-only RMSNorm over `hidden_size`, applied before the downsample.
    pub post_layernorm_w: DevicePtr,
    /// 2026-09-25: The 2×2 stride-2 conv weight read as `[out_hidden, hidden*4]`
    /// in `(in_channel, kh, kw)` order, the order `glm_vit_im2col_2x2` writes.
    pub downsample_w: DevicePtr,
    pub downsample_b: DevicePtr,
    pub proj_w: DevicePtr,
    /// 2026-09-25: `post_projection_norm`, the tower's only LayerNorm (mean
    /// subtracted, with bias); every other norm is weight-only RMSNorm.
    pub norm_w: DevicePtr,
    pub norm_b: DevicePtr,
    pub gate_w: DevicePtr,
    pub up_w: DevicePtr,
    pub down_w: DevicePtr,
}

/// 2026-09-25: Device scratch, allocated by `scratch_init` on first use rather
/// than at load. `buf_scores` and `buf_probs` hold `p_max²` elements.
pub struct GlmVitScratch {
    pub(crate) buf_f32: DevicePtr,
    pub(crate) buf_h1: DevicePtr,
    pub(crate) buf_h2: DevicePtr,
    pub(crate) buf_gate: DevicePtr,
    pub(crate) buf_up: DevicePtr,
    pub(crate) buf_qr: DevicePtr,
    pub(crate) buf_kr: DevicePtr,
    pub(crate) buf_vt: DevicePtr,
    pub(crate) buf_scores: DevicePtr,
    pub(crate) buf_probs: DevicePtr,
    pub(crate) buf_o_stage: DevicePtr,
    pub(crate) buf_rope_cos: DevicePtr,
    pub(crate) buf_rope_sin: DevicePtr,
    pub(crate) buf_merge_a: DevicePtr,
    pub(crate) buf_merge_b: DevicePtr,
    pub(crate) buf_merge_g: DevicePtr,
    pub(crate) buf_merge_u: DevicePtr,
    pub buf_out: DevicePtr,
}

pub struct GlmVit {
    pub(crate) patch_embed_w: DevicePtr,
    pub(crate) patch_embed_b: DevicePtr,
    pub(crate) blocks: Vec<GlmVitBlock>,
    pub(crate) merger: GlmVitMerger,

    pub(crate) k_gemm: KernelHandle,
    pub(crate) k_gemm_f32: KernelHandle,
    pub(crate) k_add_bias: KernelHandle,
    pub(crate) k_rmsnorm: KernelHandle,
    pub(crate) k_layernorm: KernelHandle,
    pub(crate) k_gelu: KernelHandle,
    pub(crate) k_swiglu: KernelHandle,
    pub(crate) k_qknorm_rope: KernelHandle,
    pub(crate) k_softmax: KernelHandle,
    pub(crate) k_scatter_head: KernelHandle,
    pub(crate) k_im2col: KernelHandle,
    pub(crate) k_copy: KernelHandle,
    pub(crate) k_add: KernelHandle,
    pub(crate) k_f32_bf16: KernelHandle,

    pub(crate) hidden_size: usize,
    pub(crate) num_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub(crate) projection_intermediate_size: usize,
    /// 2026-09-25: `3 × temporal_patch_size × patch_size²` floats per patch,
    /// from the config; the Qwen encoder's `PATCH_DIM` is a constant 1536.
    pub(crate) patch_dim: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) swiglu_limit: f32,
    /// 2026-09-25: Patch-row capacity of the scratch: the per-patch buffers hold
    /// `p_max` rows, the merger buffers `mp_max()` merged rows.
    pub(crate) p_max: usize,
    /// 2026-09-25: `inv_freq[k] = theta^(-2k/spatial_dim)` for `k < spatial_dim/2`,
    /// `spatial_dim = head_dim/2`. `rope::build_axial_rope_tables` uses every
    /// entry once for the h axis and once for the w axis.
    pub(crate) rope_inv_freq: Vec<f32>,

    pub(crate) scratch: std::sync::OnceLock<GlmVitScratch>,
}

mod block;
mod forward;
pub mod init;
mod merge;
mod rope;

pub use init::GlmVitGeometry;
pub use rope::{block_major_position_ids, build_axial_rope_tables};
