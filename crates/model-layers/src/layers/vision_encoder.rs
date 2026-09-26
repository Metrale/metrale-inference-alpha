// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The Qwen3-VL-shaped vision encoder: a ViT with optional DeepStack mergers.
//!
//! Owner: model-layers (vision).
//! Invariants:
//! - At most one scratch group is stored (`scratch` is a `OnceLock`); it is
//!   set on the first image or by `VisionTower::ensure_scratch`.
//!
//! Pixels go through patch embedding, the ViT blocks and a final spatial
//! merger. The first `Σmerged_p` rows of `buf_out` hold the final merger's
//! output, `[Σmerged_p, out_hidden_size]` BF16, which the LLM embeds. Each
//! DeepStack merger listed in `deepstack_indexes` writes its own region after
//! that in the batched path; the LLM does not read those regions.

use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

pub(super) const IMAGE_PAD_TOKEN: u32 = 151_655;
pub const IMAGE_PAD_TOKEN_ID: u32 = IMAGE_PAD_TOKEN;

/// 2026-09-25: Fallback `<|video_pad|>` id, used when the checkpoint's config declares
/// none (or 0). The Qwen3-VL, Qwen3.6 and Qwen3.8 checkpoints all put the
/// video token directly after the image token. A declared id always wins, as
/// for the image token.
pub const VIDEO_PAD_TOKEN_ID: u32 = IMAGE_PAD_TOKEN + 1;

/// 2026-09-25: The ViT's per-image scratch buffers, allocated as one group.
///
/// Sizes derive only from the encoder's geometry (`p_max` and the head/hidden
/// dims), not from any image, so the group is built once, lazily, and reused
/// for every image.
pub struct VisionScratch {
    pub buf_f32: DevicePtr,
    pub buf_h1: DevicePtr,
    pub buf_h2: DevicePtr,
    pub buf_wide: DevicePtr,
    pub buf_merge_in: DevicePtr,
    pub buf_merge_fc1: DevicePtr,
    pub buf_out: DevicePtr,
    pub buf_pos_resampled: DevicePtr,
    pub buf_rope_cos: DevicePtr,
    pub buf_rope_sin: DevicePtr,
    pub buf_qr: DevicePtr,
    pub buf_kr: DevicePtr,
    pub buf_vt: DevicePtr,
    pub buf_scores: DevicePtr,
    pub buf_probs: DevicePtr,
    pub buf_o_stage: DevicePtr,
}

/// 2026-09-25: Flattened per-patch pixel dimension `C × temporal_patch_size × patch_size²`
/// = 3 × 2 × 16 × 16 for this ViT. It is fixed in the encoder, not read from
/// config: `buf_f32` is allocated at `p_max × PATCH_DIM × 4` and the
/// patch-embed GEMM is issued with `K = PATCH_DIM`.
///
/// The host preprocessor (`vision_preprocess::preprocess_image`) sizes the
/// pixel buffer from `vision_config`, so a checkpoint with a different
/// `patch_size`/`temporal_patch_size` produces a buffer of a different
/// length. `patch_embed`'s `check_pixel_len` refuses such a buffer.
pub(crate) const PATCH_DIM: usize = 1536;

pub struct ViTBlock {
    pub norm1_w: DevicePtr,
    pub norm1_b: DevicePtr,
    pub qkv_w: DevicePtr,
    pub qkv_b: DevicePtr,
    pub proj_w: DevicePtr,
    pub proj_b: DevicePtr,
    pub norm2_w: DevicePtr,
    pub norm2_b: DevicePtr,
    pub fc1_w: DevicePtr,
    pub fc1_b: DevicePtr,
    pub fc2_w: DevicePtr,
    pub fc2_b: DevicePtr,
}

pub struct MergerLayer {
    pub norm_w: DevicePtr,
    pub norm_b: DevicePtr,
    pub fc1_w: DevicePtr,
    pub fc1_b: DevicePtr,
    pub fc2_w: DevicePtr,
    pub fc2_b: DevicePtr,
}

pub struct VisionEncoder {
    /// 2026-09-25: `[hidden_size, PATCH_DIM]` BF16.
    pub patch_embed_w: DevicePtr,
    /// 2026-09-25: `[hidden_size]` BF16.
    pub patch_embed_b: DevicePtr,
    /// 2026-09-25: `[num_position_embeddings, hidden_size]` BF16. The forward path
    /// reads it directly only when `METRALE_VISION_POSINTERP=0`; otherwise
    /// it uses the host copy `pos_embed_host_f32`.
    pub pos_embed: DevicePtr,
    pub blocks: Vec<ViTBlock>,
    pub deepstack: Vec<MergerLayer>,
    /// 2026-09-25: `deepstack[i]` runs after the block whose 1-based number is
    /// `deepstack_indexes[i]` (`forward_batched` compares `block_idx + 1`).
    pub deepstack_indexes: Vec<usize>,
    /// 2026-09-25: The final merger, after the last block.
    pub merger: MergerLayer,
    k_gemm: KernelHandle,
    // 2026-09-25: `k_gemm_pipelined`, `k_add_bias`, `k_rope_deint`,
    // `k_softmax`, `k_scatter_head` and `k_gemm_f32` are bound with
    // `try_kernel` and may be null. `vit_gemm_bias` uses `k_gemm` unless both
    // `k_gemm_pipelined` and `k_add_bias` are set; the ViT attention uses
    // `k_attn` when `k_rope_deint` is null.
    k_gemm_pipelined: KernelHandle,
    k_add_bias: KernelHandle,
    k_norm: KernelHandle,
    k_add: KernelHandle,
    k_gelu: KernelHandle,
    k_attn: KernelHandle,
    k_rope_deint: KernelHandle,
    k_softmax: KernelHandle,
    k_scatter_head: KernelHandle,
    k_gemm_f32: KernelHandle,
    k_merge: KernelHandle,
    k_f32_bf16: KernelHandle,
    k_copy: KernelHandle,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub intermediate_size: usize,
    /// 2026-09-25: Patch-row capacity of every scratch buffer (`derive_max_patches`).
    pub p_max: usize,
    /// 2026-09-25: `sqrt(num_position_embeddings)`; `new` refuses a non-square count.
    pub num_grid_per_side: usize,
    /// 2026-09-25: ViT scratch, allocated on the first image rather than at load, so
    /// a serve that sees no image never allocates it.
    ///
    /// A `OnceLock`, because the forward path takes `&self`; if two callers
    /// race, one group is kept.
    scratch: std::sync::OnceLock<VisionScratch>,
    /// 2026-09-25: `[num_position_embeddings × hidden_size]` row-major.
    pos_embed_host_f32: Vec<f32>,
    /// 2026-09-25: `[head_dim / 4]` RoPE inverse frequencies.
    rope_inv_freq: Vec<f32>,
}

pub(crate) mod enc_impl;
