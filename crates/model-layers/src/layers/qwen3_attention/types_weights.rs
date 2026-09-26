// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Weight structs for the attention layer's optional parts (MLA, the DeepSeek-V4 compressor, hyper-connections) and the FP8 prefill-twin selection.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::DevicePtr;

use crate::weight_map::{DenseWeight, QuantizedWeight};

/// 2026-09-25: MLA (multi-head latent attention) weights. Q is projected in two
/// steps: `input × wq_a → latent`, the `q_a_norm` RMS norm, then
/// `latent × wq_b → Q`.
pub struct MlaWeights {
    pub wq_a: DenseWeight,
    pub wq_a_nvfp4: Option<QuantizedWeight>,
    /// 2026-09-25: Native block-scaled FP8 copy (E4M3 with 128x128 block
    /// scales), read by the decode `w8a16_gemv`.
    pub wq_a_fp8: Option<crate::weight_map::Fp8Weight>,
    pub wq_b: DenseWeight,
    pub wq_b_nvfp4: Option<QuantizedWeight>,
    pub wq_b_fp8: Option<crate::weight_map::Fp8Weight>,
    pub q_a_norm: DenseWeight,
    pub wkv_a: DenseWeight,
    pub wkv_a_nvfp4: Option<QuantizedWeight>,
    pub wkv_a_fp8: Option<crate::weight_map::Fp8Weight>,
    pub wkv_b: DenseWeight,
    pub kv_a_norm: DenseWeight,
    pub wkv_a_rope: DenseWeight,
    pub wkv_a_merged: DenseWeight,
    pub wo: DenseWeight,
    pub wo_nvfp4: Option<QuantizedWeight>,
    /// 2026-09-25: Grouped low-rank O projection (`wo_a` then `wo_b`), used by
    /// the decode and prefill paths when `o_lora_rank > 0` instead of `wo`.
    pub wo_a: DenseWeight,
    pub wo_a_nvfp4: Option<QuantizedWeight>,
    /// 2026-09-25: Native block-scaled FP8 `wo_a` for the grouped decode
    /// O-projection.
    pub wo_a_fp8: Option<crate::weight_map::Fp8Weight>,
    pub wo_b: DenseWeight,
    pub wo_b_nvfp4: Option<QuantizedWeight>,
    pub wo_b_fp8: Option<crate::weight_map::Fp8Weight>,
    pub w_uk_t: DenseWeight,
    pub w_uv: DenseWeight,
    pub wq_b_rope: DenseWeight,
    pub w_qk_absorbed: DenseWeight,
    pub w_uk_block_diag: DenseWeight,
    pub w_uv_block_diag: DenseWeight,
    pub yarn_inv_freq: metrale_gpu_runtime::gpu::DevicePtr,
    /// 2026-09-25: Plain inverse frequencies without YaRN, for the Q/K RoPE on
    /// layers without a compressor; layers with one (CSA/HCA) use
    /// `yarn_inv_freq`.
    pub main_inv_freq: metrale_gpu_runtime::gpu::DevicePtr,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub o_lora_rank: usize,
    pub nope: usize,
    pub rope: usize,
    pub v_dim: usize,
    /// 2026-09-25: DeepSeek-V4 compressed-attention compressor; `None` when the
    /// layer's `compress_ratios` entry is 0 (full attention).
    pub compressor: Option<CompressorWeights>,
    /// 2026-09-25: Per-head attention sink logits, FP32 (the loader converts a
    /// BF16 checkpoint tensor); NULL when the checkpoint has none for this layer.
    pub attn_sink: metrale_gpu_runtime::gpu::DevicePtr,
}

/// 2026-09-25: DeepSeek-V4 compressed-attention compressor weights, one per layer
/// with a non-zero `compress_ratios` entry. CSA layers use a 2×ratio
/// overlapping window, HCA layers a single window.
#[derive(Debug, Clone, Copy)]
pub struct CompressorWeights {
    /// 2026-09-25: `wkv`: `[proj_dim, hidden]`.
    pub wkv: DenseWeight,
    /// 2026-09-25: `wgate`: same shape as `wkv`.
    pub wgate: DenseWeight,
    pub norm: DenseWeight,
    /// 2026-09-25: `ape`, `[ratio, proj_dim]` FP32.
    pub ape: metrale_gpu_runtime::gpu::DevicePtr,
    /// 2026-09-25: Compression ratio for this layer; `is_csa` is `ratio < 128`.
    pub ratio: usize,
    /// 2026-09-25: Output width of `wkv`/`wgate`: `2 * head_dim` for CSA,
    /// `head_dim` for HCA.
    pub proj_dim: usize,
    /// 2026-09-25: CSA (2×ratio overlapping window) when true, HCA (single
    /// window) when false.
    pub is_csa: bool,
    /// 2026-09-25: Persistent flat compressed-KV pool, `[pool_blocks × hd_mla]`
    /// FP8-E4M3 with `hd_mla = qk_nope_head_dim + qk_rope_head_dim`. Prefill
    /// fills blocks `[0, n / ratio)`; decode appends after them.
    pub pool: metrale_gpu_runtime::gpu::DevicePtr,
    /// 2026-09-25: Pool capacity in blocks:
    /// `max_position_embeddings.div_ceil(ratio)`.
    pub pool_blocks: usize,
    /// 2026-09-25: Decode-time ring of compressor inputs, `[ratio × hidden]` BF16.
    /// Each decode token's `normed` is written to slot `pos % ratio`.
    pub ring: metrale_gpu_runtime::gpu::DevicePtr,
    /// 2026-09-25: CSA only: the previous completed window's inputs,
    /// `[ratio × hidden]` BF16. NULL for HCA.
    pub prev_win: metrale_gpu_runtime::gpu::DevicePtr,
    /// 2026-09-25: CSA only: `[2×ratio × hidden]` BF16 staging for the
    /// overlapped window. NULL for HCA.
    pub stage: metrale_gpu_runtime::gpu::DevicePtr,
}

/// 2026-09-25: The low-rank hyper-connection parameters for one site (qwen4_exp).
///
/// Used instead of the Sinkhorn `hc_fn`/`hc_base`/`hc_scale` fields of
/// `HcSiteWeights`, which are NULL when this is present. The presence of this
/// struct, not the model name, selects the kernel (`HcVariant::of_site`).
#[derive(Clone, Copy)]
pub struct HcLowRank {
    /// 2026-09-25: `hc_norm`: a grouped RMS norm scale over the `hc_mult`
    /// streams.
    pub norm_w: DevicePtr,
    /// 2026-09-25: `input_mix_weight_down`.
    pub down_w: DevicePtr,
    /// 2026-09-25: `input_mix_weight_up`.
    pub up_w: DevicePtr,
    /// 2026-09-25: `block_inject_weight`; NULL on the model-level mixer, which
    /// has none.
    pub inject_w: DevicePtr,
    /// 2026-09-25: `hc_lowrank` from the config.
    pub rank: usize,
}

/// 2026-09-25: Hyper-connection (mHC) parameters for one site (attention or FFN).
/// The Sinkhorn fields are FP32; with `lowrank` set they are NULL.
pub struct HcSiteWeights {
    /// 2026-09-25: Mix projection `fn`: `[mix_hc, hc_mult*hidden]` FP32, where
    /// `mix_hc = (2 + hc_mult) * hc_mult`.
    pub hc_fn: DevicePtr,
    /// 2026-09-25: Mix bias `base`: `[mix_hc]` FP32.
    pub hc_base: DevicePtr,
    /// 2026-09-25: Mix scale: `[3]` FP32.
    pub hc_scale: DevicePtr,
    /// 2026-09-25: Low-rank variant. `Some` selects the low-rank kernels, and
    /// `hc_fn`/`hc_base`/`hc_scale` are then NULL.
    pub lowrank: Option<HcLowRank>,
}

/// 2026-09-25: Model-level HC head parameters: the final stream collapse before
/// the LM head. Only the last model layer uses them.
#[derive(Clone)]
pub struct HcHeadWeights {
    /// 2026-09-25: Mix projection `head_fn`: `[hc_mult, hc_mult*hidden]` FP32.
    pub hc_fn: DevicePtr,
    /// 2026-09-25: Mix bias `head_base`: `[hc_mult]` FP32.
    pub hc_base: DevicePtr,
    /// 2026-09-25: Mix scale: `[1]` FP32.
    pub hc_scale: DevicePtr,
    /// 2026-09-25: Low-rank variant of the model-level mixer. `Some` selects
    /// the low-rank kernels; its `inject_w` is NULL.
    pub lowrank: Option<HcLowRank>,
}

pub struct HcWeights {
    pub attn: HcSiteWeights,
    pub ffn: HcSiteWeights,
    /// 2026-09-25: Model-level head weights, used only by the last model layer's
    /// `hc_head` call.
    pub head: Option<HcHeadWeights>,
    pub hc_mult: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
    /// 2026-09-25: Whether this is model layer 0, the layer that seeds the
    /// highway with `hc_expand`.
    ///
    /// Carried here because `attn_layer_idx` counts attention layers only; on a
    /// model that interleaves GDN and attention layers the two indices differ.
    pub is_first_model_layer: bool,
    /// 2026-09-25: Whether this is the last model layer, the one that collapses
    /// the highway with `hc_head`. Carried here for the same reason. On
    /// qwen4_exp that collapse is also the final norm: the checkpoint has no
    /// `model.norm`, and the loader installs a placeholder.
    pub is_last_model_layer: bool,
}

/// 2026-09-25: Which of the four attention projections get an FP8 `[K, N]`
/// transposed twin built by [`Qwen3AttentionLayer::transpose_fp8_for_prefill`].
///
/// [`Qwen3AttentionLayer::transpose_fp8_for_prefill`]:
///     super::Qwen3AttentionLayer::transpose_fp8_for_prefill
///
/// One flag per projection because different prefill arms read them; for
/// example the first-chunk chain reads the Q twin only with
/// `METRALE_ATTN_PREFILL_Q_T=1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp8TwinSet {
    pub q: bool,
    pub k: bool,
    pub v: bool,
    pub o: bool,
}

impl Fp8TwinSet {
    pub const NONE: Self = Self {
        q: false,
        k: false,
        v: false,
        o: false,
    };
    pub const ALL: Self = Self {
        q: true,
        k: true,
        v: true,
        o: true,
    };

    pub fn any(self) -> bool {
        self.q || self.k || self.v || self.o
    }
}

/// 2026-09-25: The two `(module, function)` pairs the W8A8 block-scaled prefill
/// arm needs: the activation quantizer and `fp8_gemm_t_blockscaled`.
///
/// `init.rs` resolves the GEMM from this table, and
/// [`w8a8_prefill_kernels_loaded`] asks the backend for both, so serve load
/// can ask before any layer exists.
pub const W8A8_PREFILL_KERNELS: [(&str, &str); 2] = [
    (
        crate::layers::ops::FP8_QUANT_MODULE,
        crate::layers::ops::FP8_QUANT_ENTRY,
    ),
    ("fp8_gemm_t_blockscaled", "fp8_gemm_t_blockscaled"),
];

/// 2026-09-25: Whether both [`W8A8_PREFILL_KERNELS`] resolve on this backend
/// (through `try_kernel`), without a constructed layer.
pub fn w8a8_prefill_kernels_loaded(gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend) -> bool {
    W8A8_PREFILL_KERNELS
        .iter()
        .all(|(module, func)| crate::layers::try_kernel(gpu, module, func).0 != 0)
}
