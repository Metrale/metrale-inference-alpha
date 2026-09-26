// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-token decode forward for one Qwen3.5 decoder layer, full
//! attention or GDN, through `&dyn GpuBackend` and a `QuantWeights` impl.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.
//!
//! [`forward_full_attention`] and [`forward_linear_attention`] each run norm,
//! projections, attention or the GDN recurrence, the residual add with the
//! post-attention norm, and the MLP with its residual. One token per call: the
//! KV append writes position `cache_pos`, and attention is passed `seq_len_attn`
//! as the cache length. Tokenizer, sampler and weight loading belong to the caller.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::quant_weights::QuantWeights;

mod full_attention;
mod linear_attention;

pub use full_attention::forward_full_attention;
pub use linear_attention::forward_linear_attention;

/// 2026-09-25: Dimensions of a Qwen3.5 checkpoint, read by both layer forwards.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35ForwardConfig {
    pub hidden: u32,
    pub intermediate: u32,
    pub num_layers: u32,
    pub vocab: u32,
    pub group_size: u32,
    pub rms_eps: f32,

    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rope_theta: f32,
    /// 2026-09-25: RoPE rotates only the first `rotary_dim` elements of each head;
    /// the rest pass through unchanged (`rope_apply.metal`).
    pub rotary_dim: u32,

    pub num_k_heads_lin: u32,
    pub num_v_heads_lin: u32,
    pub k_head_dim_lin: u32,
    pub v_head_dim_lin: u32,
    pub conv_kernel_size: u32,
}

impl Qwen35ForwardConfig {
    /// 2026-09-25: The dimensions the `metal_qwen35_inference` example runs with; its
    /// default model directory is `Qwen3.5-4B-MLX-8bit`.
    pub const fn qwen3_5_4b_mlx_int8() -> Self {
        Self {
            hidden: 2560,
            intermediate: 9216,
            num_layers: 32,
            vocab: 248_320,
            group_size: 64,
            rms_eps: 1e-6,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            rotary_dim: 64,
            num_k_heads_lin: 16,
            num_v_heads_lin: 32,
            k_head_dim_lin: 128,
            v_head_dim_lin: 128,
            conv_kernel_size: 4,
        }
    }

    /// 2026-09-25: q_proj output width. Qwen3.5 packs the attention output gate
    /// into the Q projection: each head's row is `[Q_h | gate_h]`, which
    /// `qwen35_qkv_split` separates before the norm.
    #[inline]
    pub const fn q_total(&self) -> u32 {
        self.num_heads * self.head_dim * 2
    }
    /// 2026-09-25: Q width after the gate is split off: half of [`Self::q_total`].
    #[inline]
    pub const fn q_only(&self) -> u32 {
        self.num_heads * self.head_dim
    }
    #[inline]
    pub const fn kv_dim(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }
    #[inline]
    pub const fn z_dim_lin(&self) -> u32 {
        self.num_v_heads_lin * self.v_head_dim_lin
    }
    #[inline]
    pub const fn qkv_total_lin(&self) -> u32 {
        2 * self.num_k_heads_lin * self.k_head_dim_lin + self.num_v_heads_lin * self.v_head_dim_lin
    }
    /// 2026-09-25: The number of GDN heads that gate, beta, dt_bias and A_log are
    /// indexed by.
    #[inline]
    pub const fn num_state_heads(&self) -> u32 {
        self.num_v_heads_lin
    }
}

/// 2026-09-25: Kernel handles for both layer forwards, looked up once by
/// [`Self::resolve`].
pub struct Qwen35Kernels {
    pub rms: KernelHandle,
    pub rope: KernelHandle,
    pub kvap: KernelHandle,
    pub attn: KernelHandle,
    pub sg: KernelHandle,
    pub add_rms: KernelHandle,
    pub qkv_split: KernelHandle,
    pub conv1d: KernelHandle,
    pub gdn_gate: KernelHandle,
    pub sigmoid: KernelHandle,
    pub gdn_dec: KernelHandle,
    /// 2026-09-25: Turbo KV cache kernels: quantizing appends, dequantizing decode
    /// attentions, and the WHT rotations around them. `resolve` looks them up
    /// even for a bf16 cache, so a missing one fails at startup.
    pub kvap_turbo8: KernelHandle,
    pub attn_turbo8: KernelHandle,
    pub kvap_turbo4: KernelHandle,
    pub attn_turbo4: KernelHandle,
    pub kvap_turbo3: KernelHandle,
    pub attn_turbo3: KernelHandle,
    pub kvap_turbo2: KernelHandle,
    pub attn_turbo2: KernelHandle,
    pub kvap_bf16k_turbo4v: KernelHandle,
    pub attn_bf16k_turbo4v: KernelHandle,
    pub kvap_bf16k_turbo3v: KernelHandle,
    pub attn_bf16k_turbo3v: KernelHandle,
    pub kvap_bf16k_turbo2v: KernelHandle,
    pub attn_bf16k_turbo2v: KernelHandle,
    pub wht: KernelHandle,
    pub wht_inv: KernelHandle,
}

impl Qwen35Kernels {
    /// 2026-09-25: Looks up every kernel both layer forwards use; returns an error
    /// if any is missing.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            rms: gpu.kernel("rms_norm", "rms_norm")?,
            rope: gpu.kernel("rope_apply", "rope_apply")?,
            kvap: gpu.kernel("kv_cache_append", "kv_cache_append")?,
            attn: gpu.kernel("attention_decode", "attention_decode")?,
            sg: gpu.kernel("sigmoid_gate", "sigmoid_gate")?,
            add_rms: gpu.kernel("add_rms_norm", "add_rms_norm")?,
            qkv_split: gpu.kernel("qwen35_qkv_split", "qwen35_qkv_split")?,
            conv1d: gpu.kernel("causal_conv1d_update_l2norm", "causal_conv1d_update_l2norm")?,
            gdn_gate: gpu.kernel("gdn_helpers", "gdn_compute_gate")?,
            sigmoid: gpu.kernel("gdn_helpers", "sigmoid_bf16_to_f32")?,
            gdn_dec: gpu.kernel("gated_delta_rule_decode", "gated_delta_rule_decode")?,
            kvap_turbo8: gpu.kernel("kv_cache_append_turbo8", "kv_cache_append_turbo8")?,
            attn_turbo8: gpu.kernel("attention_decode_turbo8", "attention_decode_turbo8")?,
            kvap_turbo4: gpu.kernel("kv_cache_append_turbo4", "kv_cache_append_turbo4")?,
            attn_turbo4: gpu.kernel("attention_decode_turbo4", "attention_decode_turbo4")?,
            kvap_turbo3: gpu.kernel("kv_cache_append_turbo3", "kv_cache_append_turbo3")?,
            attn_turbo3: gpu.kernel("attention_decode_turbo3", "attention_decode_turbo3")?,
            kvap_turbo2: gpu.kernel("kv_cache_append_turbo2", "kv_cache_append_turbo2")?,
            attn_turbo2: gpu.kernel("attention_decode_turbo2", "attention_decode_turbo2")?,
            kvap_bf16k_turbo4v: gpu.kernel(
                "kv_cache_append_bf16k_turbov",
                "kv_cache_append_bf16k_turbo4v",
            )?,
            attn_bf16k_turbo4v: gpu.kernel(
                "attention_decode_bf16k_turbov",
                "attention_decode_bf16k_turbo4v",
            )?,
            kvap_bf16k_turbo3v: gpu.kernel(
                "kv_cache_append_bf16k_turbov",
                "kv_cache_append_bf16k_turbo3v",
            )?,
            attn_bf16k_turbo3v: gpu.kernel(
                "attention_decode_bf16k_turbov",
                "attention_decode_bf16k_turbo3v",
            )?,
            kvap_bf16k_turbo2v: gpu.kernel(
                "kv_cache_append_bf16k_turbov",
                "kv_cache_append_bf16k_turbo2v",
            )?,
            attn_bf16k_turbo2v: gpu.kernel(
                "attention_decode_bf16k_turbov",
                "attention_decode_bf16k_turbo2v",
            )?,
            wht: gpu.kernel("wht_bf16", "wht_bf16_inplace")?,
            wht_inv: gpu.kernel("wht_bf16", "wht_bf16_inplace_inv")?,
        })
    }
}

/// 2026-09-25: Storage format of the contiguous KV cache ([`LayerKvCache`]).
/// Per-element sizes are what [`LayerKvCache::alloc`] allocates, scales
/// included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalKvDtype {
    /// 2026-09-25: BF16, 2 bytes per element.
    Bf16,
    /// 2026-09-25: FP8 E4M3 data and a BF16 scale per group of 16, in the
    /// WHT-rotated basis: 1.125 bytes per element.
    Turbo8,
    /// 2026-09-25: 4-bit Lloyd-Max codebook indices and an FP8 E4M3 scale per
    /// group of 16 that keeps the group's L2 norm, in the WHT-rotated basis:
    /// 0.5625 bytes per element.
    Turbo4,
    /// 2026-09-25: 3-bit Lloyd-Max indices (8 values in 3 bytes) and FP8 E4M3
    /// group scales: 0.4375 bytes per element.
    Turbo3,
    /// 2026-09-25: 2-bit Lloyd-Max indices (4 per byte) and FP8 E4M3 group
    /// scales: 0.3125 bytes per element.
    Turbo2,
    /// 2026-09-25: K in BF16, not rotated; V as Turbo4.
    Bf16KTurbo4V,
    /// 2026-09-25: K in BF16, not rotated; V as Turbo3.
    Bf16KTurbo3V,
    /// 2026-09-25: K in BF16, not rotated; V as Turbo2.
    Bf16KTurbo2V,
}

impl MetalKvDtype {
    /// 2026-09-25: K is stored in the WHT-rotated basis. When true, the forward
    /// rotates K before the append and Q before attention.
    pub fn k_is_rotated(self) -> bool {
        matches!(
            self,
            Self::Turbo8 | Self::Turbo4 | Self::Turbo3 | Self::Turbo2
        )
    }
    /// 2026-09-25: V is stored in the WHT-rotated basis. When true, the forward
    /// rotates V before the append and applies the inverse WHT to the
    /// attention output.
    pub fn v_is_rotated(self) -> bool {
        self != Self::Bf16
    }
}

impl std::str::FromStr for MetalKvDtype {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "turbo8" => Ok(Self::Turbo8),
            "turbo4" => Ok(Self::Turbo4),
            "turbo3" => Ok(Self::Turbo3),
            "turbo2" => Ok(Self::Turbo2),
            "bf16k_turbo4v" => Ok(Self::Bf16KTurbo4V),
            "bf16k_turbo3v" => Ok(Self::Bf16KTurbo3V),
            "bf16k_turbo2v" => Ok(Self::Bf16KTurbo2V),
            other => {
                anyhow::bail!(
                    "kv dtype {other:?} not supported on metal (bf16 | turbo8 | turbo4 | turbo3 | turbo2 | bf16k_turbo4v/3v/2v)"
                )
            }
        }
    }
}

/// 2026-09-25: KV cache of one full-attention layer, for one sequence.
///
/// `dtype` selects the storage format. A quantized side holds packed data in
/// `k`/`v` and its per-16-element group scales in `k_scales`/`v_scales`
/// (BF16 for Turbo8, FP8 E4M3 otherwise).
pub struct LayerKvCache {
    pub k: DevicePtr,
    pub v: DevicePtr,
    /// 2026-09-25: Capacity in tokens: the `max_seq` given to [`Self::alloc`].
    #[allow(dead_code)]
    pub capacity: u32,
    pub dtype: MetalKvDtype,
    /// 2026-09-25: Group-scale buffers, `Some` only for a quantized side: both
    /// for Turbo8/4/3/2, V only for the `Bf16KTurbo*V` formats, neither for
    /// `Bf16`.
    pub k_scales: Option<DevicePtr>,
    pub v_scales: Option<DevicePtr>,
}

impl LayerKvCache {
    /// 2026-09-25: Allocates a cache of `max_seq` tokens by `kv_dim` elements in
    /// the given format. Panics for a turbo format when `kv_dim` is not a
    /// multiple of 16.
    pub fn alloc(
        gpu: &dyn GpuBackend,
        dtype: MetalKvDtype,
        max_seq: u32,
        kv_dim: u32,
    ) -> Result<Self> {
        assert!(
            dtype == MetalKvDtype::Bf16 || kv_dim.is_multiple_of(16),
            "turbo dtypes need KV_DIM divisible by 16"
        );
        let n = (max_seq * kv_dim) as usize;
        let scale_bytes_e4m3 = (max_seq * kv_dim / 16) as usize;
        // 2026-09-25: (k_bytes, v_bytes, k_scale_bytes, v_scale_bytes)
        let (kb, vb, ksb, vsb) = match dtype {
            MetalKvDtype::Bf16 => (n * 2, n * 2, 0, 0),
            // 2026-09-25: Turbo8 scales are BF16: 2 bytes per group of 16.
            MetalKvDtype::Turbo8 => (n, n, scale_bytes_e4m3 * 2, scale_bytes_e4m3 * 2),
            MetalKvDtype::Turbo4 => (n / 2, n / 2, scale_bytes_e4m3, scale_bytes_e4m3),
            MetalKvDtype::Turbo3 => (n * 3 / 8, n * 3 / 8, scale_bytes_e4m3, scale_bytes_e4m3),
            MetalKvDtype::Turbo2 => (n / 4, n / 4, scale_bytes_e4m3, scale_bytes_e4m3),
            MetalKvDtype::Bf16KTurbo4V => (n * 2, n / 2, 0, scale_bytes_e4m3),
            MetalKvDtype::Bf16KTurbo3V => (n * 2, n * 3 / 8, 0, scale_bytes_e4m3),
            MetalKvDtype::Bf16KTurbo2V => (n * 2, n / 4, 0, scale_bytes_e4m3),
        };
        let alloc_opt = |bytes: usize| -> Result<Option<DevicePtr>> {
            Ok(if bytes > 0 {
                Some(gpu.alloc(bytes)?)
            } else {
                None
            })
        };
        Ok(Self {
            k: gpu.alloc(kb)?,
            v: gpu.alloc(vb)?,
            capacity: max_seq,
            dtype,
            k_scales: alloc_opt(ksb)?,
            v_scales: alloc_opt(vsb)?,
        })
    }
}

pub struct FullAttentionLayer<'a, Q: QuantWeights> {
    pub input_ln: DevicePtr,
    pub q_norm: DevicePtr,
    pub k_norm: DevicePtr,
    pub post_ln: DevicePtr,
    pub q_proj: &'a Q,
    pub k_proj: &'a Q,
    pub v_proj: &'a Q,
    pub o_proj: &'a Q,
    pub gate_proj: &'a Q,
    pub up_proj: &'a Q,
    pub down_proj: &'a Q,
}

/// 2026-09-25: Scratch buffers for [`forward_full_attention`].
pub struct FullAttentionScratch {
    pub x_norm: DevicePtr,
    pub q_full: DevicePtr,
    pub q_split: DevicePtr,
    pub gate_split: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub q_norm_out: DevicePtr,
    pub k_norm_out: DevicePtr,
    pub attn_out: DevicePtr,
    pub gated_attn: DevicePtr,
    pub o: DevicePtr,
    pub x_resid: DevicePtr,
    pub x_norm2: DevicePtr,
    pub gate_act: DevicePtr,
    pub up_act: DevicePtr,
    pub x_out: DevicePtr,
}

pub struct LinearAttentionLayer<'a, Q: QuantWeights> {
    pub input_ln: DevicePtr,
    /// 2026-09-25: FP32 `[num_state_heads]`.
    pub a_log: DevicePtr,
    /// 2026-09-25: BF16 `[num_state_heads]`.
    pub dt_bias: DevicePtr,
    /// 2026-09-25: BF16 `[qkv_total_lin, conv_kernel_size]`.
    pub conv1d_weight: DevicePtr,
    pub in_proj_a: &'a Q,
    pub in_proj_b: &'a Q,
    pub in_proj_qkv: &'a Q,
    pub in_proj_z: &'a Q,
    /// 2026-09-25: BF16 `[v_head_dim_lin]`.
    pub norm_weight: DevicePtr,
    pub out_proj: &'a Q,
    pub post_ln: DevicePtr,
    pub gate_proj: &'a Q,
    pub up_proj: &'a Q,
    pub down_proj: &'a Q,
}

/// 2026-09-25: Conv and GDN state of one linear-attention layer, carried from
/// token to token. The caller allocates and zeroes it.
pub struct LinearAttentionState {
    /// 2026-09-25: FP32 `[qkv_total_lin, conv_kernel_size]`.
    pub conv1d_state: DevicePtr,
    /// 2026-09-25: FP32 `[num_v_heads_lin, k_head_dim_lin, v_head_dim_lin]`.
    pub gdn_state: DevicePtr,
}

/// 2026-09-25: Scratch buffers for [`forward_linear_attention`].
pub struct LinearAttentionScratch {
    pub x_norm: DevicePtr,
    pub dt_raw: DevicePtr,
    pub b_raw: DevicePtr,
    pub qkv: DevicePtr,
    pub qkv_smooth: DevicePtr,
    pub z: DevicePtr,
    /// 2026-09-25: FP32 `[num_state_heads]`.
    pub gate: DevicePtr,
    /// 2026-09-25: FP32 `[num_state_heads]`.
    pub beta: DevicePtr,
    pub y: DevicePtr,
    pub y_norm: DevicePtr,
    pub out: DevicePtr,
    pub x_resid: DevicePtr,
    pub x_norm2: DevicePtr,
    pub gate_act: DevicePtr,
    pub up_act: DevicePtr,
    pub x_final: DevicePtr,
}
