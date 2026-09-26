// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `Qwen3AttentionLayer` struct: its weights, per-layer settings and resolved kernel handles.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants: none beyond the types.
//!
//! `MlaWeights` and `HcWeights` are defined in `types_weights.rs` and
//! re-exported here.

use metrale_cache::kv_cache::KvCacheDtype;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use crate::layers::FfnComponent;
use crate::layers::fp8_calibration::Fp8KvCalibration;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::weight_map::{AttentionWeights, DenseWeight, QuantWeight, QuantizedWeight};

pub use super::types_weights::{HcWeights, MlaWeights};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeadGateActivation {
    Sigmoid,
    Softplus,
}

/// 2026-09-25: A full-attention transformer layer: attention plus its FFN or MoE
/// sublayer. Loaders for several architectures build it and switch features on
/// through the `set_*` methods.
#[allow(dead_code)]
pub struct Qwen3AttentionLayer {
    pub(super) input_norm: DenseWeight,
    pub attn: AttentionWeights,
    pub(super) post_attn_norm: DenseWeight,
    pub(super) ffn: FfnComponent,
    pub(super) attn_layer_idx: usize,
    /// 2026-09-25: LoRA adapter weights for the attention projections, installed
    /// by `set_lora_weights`; `None` is base weights only.
    pub(super) lora: Option<crate::layers::ops::lora_delta::LoraAttnWeights>,
    /// 2026-09-25: Whether q_proj emits an interleaved `[Q|gate]` (width
    /// `2 * q_dim`) whose gate half multiplies the attention output through a
    /// sigmoid. When false, q_proj emits `q_dim`.
    pub(super) gated: bool,
    /// 2026-09-25: MRoPE-interleaved instead of scalar RoPE; from
    /// `config.mrope_interleaved`.
    pub(crate) mrope_interleaved: bool,
    /// 2026-09-25: Per-layer head_dim and head-count overrides
    /// (`set_dimension_overrides`); `None` uses the config value.
    pub(crate) head_dim_override: Option<usize>,
    pub(crate) num_q_heads_override: Option<usize>,
    pub(crate) num_kv_heads_override: Option<usize>,
    /// 2026-09-25: Per-layer sliding-window size (`set_sliding_window`).
    pub(crate) sliding_window: Option<u32>,
    /// 2026-09-25: Per-layer RoPE theta and rotary_dim (`set_rope_overrides`).
    pub(crate) rope_theta_override: Option<f32>,
    pub(crate) rotary_dim_override: Option<u32>,
    /// 2026-09-25: Proportional RoPE (`set_rope_proportional`, Gemma-4
    /// full-attention layers).
    pub(crate) rope_proportional: bool,
    /// 2026-09-25: Attention scale override (`set_attn_scale_override`).
    pub(crate) attn_scale_override: Option<f32>,
    /// 2026-09-25: K=V mode (`set_k_eq_v`, Gemma-4 full-attention layers).
    pub(crate) k_eq_v: bool,
    /// 2026-09-25: Weight for the pure-RMS v_norm (`set_k_eq_v`, `set_v_norm`);
    /// the Gemma-4 loader passes a ones-filled buffer.
    pub(crate) v_norm_weight: Option<DenseWeight>,
    /// 2026-09-25: Per-head attention gate weight, `[num_q_heads, hidden_size]`
    /// BF16 (`set_head_gate_weight`): `attn_out *= act(g_proj @ normed)`,
    /// broadcast over head_dim, with `act` from `head_gate_activation`.
    pub(crate) head_gate_weight: Option<DenseWeight>,
    pub(crate) head_gate_activation: HeadGateActivation,
    pub(super) sigmoid_gate_head_broadcast_k: KernelHandle,
    pub(super) softplus_gate_head_broadcast_k: KernelHandle,
    /// 2026-09-25: YaRN inverse frequencies and attention factor
    /// (`set_yarn_rope`).
    pub(crate) yarn_inv_freq: DevicePtr,
    pub(crate) yarn_attention_factor: f32,
    /// 2026-09-25: Gemma-4 `post_attention_layernorm`, on the attention output.
    pub(crate) post_attn_out_norm: Option<DenseWeight>,
    /// 2026-09-25: Gemma-4 `post_feedforward_layernorm`, on the FFN output.
    pub(crate) post_ffn_out_norm: Option<DenseWeight>,
    /// 2026-09-25: Gemma-4 `layer_scalar`: the hidden state is scaled by it at
    /// the end of the layer.
    pub(crate) layer_scalar: Option<f32>,
    /// 2026-09-25: A second FFN: the Gemma-4 dual-FFN MoE (`set_moe_ffn`), or
    /// the LongCat shortcut MoE (`set_shortcut_moe`).
    pub(crate) moe_ffn: Option<FfnComponent>,
    /// 2026-09-25: LongCat shortcut-MoE producer: this sublayer runs `moe_ffn` on
    /// its post-attention normed input and stashes the result in the carry
    /// buffer `(ptr, token_capacity)` instead of adding it; the paired next
    /// sublayer adds it at its end. Separate from the Gemma-4 dual-FFN arm,
    /// which requires the Gemma norms (`pre_moe_norm` is `None` here).
    pub(crate) shortcut_carry_out: Option<(metrale_gpu_runtime::gpu::DevicePtr, usize)>,
    /// 2026-09-25: LongCat shortcut-MoE consumer: after this sublayer's FFN
    /// residual add, `hidden += carry` (the output the previous sublayer
    /// stashed).
    pub(crate) shortcut_carry_in: Option<(metrale_gpu_runtime::gpu::DevicePtr, usize)>,
    /// 2026-09-25: Gemma-4 `pre_feedforward_layernorm_2` (MoE input norm).
    pub(crate) pre_moe_norm: Option<DenseWeight>,
    /// 2026-09-25: Gemma-4 `post_feedforward_layernorm_2` (MoE output norm).
    pub(crate) post_moe_out_norm: Option<DenseWeight>,
    /// 2026-09-25: Gemma-4 `post_feedforward_layernorm_1` (dense FFN output norm).
    pub(crate) post_dense_ffn_norm: Option<DenseWeight>,
    pub(super) kv_dtype: KvCacheDtype,
    pub(super) sparse_v_threshold: f32,
    // 2026-09-25: Quantized q/k/v/o weights for decode; `None` uses the dense
    // `attn` weight.
    pub(super) q_weight: Option<QuantWeight>,
    pub(super) k_weight: Option<QuantWeight>,
    pub(super) v_weight: Option<QuantWeight>,
    pub(super) o_weight: Option<QuantWeight>,
    /// 2026-09-25: BF16 output-projection weight (`set_o_dense_bf16`), used by
    /// the decode and prefill o_proj paths when no earlier arm (MLA) applies.
    pub(super) o_dense_bf16: Option<DenseWeight>,
    pub(crate) mla: Option<MlaWeights>,
    /// 2026-09-25: Hyper-connection (mHC) weights (`set_hc_weights`: DeepSeek-V4,
    /// qwen4_exp). When `Some`, the attention and FFN residual sites run
    /// `hc_pre`/`hc_post` against the `hc_streams` buffer instead of the plain
    /// residual add.
    pub hc: Option<HcWeights>,
    /// 2026-09-25: QSA sparse-attention indexer (`set_qsa`, qwen4_exp).
    pub(crate) qsa: Option<crate::layers::qsa::QsaIndexer>,
    pub(super) hc_pre_k: KernelHandle,
    pub(super) hc_post_k: KernelHandle,
    pub(super) hc_expand_k: KernelHandle,
    pub(super) hc_head_k: KernelHandle,
    /// 2026-09-25: Fused `[q|k|v]` transposed NVFP4 weight
    /// (`set_fused_qkv_prefill_weight`). The wide multi-sequence Q/K/V GEMM uses
    /// it above 8 rows unless `METRALE_NO_FUSED_QKV=1`; `None` runs three
    /// separate GEMMs.
    pub(super) qkv_nvfp4_t: Option<QuantizedWeight>,
    pub(super) q_nvfp4_t: Option<QuantizedWeight>,
    pub(super) k_nvfp4_t: Option<QuantizedWeight>,
    pub(super) v_nvfp4_t: Option<QuantizedWeight>,
    pub(super) o_nvfp4_t: Option<QuantizedWeight>,
    pub(super) q_fp8w_t: Option<crate::weight_map::Fp8WeightTransposed>,
    pub(super) k_fp8w_t: Option<crate::weight_map::Fp8WeightTransposed>,
    pub(super) v_fp8w_t: Option<crate::weight_map::Fp8WeightTransposed>,
    pub(super) o_fp8w_t: Option<crate::weight_map::Fp8WeightTransposed>,
    pub(super) w8a16_gemm_t_k: KernelHandle,
    pub(super) w8a16_gemm_t_pipelined_k: KernelHandle,
    pub(super) w8a16_gemm_t_m128_k: KernelHandle,
    pub(super) per_token_group_quant_fp8_k: crate::layers::ops::Fp8ActQuant,
    pub(super) fp8_gemm_t_blockscaled_k: KernelHandle,
    /// 2026-09-25: `fp8_act_scale_to_kmajor`: rewrites the quantizer's
    /// `[M, K/128]` activation scales into the `[K/128, ceil16(M)]` layout
    /// cuBLASLt reads. Zero when the module is absent; the decode W8A8 arms then
    /// decline while the K-major layout is selected.
    pub(super) fp8_act_scale_kmajor_k: KernelHandle,
    /// 2026-09-25: Offset-from-1 `rms_norm` (`out = x * (1 + w) / rms`).
    pub(super) rms_norm_k: KernelHandle,
    /// 2026-09-25: The norm kernel for checkpoint norm weights: `rms_norm`, or
    /// `rms_norm_vanilla` (`out = x * w / rms`) when the model ships vanilla
    /// norm weights.
    pub(super) rms_norm_w_k: KernelHandle,
    /// 2026-09-25: Warp-per-row sibling of `rms_norm_w_k`; zero unless the model
    /// ships vanilla norm weights and the target has the kernel.
    pub(super) rms_norm_w_warp_row_k: KernelHandle,
    /// 2026-09-25: Whether `rms_norm_w_k` is the vanilla kernel
    /// (`ships_vanilla_norm_weights`).
    pub(super) norm_vanilla: bool,
    pub(super) rms_norm_residual_k: KernelHandle,
    pub(super) rms_norm_f32_in_k: KernelHandle,
    pub(super) dense_gemv_k: KernelHandle,
    /// 2026-09-25: Packed-Q2 to BF16 dequant (`dequant_gguf_bf16`), for the
    /// keep-packed attention prefill; decode uses `q2_0_gemv_vec`. Zero when
    /// absent.
    pub(super) dequant_q2_0_gn_k: KernelHandle,
    /// 2026-09-25: Q2_0 MMQ prefill kernels and the q8_1 activation quantizer.
    /// Zero at construction; `set_packed_q2_weights` resolves them.
    pub(super) q2_0_mmq_nc_k: KernelHandle,
    pub(super) q2_0_mmq_wc_k: KernelHandle,
    pub(super) q4k_quant_act_k: KernelHandle,
    /// 2026-09-25: `q2_0_gemv_vec`, the decode kernel for keep-packed q/k/v/o.
    pub(super) q2_0_gemv_k: KernelHandle,
    /// 2026-09-25: Batched BF16 GEMV (M rows, one weight pass), for
    /// multi-sequence decode q/k/v when the attention weights are plain BF16.
    /// Zero when absent.
    pub(super) dense_gemv_batchm_k: KernelHandle,
    pub(super) w4a16_gemv_k: KernelHandle,
    /// 2026-09-25: Single-warp `w4a16_gemv_sw`; zero when absent, and then the
    /// NVFP4 decode GEMV uses `w4a16_gemv_k`.
    pub(super) w4a16_gemv_sw_k: KernelHandle,
    pub(super) w8a16_gemv_k: KernelHandle,
    /// 2026-09-25: `w8a16_gemv_batch4` (M <= 4); zero when absent.
    pub(super) w8a16_gemv_batch4_k: KernelHandle,
    /// 2026-09-25: `w8a16_gemv_batch16` (M <= 16), for the contiguous o_proj;
    /// zero when absent.
    pub(super) w8a16_gemv_batch16_k: KernelHandle,
    /// 2026-09-25: Strided siblings of the above (caller-supplied A/C row
    /// pitches), for the multi-sequence decode Q/K/V, which writes into the
    /// `per_seq_qkv`-strided QKV buffer. Zero on either handle keeps the
    /// per-sequence scalar `w8a16_gemv` loop.
    pub(super) w8a16_gemv_batch4_strided_k: KernelHandle,
    pub(super) w8a16_gemv_batch16_strided_k: KernelHandle,
    /// 2026-09-25: `w8a16_gemm_m16` and its strided sibling: the MMA arm for the
    /// FP8 o_proj (contiguous) and multi-sequence Q/K/V (strided) at 5..=16
    /// rows, taken when `m16_tc`. Zero when absent, which keeps the batched
    /// GEMVs.
    pub(super) w8a16_gemm_m16_k: KernelHandle,
    pub(super) w8a16_gemm_m16_strided_k: KernelHandle,
    /// 2026-09-25: The resolved `attn_m16_tc` setting (`m16_tc_levers().attn`:
    /// the target's `[defaults]` row, overridden by `METRALE_ATTN_M16_TC`, else
    /// by `METRALE_M16_TC`). It covers the Q/K/V and o_proj arms only; the dense
    /// FFN reads `ffn_m16_tc`. Set at construction, so the route cannot change
    /// between CUDA-graph replays.
    pub(super) m16_tc: bool,
    /// 2026-09-25: N-column-blocked W8A16 GEMVs, the bit-exact sibling of
    /// `w8a16_gemv_batch16`, contiguous (o_proj) and strided (multi-sequence
    /// Q/K/V) at 5..=16 rows. Zero when absent, which keeps the batch16 GEMVs.
    /// Rule and reasons: `attn_ncol_gemv.rs`.
    pub(super) w8a16_gemv_ncol2_k: KernelHandle,
    pub(super) w8a16_gemv_ncol4_k: KernelHandle,
    pub(super) w8a16_gemv_ncol2_strided_k: KernelHandle,
    pub(super) w8a16_gemv_ncol4_strided_k: KernelHandle,
    /// 2026-09-25: The N-column arm's width, resolved once at construction.
    /// `None` when the resolved `attn_ncol_gemv` setting is off (the target's
    /// `[defaults]` row, `METRALE_ATTN_NCOL_GEMV`, and
    /// `METRALE_NO_ATTN_DECODE_BATCH`, which wins).
    pub(super) attn_ncol: Option<super::attn_ncol_gemv::NcolWidth>,
    pub(super) w8a16_gemm_k: KernelHandle,
    pub(super) w8a16_gemm_pipelined_k: KernelHandle,
    /// 2026-09-25: 32-row M-tile twin of `w8a16_gemm_pipelined`: one launch per
    /// Q/K/V projection (strided) and per o_proj (contiguous) above 16 rows.
    /// Zero unless `ModelLevers::fp8_attn_m32` is on and the target has the
    /// module; zero keeps those rows on the per-row loop.
    pub(super) w8a16_gemm_pipelined_m32_k: KernelHandle,
    pub(super) w4a16_gemv_dual_k: KernelHandle,
    pub(super) rope_k: KernelHandle,
    /// 2026-09-25: `rope_forward_strided`; zero when absent.
    pub(super) rope_strided_k: KernelHandle,
    /// 2026-09-25: `rms_norm_strided`; zero when absent.
    pub(super) rms_norm_strided_k: KernelHandle,
    pub(super) rope_mrope_interleaved_k: KernelHandle,
    pub(super) rope_mrope_interleaved_k_only_k: KernelHandle,
    pub(super) rope_yarn_k: KernelHandle,
    pub(super) rope_yarn_scaled_k: KernelHandle,
    pub(super) rope_yarn_interleaved_k: KernelHandle,
    pub(super) rope_yarn_interleaved_inv_k: KernelHandle,
    pub(super) rope_proportional_k: KernelHandle,
    pub(super) reshape_cache_k: KernelHandle,
    pub(super) fused_k_norm_rope_cache_write_bf16_k: KernelHandle,
    pub(super) fused_k_norm_rope_mrope_cache_write_bf16_k: KernelHandle,
    pub(super) reshape_and_cache_flash_v_only_k: KernelHandle,
    /// 2026-09-25: Decode fusion of k_norm, RoPE and the FP8 K/V cache write.
    /// Zero when the target lacks it; the caller then runs the unfused chain.
    pub(super) fused_k_norm_rope_cache_write_fp8_kv_k: KernelHandle,
    pub(super) wht_bf16_k: KernelHandle,
    pub(super) wht_bf16_k_inv: KernelHandle,
    /// 2026-09-25: InnerQ apply kernels for Q and K (`tq_plus_innerq_apply`);
    /// zero when the module is absent.
    pub(super) innerq_apply_q_k: KernelHandle,
    pub(super) innerq_apply_k_k: KernelHandle,
    pub(super) paged_decode_k: KernelHandle,
    pub(super) paged_decode_512_k: KernelHandle,
    pub(super) paged_decode_mla_k: KernelHandle,
    pub(super) mla_paged_decode_k: KernelHandle,
    pub(super) mla_paged_decode_fp8_k: KernelHandle,
    pub(super) mla_batched_gemv_k: KernelHandle,
    pub(super) mla_q_rope_scatter_k: KernelHandle,
    pub(super) mla_q_rope_writeback_k: KernelHandle,
    pub(super) mla_cache_assemble_k: KernelHandle,
    pub(super) mla_q_rope_extract_batched_k: KernelHandle,
    pub(super) mla_q_rope_writeback_batched_k: KernelHandle,
    pub(super) mla_kv_assemble_batched_k: KernelHandle,
    pub(super) mla_cache_assemble_batched_k: KernelHandle,
    pub(super) prefill_attn_mla320_k: KernelHandle,
    pub(super) grouped_gemm_mla_k: KernelHandle,
    pub(super) mla_q_final_assemble_k: KernelHandle,
    pub(super) mla_fused_prefill_k: KernelHandle,
    pub(super) gemm_splitk_partial_k: KernelHandle,
    pub(super) gemm_splitk_reduce_k: KernelHandle,
    pub(super) dense_gemm_tc_k: KernelHandle,
    pub(super) paged_decode_splitk_k: Option<KernelHandle>,
    pub(super) paged_decode_reduce_k: Option<KernelHandle>,
    /// 2026-09-25: GQA-packed non-split paged-decode kernels
    /// (`kernels/gb10/common/paged_decode_attn_{bf16,fp8}_gqa.cu`); `None` on a
    /// target without the sources. Used only when `gqa_pack_enabled()` (off
    /// unless `METRALE_ATTN_DECODE_GQA_PACK` turns it on) and
    /// `attn_splitk::gqa_pack_shape_ok` accepts the shape.
    pub(super) paged_decode_bf16_gqa_k: Option<KernelHandle>,
    pub(super) paged_decode_fp8_gqa_k: Option<KernelHandle>,
    /// 2026-09-25: Hopper paged-decode split-K kernels
    /// (`kernels/hopper/common/paged_decode_{fp8,bf16}_splitk_hopper.cu`);
    /// `None` on a target without those sources.
    pub(super) paged_decode_splitk_hopper_k: Option<KernelHandle>,
    pub(super) paged_decode_reduce_hopper_k: Option<KernelHandle>,
    pub(super) paged_decode_splitk_bf16_hopper_k: Option<KernelHandle>,
    pub(super) paged_decode_reduce_bf16_hopper_k: Option<KernelHandle>,
    pub(super) residual_add_k: KernelHandle,
    pub(super) sigmoid_gate_mul_k: KernelHandle,
    pub(super) deinterleave_qg_k: KernelHandle,
    pub(super) w4a16_gemv_qg_k: KernelHandle,
    pub(super) residual_add_rms_norm_k: KernelHandle,
    /// 2026-09-25: Dual-output (BF16 + FP32) MoE-input norm, used when FP32
    /// routing is active; zero when absent.
    pub(super) residual_add_rms_norm_gatef32_k: KernelHandle,
    pub(super) w4a16_gemv_qg_batch2_k: KernelHandle,
    pub(super) w4a16_gemv_dual_batch2_k: KernelHandle,
    pub(super) w4a16_gemv_batch2_k: KernelHandle,
    pub(super) w4a16_gemv_qg_batch3_k: KernelHandle,
    pub(super) w4a16_gemv_dual_batch3_k: KernelHandle,
    pub(super) w4a16_gemv_batch3_k: KernelHandle,
    /// 2026-09-25: The `w4a16_gemv_batch{M}` tiers for multi-row q/k/v/o GEMVs; a
    /// zero handle means the target lacks that tier.
    pub(super) w4a16_batchm: W4a16BatchmTiers,
    pub(super) w4a16_gemm_k: KernelHandle,
    pub(super) w4a16_gemm_t_k: KernelHandle,
    pub(super) w4a16_gemm_t_k64_k: KernelHandle,
    /// 2026-09-25: K64 with a 64-wide N tile. Zero when absent or when
    /// `METRALE_NO_K64_N64` is set.
    pub(super) w4a16_gemm_t_k64_n64_k: KernelHandle,
    pub(super) w4a16_gemm_t_m128_k: KernelHandle,
    /// 2026-09-25: BF16 variant of `w4a16_gemm_t_m128` for the Q/K/V/o prefill
    /// projections, used only when `ModelLevers::bf16_tc_proj`
    /// (`METRALE_BF16_TC_PROJ` set). Zero when absent.
    pub(super) w4a16_gemm_t_m128_bf16_k: KernelHandle,
    /// 2026-09-25: `w4a16_gemm_t_m128_v2`, resolved only for
    /// `METRALE_W4A16_VARIANT=v2` or `v3`; zero otherwise.
    pub(super) w4a16_gemm_t_m128_v2_k: KernelHandle,
    /// 2026-09-25: `w4a16_gemm_t_m128_v3`, resolved only for
    /// `METRALE_W4A16_VARIANT=v3`; zero otherwise.
    pub(super) w4a16_gemm_t_m128_v3_k: KernelHandle,
    pub(super) dense_gemm_k: KernelHandle,
    pub(super) dense_gemm_pipelined_k: KernelHandle,
    pub(super) prefill_attn_k: KernelHandle,
    pub(super) prefill_attn_512_k: KernelHandle,
    /// 2026-09-25: Whether `prefill_attn_512_k` resolved to the tensor-core
    /// instantiation or the scalar reference. Read only for the profile label,
    /// so the label names the kernel that ran.
    pub(super) prefill_attn_512_is_tc: bool,
    pub(super) csa_compress_k: KernelHandle,
    pub(super) prefill_attn_compressed_k: KernelHandle,
    /// 2026-09-25: Compressed blocks in `mla.compressor.pool` for the active
    /// sequence: prefill stores `n / ratio`, and each decode append advances it.
    /// One counter per layer, so it tracks one sequence at a time.
    pub(super) v4_comp_pool_filled: std::sync::atomic::AtomicU32,
    /// 2026-09-25: Whether the CSA `prev_win` ring holds a real previous window.
    /// Prefill sets it when the prompt has a full window, and a decode append
    /// sets it; when false the CSA append masks Ca (window-0 semantics).
    pub(super) v4_comp_prev_valid: std::sync::atomic::AtomicBool,
    pub(super) v4_decode_started: std::sync::atomic::AtomicBool,
    pub(super) v4_decode_first_pos: std::sync::atomic::AtomicU32,
    pub(super) prefill_attn_paged_512_k: KernelHandle,
    pub(super) prefill_attn_64_k: KernelHandle,
    pub(super) prefill_attn_paged_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8_k: KernelHandle,
    pub(super) prefill_attn_paged_nvfp4_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo4_k: KernelHandle,
    pub(super) prefill_attn_paged_64_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8_64_k: KernelHandle,
    pub(super) prefill_attn_paged_nvfp4_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo2_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo3_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo4_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo8_64_k: KernelHandle,
    pub(super) prefill_attn_paged_bf16k_turbo3v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_bf16k_turbo4v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_bf16k_turbo2v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8k_turbo3v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8k_turbo4v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8k_turbo2v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo4k_turbo3v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo4k_turbo8v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo3k_turbo8v_64_k: KernelHandle,
    // 2026-09-25: Batched paged-prefill kernels: one launch over several
    // streams, each with its own block table (`block_table_ptrs`). Zero when
    // absent.
    pub(super) prefill_attn_paged_batched_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8_batched_k: KernelHandle,
    pub(super) prefill_attn_paged_nvfp4_batched_k: KernelHandle,
    pub(super) prefill_attn_paged_batched_64_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8_batched_64_k: KernelHandle,
    pub(super) prefill_attn_paged_nvfp4_batched_64_k: KernelHandle,
    pub(super) deinterleave_qg_split_k: KernelHandle,
    pub(super) deinterleave_qg_split_qnorm_k: KernelHandle,
    pub(super) deinterleave_qg_split_qnorm_mrope_k: KernelHandle,
    pub(super) sigmoid_gate_mul_batched_k: KernelHandle,
    pub(super) q_fp8: Option<DevicePtr>,
    pub(super) k_fp8: Option<DevicePtr>,
    pub(super) v_fp8: Option<DevicePtr>,
    pub(super) o_fp8: Option<DevicePtr>,
    pub(super) fp8_gemm_k: KernelHandle,
    pub(super) bf16_to_fp8_k: KernelHandle,
    pub(super) fp8_fp8_gemm_k: KernelHandle,
    pub(super) fp8_gemm_t_m128_k: KernelHandle,
    pub(super) fp8_fp8_gemm_t_m128_k: KernelHandle,
    // 2026-09-25: Native FP4 prefill (`w4a4_gemm_mfast`); zero when the target's
    // kernel dir does not ship it.
    pub(super) w4a4_gemm_k: KernelHandle,
    pub(super) quantize_nvfp4_k: KernelHandle,
    pub(super) fp8_calibration: Option<Fp8KvCalibration>,
}
