// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `Qwen3SsmLayer` struct: the GDN layer's weights, kernel
//! handles and per-layer bindings.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants: none beyond the types.

use crate::layers::FfnComponent;
use crate::layers::ops;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::weight_map::{DenseWeight, Fp8Weight, QuantizedWeight, SsmWeights};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

/// 2026-09-25: The Gated DeltaNet layer. Its QKVZ projection comes in two
/// layouts: interleaved per key-head group, which `deinterleave_qkvz` (or the
/// fused `w4a16_gemv_qkvz`) turns into `[Q | K | V | Z]`, and sequential
/// (`sequential_qkvz`), already in that order.
#[allow(dead_code)]
pub struct Qwen3SsmLayer {
    /// 2026-09-25: mHC highway weights (`set_hc_weights`); `Some` routes the
    /// entry points to the hc paths (see `hc.rs`).
    pub(crate) hc: Option<crate::layers::qwen3_attention::HcWeights>,
    /// 2026-09-25: PLE n-gram injection (`set_ple`), run by the hc paths before
    /// `hc_pre_site`.
    pub(crate) ple: Option<crate::layers::ple::PleLayer>,
    /// 2026-09-25: `hyper_connection::hc_pre`. It and the two `hc_*` handles
    /// below are 0 unless `config.hc_mult > 0` (`hc_kernel`).
    pub(super) hc_pre_k: KernelHandle,
    pub(super) hc_post_k: KernelHandle,
    /// 2026-09-25: `hc_expand`, launched only on the first model layer
    /// (`HcWeights::is_first_model_layer`).
    pub(super) hc_expand_k: KernelHandle,
    pub(super) input_norm: DenseWeight,
    pub(super) ssm: SsmWeights,
    pub(super) post_attn_norm: DenseWeight,
    pub(super) ffn: FfnComponent,
    /// 2026-09-25: GDN `out_proj` LoRA delta and its kernels
    /// (`set_out_proj_lora`); `None` makes `apply_lora_out_proj` a no-op.
    pub(super) lora_out_proj: Option<(
        crate::layers::ops::lora_delta::LoraPair,
        crate::layers::ops::lora_delta::LoraKernels,
    )>,
    pub(super) qkvz_nvfp4: Option<QuantizedWeight>,
    pub(super) qkvz_nvfp4_t: Option<QuantizedWeight>,
    pub(super) out_proj_nvfp4_t: Option<QuantizedWeight>,
    pub out_proj_dense: Option<DenseWeight>,
    // 2026-09-25: Block-scaled FP8 weights (`set_fp8_decode_weights`).
    pub(super) qkvz_fp8w: Option<Fp8Weight>,
    pub(super) out_proj_fp8w: Option<Fp8Weight>,
    /// 2026-09-25: Per-row FP8 weights (`set_fp8_rowwise_prefill_weights`),
    /// read only by the row-wise prefill arms. They are kept apart from
    /// `qkvz_fp8w` / `out_proj_fp8w` because those fields' readers index the
    /// scale as a `[N/128, K/128]` grid, and a per-row scale read that way gives
    /// wrong values without faulting.
    pub(super) qkvz_fp8w_rowwise: Option<Fp8Weight>,
    pub(super) out_proj_fp8w_rowwise: Option<Fp8Weight>,
    /// 2026-09-25: Addresses of this layer's two slices of the arena slab
    /// `ssm_rowwise_w_bf16`, 0 until the first row-wise prefill carves them
    /// (`rowwise_bf16.rs`).
    pub(super) qkvz_rowwise_bf16: std::sync::atomic::AtomicU64,
    pub(super) out_proj_rowwise_bf16: std::sync::atomic::AtomicU64,
    /// 2026-09-25: Packed Q2_0 fused `in_proj_qkvz` (`set_packed_q2_qkvz`).
    /// Its only installer, the `qwen35_dense` loader, keeps `out_proj` NVFP4.
    pub(super) qkvz_q2: Option<crate::weight_map::PackedQ2Weight>,
    /// 2026-09-25: Packed-Q2 kernels: `q2_0_gemv_vec` for decode, and for
    /// `qkvz_q2_prefill_gemm` the per-call BF16 dequant, the two MMQ GEMMs and
    /// the q8_1 activation quantizer. The last three are 0 until
    /// `set_packed_q2_qkvz` looks them up.
    pub(super) q2_0_gemv_k: KernelHandle,
    pub(super) dequant_q2_0_gn_k: KernelHandle,
    pub(super) q2_0_mmq_nc_k: KernelHandle,
    pub(super) q2_0_mmq_wc_k: KernelHandle,
    pub(super) q4k_quant_act_k: KernelHandle,
    /// 2026-09-25: The QKVZ weight is already in `[Q | K | V | Z]` row order
    /// (`new_sequential`); no path launches `deinterleave_qkvz` then.
    pub(super) sequential_qkvz: bool,
    /// 2026-09-25: SM count from `GpuBackend::sm_count` at construction;
    /// `ms_proj_gemm` compares grid widths with it.
    pub(super) sm_count: u32,
    pub(super) rms_norm_residual_k: KernelHandle,
    pub(super) gated_rms_norm_k: KernelHandle,
    pub(super) gated_rms_norm_f32_k: KernelHandle,
    /// 2026-09-25: `gated_rms_norm_f32_input_strided`, one launch for all
    /// sequences in `ssm_batched_recurrent.rs`. 0 when absent and on a
    /// `gdn_norm_sigmoid` model; that file then loops per sequence.
    pub(super) gated_rms_norm_f32_strided_k: KernelHandle,
    pub(super) dense_gemv_k: KernelHandle,
    /// 2026-09-25: `dense_gemv_bf16_batch2`, the two-row GEMV over the BF16
    /// `in_proj_qkvz` in the batched decode projection.
    pub(super) dense_gemv_batch2_k: KernelHandle,
    pub(super) w4a16_gemv_k: KernelHandle,
    /// 2026-09-25: `w4a16_gemv_sw`; `w4a16_decode_gemv` picks it over
    /// `w4a16_gemv_k` only when `use_gemv_sw` accepts the lever and the handle.
    pub(super) w4a16_gemv_sw_k: KernelHandle,
    pub(super) w8a16_gemv_k: KernelHandle,
    pub(super) w4a16_gemv_qkvz_k: KernelHandle,
    pub(super) deinterleave_k: KernelHandle,
    pub(super) conv1d_k: KernelHandle,
    pub(super) conv1d_l2norm_k: KernelHandle,
    pub(super) conv1d_l2norm_f32_k: KernelHandle,
    /// 2026-09-25: `causal_conv1d_update_l2norm_f32_strided`, one launch for all
    /// sequences in `ssm_batched_recurrent.rs`; with 0 that file loops per
    /// sequence.
    pub(super) conv1d_l2norm_f32_strided_k: KernelHandle,
    pub(super) gdn_k: KernelHandle,
    pub(super) gdn_f32_k: KernelHandle,
    pub(super) gdn_f32_norm_k: KernelHandle,
    pub(super) gdn_f32_conv_norm_k: KernelHandle,
    pub(super) gdn_f32_strided_k: KernelHandle,
    pub(super) gdn_f32_strided_norm_k: KernelHandle,
    pub(super) gdn_f32_strided_norm_half_k: KernelHandle,
    /// 2026-09-25: Shared-memory-staged strided decode kernel, selected in
    /// `ssm_batched_recurrent.rs` only when kd == vd == 128 and
    /// `METRALE_GDN_SMEM_STAGE` is set (any value).
    pub(super) gdn_f32_strided_norm_smem_k: KernelHandle,
    /// 2026-09-25: FP16 h-state variant of `gdn_f32_strided_norm_half_k`; the
    /// strided recurrence in `ssm_batched_recurrent.rs` launches it, and no
    /// FP32 variant, under the FP16 h-state.
    pub(super) gdn_f16_strided_norm_half_k: KernelHandle,
    /// 2026-09-25: FP16 h-state variant of `gdn_f32_norm_k`, used by
    /// `ssm_forward` and the per-sequence arm of `ssm_batched_recurrent.rs`.
    pub(super) gdn_f16_norm_k: KernelHandle,
    pub(super) ba_gates_k: KernelHandle,
    pub(super) residual_add_k: KernelHandle,
    pub(super) l2_norm_k: KernelHandle,
    pub(super) residual_add_rms_norm_k: KernelHandle,
    /// 2026-09-25: `residual_add_rms_norm_gatef32`; single-token decode uses it
    /// when FP32 routing is active for the FFN and the handle is non-zero.
    pub(super) residual_add_rms_norm_gatef32_k: KernelHandle,
    pub(super) gated_rms_norm_prefill_k: KernelHandle,
    pub(super) w4a16_gemm_k: KernelHandle,
    pub(super) w4a16_gemm_t_k: KernelHandle,
    pub(super) w4a16_gemm_t_k64_k: KernelHandle,
    /// 2026-09-25: `w4a16_gemm_t_k64_n64_p3`, the narrow-N deep-K GEMM; 0 when
    /// absent or when `METRALE_NO_K64_N64` is set (any value).
    pub(super) w4a16_gemm_t_k64_n64_k: KernelHandle,
    pub(super) w4a16_gemm_t_m128_k: KernelHandle,
    pub(super) w4a16_gemm_t_m128_v2_k: KernelHandle,
    pub(super) w4a16_gemv_batch2_k: KernelHandle,
    pub(super) dense_gemm_k: KernelHandle,
    pub(super) dense_gemm_pipelined_k: KernelHandle,
    pub(super) gdn_prefill_k: KernelHandle,
    pub(super) gdn_prefill_split_k: KernelHandle,
    pub(super) gdn_prefill_split4_k: KernelHandle,
    pub(super) gdn_prefill_persistent_k: KernelHandle,
    pub(super) gdn_prefill_persistent_wy4_k: KernelHandle,
    /// 2026-09-25: `gated_delta_rule_prefill_regresident`, used by
    /// `trait_prefill_recur.rs` when the `gdn_regresident` lever is on (the
    /// default; `METRALE_NO_GDN_REGRESIDENT=1` turns it off), kd == vd == 128
    /// and the handle is non-zero.
    pub(super) gdn_prefill_regresident_k: KernelHandle,
    /// 2026-09-25: `gated_delta_rule_recompute_wu`, the first kernel of the FLA
    /// chunked prefill (`ops::gdn_prefill_fla`), which then runs a state spine
    /// and `chunk_fwd_o`.
    pub(super) gdn_prefill_fla_recompute_wu_k: KernelHandle,
    /// 2026-09-25: Hopper twins of `recompute_wu` and `chunk_fwd_o`
    /// (`init_kernels::prefill_wu_hopper_k`, `prefill_fwd_o_hopper_k`); 0 on
    /// targets without the Hopper sources, where the parents run.
    pub(super) gdn_prefill_fla_recompute_wu_hopper_k: KernelHandle,
    pub(super) gdn_prefill_fla_chunk_fwd_o_hopper_k: KernelHandle,
    pub(super) gdn_prefill_fla_chunk_delta_h_k: KernelHandle,
    /// 2026-09-25: `gated_delta_rule_chunk_delta_h_tc_vblock`;
    /// `ops::gdn_prefill_fla` uses it only when the fused spine is not used and
    /// `METRALE_GDN_TC_VBLOCK=1`.
    #[allow(dead_code)]
    pub(super) gdn_prefill_fla_chunk_delta_h_tc_vblock_k: KernelHandle,
    /// 2026-09-25: The tensor-core state spine, `ops::GDN_TC_SPINE_ENTRY`
    /// (`init_kernels::gdn_prefill_tc_kernel`); 0 unless the target default
    /// `gdn_prefill_tc` is on and the target compiles the module.
    pub(super) gdn_prefill_fla_chunk_delta_h_tcfuse_k: KernelHandle,
    /// 2026-09-25: The scalar fused state spine that
    /// `init_kernels::fused_spine_kernel` picks.
    pub(super) gdn_prefill_fla_chunk_delta_h_fused_k: KernelHandle,
    /// 2026-09-25: `gated_delta_rule_chunk_delta_h_tma`, requested by
    /// `METRALE_GDN_TMA=1` (`ops::ssm_gdn_a3`); 0 when absent.
    pub(super) gdn_prefill_fla_chunk_delta_h_tma_k: KernelHandle,
    pub(super) gdn_prefill_fla_chunk_fwd_o_k: KernelHandle,
    /// 2026-09-25: `gated_delta_rule_prefill_wy64` (module
    /// `gated_delta_rule_wy64_prefill`), which takes WY chunks of 32 tokens
    /// with H in shared memory.
    pub(super) gdn_prefill_wy32_k: KernelHandle,
    // 2026-09-25: Multi-stream GDN prefill kernels for
    // `trait_prefill_gdn/batched.rs`; 0 when the target lacks them.
    pub(super) gdn_prefill_wy32_batched_k: KernelHandle,
    pub(super) gdn_prefill_persistent_batched_k: KernelHandle,
    pub(super) gdn_prefill_persistent_wy4_batched_k: KernelHandle,
    pub(super) gdn_prefill_split4_batched_k: KernelHandle,
    pub(super) compute_gdn_gates_k: KernelHandle,
    pub(super) ba_gates_prefill_k: KernelHandle,
    /// 2026-09-25: Hopper twin of `ba_gates_prefill_k`
    /// (`init_kernels::ba_gates_hopper_k`); `ops::ba_gates_pick` chooses
    /// between the two per launch.
    pub(super) ba_gates_prefill_hopper_k: KernelHandle,
    pub(super) conv1d_prefill_k: KernelHandle,
    /// 2026-09-25: `causal_conv1d_update_prefill_tp`; 0 when absent.
    pub(super) conv1d_prefill_tp_k: KernelHandle,
    pub(super) gdn_chunk2_k: KernelHandle,
    pub(super) conv1d_chunk2_k: KernelHandle,
    pub(super) gdn_chunk3_k: KernelHandle,
    pub(super) w4a16_gemv_batch3_k: KernelHandle,
    // 2026-09-25: The `w4a16_gemv_batch{4..8}` tiers, and below them
    // `w4a16_gemv_batch16`.
    pub(super) w4a16_batchm: W4a16BatchmTiers,
    pub(super) w4a16_gemv_batch16_k: KernelHandle,
    pub(super) gdn_wy2_k: KernelHandle,
    /// 2026-09-25: Register-resident variant of `gdn_wy2_k`. `wy2_kernel` picks
    /// it when kd == vd == 128, `n >= wy_resident_min_width()`, the handle is
    /// non-zero and `METRALE_NO_GDN_WY2_RESIDENT` is unset, and logs its choice
    /// at the first dispatch.
    pub(super) gdn_wy2_resident_k: KernelHandle,
    pub(super) gdn_wy3_k: KernelHandle,
    /// 2026-09-25: Register-resident variant of `gdn_wy3_k`, picked by
    /// `wy3_kernel` under the same conditions with
    /// `METRALE_NO_GDN_WY3_RESIDENT`.
    pub(super) gdn_wy3_resident_k: KernelHandle,
    pub(super) gdn_wy4_k: KernelHandle,
    /// 2026-09-25: Write-on-accept K=4 verify kernel, with its fold and
    /// flag-clear kernels below (`woa::woa_kernels`, each 0 when absent).
    /// `woa_decision` runs it only when all three resolved, the call requests
    /// it, `METRALE_GDN_WOA=1`, the h-state is FP32, kd == vd == 128 and a stash
    /// is bound for at least `n` sequences.
    pub(super) gdn_wy4_woa_k: KernelHandle,
    pub(super) gdn_wy4_fold_k: KernelHandle,
    pub(super) gdn_wy4_clear_k: KernelHandle,
    /// 2026-09-25: Set by `gdn_woa_bind`, 0 until then: the device address of
    /// the engaged word, this layer's stash slab, and the number of sequences
    /// the slab holds.
    pub(super) woa_flag: std::sync::atomic::AtomicU64,
    pub(super) woa_stash: std::sync::atomic::AtomicU64,
    pub(super) woa_seqs: std::sync::atomic::AtomicUsize,
    pub(super) woa_dims: [usize; 4],
    /// 2026-09-25: FP16 h-state variants of the WY verify kernels, 0 when
    /// absent. Under `ssm_h_fp16_enabled()`, `wy2_kernel` and `wy3_kernel`
    /// return these, even when 0, and never the FP32 kernels.
    pub(super) gdn_wy2_f16_k: KernelHandle,
    pub(super) gdn_wy2_resident_f16_k: KernelHandle,
    pub(super) gdn_wy3_f16_k: KernelHandle,
    pub(super) gdn_wy3_resident_f16_k: KernelHandle,
    pub(super) gdn_wy4_f16_k: KernelHandle,
    /// 2026-09-25: The h-state width converters (`ssm_h_dtype`). Under the
    /// 2-byte-sized pool, `prefill_h_begin` / `prefill_h_end` (`ssm_h_fp16.rs`)
    /// widen a sequence's slot into its FP32 staging buffer before the FP32
    /// prefill kernels and narrow it back after; a 0 handle there is an error.
    pub(super) ssm_h_f16_to_f32_k: KernelHandle,
    pub(super) ssm_h_f32_to_f16_k: KernelHandle,
    /// 2026-09-25: Fused K=2 verify conv and norm kernels, used only when both
    /// are non-zero and `METRALE_GDN_FUSED_VERIFY=1` (`fused_verify_k2_enabled`).
    pub(super) gdn_verify_fused_conv_k2_k: KernelHandle,
    pub(super) gdn_verify_fused_norm_k2_k: KernelHandle,
    /// 2026-09-25: Fused verify conv of the wyN / wy17 arm
    /// (`decode_batched_conv_gdn_wyn`), used when non-zero, the conv
    /// intermediates are contiguous and `METRALE_GDN_FUSED_CONV17` is not `0`;
    /// otherwise that arm runs the conv per token.
    pub(super) gdn_verify_fused_conv_kn_k: KernelHandle,
    /// 2026-09-25: Multi-sequence variant of the above, for
    /// `trait_decode_batched_conv_gdn_multi.rs`; 0 when absent.
    pub(super) gdn_verify_fused_conv_kn_batched_k: KernelHandle,
    /// 2026-09-25: Exact-verify kernels that write the per-token h-state
    /// snapshot inline, and the FP32-output fused verify conv. With a 0 handle,
    /// `trait_decode_batched_conv_gdn_exact.rs` copies snapshots with
    /// `copy_d2d_async` and `_multi_exact.rs` declines to the per-sequence loop.
    pub(super) gdn_f32_norm_snap_k: KernelHandle,
    pub(super) gdn_f32_strided_norm_snap_k: KernelHandle,
    pub(super) gdn_verify_fused_conv_kn_f32_k: KernelHandle,
    /// 2026-09-25: `gated_delta_rule_wy17`, used for K = 17 when non-zero, the
    /// `gdn_wy17` lever is on and the h-state is FP32; otherwise K = 17 takes
    /// the sequential path.
    pub(super) gdn_wy17_k: KernelHandle,
    /// 2026-09-25: The K = 5..16 verify kernels (`init_kernels::wyn_kernels`),
    /// index = K - 5, read by `wyn_kernel` when the `gdn_wyn` lever is on
    /// (the default; `METRALE_GDN_WYN=0` turns it off).
    pub(super) gdn_wyn_k: [KernelHandle; 12],
    /// 2026-09-25: FP16 h-state variants of `gdn_wyn_k`, same index. Under the
    /// FP16 h-state a 0 entry makes `decode_batched_conv_gdn` return an error
    /// rather than run an FP32 kernel on the FP16
    /// state). provenance-id: 526f6e616c6420522e205374657369616b
    pub(super) gdn_wyn_f16_k: [KernelHandle; 12],
    /// 2026-09-25: Pointer-table variants for the cross-sequence batched verify
    /// (`init_kernels::wyn_table_kernels`, read by `wyn_table_kernel`), index =
    /// K - 5.
    pub(super) gdn_wyn_table_k: [KernelHandle; 12],
    /// 2026-09-25: FP16 variants of `gdn_wyn_table_k`, same index.
    pub(super) gdn_wyn_f16_table_k: [KernelHandle; 12],
    // 2026-09-25: Per-sequence FP32 state sizes in bytes: `nv * kd * vd` values
    // of h, `conv_dim * d_conv` of conv state.
    pub(super) h_state_bytes: usize,
    pub(super) conv_state_bytes: usize,
    // 2026-09-25: Unscaled FP8 weights for the `fp8_gemm_n128` arms
    // (`set_fp8_prefill_only_weights`, `predequant_for_prefill`).
    pub(super) qkvz_fp8: Option<DevicePtr>,
    pub(super) out_proj_fp8: Option<DevicePtr>,
    pub(super) fp8_gemm_k: KernelHandle,
    pub(super) fp8_gemm_t_m128_k: KernelHandle,
    // 2026-09-25: Block-scaled W8A16 GEMMs, which the prefill tries before the
    // unscaled `fp8_gemm_n128` arm.
    pub(super) w8a16_gemm_k: KernelHandle,
    // 2026-09-25: `w8a16_gemm_pipelined`; no environment variable gates it.
    pub(super) w8a16_gemm_pipelined_k: KernelHandle,
    // 2026-09-25: 32-row M-tile variant; `ops::w8a16_gemm_pipelined_by_m`
    // chooses between it and `w8a16_gemm_pipelined` by row count.
    pub(super) w8a16_gemm_pipelined_m32_k: KernelHandle,
    // 2026-09-25: `w8a16_gemv_batch4` (M <= 4) and, below it,
    // `w8a16_gemv_batch16`, both from module `w8a16_gemv_batch4`.
    pub(super) w8a16_gemv_batch4_k: KernelHandle,
    pub(super) w8a16_gemv_batch16_k: KernelHandle,
    pub(super) w8a16_gemm_t_k: KernelHandle,
    // 2026-09-25: W8A8 prefill: `per_token_group_quant_fp8` writes FP8
    // activations with one FP32 scale per token per 128 columns, and
    // `fp8_gemm_t_blockscaled` applies both scale sets in an FP32 epilogue.
    // The QKVZ arm runs unless `METRALE_FP8_SINGLE_SCALE=1`; the `out_proj`
    // arm only under `METRALE_FP8_W8A8=1`.
    pub(super) per_token_group_quant_fp8_k: ops::Fp8ActQuant,
    pub(super) fp8_gemm_t_blockscaled_k: KernelHandle,
    /// 2026-09-25: `fp8_act_scale_to_kmajor`: copies the quantizer's
    /// `[M, K/128]` activation scales into the `[K/128, ceil16(M)]` layout the
    /// cuBLASLt arms read. With 0, both cuBLASLt W8A8 prefill arms decline under
    /// the default k-major layout.
    pub(super) fp8_act_scale_kmajor_k: KernelHandle,
}
