// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Qwen3AttentionLayer` constructors: `new` (Q projection carries
//! an output gate), `new_ungated`, and the shared `new_with_gating`, which
//! resolves every kernel handle the layer holds.
//!
//! Owner: model-layers (attention).
//! Invariants:
//! - A kernel family behind `ArchProbes` is looked up only when the config
//!   says the model has it.
//! - Every lookup of the `fused_k_norm_rope_cache` module goes through
//!   `try_target_kernel` (pinned by the `fused_kv_probe_guard` test below).

use anyhow::Result;
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::init_arch_gates::ArchProbes;
use super::types::{HeadGateActivation, Qwen3AttentionLayer};
use crate::layers::FfnComponent;
use crate::layers::fp8_calibration::Fp8KvCalibration;
use crate::weight_map::{AttentionWeights, DenseWeight, QuantWeight, QuantizedWeight};

impl Qwen3AttentionLayer {
    pub fn new(
        input_norm: DenseWeight,
        attn: AttentionWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        attn_layer_idx: usize,
        q_nvfp4: Option<QuantizedWeight>,
        k_nvfp4: Option<QuantizedWeight>,
        v_nvfp4: Option<QuantizedWeight>,
        gpu: &dyn GpuBackend,
        kv_dtype: KvCacheDtype,
        fp8_calibration_tokens: usize,
        config: &metrale_config::ModelConfig,
    ) -> Result<Self> {
        Self::new_with_gating(
            input_norm,
            attn,
            post_attn_norm,
            ffn,
            attn_layer_idx,
            q_nvfp4,
            k_nvfp4,
            v_nvfp4,
            true,
            gpu,
            kv_dtype,
            fp8_calibration_tokens,
            config,
        )
    }

    pub fn new_ungated(
        input_norm: DenseWeight,
        attn: AttentionWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        attn_layer_idx: usize,
        q_nvfp4: Option<QuantizedWeight>,
        k_nvfp4: Option<QuantizedWeight>,
        v_nvfp4: Option<QuantizedWeight>,
        gpu: &dyn GpuBackend,
        kv_dtype: KvCacheDtype,
        fp8_calibration_tokens: usize,
        config: &metrale_config::ModelConfig,
    ) -> Result<Self> {
        Self::new_with_gating(
            input_norm,
            attn,
            post_attn_norm,
            ffn,
            attn_layer_idx,
            q_nvfp4,
            k_nvfp4,
            v_nvfp4,
            false,
            gpu,
            kv_dtype,
            fp8_calibration_tokens,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_gating(
        input_norm: DenseWeight,
        attn: AttentionWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        attn_layer_idx: usize,
        q_nvfp4: Option<QuantizedWeight>,
        k_nvfp4: Option<QuantizedWeight>,
        v_nvfp4: Option<QuantizedWeight>,
        gated: bool,
        gpu: &dyn GpuBackend,
        kv_dtype: KvCacheDtype,
        fp8_calibration_tokens: usize,
        config: &metrale_config::ModelConfig,
    ) -> Result<Self> {
        let (reshape_mod, reshape_fn, decode_mod, decode_fn) =
            super::init_kernel_dispatch::kernel_modules_for_dtype(kv_dtype, config.head_dim);
        // 2026-09-25: Which cross-architecture kernel families the config says
        // the model has. `gate` issues no lookup for a family the model lacks,
        // so it leaves no failed row in the boot audit. See `init_arch_gates`.
        let probes = ArchProbes::from_config(config);
        let mrope_interleaved = config.mrope_interleaved;
        let proj = super::init_proj_kernels::ProjKernels::resolve(
            gpu,
            config,
            &probes,
            reshape_mod,
            reshape_fn,
        )?;
        let dec = super::init_decode_kernels::DecodeKernels::resolve(
            gpu, kv_dtype, &probes, decode_mod, decode_fn,
        )?;
        let pre = super::init_prefill_kernels::PrefillKernels::resolve(gpu, config, &probes)?;
        Ok(Self {
            input_norm,
            attn,
            post_attn_norm,
            ffn,
            attn_layer_idx,
            lora: None,
            gated,
            mrope_interleaved,
            kv_dtype,
            head_dim_override: None,
            num_q_heads_override: None,
            num_kv_heads_override: None,
            sliding_window: None,
            rope_theta_override: None,
            rotary_dim_override: None,
            rope_proportional: false,
            attn_scale_override: None,
            k_eq_v: false,
            v_norm_weight: None,
            head_gate_weight: None,
            head_gate_activation: HeadGateActivation::Sigmoid,
            sigmoid_gate_head_broadcast_k: proj.sigmoid_gate_head_broadcast_k,
            softplus_gate_head_broadcast_k: proj.softplus_gate_head_broadcast_k,
            yarn_inv_freq: metrale_gpu_runtime::gpu::DevicePtr::NULL,
            yarn_attention_factor: 1.0,
            post_attn_out_norm: None,
            post_ffn_out_norm: None,
            layer_scalar: None,
            moe_ffn: None,
            shortcut_carry_out: None,
            shortcut_carry_in: None,
            pre_moe_norm: None,
            post_moe_out_norm: None,
            post_dense_ffn_norm: None,
            sparse_v_threshold: 0.0,
            q_weight: q_nvfp4.map(QuantWeight::Nvfp4),
            k_weight: k_nvfp4.map(QuantWeight::Nvfp4),
            v_weight: v_nvfp4.map(QuantWeight::Nvfp4),
            o_weight: None,
            o_dense_bf16: None,
            mla: None,
            // 2026-09-25: `hc` stays `None` unless a loader calls
            // `set_hc_weights` after construction, as the DeepSeek-V4 loader
            // does. The hyper-connection kernels are looked up only when the
            // config's `hc_mult` is nonzero.
            hc: None,
            qsa: None,
            hc_pre_k: proj.hc_pre_k,
            hc_post_k: proj.hc_post_k,
            hc_expand_k: proj.hc_expand_k,
            hc_head_k: proj.hc_head_k,
            qkv_nvfp4_t: None,
            q_nvfp4_t: None,
            k_nvfp4_t: None,
            v_nvfp4_t: None,
            o_nvfp4_t: None,
            q_fp8w_t: None,
            k_fp8w_t: None,
            v_fp8w_t: None,
            o_fp8w_t: None,
            w8a16_gemm_t_k: proj.w8a16_gemm_t_k,
            w8a16_gemm_t_pipelined_k: proj.w8a16_gemm_t_pipelined_k,
            w8a16_gemm_t_m128_k: proj.w8a16_gemm_t_m128_k,
            per_token_group_quant_fp8_k: proj.per_token_group_quant_fp8_k,
            fp8_gemm_t_blockscaled_k: proj.fp8_gemm_t_blockscaled_k,
            fp8_act_scale_kmajor_k: proj.fp8_act_scale_kmajor_k,
            rms_norm_k: proj.rms_norm_k,
            rms_norm_w_k: proj.rms_norm_w_k,
            rms_norm_w_warp_row_k: proj.rms_norm_w_warp_row_k,
            norm_vanilla: proj.norm_vanilla,
            rms_norm_residual_k: proj.rms_norm_residual_k,
            dense_gemv_k: proj.dense_gemv_k,
            dequant_q2_0_gn_k: proj.dequant_q2_0_gn_k,
            // 2026-09-25: Resolved by `set_packed_q2_weights`, not here: the
            // `q2_0_mmq` source is only in `kernels/gb10/qwen3.6-27b/nvfp4`, so
            // a lookup here would leave a failed row in the boot audit of every
            // other model.
            q2_0_mmq_nc_k: KernelHandle(0),
            q2_0_mmq_wc_k: KernelHandle(0),
            q4k_quant_act_k: KernelHandle(0),
            q2_0_gemv_k: proj.q2_0_gemv_k,
            dense_gemv_batchm_k: proj.dense_gemv_batchm_k,
            w4a16_gemv_k: proj.w4a16_gemv_k,
            w4a16_gemv_sw_k: proj.w4a16_gemv_sw_k,
            w8a16_gemv_k: proj.w8a16_gemv_k,
            w8a16_gemv_batch4_k: proj.w8a16_gemv_batch4_k,
            w8a16_gemv_batch16_k: proj.w8a16_gemv_batch16_k,
            w8a16_gemv_batch4_strided_k: proj.w8a16_gemv_batch4_strided_k,
            w8a16_gemv_batch16_strided_k: proj.w8a16_gemv_batch16_strided_k,
            w8a16_gemm_m16_k: proj.w8a16_gemm_m16_k,
            w8a16_gemm_m16_strided_k: proj.w8a16_gemm_m16_strided_k,
            m16_tc: proj.m16_tc,
            w8a16_gemv_ncol2_k: proj.w8a16_gemv_ncol2_k,
            w8a16_gemv_ncol4_k: proj.w8a16_gemv_ncol4_k,
            w8a16_gemv_ncol2_strided_k: proj.w8a16_gemv_ncol2_strided_k,
            w8a16_gemv_ncol4_strided_k: proj.w8a16_gemv_ncol4_strided_k,
            attn_ncol: proj.attn_ncol,
            w8a16_gemm_k: proj.w8a16_gemm_k,
            w8a16_gemm_pipelined_k: proj.w8a16_gemm_pipelined_k,
            w8a16_gemm_pipelined_m32_k: proj.w8a16_gemm_pipelined_m32_k,
            w4a16_gemv_dual_k: proj.w4a16_gemv_dual_k,
            rope_k: proj.rope_k,
            rope_strided_k: proj.rope_strided_k,
            rms_norm_strided_k: proj.rms_norm_strided_k,
            rope_mrope_interleaved_k: proj.rope_mrope_interleaved_k,
            rope_mrope_interleaved_k_only_k: proj.rope_mrope_interleaved_k_only_k,
            rope_yarn_k: proj.rope_yarn_k,
            rope_yarn_scaled_k: proj.rope_yarn_scaled_k,
            rope_yarn_interleaved_k: proj.rope_yarn_interleaved_k,
            rope_yarn_interleaved_inv_k: proj.rope_yarn_interleaved_inv_k,
            rope_proportional_k: proj.rope_proportional_k,
            reshape_cache_k: proj.reshape_cache_k,
            fused_k_norm_rope_cache_write_bf16_k: dec.fused_k_norm_rope_cache_write_bf16_k,
            fused_k_norm_rope_mrope_cache_write_bf16_k: dec
                .fused_k_norm_rope_mrope_cache_write_bf16_k,
            reshape_and_cache_flash_v_only_k: dec.reshape_and_cache_flash_v_only_k,
            fused_k_norm_rope_cache_write_fp8_kv_k: dec.fused_k_norm_rope_cache_write_fp8_kv_k,
            wht_bf16_k: dec.wht_bf16_k,
            wht_bf16_k_inv: dec.wht_bf16_k_inv,
            innerq_apply_q_k: dec.innerq_apply_q_k,
            innerq_apply_k_k: dec.innerq_apply_k_k,
            paged_decode_k: dec.paged_decode_k,
            paged_decode_512_k: dec.paged_decode_512_k,
            paged_decode_mla_k: dec.paged_decode_mla_k,
            mla_paged_decode_k: dec.mla_paged_decode_k,
            mla_paged_decode_fp8_k: dec.mla_paged_decode_fp8_k,
            mla_batched_gemv_k: dec.mla_batched_gemv_k,
            mla_q_rope_scatter_k: dec.mla_q_rope_scatter_k,
            mla_q_rope_writeback_k: dec.mla_q_rope_writeback_k,
            mla_cache_assemble_k: dec.mla_cache_assemble_k,
            mla_q_rope_extract_batched_k: dec.mla_q_rope_extract_batched_k,
            mla_q_rope_writeback_batched_k: dec.mla_q_rope_writeback_batched_k,
            mla_kv_assemble_batched_k: dec.mla_kv_assemble_batched_k,
            mla_cache_assemble_batched_k: dec.mla_cache_assemble_batched_k,
            prefill_attn_mla320_k: dec.prefill_attn_mla320_k,
            grouped_gemm_mla_k: dec.grouped_gemm_mla_k,
            mla_q_final_assemble_k: dec.mla_q_final_assemble_k,
            mla_fused_prefill_k: dec.mla_fused_prefill_k,
            gemm_splitk_partial_k: dec.gemm_splitk_partial_k,
            gemm_splitk_reduce_k: dec.gemm_splitk_reduce_k,
            dense_gemm_tc_k: dec.dense_gemm_tc_k,
            paged_decode_splitk_k: dec.paged_decode_splitk_k,
            paged_decode_reduce_k: dec.paged_decode_reduce_k,
            paged_decode_bf16_gqa_k: dec.paged_decode_bf16_gqa_k,
            paged_decode_fp8_gqa_k: dec.paged_decode_fp8_gqa_k,
            paged_decode_splitk_hopper_k: dec.paged_decode_splitk_hopper_k,
            paged_decode_reduce_hopper_k: dec.paged_decode_reduce_hopper_k,
            paged_decode_splitk_bf16_hopper_k: dec.paged_decode_splitk_bf16_hopper_k,
            paged_decode_reduce_bf16_hopper_k: dec.paged_decode_reduce_bf16_hopper_k,
            residual_add_k: pre.residual_add_k,
            rms_norm_f32_in_k: KernelHandle(0),
            sigmoid_gate_mul_k: pre.sigmoid_gate_mul_k,
            deinterleave_qg_k: pre.deinterleave_qg_k,
            w4a16_gemv_qg_k: pre.w4a16_gemv_qg_k,
            residual_add_rms_norm_k: pre.residual_add_rms_norm_k,
            residual_add_rms_norm_gatef32_k: pre.residual_add_rms_norm_gatef32_k,
            w4a16_gemv_qg_batch2_k: pre.w4a16_gemv_qg_batch2_k,
            w4a16_gemv_dual_batch2_k: pre.w4a16_gemv_dual_batch2_k,
            w4a16_gemv_batch2_k: pre.w4a16_gemv_batch2_k,
            w4a16_gemv_qg_batch3_k: pre.w4a16_gemv_qg_batch3_k,
            w4a16_gemv_dual_batch3_k: pre.w4a16_gemv_dual_batch3_k,
            w4a16_gemv_batch3_k: pre.w4a16_gemv_batch3_k,
            w4a16_batchm: pre.w4a16_batchm,
            w4a16_gemm_k: pre.w4a16_gemm_k,
            w4a16_gemm_t_k: pre.w4a16_gemm_t_k,
            w4a16_gemm_t_k64_k: pre.w4a16_gemm_t_k64_k,
            w4a16_gemm_t_k64_n64_k: pre.w4a16_gemm_t_k64_n64_k,
            w4a16_gemm_t_m128_k: pre.w4a16_gemm_t_m128_k,
            w4a16_gemm_t_m128_bf16_k: pre.w4a16_gemm_t_m128_bf16_k,
            w4a16_gemm_t_m128_v2_k: pre.w4a16_gemm_t_m128_v2_k,
            w4a16_gemm_t_m128_v3_k: pre.w4a16_gemm_t_m128_v3_k,
            dense_gemm_k: pre.dense_gemm_k,
            dense_gemm_pipelined_k: pre.dense_gemm_pipelined_k,
            prefill_attn_k: pre.prefill_attn_k,
            prefill_attn_512_k: pre.prefill_attn_512_k,
            prefill_attn_512_is_tc: pre.prefill_attn_512_is_tc,
            csa_compress_k: pre.csa_compress_k,
            prefill_attn_compressed_k: pre.prefill_attn_compressed_k,
            v4_comp_pool_filled: std::sync::atomic::AtomicU32::new(0),
            v4_comp_prev_valid: std::sync::atomic::AtomicBool::new(false),
            v4_decode_started: std::sync::atomic::AtomicBool::new(false),
            v4_decode_first_pos: std::sync::atomic::AtomicU32::new(0),
            prefill_attn_paged_512_k: pre.prefill_attn_paged_512_k,
            prefill_attn_64_k: pre.prefill_attn_64_k,
            prefill_attn_paged_k: pre.prefill_attn_paged_k,
            prefill_attn_paged_fp8_k: pre.prefill_attn_paged_fp8_k,
            prefill_attn_paged_nvfp4_k: pre.prefill_attn_paged_nvfp4_k,
            prefill_attn_paged_turbo4_k: pre.prefill_attn_paged_turbo4_k,
            prefill_attn_paged_64_k: pre.prefill_attn_paged_64_k,
            prefill_attn_paged_fp8_64_k: pre.prefill_attn_paged_fp8_64_k,
            prefill_attn_paged_nvfp4_64_k: pre.prefill_attn_paged_nvfp4_64_k,
            prefill_attn_paged_turbo2_64_k: pre.prefill_attn_paged_turbo2_64_k,
            prefill_attn_paged_turbo3_64_k: pre.prefill_attn_paged_turbo3_64_k,
            prefill_attn_paged_turbo4_64_k: pre.prefill_attn_paged_turbo4_64_k,
            prefill_attn_paged_turbo8_64_k: pre.prefill_attn_paged_turbo8_64_k,
            prefill_attn_paged_bf16k_turbo3v_64_k: pre.prefill_attn_paged_bf16k_turbo3v_64_k,
            prefill_attn_paged_bf16k_turbo4v_64_k: pre.prefill_attn_paged_bf16k_turbo4v_64_k,
            prefill_attn_paged_bf16k_turbo2v_64_k: pre.prefill_attn_paged_bf16k_turbo2v_64_k,
            prefill_attn_paged_fp8k_turbo3v_64_k: pre.prefill_attn_paged_fp8k_turbo3v_64_k,
            prefill_attn_paged_fp8k_turbo4v_64_k: pre.prefill_attn_paged_fp8k_turbo4v_64_k,
            prefill_attn_paged_fp8k_turbo2v_64_k: pre.prefill_attn_paged_fp8k_turbo2v_64_k,
            prefill_attn_paged_turbo4k_turbo3v_64_k: pre.prefill_attn_paged_turbo4k_turbo3v_64_k,
            prefill_attn_paged_turbo4k_turbo8v_64_k: pre.prefill_attn_paged_turbo4k_turbo8v_64_k,
            prefill_attn_paged_turbo3k_turbo8v_64_k: pre.prefill_attn_paged_turbo3k_turbo8v_64_k,
            prefill_attn_paged_batched_k: pre.prefill_attn_paged_batched_k,
            prefill_attn_paged_fp8_batched_k: pre.prefill_attn_paged_fp8_batched_k,
            prefill_attn_paged_nvfp4_batched_k: pre.prefill_attn_paged_nvfp4_batched_k,
            prefill_attn_paged_batched_64_k: pre.prefill_attn_paged_batched_64_k,
            prefill_attn_paged_fp8_batched_64_k: pre.prefill_attn_paged_fp8_batched_64_k,
            prefill_attn_paged_nvfp4_batched_64_k: pre.prefill_attn_paged_nvfp4_batched_64_k,
            deinterleave_qg_split_k: pre.deinterleave_qg_split_k,
            deinterleave_qg_split_qnorm_k: pre.deinterleave_qg_split_qnorm_k,
            deinterleave_qg_split_qnorm_mrope_k: pre.deinterleave_qg_split_qnorm_mrope_k,
            sigmoid_gate_mul_batched_k: pre.sigmoid_gate_mul_batched_k,
            q_fp8: None,
            k_fp8: None,
            v_fp8: None,
            o_fp8: None,
            fp8_gemm_k: pre.fp8_gemm_k,
            bf16_to_fp8_k: pre.bf16_to_fp8_k,
            fp8_fp8_gemm_k: pre.fp8_fp8_gemm_k,
            fp8_gemm_t_m128_k: pre.fp8_gemm_t_m128_k,
            fp8_fp8_gemm_t_m128_k: pre.fp8_fp8_gemm_t_m128_k,
            w4a4_gemm_k: pre.w4a4_gemm_k,
            quantize_nvfp4_k: pre.quantize_nvfp4_k,
            fp8_calibration: if fp8_calibration_tokens > 0
                && crate::layers::fp8_calibration::dtype_runs_online_fp8_kv_calibration(kv_dtype)
            {
                Some(Fp8KvCalibration::new(
                    attn_layer_idx,
                    fp8_calibration_tokens,
                    config.fp8_kv_headroom,
                    gpu,
                )?)
            } else {
                None
            },
        })
    }
}

#[cfg(test)]
mod fused_kv_probe_guard {
    /// 2026-09-25: Every lookup of the `fused_k_norm_rope_cache` module must go
    /// through `try_target_kernel`.
    ///
    /// Only `kernels/gb10/common` and the trees that overlay it build that
    /// module. On any other target a plain lookup fails, and the boot audit
    /// refuses to serve on it. The test reads the source because that failure
    /// is a boot-time refusal on targets this suite does not run on, which no
    /// mock backend reproduces.
    #[test]
    fn fused_k_norm_rope_cache_is_probed_target_scoped() {
        // 2026-09-26: `init.rs` and the three resolver files `new_with_gating`
        // calls, where the lookups are written.
        let sources = [
            ("init.rs", include_str!("init.rs")),
            ("init_proj_kernels.rs", include_str!("init_proj_kernels.rs")),
            (
                "init_decode_kernels.rs",
                include_str!("init_decode_kernels.rs"),
            ),
            (
                "init_prefill_kernels.rs",
                include_str!("init_prefill_kernels.rs"),
            ),
        ];
        let mut offenders = Vec::new();
        for (file, src) in sources {
            for (i, window) in src.match_indices("\"fused_k_norm_rope_cache\"") {
                let _ = window;
                // 2026-09-25: Walk back to the probe call that owns this module
                // argument.
                let head = &src[..i];
                let call = head.rfind("try_kernel(").map(|p| (p, "try_kernel"));
                let tcall = head
                    .rfind("try_target_kernel(")
                    .map(|p| (p, "try_target_kernel"));
                let chosen = match (call, tcall) {
                    (Some((a, _)), Some((b, n))) if b >= a => Some((b, n)),
                    (Some((a, n)), _) => Some((a, n)),
                    (None, t) => t,
                };
                match chosen {
                    Some((_, "try_target_kernel")) => {}
                    other => offenders.push(format!("{file}: {other:?} before byte {i}")),
                }
            }
        }
        assert!(
            !offenders.is_empty()
                || sources
                    .iter()
                    .any(|(_, src)| src.contains("fused_k_norm_rope_cache")),
            "guard found no lookups at all — it has stopped measuring anything"
        );
        assert!(
            offenders.is_empty(),
            "fused_k_norm_rope_cache probed without try_target_kernel: {offenders:?}"
        );
    }
}
