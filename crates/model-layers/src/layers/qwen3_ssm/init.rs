// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Qwen3SsmLayer::new`, which looks up the layer's kernel handles
//! and sizes its per-sequence state.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants: none beyond the types.

use super::*;

impl Qwen3SsmLayer {
    pub fn new(
        input_norm: DenseWeight,
        ssm: SsmWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        qkvz_nvfp4: Option<QuantizedWeight>,
        config: &metrale_config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let nv = config.linear_num_value_heads;
        let vd = config.linear_value_head_dim;
        let nk = config.linear_num_key_heads;
        let kd = config.linear_key_head_dim;
        let d_conv = config.linear_conv_kernel_dim;

        let conv_dim = nk * kd * 2 + nv * vd;

        // 2026-09-25: Looked up before the struct literal because two fields use
        // it: `gdn_prefill_fla_chunk_delta_h_tcfuse_k` itself, and
        // `fused_spine_kernel`, whose route line names the tensor-core spine
        // when this handle is non-zero.
        let gdn_tc_spine = gdn_prefill_tc_kernel(gpu);
        let woa = super::woa::woa_kernels(gpu);

        Ok(Self {
            // 2026-09-25: The mHC weights arrive later through `set_hc_weights`;
            // `hc_kernel` looks the handles up only when `config.hc_mult > 0`.
            hc: None,
            ple: None,
            hc_pre_k: hc_kernel(config, gpu, "hc_pre"),
            hc_post_k: hc_kernel(config, gpu, "hc_post"),
            hc_expand_k: hc_kernel(config, gpu, "hc_expand"),
            input_norm,
            ssm,
            post_attn_norm,
            ffn,
            lora_out_proj: None,
            qkvz_nvfp4,
            qkvz_nvfp4_t: None,
            out_proj_nvfp4_t: None,
            out_proj_dense: None,
            qkvz_fp8w: None,
            out_proj_fp8w: None,
            qkvz_fp8w_rowwise: None,
            out_proj_fp8w_rowwise: None,
            qkvz_rowwise_bf16: std::sync::atomic::AtomicU64::new(0),
            out_proj_rowwise_bf16: std::sync::atomic::AtomicU64::new(0),
            qkvz_q2: None,
            q2_0_gemv_k: super::super::try_kernel(gpu, "q2_0_gemv_vec", "q2_0_gemv_vec"),
            dequant_q2_0_gn_k: super::super::try_kernel(
                gpu,
                "dequant_gguf_bf16",
                "dequant_q2_0_gn_to_bf16",
            ),
            // 2026-09-25: Looked up by `set_packed_q2_qkvz`, so a layer without a
            // packed weight issues no lookup for them.
            q2_0_mmq_nc_k: KernelHandle(0),
            q2_0_mmq_wc_k: KernelHandle(0),
            q4k_quant_act_k: KernelHandle(0),
            sequential_qkvz: false,
            // 2026-09-25: Read once from the driver; `ms_proj_gemm` compares grid
            // widths with it.
            sm_count: gpu.sm_count()?,
            rms_norm_residual_k: gpu.kernel("norm", "rms_norm_residual")?,
            // 2026-09-25: `config.gdn_norm_sigmoid` selects the sigmoid gated-norm
            // kernels here; no forward call site reads the flag.
            gated_rms_norm_k: if config.gdn_norm_sigmoid {
                gpu.kernel("gated_norm_sigmoid", "gated_rms_norm_sigmoid")?
            } else {
                gpu.kernel("norm", "gated_rms_norm")?
            },
            gated_rms_norm_f32_k: if config.gdn_norm_sigmoid {
                super::super::try_kernel(
                    gpu,
                    "gated_norm_sigmoid",
                    "gated_rms_norm_f32_input_sigmoid",
                )
            } else {
                super::super::try_kernel(gpu, "norm", "gated_rms_norm_f32_input")
            },
            // 2026-09-25: No sigmoid strided variant is looked up. With 0,
            // `trait_decode_multi_seq/ssm_batched_recurrent.rs` runs the gated
            // norm once per sequence.
            gated_rms_norm_f32_strided_k: if config.gdn_norm_sigmoid {
                KernelHandle(0)
            } else {
                super::super::try_kernel(gpu, "norm", "gated_rms_norm_f32_input_strided")
            },
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemv_batch2_k: gpu.kernel("dense_gemv_bf16_batch2", "dense_gemv_bf16_batch2")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: super::super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w8a16_gemv_k: gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
            w4a16_gemv_qkvz_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qkvz")?,
            deinterleave_k: gpu.kernel("ssm_preprocess", "deinterleave_qkvz")?,
            conv1d_k: gpu.kernel("causal_conv1d", "causal_conv1d_update")?,
            conv1d_l2norm_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?,
            // 2026-09-25: With 0, `trait_decode_multi_seq/ssm_batched_recurrent.rs`
            // runs the conv once per sequence.
            conv1d_l2norm_f32_strided_k: super::super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_l2norm_f32_strided",
            ),
            conv1d_l2norm_f32_k: {
                let h = super::super::try_kernel(
                    gpu,
                    "causal_conv1d",
                    "causal_conv1d_update_l2norm_f32",
                );
                if h.0 == 0 {
                    tracing::warn!(
                        "FP32 conv1d kernel not loaded; SSM uses BF16 conv \
                         output. Expect long-context coherence drift past ~8k \
                         tokens on this backend."
                    );
                }
                h
            },
            gdn_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_decode")?,
            gdn_f32_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32",
            ),
            gdn_f32_norm_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_norm",
            ),
            gdn_f32_conv_norm_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_conv_norm",
            ),
            gdn_f32_strided_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_strided",
            ),
            gdn_f32_strided_norm_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_strided_norm",
            ),
            gdn_f32_strided_norm_half_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_strided_norm_half",
            ),
            gdn_f32_strided_norm_smem_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_strided_norm_smem",
            ),
            gdn_f16_strided_norm_half_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f16_strided_norm_half",
            ),
            gdn_f16_norm_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f16_norm",
            ),
            ssm_h_f16_to_f32_k: super::super::try_kernel(
                gpu,
                "ssm_h_dtype",
                "ssm_h_state_f16_to_f32",
            ),
            ssm_h_f32_to_f16_k: super::super::try_kernel(
                gpu,
                "ssm_h_dtype",
                "ssm_h_state_f32_to_f16",
            ),
            ba_gates_k: gpu.kernel("ssm_preprocess", "dense_gemv_ba_gates")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            l2_norm_k: gpu.kernel("norm", "l2_norm_bf16")?,
            residual_add_rms_norm_k: gpu.kernel("norm", "residual_add_rms_norm")?,
            residual_add_rms_norm_gatef32_k: crate::layers::try_kernel(
                gpu,
                "norm",
                "residual_add_rms_norm_gatef32",
            ),
            gated_rms_norm_prefill_k: if config.gdn_norm_sigmoid {
                gpu.kernel("gated_norm_sigmoid", "gated_rms_norm_prefill_sigmoid")?
            } else {
                gpu.kernel("norm", "gated_rms_norm_prefill")?
            },
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_t_k: crate::layers::tgemm_kernel(gpu),
            w4a16_gemm_t_k64_k: crate::layers::k64_kernel(gpu)?,
            w4a16_gemm_t_k64_n64_k: crate::layers::k64_n64_kernel(gpu),
            w4a16_gemm_t_m128_k: gpu.kernel("w4a16", "w4a16_gemm_t_m128")?,
            // 2026-09-25: 0 unless `METRALE_W4A16_VARIANT` is `v2` or `v3`; then
            // `w4a16_v2_kernel` panics if the kernel is absent.
            w4a16_gemm_t_m128_v2_k: super::super::w4a16_v2_kernel(gpu),
            w4a16_gemv_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            dense_gemm_pipelined_k: super::super::try_kernel(
                gpu,
                "gemm",
                "dense_gemm_bf16_pipelined",
            ),
            gdn_prefill_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_prefill")?,
            gdn_prefill_split_k: gpu
                .kernel("gated_delta_rule", "gated_delta_rule_prefill_split")?,
            gdn_prefill_split4_k: gpu
                .kernel("gated_delta_rule", "gated_delta_rule_prefill_split4")?,
            gdn_prefill_persistent_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_persistent",
                "gated_delta_rule_prefill_persistent",
            ),
            gdn_prefill_persistent_wy4_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_persistent",
                "gated_delta_rule_prefill_persistent_wy4",
            ),
            gdn_prefill_regresident_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_regresident",
                "gated_delta_rule_prefill_regresident",
            ),
            gdn_prefill_fla_recompute_wu_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_fla",
                "gated_delta_rule_recompute_wu",
            ),
            gdn_prefill_fla_recompute_wu_hopper_k: init_kernels::prefill_wu_hopper_k(gpu),
            gdn_prefill_fla_chunk_fwd_o_hopper_k: init_kernels::prefill_fwd_o_hopper_k(gpu),
            gdn_prefill_fla_chunk_delta_h_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_fla",
                "gated_delta_rule_chunk_delta_h_ksplit",
            ),
            gdn_prefill_fla_chunk_delta_h_tc_vblock_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_fla",
                "gated_delta_rule_chunk_delta_h_tc_vblock",
            ),
            gdn_prefill_fla_chunk_delta_h_tcfuse_k: gdn_tc_spine,
            gdn_prefill_fla_chunk_delta_h_fused_k: fused_spine_kernel(gpu, gdn_tc_spine),
            gdn_prefill_fla_chunk_delta_h_tma_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_fla",
                "gated_delta_rule_chunk_delta_h_tma",
            ),
            gdn_prefill_fla_chunk_fwd_o_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_fla",
                "gated_delta_rule_chunk_fwd_o",
            ),
            gdn_prefill_wy32_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy64_prefill",
                "gated_delta_rule_prefill_wy64",
            ),
            gdn_prefill_wy32_batched_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy64_prefill",
                "gated_delta_rule_prefill_wy64_batched",
            ),
            gdn_prefill_persistent_batched_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_persistent",
                "gated_delta_rule_prefill_persistent_batched",
            ),
            gdn_prefill_persistent_wy4_batched_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_persistent",
                "gated_delta_rule_prefill_persistent_wy4_batched",
            ),
            gdn_prefill_split4_batched_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_prefill_split4_batched",
            ),
            compute_gdn_gates_k: gpu.kernel("ssm_preprocess", "compute_gdn_gates")?,
            ba_gates_prefill_k: gpu.kernel("ssm_preprocess", "dense_gemm_ba_gates_prefill")?,
            ba_gates_prefill_hopper_k: init_kernels::ba_gates_hopper_k(gpu),
            conv1d_prefill_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill")?,
            conv1d_prefill_tp_k: super::super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_prefill_tp",
            ),
            gdn_chunk2_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_chunk2")?,
            conv1d_chunk2_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_chunk2")?,
            gdn_chunk3_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_chunk3")?,
            w4a16_gemv_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            gdn_wy2_k: gpu.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?,
            gdn_wy2_resident_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy2_resident",
                "gated_delta_rule_wy2_resident",
            ),
            gdn_wy3_k: gpu.kernel("gated_delta_rule_wy3", "gated_delta_rule_wy3")?,
            gdn_wy3_resident_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy3_resident",
                "gated_delta_rule_wy3_resident",
            ),
            gdn_wy4_k: gpu.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?,
            gdn_wy4_woa_k: woa.woa_k,
            gdn_wy4_fold_k: woa.fold_k,
            gdn_wy4_clear_k: woa.clear_k,
            woa_flag: std::sync::atomic::AtomicU64::new(0),
            woa_stash: std::sync::atomic::AtomicU64::new(0),
            woa_seqs: std::sync::atomic::AtomicUsize::new(0),
            woa_dims: [nk, nv, kd, vd],
            gdn_wy2_f16_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy_f16",
                "gated_delta_rule_wy2_f16",
            ),
            gdn_wy2_resident_f16_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy2_resident_f16",
                "gated_delta_rule_wy2_resident_f16",
            ),
            gdn_wy3_f16_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy3_f16",
                "gated_delta_rule_wy3_f16",
            ),
            gdn_wy3_resident_f16_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy3_resident_f16",
                "gated_delta_rule_wy3_resident_f16",
            ),
            gdn_wy4_f16_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy4_f16",
                "gated_delta_rule_wy4_f16",
            ),
            gdn_verify_fused_conv_k2_k: super::super::try_kernel(
                gpu,
                "gdn_verify_fused_k2",
                "gdn_verify_fused_conv_k2",
            ),
            gdn_verify_fused_norm_k2_k: super::super::try_kernel(
                gpu,
                "gdn_verify_fused_k2",
                "gdn_verify_fused_norm_k2",
            ),
            gdn_verify_fused_conv_kn_k: super::super::try_kernel(
                gpu,
                "gdn_verify_fused_conv_kn",
                "gdn_verify_fused_conv_kn",
            ),
            gdn_verify_fused_conv_kn_batched_k: super::super::try_kernel(
                gpu,
                "gdn_verify_fused_conv_kn",
                "gdn_verify_fused_conv_kn_batched",
            ),
            gdn_f32_norm_snap_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_snap",
                "gated_delta_rule_decode_f32_norm_snap",
            ),
            gdn_f32_strided_norm_snap_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_snap",
                "gated_delta_rule_decode_f32_strided_norm_snap",
            ),
            gdn_verify_fused_conv_kn_f32_k: super::super::try_kernel(
                gpu,
                "gdn_verify_fused_conv_kn_f32",
                "gdn_verify_fused_conv_kn_f32",
            ),
            gdn_wy17_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy17",
                "gated_delta_rule_wy17",
            ),
            gdn_wyn_k: init_kernels::wyn_kernels(gpu),
            gdn_wyn_f16_k: init_kernels::wyn_f16_kernels(gpu),
            // 2026-09-25: Pointer-table twins for the cross-sequence batched verify.
            // provenance-id: 526f6e616c6420522e205374657369616b
            gdn_wyn_table_k: init_kernels::wyn_table_kernels(gpu),
            gdn_wyn_f16_table_k: init_kernels::wyn_f16_table_kernels(gpu),
            h_state_bytes: nv * vd * kd * 4,
            conv_state_bytes: conv_dim * d_conv * 4,
            qkvz_fp8: None,
            out_proj_fp8: None,
            fp8_gemm_k: gpu.kernel("w4a16", "fp8_gemm_t")?,
            fp8_gemm_t_m128_k: gpu.kernel("w4a16", "fp8_gemm_t_m128")?,
            w8a16_gemm_k: super::super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemm_pipelined_k: super::super::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            w8a16_gemm_pipelined_m32_k: super::super::try_target_kernel(
                gpu,
                "w8a16_gemm_pipelined_m32",
                "w8a16_gemm_pipelined_m32",
            ),
            w8a16_gemv_batch4_k: super::super::try_kernel(
                gpu,
                "w8a16_gemv_batch4",
                "w8a16_gemv_batch4",
            ),
            w8a16_gemv_batch16_k: super::super::try_kernel(
                gpu,
                "w8a16_gemv_batch4",
                "w8a16_gemv_batch16",
            ),
            w4a16_batchm: crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(gpu),
            w4a16_gemv_batch16_k: super::super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch16"),
            w8a16_gemm_t_k: super::super::try_kernel(gpu, "w8a16_gemm_t", "w8a16_gemm_t"),
            per_token_group_quant_fp8_k: ops::Fp8ActQuant::resolve(gpu),
            fp8_gemm_t_blockscaled_k: super::super::try_kernel(
                gpu,
                "fp8_gemm_t_blockscaled",
                "fp8_gemm_t_blockscaled",
            ),
            fp8_act_scale_kmajor_k: super::super::try_kernel(
                gpu,
                "fp8_scale_transpose",
                "fp8_act_scale_to_kmajor",
            ),
        })
    }
}

#[path = "init_kernels.rs"]
mod init_kernels;
use init_kernels::{fused_spine_kernel, gdn_prefill_tc_kernel, hc_kernel};

#[path = "init_sequential.rs"]
mod init_sequential;
