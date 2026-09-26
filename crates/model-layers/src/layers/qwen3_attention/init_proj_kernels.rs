// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ProjKernels`: the output-gate, hyper-connection, projection
//! GEMM/GEMV, norm and RoPE kernels, and the KV-cache reshape kernel of a
//! `Qwen3AttentionLayer`. `new_with_gating` (`init.rs`) calls the three
//! resolvers in the order `ProjKernels`, `DecodeKernels`, `PrefillKernels` and
//! moves every field into the layer.
//!
//! Owner: model-layers (attention).
//! Invariants:
//! - `resolve` looks the kernels up in the order the fields are written.
//! - A kernel family behind `ArchProbes` is looked up only when the config
//!   says the model has it.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

// 2026-09-26: `gate` is called by path, never through a `let`-bound function
// pointer: a call through a pointer does not pass the caller's location to a
// `#[track_caller]` fn, so the boot audit would name `gated` itself instead of
// the dispatch site below.
use super::init_arch_gates::{ArchProbes, gated as gate};

/// 2026-09-26: The handles `ProjKernels::resolve` looked up, one field per
/// `Qwen3AttentionLayer` field of the same name.
pub(super) struct ProjKernels {
    pub(super) sigmoid_gate_head_broadcast_k: KernelHandle,
    pub(super) softplus_gate_head_broadcast_k: KernelHandle,
    pub(super) hc_pre_k: KernelHandle,
    pub(super) hc_post_k: KernelHandle,
    pub(super) hc_expand_k: KernelHandle,
    pub(super) hc_head_k: KernelHandle,
    pub(super) w8a16_gemm_t_k: KernelHandle,
    pub(super) w8a16_gemm_t_pipelined_k: KernelHandle,
    pub(super) w8a16_gemm_t_m128_k: KernelHandle,
    pub(super) per_token_group_quant_fp8_k: crate::layers::ops::Fp8ActQuant,
    pub(super) fp8_gemm_t_blockscaled_k: KernelHandle,
    pub(super) fp8_act_scale_kmajor_k: KernelHandle,
    pub(super) rms_norm_k: KernelHandle,
    pub(super) rms_norm_w_k: KernelHandle,
    pub(super) rms_norm_w_warp_row_k: KernelHandle,
    pub(super) norm_vanilla: bool,
    pub(super) rms_norm_residual_k: KernelHandle,
    pub(super) dense_gemv_k: KernelHandle,
    pub(super) dequant_q2_0_gn_k: KernelHandle,
    pub(super) q2_0_gemv_k: KernelHandle,
    pub(super) dense_gemv_batchm_k: KernelHandle,
    pub(super) w4a16_gemv_k: KernelHandle,
    pub(super) w4a16_gemv_sw_k: KernelHandle,
    pub(super) w8a16_gemv_k: KernelHandle,
    pub(super) w8a16_gemv_batch4_k: KernelHandle,
    pub(super) w8a16_gemv_batch16_k: KernelHandle,
    pub(super) w8a16_gemv_batch4_strided_k: KernelHandle,
    pub(super) w8a16_gemv_batch16_strided_k: KernelHandle,
    pub(super) w8a16_gemm_m16_k: KernelHandle,
    pub(super) w8a16_gemm_m16_strided_k: KernelHandle,
    pub(super) m16_tc: bool,
    pub(super) w8a16_gemv_ncol2_k: KernelHandle,
    pub(super) w8a16_gemv_ncol4_k: KernelHandle,
    pub(super) w8a16_gemv_ncol2_strided_k: KernelHandle,
    pub(super) w8a16_gemv_ncol4_strided_k: KernelHandle,
    pub(super) attn_ncol: Option<super::attn_ncol_gemv::NcolWidth>,
    pub(super) w8a16_gemm_k: KernelHandle,
    pub(super) w8a16_gemm_pipelined_k: KernelHandle,
    pub(super) w8a16_gemm_pipelined_m32_k: KernelHandle,
    pub(super) w4a16_gemv_dual_k: KernelHandle,
    pub(super) rope_k: KernelHandle,
    pub(super) rope_strided_k: KernelHandle,
    pub(super) rms_norm_strided_k: KernelHandle,
    pub(super) rope_mrope_interleaved_k: KernelHandle,
    pub(super) rope_mrope_interleaved_k_only_k: KernelHandle,
    pub(super) rope_yarn_k: KernelHandle,
    pub(super) rope_yarn_scaled_k: KernelHandle,
    pub(super) rope_yarn_interleaved_k: KernelHandle,
    pub(super) rope_yarn_interleaved_inv_k: KernelHandle,
    pub(super) rope_proportional_k: KernelHandle,
    pub(super) reshape_cache_k: KernelHandle,
}

impl ProjKernels {
    /// 2026-09-26: Looks up every field's kernel, in field order. The first
    /// failed required lookup returns its error and issues no later lookup.
    pub(super) fn resolve(
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
        probes: &ArchProbes,
        reshape_mod: &'static str,
        reshape_fn: &'static str,
    ) -> Result<Self> {
        Ok(Self {
            sigmoid_gate_head_broadcast_k: super::super::try_kernel(
                gpu,
                "residual_add",
                "sigmoid_gate_mul_head_broadcast",
            ),
            softplus_gate_head_broadcast_k: super::super::try_kernel(
                gpu,
                "residual_add",
                "softplus_gate_mul_head_broadcast",
            ),
            hc_pre_k: gate(probes.hyper_connection, gpu, "hyper_connection", "hc_pre"),
            hc_post_k: gate(probes.hyper_connection, gpu, "hyper_connection", "hc_post"),
            hc_expand_k: gate(
                probes.hyper_connection,
                gpu,
                "hyper_connection",
                "hc_expand",
            ),
            hc_head_k: gate(probes.hyper_connection, gpu, "hyper_connection", "hc_head"),
            w8a16_gemm_t_k: super::super::try_kernel(gpu, "w8a16_gemm_t", "w8a16_gemm_t"),
            w8a16_gemm_t_pipelined_k: super::super::try_kernel(
                gpu,
                "w8a16_gemm_t",
                "w8a16_gemm_t_pipelined",
            ),
            w8a16_gemm_t_m128_k: super::super::try_kernel(
                gpu,
                "w8a16_gemm_t_m128",
                "w8a16_gemm_t_m128",
            ),
            // 2026-09-25: `Fp8ActQuant` resolves the shared quantizer and the
            // Hopper twin, which only `kernels/hopper` has, and carries both
            // handles, so the grid is derived from the same pick as the entry
            // point (`ops::fp8_quant_grid`). `W8A8_PREFILL_KERNELS[0]` is built
            // from the same name constants.
            per_token_group_quant_fp8_k: crate::layers::ops::Fp8ActQuant::resolve(gpu),
            fp8_gemm_t_blockscaled_k: super::super::try_kernel(
                gpu,
                super::types_weights::W8A8_PREFILL_KERNELS[1].0,
                super::types_weights::W8A8_PREFILL_KERNELS[1].1,
            ),
            // 2026-09-25: The same optional adapter `qwen3_ssm/init.rs` loads.
            // On a target without the `fp8_scale_transpose` module the handle is
            // 0, and the cuBLASLt W8A8 prefill arm is not taken while cuBLASLt
            // expects K-major scales (`prefill_w8a8.rs`, `kmajor_ready`).
            fp8_act_scale_kmajor_k: super::super::try_kernel(
                gpu,
                "fp8_scale_transpose",
                "fp8_act_scale_to_kmajor",
            ),
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            rms_norm_w_k: if crate::ships_vanilla_norm_weights(config) {
                gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?
            } else {
                gpu.kernel("norm", "rms_norm")?
            },
            rms_norm_w_warp_row_k: if crate::ships_vanilla_norm_weights(config) {
                gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla_warp_row")
                    .unwrap_or(KernelHandle(0))
            } else {
                KernelHandle(0)
            },
            norm_vanilla: crate::ships_vanilla_norm_weights(config),
            rms_norm_residual_k: if crate::ships_vanilla_norm_weights(config) {
                gpu.kernel("norm", "rms_norm_residual_vanilla")?
            } else {
                gpu.kernel("norm", "rms_norm_residual")?
            },
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dequant_q2_0_gn_k: super::super::try_kernel(
                gpu,
                "dequant_gguf_bf16",
                "dequant_q2_0_gn_to_bf16",
            ),
            q2_0_gemv_k: super::super::try_kernel(gpu, "q2_0_gemv_vec", "q2_0_gemv_vec"),
            dense_gemv_batchm_k: gpu
                .kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")
                .unwrap_or(KernelHandle(0)),
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: super::super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w8a16_gemv_k: gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
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
            w8a16_gemv_batch4_strided_k: super::super::try_kernel(
                gpu,
                "w8a16_gemv_batch4",
                "w8a16_gemv_batch4_strided",
            ),
            w8a16_gemv_batch16_strided_k: super::super::try_kernel(
                gpu,
                "w8a16_gemv_batch4",
                "w8a16_gemv_batch16_strided",
            ),
            w8a16_gemm_m16_k: super::super::try_target_kernel(
                gpu,
                "w8a16_gemm_m16",
                "w8a16_gemm_m16",
            ),
            w8a16_gemm_m16_strided_k: super::super::try_target_kernel(
                gpu,
                "w8a16_gemm_m16",
                "w8a16_gemm_m16_strided",
            ),
            m16_tc: crate::layers::dense_ffn::m16_tc::m16_tc_levers().attn,
            w8a16_gemv_ncol2_k: super::super::try_target_kernel(
                gpu,
                "w8a16_gemv_ncol",
                "w8a16_gemv_batch16_ncol2",
            ),
            w8a16_gemv_ncol4_k: super::super::try_target_kernel(
                gpu,
                "w8a16_gemv_ncol",
                "w8a16_gemv_batch16_ncol4",
            ),
            w8a16_gemv_ncol2_strided_k: super::super::try_target_kernel(
                gpu,
                "w8a16_gemv_ncol",
                "w8a16_gemv_batch16_ncol2_strided",
            ),
            w8a16_gemv_ncol4_strided_k: super::super::try_target_kernel(
                gpu,
                "w8a16_gemv_ncol",
                "w8a16_gemv_batch16_ncol4_strided",
            ),
            attn_ncol: super::attn_ncol_gemv::ncol_gemv_enabled()
                .then(super::attn_ncol_gemv::ncol_gemv_width),
            w8a16_gemm_k: super::super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemm_pipelined_k: super::super::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            // 2026-09-25: Looked up only when `ModelLevers::fp8_attn_m32` is on
            // (`METRALE_FP8_ATTN_M32=1`). Every reader checks for a nonzero
            // handle before it uses it.
            w8a16_gemm_pipelined_m32_k: if crate::layers::ops::ModelLevers::get().fp8_attn_m32 {
                super::super::try_target_kernel(
                    gpu,
                    "w8a16_gemm_pipelined_m32",
                    "w8a16_gemm_pipelined_m32",
                )
            } else {
                KernelHandle(0)
            },
            w4a16_gemv_dual_k: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            rope_k: gpu.kernel("rope", "rope_forward")?,
            rope_strided_k: super::super::try_kernel(gpu, "rope", "rope_forward_strided"),
            rms_norm_strided_k: super::super::try_kernel(gpu, "norm", "rms_norm_strided"),
            rope_mrope_interleaved_k: super::super::try_kernel(
                gpu,
                "rope_mrope_interleaved",
                "rope_forward_mrope_interleaved",
            ),
            rope_mrope_interleaved_k_only_k: super::super::try_kernel(
                gpu,
                "rope_mrope_interleaved",
                "rope_forward_mrope_interleaved_k_only",
            ),
            rope_yarn_k: super::super::try_kernel(gpu, "rope", "rope_forward_yarn"),
            rope_yarn_scaled_k: super::super::try_kernel(gpu, "rope", "rope_forward_yarn_scaled"),
            // 2026-09-25: YaRN RoPE on adjacent pairs (2i, 2i+1), and its
            // inverse below; the DeepSeek-V4 paths use them.
            rope_yarn_interleaved_k: super::super::try_kernel(
                gpu,
                "rope",
                "rope_forward_yarn_interleaved",
            ),
            rope_yarn_interleaved_inv_k: super::super::try_kernel(
                gpu,
                "rope",
                "rope_forward_yarn_interleaved_inv",
            ),
            rope_proportional_k: super::super::try_kernel(gpu, "rope", "rope_forward_proportional"),
            reshape_cache_k: gpu.kernel(reshape_mod, reshape_fn)?,
        })
    }
}
