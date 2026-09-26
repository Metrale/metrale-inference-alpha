// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `PrefillKernels`: the residual/gate kernels, the W4A16 and FP8
//! GEMM/GEMV tiers, and the prefill attention kernels of a
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
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;

/// 2026-09-26: The handles `PrefillKernels::resolve` looked up, one field per
/// `Qwen3AttentionLayer` field of the same name.
pub(super) struct PrefillKernels {
    pub(super) residual_add_k: KernelHandle,
    pub(super) sigmoid_gate_mul_k: KernelHandle,
    pub(super) deinterleave_qg_k: KernelHandle,
    pub(super) w4a16_gemv_qg_k: KernelHandle,
    pub(super) residual_add_rms_norm_k: KernelHandle,
    pub(super) residual_add_rms_norm_gatef32_k: KernelHandle,
    pub(super) w4a16_gemv_qg_batch2_k: KernelHandle,
    pub(super) w4a16_gemv_dual_batch2_k: KernelHandle,
    pub(super) w4a16_gemv_batch2_k: KernelHandle,
    pub(super) w4a16_gemv_qg_batch3_k: KernelHandle,
    pub(super) w4a16_gemv_dual_batch3_k: KernelHandle,
    pub(super) w4a16_gemv_batch3_k: KernelHandle,
    pub(super) w4a16_batchm: W4a16BatchmTiers,
    pub(super) w4a16_gemm_k: KernelHandle,
    pub(super) w4a16_gemm_t_k: KernelHandle,
    pub(super) w4a16_gemm_t_k64_k: KernelHandle,
    pub(super) w4a16_gemm_t_k64_n64_k: KernelHandle,
    pub(super) w4a16_gemm_t_m128_k: KernelHandle,
    pub(super) w4a16_gemm_t_m128_bf16_k: KernelHandle,
    pub(super) w4a16_gemm_t_m128_v2_k: KernelHandle,
    pub(super) w4a16_gemm_t_m128_v3_k: KernelHandle,
    pub(super) dense_gemm_k: KernelHandle,
    pub(super) dense_gemm_pipelined_k: KernelHandle,
    pub(super) prefill_attn_k: KernelHandle,
    pub(super) prefill_attn_512_k: KernelHandle,
    pub(super) prefill_attn_512_is_tc: bool,
    pub(super) csa_compress_k: KernelHandle,
    pub(super) prefill_attn_compressed_k: KernelHandle,
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
    pub(super) fp8_gemm_k: KernelHandle,
    pub(super) bf16_to_fp8_k: KernelHandle,
    pub(super) fp8_fp8_gemm_k: KernelHandle,
    pub(super) fp8_gemm_t_m128_k: KernelHandle,
    pub(super) fp8_fp8_gemm_t_m128_k: KernelHandle,
    pub(super) w4a4_gemm_k: KernelHandle,
    pub(super) quantize_nvfp4_k: KernelHandle,
}

impl PrefillKernels {
    /// 2026-09-26: Looks up every field's kernel, in field order. The first
    /// failed required lookup returns its error and issues no later lookup.
    pub(super) fn resolve(
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
        probes: &ArchProbes,
    ) -> Result<Self> {
        Ok(Self {
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            sigmoid_gate_mul_k: gpu.kernel("residual_add", "sigmoid_gate_mul")?,
            deinterleave_qg_k: gpu.kernel("ssm_preprocess", "deinterleave_qg")?,
            w4a16_gemv_qg_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg")?,
            residual_add_rms_norm_k: if crate::ships_vanilla_norm_weights(config) {
                gpu.kernel("norm", "residual_add_rms_norm_vanilla")?
            } else {
                gpu.kernel("norm", "residual_add_rms_norm")?
            },
            residual_add_rms_norm_gatef32_k: crate::layers::try_kernel(
                gpu,
                "norm",
                "residual_add_rms_norm_gatef32",
            ),
            w4a16_gemv_qg_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg_batch2")?,
            w4a16_gemv_dual_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch2")?,
            w4a16_gemv_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_qg_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg_batch3")?,
            w4a16_gemv_dual_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch3")?,
            w4a16_gemv_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_batchm: crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(gpu),
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_t_k: crate::layers::tgemm_kernel(gpu),
            w4a16_gemm_t_k64_k: crate::layers::k64_kernel(gpu)?,
            w4a16_gemm_t_k64_n64_k: crate::layers::k64_n64_kernel(gpu),
            w4a16_gemm_t_m128_k: gpu.kernel("w4a16", "w4a16_gemm_t_m128")?,
            w4a16_gemm_t_m128_bf16_k: super::super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m128_bf16",
            ),
            w4a16_gemm_t_m128_v2_k: super::super::w4a16_v2_kernel(gpu),
            w4a16_gemm_t_m128_v3_k: super::super::w4a16_v3_kernel(gpu),
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            dense_gemm_pipelined_k: super::super::try_kernel(
                gpu,
                "gemm",
                "dense_gemm_bf16_pipelined",
            ),
            prefill_attn_k: gpu.kernel("attn_prefill", "attn_prefill")?,
            // 2026-09-25: `ops::wide_prefill_kernel` returns both the kernel and
            // the BR its launcher builds the grid for: the tensor-core
            // `attn_prefill_512tc` (BR=32) unless `METRALE_ATTN_512_TC=0` or the
            // target lacks it, otherwise the scalar `attn_prefill_512` (BR=16).
            prefill_attn_512_k: if probes.wide_head_dim {
                crate::layers::ops::wide_prefill_kernel(gpu).0
            } else {
                metrale_gpu_runtime::gpu::KernelHandle(0)
            },
            prefill_attn_512_is_tc: probes.wide_head_dim
                && crate::layers::ops::wide_prefill_kernel(gpu).1 == 32,
            // 2026-09-25: The compressed-attention kernels, looked up only when
            // the config's `compress_ratios` has a nonzero entry.
            csa_compress_k: gate(probes.compressed_attn, gpu, "csa_compress", "csa_compress"),
            prefill_attn_compressed_k: gate(
                probes.compressed_attn,
                gpu,
                "prefill_attn_compressed",
                "prefill_attn_compressed",
            ),
            prefill_attn_paged_512_k: gate(
                probes.wide_head_dim,
                gpu,
                "attn_prefill_paged_512",
                "attn_prefill_paged_512",
            ),
            prefill_attn_64_k: gpu.kernel("attn_prefill", "attn_prefill_64")?,
            prefill_attn_paged_k: gpu.kernel("prefill_paged", "attn_prefill_paged")?,
            prefill_attn_paged_fp8_k: gpu.kernel("prefill_paged_fp8", "attn_prefill_paged_fp8")?,
            prefill_attn_paged_nvfp4_k: gpu
                .kernel("prefill_paged_nvfp4", "attn_prefill_paged_nvfp4")?,
            prefill_attn_paged_turbo4_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo4",
                "attn_prefill_paged_turbo4",
            ),
            prefill_attn_paged_64_k: gpu.kernel("prefill_paged", "attn_prefill_paged_64")?,
            prefill_attn_paged_fp8_64_k: gpu
                .kernel("prefill_paged_fp8", "attn_prefill_paged_fp8_64")?,
            prefill_attn_paged_nvfp4_64_k: gpu
                .kernel("prefill_paged_nvfp4", "attn_prefill_paged_nvfp4_64")?,
            prefill_attn_paged_turbo2_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo2",
                "attn_prefill_paged_turbo2",
            ),
            prefill_attn_paged_turbo3_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo3",
                "attn_prefill_paged_turbo3_64",
            ),
            prefill_attn_paged_turbo4_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo4",
                "attn_prefill_paged_turbo4_64",
            ),
            prefill_attn_paged_turbo8_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo8",
                "attn_prefill_paged_turbo8_64",
            ),
            prefill_attn_paged_bf16k_turbo3v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_bf16k_turbo3v",
                "attn_prefill_paged_bf16k_turbo3v_64",
            ),
            prefill_attn_paged_bf16k_turbo4v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_bf16k_turbo4v",
                "attn_prefill_paged_bf16k_turbo4v_64",
            ),
            prefill_attn_paged_bf16k_turbo2v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_bf16k_turbo2v",
                "attn_prefill_paged_bf16k_turbo2v_64",
            ),
            prefill_attn_paged_fp8k_turbo3v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_fp8k_turbo3v",
                "attn_prefill_paged_fp8k_turbo3v_64",
            ),
            prefill_attn_paged_fp8k_turbo4v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_fp8k_turbo4v",
                "attn_prefill_paged_fp8k_turbo4v_64",
            ),
            prefill_attn_paged_fp8k_turbo2v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_fp8k_turbo2v",
                "attn_prefill_paged_fp8k_turbo2v_64",
            ),
            prefill_attn_paged_turbo4k_turbo3v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo4k_turbo3v",
                "attn_prefill_paged_turbo4k_turbo3v_64",
            ),
            prefill_attn_paged_turbo4k_turbo8v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo4k_turbo8v",
                "attn_prefill_paged_turbo4k_turbo8v_64",
            ),
            prefill_attn_paged_turbo3k_turbo8v_64_k: super::super::try_kernel(
                gpu,
                "prefill_paged_turbo3k_turbo8v",
                "attn_prefill_paged_turbo3k_turbo8v_64",
            ),
            prefill_attn_paged_batched_k: super::super::try_kernel(
                gpu,
                "attn_prefill_paged_batched",
                "attn_prefill_paged_batched",
            ),
            prefill_attn_paged_fp8_batched_k: super::super::try_kernel(
                gpu,
                "attn_prefill_paged_fp8_batched",
                "attn_prefill_paged_fp8_batched",
            ),
            prefill_attn_paged_nvfp4_batched_k: super::super::try_kernel(
                gpu,
                "attn_prefill_paged_nvfp4_batched",
                "attn_prefill_paged_nvfp4_batched",
            ),
            prefill_attn_paged_batched_64_k: super::super::try_kernel(
                gpu,
                "attn_prefill_paged_batched",
                "attn_prefill_paged_batched_64",
            ),
            prefill_attn_paged_fp8_batched_64_k: super::super::try_kernel(
                gpu,
                "attn_prefill_paged_fp8_batched",
                "attn_prefill_paged_fp8_batched_64",
            ),
            prefill_attn_paged_nvfp4_batched_64_k: super::super::try_kernel(
                gpu,
                "attn_prefill_paged_nvfp4_batched",
                "attn_prefill_paged_nvfp4_batched_64",
            ),
            deinterleave_qg_split_k: gpu.kernel("ssm_preprocess", "deinterleave_qg_split")?,
            deinterleave_qg_split_qnorm_k: gpu
                .kernel("ssm_preprocess", "deinterleave_qg_split_qnorm")?,
            deinterleave_qg_split_qnorm_mrope_k: super::super::try_kernel(
                gpu,
                "ssm_preprocess",
                "deinterleave_qg_split_qnorm_mrope",
            ),
            sigmoid_gate_mul_batched_k: gpu.kernel("residual_add", "sigmoid_gate_mul_batched")?,
            fp8_gemm_k: gpu.kernel("w4a16", "fp8_gemm_t")?,
            bf16_to_fp8_k: gpu.kernel("w4a16", "bf16_to_fp8")?,
            fp8_fp8_gemm_k: gpu.kernel("w4a16", "fp8_fp8_gemm_t")?,
            fp8_gemm_t_m128_k: gpu.kernel("w4a16", "fp8_gemm_t_m128")?,
            fp8_fp8_gemm_t_m128_k: gpu.kernel("w4a16", "fp8_fp8_gemm_t_m128")?,
            w4a4_gemm_k: crate::layers::try_kernel(gpu, "w4a4", "w4a4_gemm_mfast"),
            quantize_nvfp4_k: crate::layers::try_kernel(
                gpu,
                "quantize_nvfp4",
                "quantize_bf16_to_nvfp4",
            ),
        })
    }
}
