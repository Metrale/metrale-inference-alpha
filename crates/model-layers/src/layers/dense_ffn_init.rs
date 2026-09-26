// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `DenseFfnLayer::new_with_activation`: looks up every kernel handle the layer
//! holds, with no FP8, BF16, packed-Q2 or LoRA overlay installed.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - A kernel looked up with `gpu.kernel` must resolve or construction fails; one looked up
//!   with `try_kernel` or `try_target_kernel` leaves a zero handle when absent.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::{DenseFfnLayer, DenseFfnWeights, FfnActivation, batch16_decode, gateup_fused, m16_tc};
use crate::layers::ops;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;

impl DenseFfnLayer {
    pub fn new_with_activation(
        weights: DenseFfnWeights,
        activation: FfnActivation,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let act_mul = match activation {
            FfnActivation::SiLU => gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            FfnActivation::GeLU => gpu.kernel("gelu", "gelu_mul")?,
        };
        // 2026-09-25: One value gates both the `silu_mul_strided` lookup below and
        // `gateup_fused_plan`.
        let gateup_fused = gateup_fused::ffn_gateup_fused();
        // 2026-09-25: Optional BF16 kernels: `try_kernel` gives a zero handle when the target lacks
        // them.
        let dense_gemv_bf16_k = super::try_kernel(gpu, "gemv", "dense_gemv_bf16");
        let dense_gemm_bf16_k = super::try_kernel(gpu, "gemm", "dense_gemm_bf16");
        let dense_gemm_tc_k = super::try_kernel(gpu, "gemm_tc", "dense_gemm_tc");

        let layer = Self {
            weights,
            activation,
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w4a16_gemv_dual: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            w4a16_gemv_silu_input: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_silu_input")?,
            w4a16_gemv_dual_sw: super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_sw"),
            w4a16_gemv_silu_input_sw: super::try_kernel(
                gpu,
                "w4a16_gemv_fused",
                "w4a16_gemv_silu_input_sw",
            ),
            w4a16_gemv_dual_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch2")?,
            w4a16_gemv_dual_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch3")?,
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_batchm: W4a16BatchmTiers::resolve(gpu),
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_t_m128_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128"),
            w4a16_gemm_t_m128_v2_k: super::w4a16_v2_kernel(gpu),
            w4a16_gemm_t_m128_bf16_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128_bf16"),
            w4a16_gemm_t_m128_bf16_v2_k: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m128_bf16_v2",
            ),
            w4a16_gemm_t_k: super::tgemm_kernel(gpu),
            int8_faith2_k: super::try_kernel(gpu, "w4a16", "int8_gemm_faith2"),
            int8_faith5_k: super::try_kernel(gpu, "w4a16", "int8_gemm_i32acc"),
            requant_w_int8_k: super::try_kernel(gpu, "w4a16", "requant_w_nvfp4_int8"),
            requant_a_int8_k: super::try_kernel(gpu, "w4a16", "requant_a_bf16_int8"),
            int8_gate: std::sync::OnceLock::new(),
            int8_up: std::sync::OnceLock::new(),
            int8_down: std::sync::OnceLock::new(),
            w4a4_gemm_k: super::try_kernel(gpu, "w4a4", "w4a4_gemm"),
            quantize_nvfp4_k: super::try_kernel(gpu, "quantize_nvfp4", "quantize_bf16_to_nvfp4"),
            q4k_mmq_nc_k: super::try_kernel(gpu, "q4k_mmq", "metrale_q4k_mmq128_nc"),
            q4k_mmq_wc_k: super::try_kernel(gpu, "q4k_mmq", "metrale_q4k_mmq128_wc"),
            q4k_quant_act_k: super::try_kernel(gpu, "q4k_mmq", "metrale_q8_1_quantize_ds4_bf16"),
            q4k_quant_w_k: super::try_kernel(gpu, "q4k_quantize", "q4k_quantize"),
            dequant_nvfp4_bf16_k: super::try_kernel(
                gpu,
                "dequant_nvfp4_bf16",
                "dequant_nvfp4_to_bf16",
            ),
            q4k_gate: std::sync::OnceLock::new(),
            q4k_up: std::sync::OnceLock::new(),
            q4k_down: std::sync::OnceLock::new(),
            nvfp4_mmq_nc_k: super::try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_mmq128_nc"),
            nvfp4_mmq_wc_k: super::try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_mmq128_wc"),
            nvfp4_mmq16_nc_k: super::try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_mmq16_nc"),
            nvfp4_mmq16_wc_k: super::try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_mmq16_wc"),
            nvfp4_mmq32_nc_k: super::try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_mmq32_nc"),
            nvfp4_mmq32_wc_k: super::try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_mmq32_wc"),
            nvfp4_mmq64_nc_k: super::try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_mmq64_nc"),
            nvfp4_mmq64_wc_k: super::try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_mmq64_wc"),
            nvfp4_quant_act_k: super::try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_quantize_bf16"),
            nvfp4_repack_k: super::try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_repack"),
            nvfp4_silu_scaled_k: super::try_kernel(
                gpu,
                "nvfp4_mmq",
                "metrale_nvfp4_silu_mul_scaled",
            ),
            nvfp4_silu_quant_k: super::try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_silu_mul_quant"),
            nvfp4_scale_k: super::try_kernel(gpu, "nvfp4_mmq", "metrale_nvfp4_scale_bf16"),
            fp4mmq_gate: std::sync::OnceLock::new(),
            fp4mmq_up: std::sync::OnceLock::new(),
            fp4mmq_down: std::sync::OnceLock::new(),
            w4a16_gemm_t_k64_k: super::k64_kernel(gpu).unwrap_or(KernelHandle(0)),
            act_mul,
            bf16_weights: None,
            dense_gemv_bf16_k,
            dense_gemm_bf16_k,
            dense_gemm_tc_k,
            fp8_weights: None,
            w8a16_gemv_k: super::try_kernel(gpu, "w8a16_gemv", "w8a16_gemv"),
            w8a16_gemm_k: super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemv_batch4_k: super::try_kernel(gpu, "w8a16_gemv_batch4", "w8a16_gemv_batch4"),
            w8a16_gemv_batch16_k: super::try_kernel(gpu, "w8a16_gemv_batch4", "w8a16_gemv_batch16"),
            batch16_enabled: batch16_decode::ffn_batch16_enabled(),
            w8a16_gemm_m16_k: super::try_target_kernel(gpu, "w8a16_gemm_m16", "w8a16_gemm_m16"),
            w8a16_gemm_m16_n64_k: super::try_target_kernel(
                gpu,
                "w8a16_gemm_m16",
                "w8a16_gemm_m16_n64",
            ),
            m16_tc: m16_tc::m16_tc_levers().ffn,
            m16_tc_n_tile: m16_tc::m16_tc_levers().ffn_n_tile,
            w8a16_gemm_pipelined_k: super::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            w8a16_gemv_dual_k: super::try_kernel(gpu, "w8a16_gemv_fused", "w8a16_gemv_dual"),
            w8a16_gemv_silu_input_k: super::try_kernel(
                gpu,
                "w8a16_gemv_fused",
                "w8a16_gemv_silu_input",
            ),
            w8a16_gemm_t_m128_k: super::try_kernel(gpu, "w8a16_gemm_t_m128", "w8a16_gemm_t_m128"),
            per_token_group_quant_fp8_k: ops::Fp8ActQuant::resolve(gpu),
            fp8_gemm_t_blockscaled_k: super::try_kernel(
                gpu,
                "fp8_gemm_t_blockscaled",
                "fp8_gemm_t_blockscaled",
            ),
            fp8_act_scale_kmajor_k: super::try_kernel(
                gpu,
                "fp8_scale_transpose",
                "fp8_act_scale_to_kmajor",
            ),
            fp8_gate_up_fused: None,
            gateup_fused,
            silu_mul_strided_k: if gateup_fused {
                super::try_target_kernel(gpu, "silu_mul_strided", "silu_mul_strided")
            } else {
                KernelHandle(0)
            },
            lora: None,
            q2_weights: None,
            q2_0_gemv_k: super::try_kernel(gpu, "q2_0_gemv_vec", "q2_0_gemv_vec"),
            q2_0_gemv_batchm_k: super::try_kernel(gpu, "q2_0_gemv_vec", "q2_0_gemv_vec_batchm"),
            dequant_q2_0_gn_k: super::try_kernel(
                gpu,
                "dequant_gguf_bf16",
                "dequant_q2_0_gn_to_bf16",
            ),
            // 2026-09-25: Looked up by `set_q2_weights`, so a layer without packed-Q2 weights never
            // probes `q2_0_mmq`.
            q2_0_mmq_nc_k: KernelHandle(0),
            q2_0_mmq_wc_k: KernelHandle(0),
        };
        Ok(layer)
    }
}
