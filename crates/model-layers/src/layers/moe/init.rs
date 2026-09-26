// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MoeLayer::new` / `new_with_hash`: validate the routing config, build
//! the expert pointer tables and resolve the kernels.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - A constructed layer has `1 <= num_experts_per_tok <= num_experts`,
//!   `num_experts_per_tok <= MOE_TOPK_SIGMOID_MAX_TOP_K` and
//!   `num_experts <= MOE_TOPK_SIGMOID_MAX_EXPERTS`; otherwise construction errors.

use super::*;

impl MoeLayer {
    pub fn new(
        weights: MoeWeights,
        num_experts: usize,
        gate_nvfp4: Option<QuantizedWeight>,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
    ) -> Result<Self> {
        Self::new_with_hash(weights, num_experts, gate_nvfp4, None, gpu, config)
    }

    /// 2026-09-25: Like [`MoeLayer::new`], with an optional hash-routing table
    /// `tid2eid` (`[vocab_size, top_k]` i64 on the device). `Some` makes this a
    /// hash-routed layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_hash(
        weights: MoeWeights,
        num_experts: usize,
        gate_nvfp4: Option<QuantizedWeight>,
        tid2eid_dev: Option<DevicePtr>,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
    ) -> Result<Self> {
        anyhow::ensure!(
            config.num_experts_per_tok <= num_experts && num_experts > 0,
            "MoE config invalid: num_experts_per_tok={} must be in 1..={}",
            config.num_experts_per_tok,
            num_experts,
        );
        // 2026-09-25: The sigmoid routing kernels hold at most MAX_TOP_K
        // selections and MAX_EXPERTS experts in shared memory and ignore the
        // rest, so such configs are refused here.
        anyhow::ensure!(
            config.num_experts_per_tok <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K
                && num_experts <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
            "MoE config exceeds the routing kernels' fixed shared-memory bounds: \
             num_experts_per_tok={} (max {}), num_experts={} (max {}). Raise \
             MAX_TOP_K / MAX_EXPERTS in kernels/gb10/common/moe_topk_sigmoid.cu \
             and their mirrors in layers::ops together.",
            config.num_experts_per_tok,
            crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K,
            num_experts,
            crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
        );
        let gate_ptrs = build_ptr_table(&weights.experts, |e| &e.gate_proj, gpu)?;
        let up_ptrs = build_ptr_table(&weights.experts, |e| &e.up_proj, gpu)?;
        let down_ptrs = build_ptr_table(&weights.experts, |e| &e.down_proj, gpu)?;

        // 2026-09-25: Read before the struct literal moves `weights`.
        let weights_correction_bias: Option<DevicePtr> =
            weights.correction_bias.map(|dw| dw.weight);

        let _ = num_experts;
        let rms_norm_k = gpu.kernel("norm", "rms_norm")?;
        let grouped = super::forward_fp8_grouped_decode::GroupedKernels::resolve(gpu);
        Ok(Self {
            weights,
            // 2026-09-25: NVFP4 until a loader says otherwise; the DeepSeek-V4
            // loader sets both kinds after construction (deepseek_v4/assemble/layer.rs).
            experts_scale_kind: crate::weight_map::WeightQuantFormat::Nvfp4,
            shared_experts_scale_kind: crate::weight_map::WeightQuantFormat::Nvfp4,
            gate_nvfp4,
            pre_expert_norm: None,
            pre_expert_norm_k: rms_norm_k,
            dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw: super::super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            dense_gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            dense_gemm_router: super::super::try_kernel(gpu, "gemm", "dense_gemm_bf16_router"),
            dense_gemm_pipelined: super::super::try_kernel(
                gpu,
                "gemm",
                "dense_gemm_bf16_pipelined",
            ),
            // 2026-09-25: FP32 router kernels (METRALE_FP32_GATE, METRALE_FP32_ROUTING);
            // 0 where the target lacks them.
            dense_gemm_f32out: super::super::try_kernel(gpu, "gemm", "dense_gemm_bf16_f32out"),
            dense_gemm_f32in: super::super::try_kernel(gpu, "gemm", "dense_gemm_f32in_f32out"),
            moe_topk_f32: super::super::try_kernel(gpu, "moe_topk", "moe_topk_softmax_f32"),
            moe_expert_gate_up_shared: gpu
                .kernel("moe_shared_expert_fused", "moe_expert_gate_up_shared")?,
            moe_expert_silu_down_shared: gpu
                .kernel("moe_shared_expert_fused", "moe_expert_silu_down_shared")?,
            moe_topk: gpu.kernel("moe_topk", "moe_topk_softmax")?,
            moe_weighted_sum_blend: gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?,
            residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
            moe_topk_batched: gpu.kernel("moe_topk", "moe_topk_softmax_batched")?,
            moe_expert_gate_up_shared_batch2: gpu
                .kernel("moe_fused_batch2", "moe_expert_gate_up_shared_batch2")?,
            moe_expert_silu_down_shared_batch2: gpu
                .kernel("moe_fused_batch2", "moe_expert_silu_down_shared_batch2")?,
            moe_weighted_sum_blend_batch2: gpu
                .kernel("moe_fused_batch2", "moe_weighted_sum_blend_batch2")?,
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            moe_expert_gate_up_shared_batch3: gpu
                .kernel("moe_fused_batch3", "moe_expert_gate_up_shared_batch3")?,
            moe_expert_silu_down_shared_batch3: gpu
                .kernel("moe_fused_batch3", "moe_expert_silu_down_shared_batch3")?,
            moe_weighted_sum_blend_batch3: gpu
                .kernel("moe_fused_batch3", "moe_weighted_sum_blend_batch3")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            moe_expert_gate_up_shared_token_major: gpu
                .kernel("moe_prefill", "moe_expert_gate_up_shared_prefill")?,
            moe_expert_silu_down_shared_token_major: gpu
                .kernel("moe_prefill", "moe_expert_silu_down_shared_prefill")?,
            moe_weighted_sum_blend_token_major: gpu
                .kernel("moe_prefill", "moe_weighted_sum_blend_prefill")?,
            moe_decode_atomic_c4_silu_down_accum_k: super::super::try_kernel(
                gpu,
                "moe_decode_atomic_c4",
                "moe_decode_atomic_c4_silu_down_accum",
            ),
            moe_decode_atomic_c4_finalize_k: super::super::try_kernel(
                gpu,
                "moe_decode_atomic_c4",
                "moe_decode_atomic_c4_finalize",
            ),
            moe_sort_by_expert: gpu.kernel("moe", "moe_sort_by_expert")?,
            moe_sorted_gate_up: gpu.kernel("moe_sorted", "moe_sorted_gate_up")?,
            moe_sorted_silu_down: gpu.kernel("moe_sorted", "moe_sorted_silu_down")?,
            moe_grouped_gemm: gpu.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable")?,
            moe_grouped_gemm_k32: if std::env::var("METRALE_MOE_GROUPED_K32").as_deref() == Ok("1")
            {
                super::super::try_kernel(gpu, "moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_k32")
            } else {
                KernelHandle(0)
            },
            moe_grouped_gemm_m256: if std::env::var("METRALE_MOE_GROUPED_M256").as_deref()
                == Ok("1")
            {
                super::super::try_kernel(gpu, "moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_m256")
            } else {
                KernelHandle(0)
            },
            moe_grouped_gemm_t: gpu.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_t")?,
            moe_grouped_gemm_t_k64: gpu
                .kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_t_k64")?,
            moe_fused_gate_up_t: gpu.kernel("moe_w4a16", "moe_w4a16_fused_gate_up_t")?,
            moe_fused_gate_up_t_k64: gpu.kernel("moe_w4a16", "moe_w4a16_fused_gate_up_t_k64")?,
            // 2026-09-25: E8M0 prefill kernels; only
            // kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu defines them.
            moe_grouped_gemm_e8m0: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_e8m0",
            ),
            moe_grouped_gemm_t_e8m0: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_t_e8m0",
            ),
            moe_grouped_gemm_t_k64_e8m0: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_t_k64_e8m0",
            ),
            moe_fused_gate_up_t_e8m0: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_fused_gate_up_t_e8m0",
            ),
            moe_fused_gate_up_t_k64_e8m0: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_fused_gate_up_t_k64_e8m0",
            ),
            // 2026-09-25: Only kernels/gb10/minimax-m2-229b defines the M=128 kernel;
            // elsewhere the handle is 0 and prefill uses the M=64 kernel.
            moe_fused_gate_up_t_k64_m128: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_fused_gate_up_t_k64_m128",
            ),
            // 2026-09-25: FP4 gate_up kernel (METRALE_HOLO_MOE_GATEUP_FP4); the
            // dispatch requires a non-zero handle.
            moe_fused_gate_up_t_k64_fp4: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_fused_gate_up_t_k64_fp4",
            ),
            moe_fp8_grouped_gemm_t: gpu.kernel("moe_w4a16", "moe_fp8_grouped_gemm_ptrtable_t")?,
            // 2026-09-25: The routed FP8 prefill GEMM over the compacted work-list;
            // 0 where the target lacks it (FP8 prefill then takes forward_batched).
            moe_fp8_grouped_gemm_k: super::super::try_kernel(
                gpu,
                "moe_fp8_grouped_gemm",
                "moe_fp8_grouped_gemm",
            ),
            // 2026-09-25: The work-list builder, in moe_permute.cu (module "moe").
            moe_build_tile_worklist_k: super::super::try_kernel(
                gpu,
                "moe",
                "moe_build_tile_worklist",
            ),
            moe_w8a8_grouped_gemm_k: super::super::try_kernel(
                gpu,
                "moe_w8a8_grouped_gemm",
                "moe_w8a8_grouped_gemm",
            ),
            // 2026-09-25: PM4 W8A8 grouped GEMM; with a 0 handle the W8A8 path uses
            // `moe_w8a8_grouped_gemm` over the dense grid.
            moe_w8a8_grouped_gemm_pm4_k: super::super::try_kernel(
                gpu,
                "moe_w8a8_grouped_gemm",
                "moe_w8a8_grouped_gemm_pm4",
            ),
            per_token_group_quant_fp8_k: ops::Fp8ActQuant::resolve(gpu),
            // 2026-09-25: 0 when the model's moe_silu_mul module lacks this entry;
            // the unfused pair runs then.
            silu_mul_quant_fp8_k: super::super::try_kernel(
                gpu,
                "moe_silu_mul",
                "silu_mul_quant_fp8",
            ),
            fp8_gemm_t_blockscaled_k: super::super::try_kernel(
                gpu,
                "fp8_gemm_t_blockscaled",
                "fp8_gemm_t_blockscaled",
            ),
            moe_bf16_grouped_gemm_k: super::super::try_kernel(
                gpu,
                "moe_bf16_grouped_gemm",
                "moe_bf16_grouped_gemm",
            ),
            moe_expert_gate_up_shared_bf16_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_bf16",
                "moe_expert_gate_up_shared_bf16",
            ),
            moe_expert_silu_down_shared_bf16_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_bf16",
                "moe_expert_silu_down_shared_bf16",
            ),
            moe_expert_gate_up_shared_bf16_batch2_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_bf16_batch2",
                "moe_expert_gate_up_shared_bf16_batch2",
            ),
            moe_expert_silu_down_shared_bf16_batch2_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_bf16_batch2",
                "moe_expert_silu_down_shared_bf16_batch2",
            ),
            w8a16_gemm_k: super::super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemm_pipelined_k: super::super::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            moe_gate_topk_fused_k: super::super::try_kernel(
                gpu,
                "moe_gate_topk",
                "moe_gate_topk_fused",
            ),
            w4a16_gemm_t: gpu.kernel("w4a16", "w4a16_gemm_t")?,
            bf16_to_fp8_k: gpu.kernel("w4a16", "bf16_to_fp8")?,
            fp8_gemm_k: gpu.kernel("w4a16", "fp8_gemm_t")?,
            moe_silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            moe_act_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            gelu_activation: false,
            moe_unpermute_reduce: gpu.kernel("moe", "moe_unpermute_reduce_indexed")?,
            moe_batched_blend: gpu.kernel("moe", "moe_batched_blend")?,
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            gate_ptrs_t: None,
            up_ptrs_t: None,
            down_ptrs_t: None,
            cutlass_grouped_host: None,
            _cutlass_sfb_owned: Vec::new(),
            down_t_scratch_packed: None,
            down_t_scratch_scale: None,
            moe_transpose_u8_batched_k: gpu
                .kernel("moe_transpose_batched", "moe_transpose_u8_batched")?,
            moe_expert_gate_up_shared_t_k: gpu
                .kernel("moe_shared_expert_fused_t", "moe_expert_gate_up_shared_t")?,
            moe_expert_silu_down_shared_t_k: gpu
                .kernel("moe_shared_expert_fused_t", "moe_expert_silu_down_shared_t")?,
            // 2026-09-25: E8M0-routed / NVFP4-shared decode kernels from
            // moe_shared_expert_fused_t.cu; 0 where the target lacks them.
            moe_expert_gate_up_shared_t_e8m0_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_t",
                "moe_expert_gate_up_shared_t_e8m0",
            ),
            moe_expert_silu_down_shared_t_e8m0_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_t",
                "moe_expert_silu_down_shared_t_e8m0",
            ),
            // 2026-09-25: Optional routing kernels (sqrtsoftplus here, softmax + bias
            // and hash routing below): 0 where the target lacks them.
            moe_topk_sqrtsoftplus_k: super::super::try_kernel(
                gpu,
                "moe_topk_sqrt",
                "moe_topk_sqrtsoftplus",
            ),
            moe_topk_sqrtsoftplus_batched_k: super::super::try_kernel(
                gpu,
                "moe_topk_sqrt",
                "moe_topk_sqrtsoftplus_batched",
            ),
            router_logits_n: (config.num_experts + config.zero_expert_num) as u32,
            moe_topk_softmax_bias_k: super::super::try_kernel(
                gpu,
                "moe_topk_softmax_bias",
                "moe_topk_softmax_bias",
            ),
            moe_topk_softmax_bias_batched_k: super::super::try_kernel(
                gpu,
                "moe_topk_softmax_bias",
                "moe_topk_softmax_bias_batched",
            ),
            moe_zero_expert_add_k: super::super::try_kernel(
                gpu,
                "moe_topk_softmax_bias",
                "moe_zero_expert_add",
            ),
            // 2026-09-25: 16384 f32 (64 KiB), one per row, on every layer; the
            // softmax + bias router writes it.
            zero_accum_dev: gpu.alloc(16384 * 4)?,
            moe_hash_route_k: super::super::try_kernel(gpu, "moe_hash_route", "moe_hash_route"),
            moe_hash_route_batched_k: super::super::try_kernel(
                gpu,
                "moe_hash_route",
                "moe_hash_route_batched",
            ),
            tid2eid_dev,
            moe_expert_gate_up_shared_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch2_t",
                "moe_expert_gate_up_shared_batch2_t",
            )?,
            moe_expert_silu_down_shared_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch2_t",
                "moe_expert_silu_down_shared_batch2_t",
            )?,
            moe_expert_gate_up_shared_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch3_t",
                "moe_expert_gate_up_shared_batch3_t",
            )?,
            moe_expert_silu_down_shared_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch3_t",
                "moe_expert_silu_down_shared_batch3_t",
            )?,
            moe_expert_gate_up_shared_fp8_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_t",
                "moe_expert_gate_up_shared_fp8_t",
            )?,
            moe_expert_silu_down_shared_fp8_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_t",
                "moe_expert_silu_down_shared_fp8_t",
            )?,
            moe_expert_gate_up_shared_fp8_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2_t",
                "moe_expert_gate_up_shared_fp8_batch2_t",
            )?,
            moe_expert_silu_down_shared_fp8_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2_t",
                "moe_expert_silu_down_shared_fp8_batch2_t",
            )?,
            moe_expert_gate_up_shared_fp8_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3_t",
                "moe_expert_gate_up_shared_fp8_batch3_t",
            )?,
            moe_expert_silu_down_shared_fp8_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3_t",
                "moe_expert_silu_down_shared_fp8_batch3_t",
            )?,
            unified_layout: std::env::var("METRALE_UNIFIED_MOE_LAYOUT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            hybrid_layout: std::env::var("METRALE_HYBRID_MOE_LAYOUT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            nvfp4_gate_up_m128: std::env::var("METRALE_NVFP4_GATE_UP_M128")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            gateup_fp4: std::env::var("METRALE_HOLO_MOE_GATEUP_FP4")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            down_fp4: std::env::var("METRALE_HOLO_MOE_DOWN_FP4")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            shared_gate_t: None,
            shared_up_t: None,
            shared_down_t: None,
            gate_fp8: None,
            shared_gate_fp8: None,
            shared_up_fp8: None,
            shared_down_fp8: None,
            prefill_stream: gpu.create_stream()?,
            event_a: gpu.create_event()?,
            event_b: gpu.create_event()?,
            moe_expert_gate_up_shared_fp8: gpu.kernel(
                "moe_shared_expert_fused_fp8",
                "moe_expert_gate_up_shared_fp8",
            )?,
            moe_expert_silu_down_shared_fp8: gpu.kernel(
                "moe_shared_expert_fused_fp8",
                "moe_expert_silu_down_shared_fp8",
            )?,
            // 2026-09-25: FP8 batch2/3 kernels for `forward_k2` / `forward_k3`.
            moe_expert_gate_up_shared_fp8_batch2: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2",
                "moe_expert_gate_up_shared_fp8_batch2",
            )?,
            moe_expert_silu_down_shared_fp8_batch2: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2",
                "moe_expert_silu_down_shared_fp8_batch2",
            )?,
            moe_weighted_sum_blend_fp8_batch2: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2",
                "moe_weighted_sum_blend_fp8_batch2",
            )?,
            moe_expert_gate_up_shared_fp8_batch3: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3",
                "moe_expert_gate_up_shared_fp8_batch3",
            )?,
            moe_expert_silu_down_shared_fp8_batch3: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3",
                "moe_expert_silu_down_shared_fp8_batch3",
            )?,
            moe_weighted_sum_blend_fp8_batch3: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3",
                "moe_weighted_sum_blend_fp8_batch3",
            )?,
            moe_expert_gate_up_shared_fp8_grouped_k: grouped.gate_up,
            moe_expert_silu_down_shared_fp8_grouped_k: grouped.silu_down,
            moe_weighted_sum_blend_fp8_grouped_k: grouped.blend,
            moe_fp8_grouped_compact_k: grouped.compact,
            fp8_gate_weight_ptrs: None,
            fp8_up_weight_ptrs: None,
            fp8_down_weight_ptrs: None,
            bf16_gate_weight_ptrs: None,
            bf16_up_weight_ptrs: None,
            bf16_down_weight_ptrs: None,
            bf16_shared_expert: None,
            fp8_shared_expert: None,
            moe_down_t_k64_fp4: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_down_t_k64_fp4",
            ),
            moe_permute_tokens_k: super::super::try_kernel(gpu, "moe", "moe_permute_tokens"),
            // 2026-09-25: Set after construction by the qwen35 loader.
            is_dflash_capture_layer: false,
            lora: None,
            correction_bias_dev: weights_correction_bias,
            // 2026-09-25: Optional, so a target without the `moe_topk_sig` module
            // still constructs; only a layer with a correction bias dispatches it.
            moe_topk_sigmoid_k: super::super::try_kernel(gpu, "moe_topk_sig", "moe_topk_sigmoid"),
            moe_topk_sigmoid_batched_k: super::super::try_kernel(
                gpu,
                "moe_topk_sig",
                "moe_topk_sigmoid_batched",
            ),
        })
    }
}
