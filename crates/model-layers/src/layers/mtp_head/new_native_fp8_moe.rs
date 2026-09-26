// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Construction of the MTP drafter's MoE layer on the
//! checkpoint's native FP8 expert tables.
//!
//! Owner: model-layers (MTP head).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::MtpHead;
use crate::layers::MoeLayer;
use crate::weight_map::{ExpertWeight, MoeWeights, MtpWeights};

impl MtpHead {
    /// 2026-09-25: The drafter's MoE on the checkpoint's FP8 tables: null
    /// NVFP4 routed and shared expert slots, the BF16 router
    /// (`gate_nvfp4 = None`), then `MoeLayer::set_fp8_experts`. The loader's
    /// BF16 expert dequants are then freed
    /// (`MtpWeights::release_bf16_expert_dequants`). Errors when the weights
    /// carry no FP8 tables or the tables' expert count differs from the
    /// config's.
    pub(super) fn new_native_fp8_moe(
        weights: &mut MtpWeights,
        config: &metrale_config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<MoeLayer> {
        let fp8 = weights
            .fp8_experts
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("new_native_fp8_moe: no FP8 tables"))?;
        anyhow::ensure!(
            fp8.experts.len() == config.num_experts,
            "MTP FP8 tables cover {} experts, config says {}",
            fp8.experts.len(),
            config.num_experts
        );
        let moe_weights = MoeWeights {
            gate: weights.moe_gate,
            shared_expert: ExpertWeight::null(),
            shared_expert_gate: weights.shared_expert_gate,
            experts: vec![ExpertWeight::null(); config.num_experts],
            router_pre_norm: None,
            correction_bias: None,
        };
        let mut moe = MoeLayer::new(moe_weights, config.num_experts, None, gpu, config)?;
        moe.set_fp8_experts(&fp8.experts, fp8.shared_expert, gpu)?;
        weights.release_bf16_expert_dequants(gpu)?;
        Ok(moe)
    }
}
