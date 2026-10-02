// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Construction of the MTP drafter's MoE layer on the checkpoint's BF16 experts (an
//! MTP head excluded from quantization, as nvidia/Qwen3.6-35B-A3B-NVFP4 ships it), so the batched
//! propose runs the drafter's MoE grouped (`MoeLayer::forward_bf16_grouped_decode`) instead of
//! per-expert GEMVs per sequence (`moe_forward_generic`).
//!
//! `--mtp-experts-nvfp4` instead requantizes those experts to NVFP4 at load, for drafting only.
//!
//! Owner: model-layers (MTP head).
//! Invariants: the switch is read by `MtpHead::new` only after the serve published it.

use std::sync::OnceLock;

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::MtpHead;
use crate::layers::MoeLayer;
use crate::weight_map::{DenseWeight, ExpertWeight, MoeWeights, MtpWeights, quantize_to_nvfp4};

static MTP_EXPERTS_NVFP4: OnceLock<bool> = OnceLock::new();

/// 2026-10-02: Publish `--mtp-experts-nvfp4`; returns the value in force, which differs when
/// something read it first.
pub fn set_mtp_experts_nvfp4_from_cli(on: bool) -> bool {
    let _ = MTP_EXPERTS_NVFP4.set(on);
    *MTP_EXPERTS_NVFP4.get().expect("just set")
}

/// 2026-10-02: Whether a BF16 MoE MTP head drafts on NVFP4 copies of its experts. Off unless the
/// serve published otherwise: the head's declared BF16 is the default.
pub fn mtp_experts_nvfp4() -> bool {
    *MTP_EXPERTS_NVFP4.get_or_init(|| false)
}

impl MtpHead {
    /// 2026-10-02: The drafter's MoE on the checkpoint's BF16 tensors: null NVFP4 routed and shared
    /// slots, the BF16 router (`gate_nvfp4 = None`), then `MoeLayer::set_bf16_experts` with the
    /// routed experts' and the shared expert's BF16 weights (no copy). Errors when the expert count
    /// differs from the config's.
    pub(super) fn new_bf16_moe(
        weights: &MtpWeights,
        config: &metrale_config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<MoeLayer> {
        anyhow::ensure!(
            weights.experts.len() == config.num_experts,
            "MTP BF16 experts: {} loaded, config says {}",
            weights.experts.len(),
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
        let col =
            |f: fn(&crate::weight_map::DenseExpertWeight) -> DenseWeight| -> Vec<DenseWeight> {
                weights.experts.iter().map(f).collect()
            };
        moe.set_bf16_experts(
            &col(|e| e.gate_proj),
            &col(|e| e.up_proj),
            &col(|e| e.down_proj),
            weights.shared_expert.gate_proj.weight,
            weights.shared_expert.up_proj.weight,
            weights.shared_expert.down_proj.weight,
            gpu,
        )?;
        Ok(moe)
    }
}

impl MtpHead {
    /// 2026-10-02: The drafter's MoE with its BF16 experts requantized to NVFP4 at load (a
    /// draft-only precision: the drafter proposes, the target verifies, so output tokens do not
    /// change, only how many drafts are accepted). The router stays BF16. The layer runs the grouped
    /// NVFP4 tensor-core decode at every width (`set_draft_nvfp4_experts`), reading 3.5x fewer
    /// bytes per expert than the BF16 point.
    pub(super) fn new_nvfp4_draft_moe(
        weights: &MtpWeights,
        config: &metrale_config::ModelConfig,
        gpu: &dyn GpuBackend,
        absmax_k: metrale_gpu_runtime::gpu::KernelHandle,
        nvfp4_k: metrale_gpu_runtime::gpu::KernelHandle,
        stream: u64,
    ) -> Result<MoeLayer> {
        let (h, inter) = (config.hidden_size, config.moe_intermediate_size);
        let shared_inter = config.shared_expert_intermediate_size;
        let q = |w: &DenseWeight, n: usize, k: usize| {
            quantize_to_nvfp4(w, n, k, gpu, absmax_k, nvfp4_k, stream)
        };
        let experts = weights
            .experts
            .iter()
            .map(|e| -> Result<ExpertWeight> {
                Ok(ExpertWeight {
                    gate_proj: q(&e.gate_proj, inter, h)?,
                    up_proj: q(&e.up_proj, inter, h)?,
                    down_proj: q(&e.down_proj, h, inter)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let sh = &weights.shared_expert;
        let moe_weights = MoeWeights {
            gate: weights.moe_gate,
            shared_expert: ExpertWeight {
                gate_proj: q(&sh.gate_proj, shared_inter, h)?,
                up_proj: q(&sh.up_proj, shared_inter, h)?,
                down_proj: q(&sh.down_proj, h, shared_inter)?,
            },
            shared_expert_gate: weights.shared_expert_gate,
            experts,
            router_pre_norm: None,
            correction_bias: None,
        };
        let mut moe = MoeLayer::new(moe_weights, config.num_experts, None, gpu, config)?;
        moe.set_draft_nvfp4_experts()?;
        tracing::info!(
            "--mtp-experts-nvfp4: the MTP head's {} routed experts and shared expert draft as NVFP4 \
             (requantized from BF16, draft-only); router BF16",
            weights.experts.len()
        );
        Ok(moe)
    }
}
