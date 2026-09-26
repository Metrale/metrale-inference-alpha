// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Weight structs for MoE layers, whole transformer layers and the MTP head.
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

/// 2026-09-25: Weights of one MoE block.
pub struct MoeWeights {
    /// 2026-09-25: Router gate, BF16, one row of `hidden_size` per router logit.
    pub gate: DenseWeight,
    pub shared_expert: ExpertWeight,
    /// 2026-09-25: Shared-expert gate `[1, hidden_size]` BF16: the shared
    /// expert's output is scaled by `sigmoid(dot(input, gate))`, or by 1 when
    /// this pointer is NULL.
    pub shared_expert_gate: DenseWeight,
    /// 2026-09-25: One entry per expert id; experts this rank does not hold are
    /// `ExpertWeight::null()`.
    pub experts: Vec<ExpertWeight>,
    /// 2026-09-25: RMSNorm weight applied to the router input before the gate
    /// GEMV (`MoeLayer::router_input`). The Gemma-4 loader sets it to the BF16
    /// vector `router.scale * hidden_size^-0.5`; `None` feeds the gate the
    /// input unchanged.
    pub router_pre_norm: Option<DenseWeight>,
    /// 2026-09-25: Per-expert F32 bias `[num_experts]` added to the router
    /// scores for top-k selection.
    pub correction_bias: Option<DenseWeight>,
}

impl MoeWeights {
    /// 2026-09-25: `num_experts` experts with NULL pointers, for tests.
    #[cfg(test)]
    pub fn empty(num_experts: usize) -> Self {
        let null_dense = DenseWeight {
            weight: DevicePtr::NULL,
        };
        let null_quant = QuantizedWeight {
            weight: DevicePtr::NULL,
            weight_scale: DevicePtr::NULL,
            weight_scale_2: 1.0,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        };
        let null_expert = ExpertWeight {
            gate_proj: null_quant,
            up_proj: null_quant,
            down_proj: null_quant,
        };
        Self {
            gate: null_dense,
            shared_expert: null_expert,
            shared_expert_gate: null_dense,
            experts: vec![null_expert; num_experts],
            router_pre_norm: None,
            correction_bias: None,
        }
    }
}

/// 2026-09-25: All weights of one attention or linear-attention layer.
pub enum LayerWeights {
    FullAttention {
        input_norm: DenseWeight,
        attn: AttentionWeights,
        post_attn_norm: DenseWeight,
        moe: MoeWeights,
    },
    LinearAttention {
        input_norm: DenseWeight,
        ssm: SsmWeights,
        post_attn_norm: DenseWeight,
        moe: MoeWeights,
    },
}

/// 2026-09-25: BF16 gate, up and down projections of one expert.
#[derive(Debug, Clone, Copy)]
pub struct DenseExpertWeight {
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
}

/// 2026-09-25: MTP (multi-token prediction) head weights: one decoder layer
/// plus the concat projection `fc`.
///
/// Every `DenseWeight` here is BF16 (`load_mtp` dequantizes FP8 or NVFP4
/// projections). `MtpHead::new` quantizes the projections when it builds the head.
pub struct MtpWeights {
    /// 2026-09-25: RMSNorm weight for the token embedding before the concat.
    pub pre_fc_norm_embedding: DenseWeight,
    /// 2026-09-25: RMSNorm weight for the target hidden state before the concat.
    pub pre_fc_norm_hidden: DenseWeight,
    /// 2026-09-25: Concat projection `[hidden_size, 2*hidden_size]` BF16.
    pub fc: DenseWeight,
    /// 2026-09-25: RMSNorm weight before attention.
    pub input_layernorm: DenseWeight,
    /// 2026-09-25: Attention projections, BF16.
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    /// 2026-09-25: RMSNorm weight after attention (fused with the residual add).
    pub post_attn_layernorm: DenseWeight,
    /// 2026-09-25: MoE router gate, BF16. NULL when `dense_ffn` is `Some`.
    pub moe_gate: DenseWeight,
    /// 2026-09-25: Shared expert, BF16. NULL fields when `dense_ffn` is `Some`.
    pub shared_expert: DenseExpertWeight,
    /// 2026-09-25: Shared-expert gate `[1, hidden_size]` BF16. NULL when `dense_ffn` is `Some`.
    pub shared_expert_gate: DenseWeight,
    /// 2026-09-25: One BF16 entry per expert. Empty when `dense_ffn` is `Some`.
    pub experts: Vec<DenseExpertWeight>,
    /// 2026-09-25: Dense FFN (`gate_proj`, `up_proj`, `down_proj`) of an MTP
    /// head without a router. When `Some`, `MtpHead::new` builds no MoE and
    /// the MoE fields above are unused.
    pub dense_ffn: Option<DenseExpertWeight>,
    /// 2026-09-25: Final RMSNorm weight.
    pub norm: DenseWeight,
    /// 2026-09-25: The routed and shared experts as the checkpoint's own FP8
    /// block-scaled tables (`mtp.layers.0.mlp.experts.{e}.*_proj.{weight,weight_scale_inv}`).
    /// When `Some`, `experts` and `shared_expert` above are fresh BF16 dequants
    /// of the same tensors; a head that builds its MoE from these tables calls
    /// [`Self::release_bf16_expert_dequants`]. `None` when the experts are not
    /// FP8 on disk, are stacked, or the variant is `Bf16Raw`.
    pub fp8_experts: Option<MtpFp8Experts>,
}

/// 2026-09-25: Native FP8 block-scaled MTP MoE tables: `experts[e]` is routed
/// expert `e` (all `num_experts` are loaded), plus the shared expert. The FP8
/// bytes belong to the weight store; only the FP32 block scales are owned
/// allocations.
pub struct MtpFp8Experts {
    pub experts: Vec<Fp8ExpertWeight>,
    pub shared_expert: Fp8ExpertWeight,
}

impl MtpWeights {
    /// 2026-09-25: Free the BF16 dequants of the routed and shared experts once
    /// an FP8 `MoeLayer` serves them. Returns an error when
    /// [`Self::fp8_experts`] is `None`. `load_mtp` attaches the tables only for
    /// unstacked FP8 experts and a variant other than `Bf16Raw`; every expert
    /// entry is then a fresh dequant allocation, never a store pointer. Leaves
    /// `experts` empty and `shared_expert` NULL.
    pub fn release_bf16_expert_dequants(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        ensure!(
            self.fp8_experts.is_some(),
            "release_bf16_expert_dequants: no FP8 tables — the BF16 experts are the only copy"
        );
        let mut released = 0usize;
        for de in self.experts.drain(..) {
            for w in [de.gate_proj, de.up_proj, de.down_proj] {
                gpu.free(w.weight)?;
                released += 1;
            }
        }
        let null = DenseWeight {
            weight: DevicePtr::NULL,
        };
        let shared = std::mem::replace(
            &mut self.shared_expert,
            DenseExpertWeight {
                gate_proj: null,
                up_proj: null,
                down_proj: null,
            },
        );
        for w in [shared.gate_proj, shared.up_proj, shared.down_proj] {
            if !w.weight.is_null() {
                gpu.free(w.weight)?;
                released += 1;
            }
        }
        tracing::info!(
            "MTP head: released {released} BF16 expert dequants (native FP8 tables serve the MoE)"
        );
        Ok(())
    }
}
