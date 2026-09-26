// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: MoE and MTP-head weight loaders: the `mlp` MoE block, the Mistral `w1/w2/w3` MoE block, and the MTP head.
//!
//! Owner: model-layers (weight loading).
//! Invariants:
//! - `MtpWeights::fp8_experts` is `Some` only when the experts are unstacked,
//!   the variant is not `Bf16Raw`, and expert 0 is FP8 with `weight_scale_inv`.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

pub(super) fn load_moe_inner(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &metrale_config::ModelConfig,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    skip_routed_experts: bool,
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.mlp");
    let inter = config.moe_intermediate_size;
    let h = config.hidden_size;

    let gate = dense(store, &format!("{p}.gate.weight"))?;
    let shared_expert_gate = dense(store, &format!("{p}.shared_expert_gate.weight"))?;

    // 2026-09-25: The shared expert is loaded even when `skip_routed_experts` is set.
    let shared_expert = ExpertWeight {
        gate_proj: quantized_any(
            store,
            &format!("{p}.shared_expert.gate_proj"),
            inter,
            h,
            gpu,
            variant,
            qctx,
        )?,
        up_proj: quantized_any(
            store,
            &format!("{p}.shared_expert.up_proj"),
            inter,
            h,
            gpu,
            variant,
            qctx,
        )?,
        down_proj: quantized_any(
            store,
            &format!("{p}.shared_expert.down_proj"),
            h,
            inter,
            gpu,
            variant,
            qctx,
        )?,
    };

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if skip_routed_experts || !config.is_local_expert(e) {
            experts.push(ExpertWeight::null());
        } else {
            experts.push(ExpertWeight {
                gate_proj: quantized_any(
                    store,
                    &format!("{p}.experts.{e}.gate_proj"),
                    inter,
                    h,
                    gpu,
                    variant,
                    qctx,
                )?,
                up_proj: quantized_any(
                    store,
                    &format!("{p}.experts.{e}.up_proj"),
                    inter,
                    h,
                    gpu,
                    variant,
                    qctx,
                )?,
                down_proj: quantized_any(
                    store,
                    &format!("{p}.experts.{e}.down_proj"),
                    h,
                    inter,
                    gpu,
                    variant,
                    qctx,
                )?,
            });
        }
    }

    Ok(MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias: None,
    })
}

/// 2026-09-25: Load a Mistral MoE block (`w1/w2/w3` naming: w1 = gate_proj,
/// w2 = down_proj, w3 = up_proj) through `quantized_v2`.
pub fn load_moe_mistral(
    store: &WeightStore,
    layer: usize,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &metrale_config::ModelConfig,
) -> Result<MoeWeights> {
    // 2026-09-25: Keys are `layers.{i}.…`; only the gate also accepts `model.layers.{i}.gate.weight`.
    let p = format!("layers.{layer}");

    let gate = dense(store, &format!("{p}.gate.weight"))
        .or_else(|_| dense(store, &format!("model.layers.{layer}.gate.weight")))
        .context("Mistral: MoE gate weight not found")?;

    // 2026-09-25: The shared expert is loaded only when `shared_experts.w1.weight_packed`
    // exists; `shared_expert_gate` is always NULL.
    let se_prefix = format!("{p}.shared_experts");
    let shared_expert = if store.contains(&format!("{se_prefix}.w1.weight_packed")) {
        ExpertWeight {
            gate_proj: quantized_v2(store, &format!("{se_prefix}.w1"), gpu)
                .context("Mistral: shared expert w1")?,
            up_proj: quantized_v2(store, &format!("{se_prefix}.w3"), gpu)
                .context("Mistral: shared expert w3")?,
            down_proj: quantized_v2(store, &format!("{se_prefix}.w2"), gpu)
                .context("Mistral: shared expert w2")?,
        }
    } else {
        tracing::warn!("L{layer}: no shared expert weights, using NULL");
        ExpertWeight::null()
    };
    let shared_expert_gate = DenseWeight {
        weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
    };

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if config.is_local_expert(e) {
            let ep = format!("{p}.experts.{e}");
            let gate_proj = quantized_v2(store, &format!("{ep}.w1"), gpu)
                .with_context(|| format!("Mistral expert {e}: w1 not found"))?;
            let up_proj = quantized_v2(store, &format!("{ep}.w3"), gpu)
                .with_context(|| format!("Mistral expert {e}: w3 not found"))?;
            let down_proj = quantized_v2(store, &format!("{ep}.w2"), gpu)
                .with_context(|| format!("Mistral expert {e}: w2 not found"))?;
            experts.push(ExpertWeight {
                gate_proj,
                up_proj,
                down_proj,
            });
        } else {
            experts.push(ExpertWeight::null());
        }
    }

    Ok(MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias: None,
    })
}

/// 2026-09-25: Load the MTP head (`mtp.*` keys) as BF16 `DenseWeight`s.
///
/// Projections go through `dense_auto` (FP8 and packed NVFP4 are dequantized),
/// except under `Bf16Raw`, which hands out the store pointers. Two FFN shapes:
/// - **MoE**: router `mtp.layers.0.mlp.gate.weight`, per-expert or stacked
///   expert tensors, and a shared expert.
/// - **Dense**: `mtp.layers.0.mlp.{gate,up,down}_proj.weight` with no
///   `.gate.weight` router, returned in `dense_ffn`.
pub fn load_mtp(
    store: &WeightStore,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
) -> Result<MtpWeights> {
    let p = "mtp.layers.0.self_attn";
    let mlp = "mtp.layers.0.mlp";

    let load = |name: &str| -> Result<DenseWeight> {
        match variant {
            Nvfp4Variant::Fp8Dequanted => dense_auto(store, name, gpu),
            Nvfp4Variant::Bf16Raw => dense(store, name),
            // 2026-09-25: A BF16 tensor comes back as the store pointer, like
            // `dense`; a packed NVFP4 (U8) head is dequantized to BF16.
            _ => dense_auto(store, name, gpu),
        }
    };

    // 2026-09-25: Return before any MoE-shaped load, so a dense head does not
    // fail on its missing `shared_expert.*` tensors.
    let dense_gate_proj = format!("{mlp}.gate_proj.weight");
    let moe_router = format!("{mlp}.gate.weight");
    if store.contains(&dense_gate_proj) && !store.contains(&moe_router) {
        let dense_ffn = DenseExpertWeight {
            gate_proj: load(&dense_gate_proj)?,
            up_proj: load(&format!("{mlp}.up_proj.weight"))?,
            down_proj: load(&format!("{mlp}.down_proj.weight"))?,
        };
        let null = DenseWeight {
            weight: DevicePtr::NULL,
        };
        return Ok(MtpWeights {
            pre_fc_norm_embedding: dense(store, "mtp.pre_fc_norm_embedding.weight")?,
            pre_fc_norm_hidden: dense(store, "mtp.pre_fc_norm_hidden.weight")?,
            fc: load("mtp.fc.weight")?,
            input_layernorm: dense(store, "mtp.layers.0.input_layernorm.weight")?,
            q_proj: load(&format!("{p}.q_proj.weight"))?,
            k_proj: load(&format!("{p}.k_proj.weight"))?,
            v_proj: load(&format!("{p}.v_proj.weight"))?,
            o_proj: load(&format!("{p}.o_proj.weight"))?,
            q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
            k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
            post_attn_layernorm: dense(store, "mtp.layers.0.post_attention_layernorm.weight")?,
            moe_gate: null,
            shared_expert: DenseExpertWeight {
                gate_proj: null,
                up_proj: null,
                down_proj: null,
            },
            shared_expert_gate: null,
            experts: Vec::new(),
            dense_ffn: Some(dense_ffn),
            norm: dense(store, "mtp.norm.weight")?,
            fp8_experts: None,
        });
    }

    let shared_expert = DenseExpertWeight {
        gate_proj: load(&format!("{mlp}.shared_expert.gate_proj.weight"))?,
        up_proj: load(&format!("{mlp}.shared_expert.up_proj.weight"))?,
        down_proj: load(&format!("{mlp}.shared_expert.down_proj.weight"))?,
    };

    // 2026-09-25: Two expert layouts: per expert
    // (`experts.{e}.{gate,up,down}_proj.weight`), or stacked
    // (`experts.gate_up_proj` `[E, 2*I, H]` and `experts.down_proj` `[E, H, I]`,
    // both present), which `load_mtp_experts_stacked` slices without copying.
    let stacked_gate_up = format!("{mlp}.experts.gate_up_proj");
    let stacked_down = format!("{mlp}.experts.down_proj");
    let stacked = store.contains(&stacked_gate_up) && store.contains(&stacked_down);
    let experts = if stacked {
        load_mtp_experts_stacked(store, mlp, num_experts)?
    } else {
        let mut v = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            v.push(DenseExpertWeight {
                gate_proj: load(&format!("{mlp}.experts.{e}.gate_proj.weight"))?,
                up_proj: load(&format!("{mlp}.experts.{e}.up_proj.weight"))?,
                down_proj: load(&format!("{mlp}.experts.{e}.down_proj.weight"))?,
            });
        }
        v
    };
    // 2026-09-25: Not under `Bf16Raw`: there `load` hands out store pointers,
    // which `release_bf16_expert_dequants` would free. Not for stacked experts,
    // whose entries alias the stacked tensors.
    let fp8_experts =
        if !stacked && variant != Nvfp4Variant::Bf16Raw && mtp_experts_native_fp8(store, mlp) {
            Some(load_mtp_fp8_experts(store, mlp, num_experts, gpu)?)
        } else {
            None
        };

    Ok(MtpWeights {
        pre_fc_norm_embedding: dense(store, "mtp.pre_fc_norm_embedding.weight")?,
        pre_fc_norm_hidden: dense(store, "mtp.pre_fc_norm_hidden.weight")?,
        fc: load("mtp.fc.weight")?,
        input_layernorm: dense(store, "mtp.layers.0.input_layernorm.weight")?,
        q_proj: load(&format!("{p}.q_proj.weight"))?,
        k_proj: load(&format!("{p}.k_proj.weight"))?,
        v_proj: load(&format!("{p}.v_proj.weight"))?,
        o_proj: load(&format!("{p}.o_proj.weight"))?,
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        post_attn_layernorm: dense(store, "mtp.layers.0.post_attention_layernorm.weight")?,
        moe_gate: dense(store, &format!("{mlp}.gate.weight"))?,
        shared_expert,
        shared_expert_gate: dense(store, &format!("{mlp}.shared_expert_gate.weight"))?,
        experts,
        dense_ffn: None,
        norm: dense(store, "mtp.norm.weight")?,
        fp8_experts,
    })
}

/// 2026-09-25: True when expert 0's `gate_proj.weight` is FP8 E4M3 with a
/// `weight_scale_inv` sibling. Only expert 0 is probed; an expert in another
/// format then fails the FP8 dtype check in `load_fp8_block_scaled_as_fp8weight`.
pub(crate) fn mtp_experts_native_fp8(store: &WeightStore, mlp: &str) -> bool {
    let weight = format!("{mlp}.experts.0.gate_proj.weight");
    store
        .get(&weight)
        .map(|w| w.dtype == WeightDtype::FP8E4M3)
        .unwrap_or(false)
        && store.contains(&format!("{mlp}.experts.0.gate_proj.weight_scale_inv"))
}

/// 2026-09-25: The MTP routed and shared experts as FP8 block-scaled tables.
/// The FP8 bytes are the store's; the FP32 block scales are new allocations.
fn load_mtp_fp8_experts(
    store: &WeightStore,
    mlp: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
) -> Result<MtpFp8Experts> {
    let proj = |prefix: &str, name: &str| -> Result<Fp8Weight> {
        load_fp8_block_scaled_as_fp8weight(store, &format!("{prefix}.{name}"), gpu)
    };
    let load = |prefix: &str| -> Result<Fp8ExpertWeight> {
        Ok(Fp8ExpertWeight {
            gate_proj: proj(prefix, "gate_proj")?,
            up_proj: proj(prefix, "up_proj")?,
            down_proj: proj(prefix, "down_proj")?,
        })
    };
    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        experts.push(
            load(&format!("{mlp}.experts.{e}"))
                .with_context(|| format!("MTP expert {e}: native FP8 tables"))?,
        );
    }
    let shared_expert =
        load(&format!("{mlp}.shared_expert")).context("MTP shared expert: native FP8 tables")?;
    Ok(MtpFp8Experts {
        experts,
        shared_expert,
    })
}

#[cfg(test)]
mod mtp_fp8_detect_tests {
    use super::*;
    use metrale_model_weights::weights::WeightTensor;
    use std::collections::HashMap;

    const MLP: &str = "mtp.layers.0.mlp";

    fn store(entries: Vec<(String, WeightDtype)>) -> WeightStore {
        let mut m = HashMap::new();
        for (name, dtype) in entries {
            m.insert(
                name,
                WeightTensor {
                    ptr: DevicePtr::NULL,
                    shape: vec![512, 2048],
                    dtype,
                },
            );
        }
        WeightStore::from_map(m)
    }

    fn gate0(suffix: &str) -> String {
        format!("{MLP}.experts.0.gate_proj.{suffix}")
    }

    #[test]
    fn fp8_weight_with_block_scale_is_native_fp8() {
        let s = store(vec![
            (gate0("weight"), WeightDtype::FP8E4M3),
            (gate0("weight_scale_inv"), WeightDtype::BF16),
        ]);
        assert!(mtp_experts_native_fp8(&s, MLP));
    }

    #[test]
    fn bf16_experts_are_not_native_fp8() {
        let s = store(vec![(gate0("weight"), WeightDtype::BF16)]);
        assert!(!mtp_experts_native_fp8(&s, MLP));
    }

    #[test]
    fn fp8_without_a_block_scale_is_refused() {
        let s = store(vec![(gate0("weight"), WeightDtype::FP8E4M3)]);
        assert!(!mtp_experts_native_fp8(&s, MLP));
    }

    #[test]
    fn absent_experts_are_not_native_fp8() {
        assert!(!mtp_experts_native_fp8(&store(vec![]), MLP));
    }
}
