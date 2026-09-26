// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: What a PEFT LoRA tensor targets: a [`LoraModule`], the MoE
//! router (`mlp.gate`), or one routed expert's projection
//! (`mlp.experts.{n}.{gate,up,down}_proj`), plus the dims and pool sizing for
//! the router and expert targets.
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use metrale_config::ModelConfig;

use crate::layers::ops::lora_delta::LoraPair;

use super::LoraModule;

/// 2026-09-25: Which routed-expert projection a delta targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExpertProj {
    Gate,
    Up,
    Down,
}

impl ExpertProj {
    /// 2026-09-25: The projection's name in a PEFT tensor key.
    pub fn peft_name(&self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }

    /// 2026-09-25: `(out_dim, in_dim)` of the routed-expert projection on
    /// `layer`. The width is [`ModelConfig::moe_intermediate_size_for`], not
    /// the dense `intermediate_size`.
    pub fn dims(&self, cfg: &ModelConfig, layer: usize) -> (usize, usize) {
        let h = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size_for(layer);
        match self {
            Self::Gate | Self::Up => (inter, h),
            Self::Down => (h, inter),
        }
    }
}

/// 2026-09-25: `(out_dim, in_dim)` of the router (`mlp.gate`) projection. Its
/// delta is added to the routing logits before top-k.
pub fn router_dims(cfg: &ModelConfig) -> (usize, usize) {
    (cfg.num_experts, cfg.hidden_size)
}

/// 2026-09-25: The decoded target of one PEFT LoRA tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraTarget {
    /// 2026-09-25: A projection packed into the equal-size slot pool.
    Attn(LoraModule),
    /// 2026-09-25: The MoE router, `mlp.gate`.
    Router,
    /// 2026-09-25: One routed expert's projection, `mlp.experts.{n}.{proj}`.
    Expert { n: u16, proj: ExpertProj },
}

/// 2026-09-25: One MoE layer's routed-expert pairs, keyed by
/// `(expert_index, projection)`; only adapted experts have entries.
#[derive(Clone, Default)]
pub struct ExpertLoraLayer {
    pub pairs: BTreeMap<(u16, ExpertProj), LoraPair>,
}

impl ExpertLoraLayer {
    /// 2026-09-25: This layer's `(expert, proj)` pair, if adapted.
    pub fn pair(&self, expert: u16, proj: ExpertProj) -> Option<&LoraPair> {
        self.pairs.get(&(expert, proj))
    }

    /// 2026-09-25: The expert indices this layer adapts, ascending and without
    /// duplicates; `expert_apply` builds its delta work items from it.
    pub fn adapted_experts(&self) -> Vec<u16> {
        let mut v: Vec<u16> = self.pairs.keys().map(|(e, _)| *e).collect();
        v.dedup();
        v
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// 2026-09-25: Padded bytes of the expert/router pool for the given audited
/// keys: `(stride * in + out * stride) * 2` per expert key and per router
/// layer. `stride` is `expert_pack::packed_stride(max_rank)`, the stride
/// `expert_pack::pack_into` advances by, so the size matches what is packed.
pub fn expert_router_bytes(
    cfg: &ModelConfig,
    expert_keys: &[(usize, ExpertProj)],
    router_layers: &[usize],
    max_rank: usize,
) -> usize {
    let stride = super::expert_pack::packed_stride(max_rank);
    let per = |out: usize, inp: usize| (stride * inp + out * stride) * 2;
    let experts: usize = expert_keys
        .iter()
        .map(|(layer, proj)| {
            let (out, inp) = proj.dims(cfg, *layer);
            per(out, inp)
        })
        .sum();
    let routers: usize = router_layers
        .iter()
        .map(|_| {
            let (out, inp) = router_dims(cfg);
            per(out, inp)
        })
        .sum();
    experts + routers
}

#[cfg(test)]
#[path = "target_tests.rs"]
mod tests;
