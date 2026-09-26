// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `StateBudget` (per-sequence state bytes) and `StructuralAccounting` (checkpoint names against the skeleton's contract).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

/// 2026-09-25: Per-sequence state bytes, built by `Glm5NextTextSkeleton::state_budget` from
/// the config's geometry. `per_rank` divides only the KDA state and the DSA KV.
#[derive(Debug, Clone, Copy)]
pub struct StateBudget {
    /// 2026-09-25: KDA recurrent state, `[heads, head_dim, head_dim]` fp32 per KDA layer,
    /// summed over the KDA layers. Does not grow with sequence length.
    pub kda_recurrent: usize,
    /// 2026-09-25: KDA causal-conv window, bf16, `conv_dim x (kernel - 1 + num_spec)` per KDA
    /// layer. Does not grow with sequence length.
    pub kda_conv: usize,
    /// 2026-09-25: DSA MLA KV per token across the text-stack DSA layers, `kv_lora_rank` x 2
    /// bytes each. Grows with sequence length.
    pub dsa_kv_per_token: usize,
    /// 2026-09-25: Indexer key and gate state per token across the text-stack DSA layers.
    /// Grows with sequence length. The indexer projections are `DsaShard::Replicated`
    /// (`glm5next_dsa/tp.rs`), so every rank holds the whole cache and [`Self::per_rank`]
    /// does not divide it.
    pub dsa_indexer_per_token: usize,
    /// 2026-09-25: mHC highway per token, `hc_mult x hidden` fp32. An activation, so neither
    /// `fixed` nor `per_token` counts it.
    pub mhc_highway_per_token: usize,
    /// 2026-09-25: MoE routing scratch per token: f32 logits plus 4-byte top-k ids and weights.
    pub moe_routing_per_token: usize,
}

impl StateBudget {
    /// 2026-09-25: Fixed (sequence-length-independent) bytes per sequence.
    pub fn fixed(&self) -> usize {
        self.kda_recurrent + self.kda_conv
    }
    /// 2026-09-25: Bytes that grow with every token of context.
    pub fn per_token(&self) -> usize {
        self.dsa_kv_per_token + self.dsa_indexer_per_token
    }
    /// 2026-09-25: Total persistent state for a sequence of `tokens`.
    pub fn for_sequence(&self, tokens: usize) -> usize {
        self.fixed() + tokens * self.per_token()
    }
    /// 2026-09-25: Divides `kda_recurrent`, `kda_conv` and `dsa_kv_per_token` by `ep`; the
    /// indexer cache, the mHC highway and the routing scratch stay whole.
    pub fn per_rank(&self, ep: usize) -> StateBudget {
        StateBudget {
            kda_recurrent: self.kda_recurrent / ep,
            kda_conv: self.kda_conv / ep,
            dsa_kv_per_token: self.dsa_kv_per_token / ep,
            // 2026-09-25: Not divided: the indexer projections are `DsaShard::Replicated`, so
            // every rank holds the whole cache.
            dsa_indexer_per_token: self.dsa_indexer_per_token,
            mhc_highway_per_token: self.mhc_highway_per_token,
            moe_routing_per_token: self.moe_routing_per_token,
        }
    }
}

#[derive(Debug)]
pub struct StructuralAccounting {
    pub required: usize,
    pub bound: usize,
    /// 2026-09-25: Structural tensors the checkpoint does not have. `is_complete` requires
    /// it empty.
    pub missing: Vec<String>,
    /// 2026-09-25: Names that are neither structural nor MLP/vision. `is_complete` requires
    /// it empty.
    pub unexpected: Vec<String>,
    /// 2026-09-25: Count of MLP/MoE and vision names, which the skeleton does not bind.
    pub deferred: usize,
}

impl StructuralAccounting {
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty() && self.bound == self.required
    }
}
