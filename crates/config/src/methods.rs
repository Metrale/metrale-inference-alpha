// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Derived getters and small predicates over `ModelConfig` fields.
//!
//! Owner: config.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use super::{LayerType, ModelConfig};

impl ModelConfig {
    /// 2026-09-26: Every configured stop-token id, primary first; `[eos_token_id]` when
    /// `eos_token_ids` is empty.
    pub fn eos_ids(&self) -> Vec<u32> {
        if self.eos_token_ids.is_empty() {
            vec![self.eos_token_id]
        } else {
            self.eos_token_ids.clone()
        }
    }

    pub fn is_eos(&self, id: u32) -> bool {
        if self.eos_token_ids.is_empty() {
            id == self.eos_token_id
        } else {
            self.eos_token_ids.contains(&id)
        }
    }

    /// 2026-09-26: Q heads per KV head; 1 when `num_key_value_heads` is 0.
    pub fn gqa_ratio(&self) -> usize {
        self.num_attention_heads
            .checked_div(self.num_key_value_heads)
            .unwrap_or(1)
    }

    /// 2026-09-26: Layer kind for any index in the checkpoint. Text-stack indices go through
    /// `layer_type`; indices `>= num_hidden_layers` are MTP/NextN layers and resolve through
    /// `mtp_layer_types`, and past both the answer is `None`.
    pub fn layer_type_at(&self, layer_idx: usize) -> Option<LayerType> {
        if layer_idx < self.num_hidden_layers {
            return Some(self.layer_type(layer_idx));
        }
        self.mtp_layer_types
            .get(layer_idx - self.num_hidden_layers)
            .copied()
    }

    /// 2026-09-26: Text-stack layers whose mixer is `LayerType::SparseAttention`.
    pub fn sparse_attention_layers(&self) -> Vec<usize> {
        self.layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::SparseAttention)
            .map(|(i, _)| i)
            .collect()
    }

    /// 2026-09-26: True when any layer, in the text stack or the MTP layers, is sparse
    /// attention.
    pub fn has_sparse_attention(&self) -> bool {
        self.layer_types.contains(&LayerType::SparseAttention)
            || self.mtp_layer_types.contains(&LayerType::SparseAttention)
    }

    /// 2026-09-26: Layer kind of a text-stack index: `layer_types[layer_idx]` (`FullAttention`
    /// past its end), or, when `layer_types` is empty, `FullAttention` on every
    /// `full_attention_interval`-th layer and `LinearAttention` elsewhere.
    pub fn layer_type(&self, layer_idx: usize) -> LayerType {
        if !self.layer_types.is_empty() {
            self.layer_types
                .get(layer_idx)
                .cloned()
                .unwrap_or(LayerType::FullAttention)
        } else if self.full_attention_interval > 0
            && (layer_idx + 1).is_multiple_of(self.full_attention_interval)
        {
            LayerType::FullAttention
        } else {
            LayerType::LinearAttention
        }
    }

    /// 2026-09-26: Number of layers that read the paged KV cache: every layer for which
    /// `LayerType::is_attention` holds (full, sliding and sparse). Without `layer_types`,
    /// `num_hidden_layers / full_attention_interval`, or every layer when the interval is 0.
    pub fn num_attention_layers(&self) -> usize {
        if !self.layer_types.is_empty() {
            self.layer_types.iter().filter(|t| t.is_attention()).count()
        } else {
            self.num_hidden_layers
                .checked_div(self.full_attention_interval)
                .unwrap_or(self.num_hidden_layers)
        }
    }

    pub fn num_ssm_layers(&self) -> usize {
        if !self.layer_types.is_empty() {
            self.layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count()
        } else {
            self.num_hidden_layers - self.num_attention_layers()
        }
    }

    /// 2026-09-26: Whether the model carries recurrent (SSM / linear-attention) state:
    /// `num_ssm_layers() > 0`. An SSM snapshot tier requested for a model without it is a
    /// startup error (`ensure_ssm_tier_capability_from`).
    pub fn has_recurrent_state(&self) -> bool {
        self.num_ssm_layers() > 0
    }

    pub fn has_experts(&self) -> bool {
        self.num_experts > 0
    }

    pub fn rotary_dim(&self) -> usize {
        if self.rotary_dim > 0 {
            self.rotary_dim
        } else {
            (self.partial_rotary_factor * self.head_dim as f64) as usize
        }
    }

    pub fn ssm_qkvz_size(&self) -> usize {
        let q = self.linear_num_key_heads * self.linear_key_head_dim;
        let k = self.linear_num_key_heads * self.linear_key_head_dim;
        let v = self.linear_num_value_heads * self.linear_value_head_dim;
        let z = self.linear_num_value_heads * self.linear_value_head_dim;
        q + k + v + z
    }

    pub fn ssm_qkv_size(&self) -> usize {
        let q = self.linear_num_key_heads * self.linear_key_head_dim;
        let k = self.linear_num_key_heads * self.linear_key_head_dim;
        let v = self.linear_num_value_heads * self.linear_value_head_dim;
        q + k + v
    }

    pub fn ssm_z_size(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn ssm_ba_size(&self) -> usize {
        self.linear_num_value_heads * 2
    }

    /// 2026-09-26: `(start, end)`, end exclusive, of the experts this EP rank owns; the last
    /// rank also takes the remainder of `num_experts / ep_world_size`.
    pub fn local_expert_range(&self) -> (usize, usize) {
        if self.ep_world_size <= 1 {
            return (0, self.num_experts);
        }
        let per_rank = self.num_experts / self.ep_world_size;
        let start = self.ep_rank * per_rank;
        let end = if self.ep_rank == self.ep_world_size - 1 {
            self.num_experts
        } else {
            start + per_rank
        };
        (start, end)
    }

    pub fn is_local_expert(&self, expert_id: usize) -> bool {
        let (start, end) = self.local_expert_range();
        expert_id >= start && expert_id < end
    }

    /// 2026-09-26: Range `[start, end)` of a `total`-sized dimension owned by this TP rank.
    /// `total` must be divisible by `tp_world_size` (debug-asserted). `(0, total)` when
    /// `tp_world_size <= 1`.
    pub fn tp_shard_range(&self, total: usize) -> (usize, usize) {
        if self.tp_world_size <= 1 {
            return (0, total);
        }
        debug_assert!(
            total.is_multiple_of(self.tp_world_size),
            "tp_shard_range: total={} not divisible by tp_world_size={}",
            total,
            self.tp_world_size,
        );
        let per_rank = total / self.tp_world_size;
        let start = self.tp_rank * per_rank;
        (start, start + per_rank)
    }

    pub fn tp_shard_dim(&self, total: usize) -> usize {
        if self.tp_world_size <= 1 {
            return total;
        }
        total / self.tp_world_size
    }

    /// 2026-09-26: `{weight_prefix}.layers.{layer_idx}`, with `model` when `weight_prefix`
    /// is empty.
    pub fn layer_prefix(&self, layer_idx: usize) -> String {
        if self.weight_prefix.is_empty() {
            format!("model.layers.{layer_idx}")
        } else {
            format!("{}.layers.{layer_idx}", self.weight_prefix)
        }
    }

    pub fn capabilities(&self) -> crate::capabilities::ModelCapabilities {
        crate::capabilities::ModelCapabilities::from_config(self)
    }

    pub fn is_qwen35(&self) -> bool {
        self.model_type == "qwen3_5_moe"
    }

    pub fn is_qwen35_dense(&self) -> bool {
        self.model_type == "qwen3_5" && self.num_experts == 0
    }

    pub fn is_qwen3_vl(&self) -> bool {
        if self.model_type == "qwen3_vl_moe" {
            return true;
        }
        if self.model_type == "qwen3_5" && self.vision.is_some() {
            return true;
        }
        false
    }

    /// 2026-09-26: Whether to keep the LM head in BF16 instead of quantizing it to NVFP4 at
    /// load. `lm_head_bf16_override` (set by serve's `--lm-head-dtype`) wins when set;
    /// otherwise yes for MLA models (`kv_lora_rank > 0`), for `laguna`, and for dense
    /// `gemma4` unless `METRALE_GEMMA4_LMHEAD_NVFP4=1`, and no for the rest.
    pub fn skip_lm_head_quantization(&self) -> bool {
        if let Some(force_bf16) = self.lm_head_bf16_override {
            return force_bf16;
        }
        if self.kv_lora_rank > 0 {
            return true;
        }
        if self.model_type == "laguna" {
            return true;
        }
        if self.model_type == "gemma4" && self.num_experts == 0 {
            return std::env::var("METRALE_GEMMA4_LMHEAD_NVFP4").ok().as_deref() != Some("1");
        }
        false
    }

    pub fn mamba2_d_inner(&self) -> usize {
        self.mamba_num_heads * self.mamba_head_dim
    }

    pub fn mamba2_d_xbc(&self) -> usize {
        self.mamba2_d_inner() + 2 * self.n_groups * self.ssm_state_size
    }

    pub fn mamba2_in_proj_size(&self) -> usize {
        self.mamba2_d_inner() + self.mamba2_d_xbc() + self.mamba_num_heads
    }

    pub fn ssm_h_state_bytes(&self) -> usize {
        if self.mamba_num_heads > 0 && self.mamba_head_dim > 0 {
            self.mamba_num_heads * self.mamba_head_dim * self.ssm_state_size * 4
        } else {
            self.linear_num_value_heads * self.linear_value_head_dim * self.linear_key_head_dim * 4
        }
    }

    pub fn ssm_conv_state_bytes(&self) -> usize {
        let d_conv = self.linear_conv_kernel_dim;
        if self.mamba_num_heads > 0 && self.mamba_head_dim > 0 {
            self.mamba2_d_xbc() * d_conv * 4
        } else {
            let conv_dim = self.linear_num_key_heads * self.linear_key_head_dim * 2
                + self.linear_num_value_heads * self.linear_value_head_dim;
            conv_dim * d_conv * 4
        }
    }

    pub fn ssm_state_norm_dims(&self) -> (usize, usize, usize) {
        if self.mamba_num_heads > 0 && self.mamba_head_dim > 0 {
            (
                self.mamba_num_heads,
                self.mamba_head_dim,
                self.ssm_state_size,
            )
        } else {
            (
                self.linear_num_value_heads,
                self.linear_key_head_dim,
                self.linear_value_head_dim,
            )
        }
    }

    pub fn moe_input_size(&self) -> usize {
        if self.moe_latent_size > 0 {
            self.moe_latent_size
        } else {
            self.hidden_size
        }
    }

    /// 2026-09-26: Routed-expert intermediate size for `layer`: its entry in
    /// `moe_intermediate_sizes` when present and non-zero (Puzzle schedules), else the scalar.
    pub fn moe_intermediate_size_for(&self, layer: usize) -> usize {
        self.moe_intermediate_sizes
            .get(layer)
            .copied()
            .filter(|&s| s > 0)
            .unwrap_or(self.moe_intermediate_size)
    }

    /// 2026-09-26: Top-k experts per token for `layer`, by the same rule over
    /// `num_experts_per_toks`.
    pub fn num_experts_per_tok_for(&self, layer: usize) -> usize {
        self.num_experts_per_toks
            .get(layer)
            .copied()
            .filter(|&k| k > 0)
            .unwrap_or(self.num_experts_per_tok)
    }

    pub fn max_moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_sizes
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(self.moe_intermediate_size)
    }

    pub fn num_moe_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| **t == LayerType::Moe)
            .count()
    }
}
