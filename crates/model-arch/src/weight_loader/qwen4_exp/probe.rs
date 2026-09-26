// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Namespace audit for `qwen4_exp`: counts what the store holds
//! per weight family, logs it, and refuses a store the loader cannot build.
//! `load_layers` runs it before building any layer.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use anyhow::{Result, ensure};
use metrale_config::ModelConfig;
use metrale_model_weights::weights::WeightStore;

/// 2026-09-25: Per-family counts of what the store holds.
#[derive(Debug, Default, Clone, Copy)]
pub struct NamespaceReport {
    pub layers: usize,
    pub gdn_layers: usize,
    pub attn_layers: usize,
    pub expert_tensors: usize,
    pub hc_tensors: usize,
    pub indexer_tensors: usize,
    pub ple_shards: usize,
    pub vision_tensors: usize,
    pub has_embed: bool,
    pub has_lm_head: bool,
    /// 2026-09-25: Layers with an `input_layernorm` or
    /// `post_attention_layernorm` tensor; `ensure_loadable` requires none.
    pub stray_layer_norms: usize,
}

pub fn audit_namespace(store: &WeightStore, config: &ModelConfig) -> NamespaceReport {
    let mut r = NamespaceReport {
        layers: config.num_hidden_layers,
        ..Default::default()
    };
    let pfx = if config.weight_prefix.is_empty() {
        "model".to_string()
    } else {
        config.weight_prefix.clone()
    };
    r.has_embed = store.contains(&format!("{pfx}.embed_tokens.weight"));
    r.has_lm_head = store.contains("lm_head.weight");

    for i in 0..config.num_hidden_layers {
        let lp = config.layer_prefix(i);
        if store.contains(&format!("{lp}.linear_attn.in_proj_qkv.weight")) {
            r.gdn_layers += 1;
        }
        if store.contains(&format!("{lp}.self_attn.q_proj.weight")) {
            r.attn_layers += 1;
        }
        if store.contains(&format!("{lp}.self_attn.indexer.index_qk_proj.weight")) {
            r.indexer_tensors += 1;
        }
        if store.contains(&format!("{lp}.attn_hyper_connection.hc_norm.weight")) {
            r.hc_tensors += 1;
        }
        if store.contains(&format!("{lp}.mlp_hyper_connection.hc_norm.weight")) {
            r.hc_tensors += 1;
        }
        // 2026-09-25: Probe the first local expert (`is_local_expert`), not
        // expert 0: with expert parallelism a rank's store may not hold expert 0.
        let first_local = (0..config.num_experts)
            .find(|e| config.is_local_expert(*e))
            .unwrap_or(0);
        if store.contains(&format!("{lp}.mlp.experts.{first_local}.gate_proj.weight")) {
            r.expert_tensors += 1;
        }
        if store.contains(&format!("{lp}.input_layernorm.weight"))
            || store.contains(&format!("{lp}.post_attention_layernorm.weight"))
        {
            r.stray_layer_norms += 1;
        }
    }

    for &l in &config.ple_layer_ids {
        let lp = config.layer_prefix(l);
        let base = format!("{lp}.ple.ple_embedding.ngram_embedding");
        let parts = if config.ngram_split_parts > 0 {
            config.ngram_split_parts
        } else {
            0
        };
        for s in 0..parts {
            if store.contains(&format!("{base}.shard_{s}.weight")) {
                r.ple_shards += 1;
            }
        }
    }

    for b in 0..64usize {
        if store.contains(&format!("model.visual.blocks.{b}.attn.qkv.weight")) {
            r.vision_tensors += 1;
        }
    }
    r
}

impl NamespaceReport {
    pub fn log(&self) {
        tracing::info!(
            "qwen4_exp namespace: {}/{} GDN + {} full-attention layers, \
             {} MoE blocks, {} hc blocks, {} indexers, {} PLE shards, \
             {} ViT blocks, embed={} lm_head={}",
            self.gdn_layers,
            self.layers,
            self.attn_layers,
            self.expert_tensors,
            self.hc_tensors,
            self.indexer_tensors,
            self.ple_shards,
            self.vision_tensors,
            self.has_embed,
            self.has_lm_head,
        );
        if self.stray_layer_norms == 0 {
            tracing::info!(
                "qwen4_exp: no per-layer input/post-attention norms, as expected — \
                 this architecture keeps normalization inside the \
                 hyper-connection blocks"
            );
        }
    }

    /// 2026-09-25: Refuse a store without `embed_tokens` or `lm_head.weight`,
    /// with GDN + full-attention layers not equal to the layer count, with a
    /// layer lacking its first local expert, or with an `input_layernorm` or
    /// `post_attention_layernorm` on any layer.
    pub fn ensure_loadable(&self) -> Result<()> {
        ensure!(
            self.has_embed && self.has_lm_head,
            "qwen4_exp: embed_tokens={} lm_head={} — checkpoint is not the \
             expected multimodal layout (is `weight_prefix` set to \
             `model.language_model`?)",
            self.has_embed,
            self.has_lm_head,
        );
        ensure!(
            self.gdn_layers + self.attn_layers == self.layers,
            "qwen4_exp: {} GDN + {} full-attention = {} but the config declares \
             {} layers — every layer must resolve to exactly one mixer",
            self.gdn_layers,
            self.attn_layers,
            self.gdn_layers + self.attn_layers,
            self.layers,
        );
        ensure!(
            self.expert_tensors == self.layers,
            "qwen4_exp: {} of {} layers have routed experts — a layer whose \
             experts silently fail to resolve would route into nothing",
            self.expert_tensors,
            self.layers,
        );
        ensure!(
            self.stray_layer_norms == 0,
            "qwen4_exp: {} layer(s) carry input/post-attention norms, which \
             this architecture should not have — normalization is supposed to \
             live inside the hyper-connection blocks. The mHC reading of this \
             checkpoint is wrong; refusing rather than loading on a bad premise",
            self.stray_layer_norms,
        );
        Ok(())
    }
}
