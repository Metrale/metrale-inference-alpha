// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`doomed_after_load`], the store tensors
//! `Qwen35DenseWeightLoader::prune_after_load` frees.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants:
//! - A tensor is named only for a layer where `ffn_gateup_fused_selected` or
//!   `gdn_fp8_arm_selected` holds, the predicates `load_layers` builds by.

use metrale_config::{LayerType, ModelConfig};
use metrale_model_layers::weight_map::detect_nvfp4_variant;
use metrale_model_weights::weights::WeightStore;

use super::{ffn_gateup_fused_selected, gdn_fp8_arm_selected};

/// 2026-09-26: The names of the store tensors `load_layers` copied into fused buffers and
/// no longer reads.
pub(super) fn doomed_after_load(
    store: &WeightStore,
    config: &ModelConfig,
) -> std::collections::HashSet<String> {
    // 2026-09-25: The same `layer_types` resolution as `load_layers`.
    let layer_types = if config.layer_types.is_empty() {
        (0..config.num_hidden_layers)
            .map(|i| config.layer_type(i))
            .collect::<Vec<_>>()
    } else {
        config.layer_types.clone()
    };
    let mut doomed: std::collections::HashSet<String> = std::collections::HashSet::new();
    // 2026-09-25: With the fused FFN gate+up, `gate_proj` and `up_proj` are views into the
    // fused buffer, so their store tensors are unused; `down_proj` is still bound
    // zero-copy and is kept. `variant` is recomputed by the same pure
    // `detect_nvfp4_variant` the load used, so the predicate sees the same inputs.
    let variant = detect_nvfp4_variant(store, config);
    for i in 0..layer_types.len() {
        let lp = config.layer_prefix(i);
        if !ffn_gateup_fused_selected(store, config, variant, &lp) {
            continue;
        }
        for proj in ["gate_proj", "up_proj"] {
            for leaf in ["weight", "weight_scale_inv", "weight_scale"] {
                doomed.insert(format!("{lp}.mlp.{proj}.{leaf}"));
            }
        }
    }
    for (i, lt) in layer_types.iter().enumerate() {
        if *lt != LayerType::LinearAttention {
            continue;
        }
        let la = format!("{}.linear_attn", config.layer_prefix(i));
        if !gdn_fp8_arm_selected(store, &la, config.tp_world_size) {
            continue;
        }
        for leaf in [
            "in_proj_qkv.weight",
            "in_proj_qkv.weight_scale_inv",
            "in_proj_qkv.weight_scale",
            "in_proj_z.weight",
            "in_proj_z.weight_scale_inv",
            "in_proj_z.weight_scale",
            "in_proj_a.weight",
            "in_proj_b.weight",
        ] {
            doomed.insert(format!("{la}.{leaf}"));
        }
    }
    doomed
}
