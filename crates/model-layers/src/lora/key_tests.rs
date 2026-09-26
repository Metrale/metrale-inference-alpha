// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `classify_key` (accepted and refused tensor names)
//! and `adapter_id_hash` (pinned values, the generation fold, the base id).
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use crate::lora::test_support::*;
use crate::lora::*;

fn reject(key: &str, cfg: &metrale_config::ModelConfig, tag: &str) {
    let err = classify_key(key, cfg).unwrap_err().to_string();
    assert!(err.contains(tag), "expected {tag} in: {err}");
}

#[test]
fn classify_key_maps_supported_and_rejects_unsupported() {
    let cfg = cfg();
    // 2026-09-25: The factory config's full-attention layers are 3, 7, 11, …, 47.
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.3.self_attn.k_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (3, LoraTarget::Attn(LoraModule::KProj), AdapterAb::A)
    );
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.3.self_attn.v_proj.lora_B.weight",
            &cfg
        )
        .unwrap(),
        (3, LoraTarget::Attn(LoraModule::VProj), AdapterAb::B)
    );
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.7.self_attn.o_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (7, LoraTarget::Attn(LoraModule::OProj), AdapterAb::A)
    );
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.11.mlp.gate_proj.lora_B.weight",
            &cfg
        )
        .unwrap(),
        (11, LoraTarget::Attn(LoraModule::GateProj), AdapterAb::B)
    );
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.47.mlp.down_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (47, LoraTarget::Attn(LoraModule::DownProj), AdapterAb::A)
    );

    assert_eq!(
        classify_key(
            "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (3, LoraTarget::Attn(LoraModule::QProj), AdapterAb::A)
    );

    // 2026-09-25: Layer 0 is a linear-attention layer.
    reject(
        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight",
        &cfg,
        "REJECT[non-full-attention-layer]",
    );
    reject(
        "base_model.model.model.layers.3.linear_attn.in_proj_qkv.lora_A.weight",
        &cfg,
        "REJECT[gdn-target]",
    );
    reject(
        "model.layers.3.self_attn.k_proj.weight",
        &cfg,
        "REJECT[not-peft-key]",
    );
}

#[test]
fn classify_key_maps_experts_and_router() {
    // 2026-09-25: The factory config has 512 experts.
    let cfg = cfg();
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.7.mlp.experts.42.gate_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (
            7,
            LoraTarget::Expert {
                n: 42,
                proj: ExpertProj::Gate
            },
            AdapterAb::A
        )
    );
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.11.mlp.experts.0.down_proj.lora_B.weight",
            &cfg
        )
        .unwrap(),
        (
            11,
            LoraTarget::Expert {
                n: 0,
                proj: ExpertProj::Down
            },
            AdapterAb::B
        )
    );
    // 2026-09-25: Whatever precedes `.layers.` is ignored.
    assert_eq!(
        classify_key(
            "base_model.model.model.language_model.layers.7.mlp.experts.3.up_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (
            7,
            LoraTarget::Expert {
                n: 3,
                proj: ExpertProj::Up
            },
            AdapterAb::A
        )
    );
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.3.mlp.gate.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (3, LoraTarget::Router, AdapterAb::A)
    );

    // 2026-09-25: Expert and router targets are accepted on linear-attention
    // layer 0; an attention target there is refused.
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.0.mlp.experts.5.down_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (
            0,
            LoraTarget::Expert {
                n: 5,
                proj: ExpertProj::Down
            },
            AdapterAb::A
        )
    );
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.0.mlp.gate.lora_B.weight",
            &cfg
        )
        .unwrap(),
        (0, LoraTarget::Router, AdapterAb::B)
    );
    reject(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        &cfg,
        "REJECT[non-full-attention-layer]",
    );

    reject(
        "base_model.model.model.layers.3.mlp.experts.999.gate_proj.lora_A.weight",
        &cfg,
        "REJECT[expert-out-of-range]",
    );
    reject(
        "base_model.model.model.layers.3.mlp.experts.gate_up_proj.lora_A.weight",
        &cfg,
        "REJECT[fused-expert-lora]",
    );
    reject(
        "base_model.model.model.layers.3.mlp.experts.5.gate_up_proj.lora_A.weight",
        &cfg,
        "REJECT[fused-expert-lora]",
    );
    reject(
        "base_model.model.model.layers.3.mlp.experts.5.wat_proj.lora_A.weight",
        &cfg,
        "REJECT[unsupported-expert-proj]",
    );
}

#[test]
fn classify_key_rejects_experts_on_dense_model() {
    let mut dense = cfg();
    dense.num_experts = 0;
    reject(
        "base_model.model.model.layers.3.mlp.experts.0.gate_proj.lora_A.weight",
        &dense,
        "REJECT[expert-lora-on-dense-model]",
    );
}

#[test]
fn adapter_id_hash_is_stable_and_base_reserved() {
    assert_eq!(adapter_id_hash("lyra", 0), adapter_id_hash("lyra", 0));
    assert_eq!(adapter_id_hash("lyra", 0), 0x48c6_19ad_b6dd_2037);
    assert_ne!(adapter_id_hash("lyra", 0), adapter_id_hash("vega", 0));
    assert_ne!(adapter_id_hash("", 0), 0);
    assert_ne!(adapter_id_hash("anything", 0), 0);
}

#[test]
fn adapter_id_hash_generation_changes_id_but_never_base() {
    for name in ["lyra", "vega", ""] {
        let g0 = adapter_id_hash(name, 0);
        let g1 = adapter_id_hash(name, 1);
        let g2 = adapter_id_hash(name, 2);
        assert_ne!(g0, g1, "generation bump must change the id ({name})");
        assert_ne!(g1, g2, "each generation is distinct ({name})");
        assert_ne!(g0, 0, "gen 0 never aliases base ({name})");
        assert_ne!(g1, 0, "gen 1 never aliases base ({name})");
        assert_ne!(g2, 0, "gen 2 never aliases base ({name})");
        assert_eq!(g1, adapter_id_hash(name, 1));
    }
    assert_eq!(adapter_id_hash("lyra", 1), 0x81d3_fdd9_ac3a_c2f6);
    assert_eq!(adapter_id_hash("lyra", 2), 0x62d9_36d0_a14b_78d5);
}
