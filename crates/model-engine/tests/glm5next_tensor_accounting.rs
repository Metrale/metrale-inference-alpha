// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Every tensor name pattern of the GLM-5.3 reference checkpoint is
//! classified by `weight_loader::glm5_next::classify`, with none unknown.
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! Fixture `fixtures/glm53-nvfp4-9e0d74e3-patterns.tsv` lists the tensor names of
//! `LibertAIDAI/GLM-5.3-Flash-NVFP4` snapshot `9e0d74e3…` with layer and expert
//! indices replaced by `N` and `E`, as `count<TAB>pattern`: 407 rows whose counts
//! sum to 113,074.

use metrale_model_arch::weight_loader::glm5_next::{TensorRole, account, classify};

const FIXTURE: &str = include_str!("fixtures/glm53-nvfp4-9e0d74e3-patterns.tsv");
const EXPECTED_TENSORS: usize = 113_074;
const EXPECTED_PATTERNS: usize = 407;

fn rows() -> Vec<(&'static str, usize)> {
    FIXTURE
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (c, name) = l.split_once('\t').expect("fixture row is count<TAB>name");
            (name, c.parse::<usize>().expect("count"))
        })
        .collect()
}

#[test]
fn fixture_is_the_reference_checkpoint() {
    let r = rows();
    assert_eq!(r.len(), EXPECTED_PATTERNS, "pattern count drifted");
    let total: usize = r.iter().map(|(_, c)| c).sum();
    assert_eq!(total, EXPECTED_TENSORS, "tensor count drifted");
}

/// 2026-09-25: All 113,074 tensors classified, none unknown.
#[test]
fn every_tensor_is_intentionally_classified() {
    let acc = account(rows());
    assert!(
        acc.unknown.is_empty(),
        "{} unclassified pattern(s), first 10: {:?}",
        acc.unknown.len(),
        &acc.unknown[..acc.unknown.len().min(10)]
    );
    assert_eq!(acc.total, EXPECTED_TENSORS);
    let sum: usize = acc.by_role.values().sum();
    assert_eq!(
        sum, EXPECTED_TENSORS,
        "role totals must reconcile to the checkpoint"
    );
}

/// 2026-09-25: Per-layer census from the mixer tensors: KDA layers carry
/// `A_log`, sparse-MLA layers carry `kv_b_proj`.
#[test]
fn layer_census_matches_reconciled_counts() {
    let r = rows();
    let count_of = |pat: &str| -> usize {
        r.iter()
            .find(|(n, _)| *n == pat)
            .map(|(_, c)| *c)
            .unwrap_or_else(|| panic!("pattern absent from fixture: {pat}"))
    };

    // 2026-09-25: 34 KDA layers, one A_log each, all text layers.
    assert_eq!(
        count_of("model.language_model.layers.N.self_attn.A_log"),
        34,
        "KDA layer count"
    );
    // 2026-09-25: 12 sparse-MLA layers in the checkpoint: 11 text and layer 45 (MTP).
    assert_eq!(
        count_of("model.language_model.layers.N.self_attn.kv_b_proj.weight"),
        12,
        "DSA layers incl. the MTP layer"
    );
    assert_eq!(
        count_of("model.language_model.layers.N.self_attn.indexer.wk.weight"),
        12,
        "indexer instances incl. the MTP layer"
    );
    // 2026-09-25: o_proj is on every mixer: 34 + 12 = 46.
    assert_eq!(
        count_of("model.language_model.layers.N.self_attn.o_proj.weight"),
        46
    );
    // 2026-09-25: 3 dense FFN layers (first_k_dense_replace = 3).
    assert_eq!(
        count_of("model.language_model.layers.N.mlp.gate_proj.weight"),
        3,
        "dense FFN layers"
    );
    // 2026-09-25: 43 MoE layers in the checkpoint: 42 text and layer 45.
    assert_eq!(
        count_of("model.language_model.layers.N.mlp.gate.weight"),
        43,
        "MoE router instances incl. the MTP layer"
    );
}

/// 2026-09-25: The MTP layer uses layer-45 names, with no `mtp.0.` tensor.
#[test]
fn mtp_lives_at_layer_45_and_not_under_mtp_prefix() {
    let r = rows();
    assert!(
        r.iter().any(|(n, _)| n.contains("eh_proj")),
        "no eh_proj in fixture"
    );
    assert!(
        !r.iter().any(|(n, _)| n.contains("mtp.0.")),
        "checkpoint must contain zero mtp.0.* tensors"
    );
    // 2026-09-25: eh_proj, enorm and hnorm occur once each, so there is one MTP layer.
    for p in ["eh_proj.weight", "enorm.weight", "hnorm.weight"] {
        let c: usize = r
            .iter()
            .filter(|(n, _)| n.ends_with(p))
            .map(|(_, c)| *c)
            .sum();
        assert_eq!(c, 1, "{p} should occur on exactly one layer");
    }
}

/// 2026-09-25: Every text tensor whose name contains `norm` classifies as a norm role.
#[test]
fn norm_sanity_pass() {
    let mut checked = 0usize;
    for (name, _) in rows() {
        let looks_like_norm = name.contains("norm") || name.ends_with(".norm.weight");
        if !looks_like_norm {
            continue;
        }
        let role = classify(name).unwrap_or_else(|| panic!("unclassified norm: {name}"));
        // 2026-09-25: Vision-tower tensors all classify as `Vision`, norms
        // included, so they are skipped here.
        if role == TensorRole::Vision {
            continue;
        }
        checked += 1;
        assert!(
            role.is_norm(),
            "text-model norm tensor {name} classified as non-norm {role:?}"
        );
    }
    assert!(
        checked >= 8,
        "expected several norm patterns, saw {checked}"
    );
}

/// 2026-09-25: The vision tower is present and classified as `Vision`, apart from
/// the text-model roles.
#[test]
fn vision_tower_is_classified_and_separable() {
    let acc = account(rows());
    let vision = acc.by_role.get("Vision").copied().unwrap_or(0);
    assert!(
        vision > 0,
        "vision tensors should be present and classified"
    );
    let text: usize = acc
        .by_role
        .iter()
        .filter(|(k, _)| k.as_str() != "Vision")
        .map(|(_, v)| *v)
        .sum();
    assert_eq!(text + vision, EXPECTED_TENSORS);
    assert!(
        text > vision,
        "text model should dominate: text={text} vision={vision}"
    );
}

/// 2026-09-25: Print the accounting table (visible with `--nocapture`).
#[test]
fn print_accounting_table() {
    let acc = account(rows());
    println!("\nGLM-5.3-Flash-NVFP4 @ 9e0d74e3 — tensor accounting");
    println!(
        "  shards 120 · patterns {EXPECTED_PATTERNS} · tensors {}",
        acc.total
    );
    for (role, n) in &acc.by_role {
        println!("  {role:>18} : {n:>7}");
    }
    println!("  {:>18} : {:>7}", "UNKNOWN", acc.unknown.len());
    println!("  MTP layer indices: {:?}", acc.mtp_layers);
    assert!(matches!(
        classify("lm_head.weight"),
        Some(TensorRole::LmHead)
    ));
}

/// 2026-09-25: Name patterns of the `nvidia/GLM-5.3-Flash-NVFP4` export that the
/// fixture's checkpoint lacks: the scales of its quantised dense MLP
/// (`weight_scale`, `weight_scale_2`, `input_scale`; see `glm5_next_load.rs`) and
/// `input_scale` on the routed and shared experts. The fixture has no
/// `input_scale` tensor. `account` reports a name `classify` does not know as
/// unknown, so each of these must classify.
#[test]
fn the_official_modelopt_export_adds_no_unclassified_pattern() {
    let mut extra: Vec<String> = Vec::new();
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        for leaf in ["weight_scale", "weight_scale_2", "input_scale"] {
            extra.push(format!("model.language_model.layers.N.mlp.{proj}.{leaf}"));
        }
        extra.push(format!(
            "model.language_model.layers.N.mlp.experts.E.{proj}.input_scale"
        ));
        extra.push(format!(
            "model.language_model.layers.N.mlp.shared_experts.{proj}.input_scale"
        ));
    }
    for name in &extra {
        assert!(
            classify(name).is_some(),
            "the official export's {name} is unclassified"
        );
    }
    let mut rows = rows();
    let owned: Vec<(&str, usize)> = extra.iter().map(|s| (s.as_str(), 1usize)).collect();
    rows.extend(owned);
    let acc = account(rows);
    assert!(
        acc.unknown.is_empty(),
        "unclassified with the official export's patterns: {:?}",
        acc.unknown
    );
    assert_eq!(acc.total, EXPECTED_TENSORS + extra.len());
}
