// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the model-variant step. `variants_for` is run against the committed
//! `BENCH.toml` files in this tree; the reducer tests use the synthetic `two_rows`.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use crossterm::event::{KeyCode, KeyEvent};

use super::*;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::from(code)
}

fn agentic_state() -> BenchState {
    let mut s = BenchState::default();
    s.target = metrale_bench::TargetEndpoint::local(8888, "test-model");
    let index = metrale_bench::registry::all()
        .iter()
        .position(|d| d.id == "agentic-webserver")
        .expect("registered");
    s.select(index);
    s
}

/// 2026-09-26: A synthetic pair, so the reducer tests do not depend on the committed tree.
fn two_rows() -> Vec<VariantRow> {
    let bound = |max| gate::Bound {
        min: None,
        max: Some(max),
        noise: None,
    };
    vec![
        VariantRow {
            hardware: "gb10".into(),
            checkpoint: "Qwen/Qwen3.6-35B-A3B-FP8".into(),
            title: "35B MoE flagship".into(),
            recipe: Some("qwen3.6/qwen3.6-35b-a3b-fp8-bf16head".into()),
            is_default: true,
            note: String::new(),
            metrics: vec![("sum_wall_s".into(), bound(1000.0))],
        },
        VariantRow {
            hardware: "gb10".into(),
            checkpoint: "unsloth/Qwen3.8-27B-NVFP4".into(),
            title: "dense 27B".into(),
            recipe: Some("qwen3.8/qwen3.8-27b-nvfp4-unsloth".into()),
            is_default: false,
            note: String::new(),
            metrics: vec![("sum_wall_s".into(), bound(2500.0))],
        },
    ]
}

/// 2026-09-26: In the committed tree the agentic gate is defined on the 35B MoE (the default,
/// listed first) and the dense Qwen3.8-27B, whose own `sum_wall_s` ceiling is 5000 s.
#[test]
fn the_agentic_gate_declares_both_variants_in_this_tree() {
    let rows = variants_for("agentic-webserver");
    assert!(
        rows.len() >= 2,
        "expected both variants, got {:?}",
        rows.iter().map(|r| &r.checkpoint).collect::<Vec<_>>()
    );
    assert!(rows[0].is_default, "the declared subject leads");
    assert_eq!(rows[0].checkpoint, "Qwen/Qwen3.6-35B-A3B-FP8");
    let dense = rows
        .iter()
        .find(|r| r.checkpoint == "unsloth/Qwen3.8-27B-NVFP4")
        .expect("the dense variant is declared");
    assert!(!dense.is_default, "the required subject is unchanged");
    assert_eq!(
        dense
            .metrics
            .iter()
            .find(|(k, _)| k == "sum_wall_s")
            .and_then(|(_, b)| b.max),
        Some(5000.0),
        "the dense variant carries its own wall ceiling"
    );
    assert!(
        dense.note.contains("PROVISIONAL") || dense.note.contains("2026-08-14"),
        "the provenance travels with the threshold"
    );
}

/// 2026-09-26: Entering a benchmark with variants shows them, titled with their BENCH.toml labels.
#[test]
fn entering_the_agentic_benchmark_opens_the_variant_step() {
    let mut s = agentic_state();
    s.enter_selected();
    assert_eq!(s.view, View::Variants);
    assert!(
        s.variants.iter().any(|r| r.title.contains("dense")),
        "labels from BENCH.toml title the rows: {:?}",
        s.variants.iter().map(|r| &r.title).collect::<Vec<_>>()
    );
}

/// 2026-09-26: Choosing a variant pins its checkpoint and sets `wall_budget_s` to its own ceiling.
#[test]
fn choosing_the_dense_variant_adopts_model_and_wall_budget() {
    let mut s = agentic_state();
    s.variants = two_rows();
    s.choose_variant(1);
    assert_eq!(s.view, View::Params);
    assert_eq!(s.target.model, "unsloth/Qwen3.8-27B-NVFP4");
    assert!(
        s.target_model_pinned,
        "follow_live_model must not undo this"
    );
    assert_eq!(s.values.float("wall_budget_s").unwrap(), 2500.0);
    let budget_row = s
        .specs
        .iter()
        .position(|p| p.key == "wall_budget_s")
        .expect("agentic declares the budget");
    assert_eq!(s.edit[budget_row], "2500", "the form shows what will run");
    let model_row = s.specs.len() + 1;
    assert_eq!(s.edit[model_row], "unsloth/Qwen3.8-27B-NVFP4");
}

/// 2026-09-26: The `min` arm: `bfcl-subset` couples `min_overall` and `min_normalized` to metrics
/// with `min` bounds, and choosing a variant adopts them.
#[test]
fn choosing_a_bfcl_variant_adopts_its_baseline_floors() {
    let bound = |min, max| gate::Bound {
        min,
        max,
        noise: None,
    };
    let state_with = |overall: gate::Bound| {
        let mut s = BenchState::default();
        s.target = metrale_bench::TargetEndpoint::local(8888, "test-model");
        let index = metrale_bench::registry::all()
            .iter()
            .position(|d| d.id == "bfcl-subset")
            .expect("registered");
        s.select(index);
        s.variants = vec![VariantRow {
            hardware: "gb10".into(),
            checkpoint: "unsloth/Qwen3.8-27B-NVFP4".into(),
            title: "dense 3.8".into(),
            recipe: Some("qwen3.8/qwen3.8-27b-nvfp4-unsloth-bfcl".into()),
            is_default: true,
            note: String::new(),
            metrics: vec![
                ("overall_accuracy".into(), overall),
                (
                    "normalized_single_turn_score".into(),
                    bound(Some(83.72), None),
                ),
            ],
        }];
        s.choose_variant(0);
        s
    };

    let s = state_with(bound(Some(83.82), None));
    assert_eq!(s.values.float("min_overall").unwrap(), 83.82);
    assert_eq!(s.values.float("min_normalized").unwrap(), 83.72);

    // 2026-09-26: A metric declaring both bounds is ambiguous, so the schema default (0) stays.
    let s = state_with(bound(Some(83.82), Some(90.0)));
    assert_eq!(
        s.values.float("min_overall").unwrap(),
        0.0,
        "ambiguous bound adopts nothing"
    );
    assert_eq!(s.values.float("min_normalized").unwrap(), 83.72);
}

/// 2026-09-26: In `two_rows` the default's ceiling equals the schema default (1000), so choosing it
/// leaves `wall_budget_s` at 1000.
#[test]
fn choosing_the_default_variant_keeps_the_35b_budget() {
    let mut s = agentic_state();
    s.variants = two_rows();
    s.choose_variant(0);
    assert_eq!(s.target.model, "Qwen/Qwen3.6-35B-A3B-FP8");
    assert_eq!(s.values.float("wall_budget_s").unwrap(), 1000.0);
}

/// 2026-09-26: j/k move within the variants, and Esc from the form goes back to the variant step.
#[test]
fn variant_navigation_and_the_way_back() {
    let mut s = agentic_state();
    s.variants = two_rows();
    s.view = View::Variants;
    s.variants_key(key(KeyCode::Char('j')));
    assert_eq!(s.variant_row, 1);
    s.variants_key(key(KeyCode::Char('j')));
    assert_eq!(s.variant_row, 1, "clamped at the end");
    s.variants_key(key(KeyCode::Char('k')));
    assert_eq!(s.variant_row, 0);
    s.variants_key(key(KeyCode::Esc));
    assert_eq!(s.view, View::List);

    s.view = View::Variants;
    s.variants_key(key(KeyCode::Enter));
    assert_eq!(s.view, View::Params);
    use crate::tui::app::BenchSub;
    s.on_key(key(KeyCode::Esc), BenchSub::Suite);
    assert_eq!(s.view, View::Variants);
}

/// 2026-09-26: A variant pin is released when another benchmark is selected.
#[test]
fn a_variant_pin_is_released_when_another_benchmark_is_selected() {
    let mut s = agentic_state();
    s.variants = two_rows();
    s.choose_variant(1);
    assert_eq!(s.target.model, "unsloth/Qwen3.8-27B-NVFP4");
    assert!(s.target_model_pinned && s.variant_pinned);

    let matrix = metrale_bench::registry::all()
        .iter()
        .position(|d| d.id == "serve-matrix")
        .expect("registered");
    s.select(matrix);
    assert!(
        !s.target_model_pinned && !s.variant_pinned,
        "the variant pin must not survive the benchmark it was chosen for"
    );
    s.follow_live_model("live/model");
    assert_eq!(
        s.target.model, "live/model",
        "the form follows the live server again"
    );
    assert_eq!(
        s.edit.last().map(String::as_str),
        Some("live/model"),
        "the model row shows what the target now is"
    );
}

/// 2026-09-26: A typed pin survives benchmark switches.
#[test]
fn an_operator_typed_pin_survives_benchmark_switches() {
    let mut s = agentic_state();
    // 2026-09-26: Type a model into the target field (last form row).
    let model_row = s.specs.len() + 1;
    s.edit[model_row] = "my/endpoint-model".into();
    s.commit_row(model_row);
    assert!(s.target_model_pinned && !s.variant_pinned);

    let matrix = metrale_bench::registry::all()
        .iter()
        .position(|d| d.id == "serve-matrix")
        .expect("registered");
    s.select(matrix);
    assert!(s.target_model_pinned, "typed pin survives");
    s.follow_live_model("live/model");
    assert_eq!(s.target.model, "my/endpoint-model");
}

/// 2026-09-26: A benchmark without variants goes straight to the form, and selecting another
/// benchmark clears the previous one's variants.
#[test]
fn a_variantless_benchmark_skips_the_step_and_selection_clears_rows() {
    let mut s = agentic_state();
    s.variants = two_rows();
    let matrix = metrale_bench::registry::all()
        .iter()
        .position(|d| d.id == "serve-matrix")
        .expect("registered");
    s.select(matrix);
    assert!(
        s.variants.is_empty(),
        "another benchmark's variants cleared"
    );
    s.enter_selected();
    assert_eq!(
        s.view,
        View::Params,
        "no baseline entries -> straight to the form"
    );
}
