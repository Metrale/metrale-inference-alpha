// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the starting-point cards in `lib_start.rs`.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use crate::recipe::fetch::Index;
use crate::tui::data::library::LibraryEntry;
use crate::tui::lib_state::View;

fn donor(id: &str, model: &str) -> Recipe {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    let mut r =
        Recipe::parse(id, &std::fs::read_to_string(path).expect("fixture")).expect("parses");
    r.model = model.to_string();
    r
}

fn local(id: &str, model_type: &str) -> LibraryEntry {
    LibraryEntry {
        id: id.into(),
        snapshot_dir: Default::default(),
        size_bytes: 1024,
        has_weights: true,
        model_type: model_type.into(),
        quant: "nvfp4".into(),
        layers: 40,
        hidden: 4096,
        heads: 32,
        experts: 128,
        context: 65536,
        optimized: false,
    }
}

fn state(recipes: Vec<Recipe>, locals: &[LibraryEntry]) -> LibState {
    let mut s = LibState::default();
    s.index = Index {
        recipes,
        ..Index::default()
    };
    s.rebuild(locals);
    s
}

/// 2026-09-26: Select the row for `model` by name rather than by position.
fn select(s: &mut LibState, model: &str) {
    s.selected = s
        .visible()
        .iter()
        .position(|e| e.model == model)
        .expect("the row exists");
}

#[test]
fn a_no_recipe_row_opens_on_starting_points_instead_of_refusing() {
    let d = donor(
        "qwen3.6/qwen3.6-35b-a3b-fp8-mtp",
        "Qwen/Qwen3.6-35B-A3B-FP8",
    );
    let mut s = state(vec![d], &[local("org/orphan", "qwen3_6_moe")]);
    select(&mut s, "org/orphan");
    s.open_cards().expect("no longer a dead end");
    assert_eq!(s.view, View::Cards);
    let cards = s.cards();
    assert!(!cards.is_empty());
    assert!(cards.iter().all(|c| c.model == "org/orphan"));
    assert!(cards.iter().all(|c| c.starting_point.is_some()));
    assert_eq!(
        cards.last().expect("blank").id,
        "starting-point/metrale-defaults"
    );
}

#[test]
fn family_matched_donors_rank_before_foreign_ones() {
    let q = donor("qwen3.6/qwen3.6-27b-nvfp4", "unsloth/Qwen3.6-27B-NVFP4");
    let g = donor("gemma4/gemma-4-26b-a4b-nvfp4", "google/gemma-4-26b");
    // 2026-09-26: `aaa/...` sorts first by id, so this fails if the ranking ignores family.
    let a = donor("aaa/unrelated", "org/unrelated");
    let mut s = state(
        vec![g, a, q],
        &[local("someorg/Qwen3.6-27B-renamed", "qwen3_6_moe")],
    );
    select(&mut s, "someorg/Qwen3.6-27B-renamed");
    s.open_cards().expect("opens");
    let first = &s.cards()[0];
    assert_eq!(
        first.starting_point.as_deref(),
        Some("qwen3.6/qwen3.6-27b-nvfp4"),
        "the architecture match must outrank id order"
    );
}

#[test]
fn donors_the_dashboard_cannot_launch_are_never_offered() {
    let mut vllm = donor("qwen3.6/vllm-only", "Qwen/Qwen3.6-35B-A3B-FP8");
    vllm.runtime = Some("vllm".into());
    let mut ep2 = donor("qwen3.6/ep2", "Qwen/Qwen3.6-35B-A3B-FP8");
    ep2.min_nodes = 2;
    let mut s = state(vec![vllm, ep2], &[local("org/orphan", "qwen3_6_moe")]);
    select(&mut s, "org/orphan");
    s.open_cards().expect("opens");
    let cards = s.cards();
    assert_eq!(cards.len(), 1);
    assert_eq!(cards[0].id, "starting-point/metrale-defaults");
}

#[test]
fn the_blank_card_is_launchable_and_renders_the_bare_serve_command() {
    let mut s = state(Vec::new(), &[local("org/orphan", "qwen3_6_moe")]);
    select(&mut s, "org/orphan");
    s.open_cards().expect("opens");
    s.card = s.cards().len() - 1;
    s.open_config().expect("a starting point is configurable");
    let argv = s.preview_argv().expect("valid");
    assert_eq!(argv, vec!["met", "serve", "org/orphan"]);
}

#[test]
fn a_template_sheds_every_claim_that_described_the_donor() {
    let d = donor(
        "qwen3.6/qwen3.6-35b-a3b-fp8-mtp",
        "Qwen/Qwen3.6-35B-A3B-FP8",
    );
    let t = template_from(&d, "org/orphan");
    assert!(t.updated.is_empty());
    assert!(t.maintainer.is_empty());
    assert!(t.model_params.is_empty());
    assert!(t.quantization.is_empty());
    assert!(t.kv_dtype.is_empty());
    assert_eq!(t.defaults, d.defaults);
    assert!(t.description.contains("not a measurement"));
    assert!(t.description.contains(&d.id), "provenance is named");
}

#[test]
fn a_starting_point_never_reports_the_donors_date() {
    let d = donor(
        "qwen3.6/qwen3.6-35b-a3b-fp8-mtp",
        "Qwen/Qwen3.6-35B-A3B-FP8",
    );
    let t = template_from(&d, "org/orphan");
    let mut s = state(Vec::new(), &[local("org/orphan", "qwen3_6_moe")]);
    s.fetched_dates.insert(d.id.clone(), "2026-05-01".into());
    assert_eq!(s.date_text(&t), "");
}

#[test]
fn a_set_built_for_another_row_is_never_served() {
    let mut s = state(
        Vec::new(),
        &[
            local("org/first", "qwen3_6_moe"),
            local("org/second", "qwen3_6_moe"),
        ],
    );
    select(&mut s, "org/first");
    s.open_cards().expect("opens");
    assert!(!s.cards().is_empty());
    // 2026-09-26: `cards()` refuses a set built for another model.
    select(&mut s, "org/second");
    assert!(s.cards().is_empty());
}

#[test]
fn a_real_recipe_arriving_for_the_model_outranks_the_guess() {
    let mut s = state(Vec::new(), &[local("org/orphan", "qwen3_6_moe")]);
    select(&mut s, "org/orphan");
    s.open_cards().expect("opens");
    assert!(s.cards()[0].starting_point.is_some());
    s.index.recipes = vec![donor("qwen3.6/now-published", "org/orphan")];
    let locals = [local("org/orphan", "qwen3_6_moe")];
    s.rebuild(&locals);
    select(&mut s, "org/orphan");
    let cards = s.cards();
    assert_eq!(cards.len(), 1);
    assert!(
        cards[0].starting_point.is_none(),
        "the measurement replaces the guess"
    );
}
