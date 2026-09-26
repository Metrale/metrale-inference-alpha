// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the Library's missing-weights refusal and the download keys its message points to.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.
//!
//! The refusal text ("press Esc, then d", in `lib_state`) and the key map (in
//! `lib_keys`) live in different files; these tests tie them together.

use super::*;
use crate::recipe::Recipe;
use crate::recipe::fetch::Index;
use crate::tui::data::library::LibraryEntry;

fn real_recipe() -> Recipe {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    let text = std::fs::read_to_string(&path).expect("fixture");
    Recipe::parse("qwen3.6/flagship", &text).expect("parses")
}

fn with_weights(model: &str) -> LibraryEntry {
    LibraryEntry {
        id: model.into(),
        snapshot_dir: Default::default(),
        size_bytes: 1024,
        has_weights: true,
        model_type: "qwen3_6_moe".into(),
        quant: "fp8".into(),
        layers: 40,
        hidden: 4096,
        heads: 32,
        experts: 128,
        context: 65536,
        optimized: true,
    }
}

fn state_with_recipe() -> LibState {
    let recipe = real_recipe();
    let local = vec![with_weights(&recipe.model)];
    let mut s = LibState {
        index: Index {
            recipes: vec![recipe],
            ..Index::default()
        },
        ..LibState::default()
    };
    s.rebuild(&local);
    s
}

#[test]
fn starting_a_recipe_whose_weights_are_missing_is_refused_before_anything_is_torn_down() {
    // 2026-09-26: `launch` refuses a recipe with no local weights before any
    // swap starts, and names the download key.
    let recipe = real_recipe();
    let mut s = LibState {
        index: Index {
            recipes: vec![recipe],
            ..Index::default()
        },
        ..LibState::default()
    };
    s.rebuild(&[]);
    s.view = View::Config;
    assert!(!s.selected_has_weights(), "nothing is on disk");

    let err = s
        .launch(std::sync::Arc::new(
            crate::main_modules::model_host::ModelHost::empty(),
        ))
        .expect_err("a model that is not downloaded cannot be started");
    assert!(err.contains("not downloaded"), "{err}");
    assert!(err.contains('d'), "and names the download key: {err}");

    // 2026-09-26: With weights on disk the guard passes.
    let mut ok = state_with_recipe();
    ok.view = View::Config;
    assert!(ok.selected_has_weights(), "the fixture has weights");
}

#[test]
fn the_download_advice_matches_the_number_of_escapes_it_actually_takes() {
    // 2026-09-26: The refusal says "press Esc, then d": one Esc from Config
    // lands in Cards, and `d` must start a download there.
    use crossterm::event::{KeyCode, KeyEvent};
    let mut s = state_with_recipe();
    s.view = View::Config;

    s.on_key(KeyEvent::from(KeyCode::Esc));
    assert_eq!(s.view, View::Cards, "one Esc leaves the form");

    let outcome = s.on_key(KeyEvent::from(KeyCode::Char('d')));
    assert!(
        matches!(outcome, crate::tui::lib_keys::Outcome::Download),
        "d must start a download from the pane one Esc away from the form"
    );
}

#[test]
fn the_download_keys_work_in_both_panes_that_show_a_model() {
    use crossterm::event::{KeyCode, KeyEvent};
    for view in [View::List, View::Cards] {
        let mut s = state_with_recipe();
        s.view = view;
        for (key, want) in [
            ('d', crate::tui::lib_keys::Outcome::Download),
            ('x', crate::tui::lib_keys::Outcome::CancelDownload),
            ('u', crate::tui::lib_keys::Outcome::CheckFresh),
        ] {
            let got = s.on_key(KeyEvent::from(KeyCode::Char(key)));
            assert_eq!(
                std::mem::discriminant(&got),
                std::mem::discriminant(&want),
                "{key:?} in {view:?}"
            );
            assert_eq!(s.view, view, "{key:?} must not navigate");
        }
    }
}
