// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Render tests for starting points (`lib_start`); the fixtures
//! come from `library_tests.rs`, and the state-level tests are in
//! `tui/lib_start_tests.rs`.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use crate::tui::render::harness::{has, screen};

use super::tests::{lib, local, recipe};

/// 2026-09-26: The chip, the pane title, the description and the form all
/// mark a starting point as one.
#[test]
fn starting_points_are_marked_as_guesses_everywhere_they_render() {
    let donor = recipe("qwen3.6-35b-a3b-fp8-mtp");
    let mut a = lib(vec![donor], vec![local("org/orphan", true)]);
    // 2026-09-26: Select the no-recipe row; `Entry::rank` sorts it after the
    // recipe row.
    a.lib.selected = a
        .lib
        .visible()
        .iter()
        .position(|e| e.model == "org/orphan")
        .expect("row");
    a.lib.open_cards().expect("opens on starting points");
    let rows = screen(&a, 200, 50);
    assert!(has(&rows, " starting point "), "the chip:\n{rows:#?}");
    assert!(
        has(&rows, "starting points ─"),
        "the pane title uses the honest noun:\n{rows:#?}"
    );
    assert!(
        has(&rows, "not a measurement"),
        "the description says what it is:\n{rows:#?}"
    );
    assert!(
        !has(&rows, "THE FLAGSHIP"),
        "the donor's measured rationale must not survive onto the guess:\n{rows:#?}"
    );

    a.lib.open_config().expect("configurable");
    let rows = screen(&a, 200, 50);
    assert!(
        has(&rows, "starting point — "),
        "the form is the last honest moment:\n{rows:#?}"
    );
    assert!(has(&rows, "unverified on this model"), "{rows:#?}");
}

/// 2026-09-26: A row with no recipe names Enter's starting points.
#[test]
fn a_no_recipe_row_names_the_way_forward() {
    let a = lib(Vec::new(), vec![local("org/orphan", true)]);
    let rows = screen(&a, 200, 50);
    assert!(has(&rows, "no recipe — ⏎ starting points"), "{rows:#?}");
    assert!(has(&rows, "⏎ pick a starting point"), "{rows:#?}");
}
