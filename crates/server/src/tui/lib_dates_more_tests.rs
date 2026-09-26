// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the cases where `lib_dates.rs` must not start a lookup, and for date precedence.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use crate::recipe::fetch::Index;

fn recipe_with(id: &str, updated: &str) -> crate::recipe::Recipe {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    let text = std::fs::read_to_string(&path).expect("fixture");
    let mut r = crate::recipe::Recipe::parse(id, &text).expect("parses");
    r.updated = updated.to_string();
    r
}

fn state(recipes: Vec<crate::recipe::Recipe>) -> LibState {
    let mut s = LibState::default();
    s.index = Index {
        recipes,
        ..Default::default()
    };
    s.rebuild(&[]);
    s
}

#[test]
fn a_recipe_the_index_has_never_heard_of_is_not_looked_up() {
    let mut s = state(vec![recipe_with("fam/stem", "")]);
    s.want_date_for("fam/vanished");
    assert!(s.dating.is_none(), "no request for a recipe that is gone");
    assert!(s.pending_date.is_none());
}

#[test]
fn a_lookup_in_flight_blocks_a_lookup_for_a_different_recipe() {
    let mut s = state(vec![
        recipe_with("fam/first", ""),
        recipe_with("fam/second", ""),
    ]);
    let (_tx, rx) = std::sync::mpsc::channel::<(String, Option<String>)>();
    s.pending_date = Some(rx);
    s.dating = Some("fam/first".into());
    s.dated.insert("fam/first".into());

    s.want_date_for("fam/second");
    assert_eq!(
        s.dating.as_deref(),
        Some("fam/first"),
        "the second must wait rather than replace it"
    );
    assert!(
        !s.dated.contains("fam/second"),
        "and is not marked as asked"
    );
}

#[test]
fn nothing_is_visible_to_date_before_a_row_is_selected() {
    let s = LibState::default();
    assert!(s.visible_recipe_id().is_none());

    let mut s = LibState::default();
    s.view = View::Cards;
    assert!(s.visible_recipe_id().is_none());
    s.view = View::Config;
    assert!(s.visible_recipe_id().is_none());
}

#[test]
fn the_pane_decides_which_of_a_models_recipes_gets_the_request() {
    let mut first = recipe_with("fam/aaa", "");
    let mut second = recipe_with("fam/zzz", "");
    first.runtime = Some("vllm".into());
    second.runtime = Some("metrale".into());
    let mut s = state(vec![first, second]);

    s.view = View::List;
    assert_eq!(
        s.visible_recipe_id().as_deref(),
        Some("fam/zzz"),
        "the list describes the launchable recipe"
    );

    s.view = View::Cards;
    s.card = 0;
    assert_eq!(s.visible_recipe_id().as_deref(), Some("fam/aaa"));
    s.card = 1;
    assert_eq!(s.visible_recipe_id().as_deref(), Some("fam/zzz"));
}

#[test]
fn a_recipes_own_date_wins_over_one_fetched_for_it() {
    let mut s = state(vec![recipe_with("fam/stem", "2026-08-01")]);
    s.fetched_dates
        .insert("fam/stem".into(), "1999-01-01".into());
    assert_eq!(
        s.date_text(&recipe_with("fam/stem", "2026-08-01")),
        "2026-08-01"
    );
}

#[test]
fn a_skeleton_never_outranks_a_date_that_is_already_known() {
    let mut s = state(vec![recipe_with("fam/stem", "")]);
    s.fetched_dates
        .insert("fam/stem".into(), "2026-07-04".into());
    s.dating = Some("fam/stem".into());
    assert_eq!(s.date_text(&recipe_with("fam/stem", "")), "2026-07-04");
}

#[test]
fn an_empty_date_from_the_worker_is_treated_as_no_answer() {
    let mut s = state(vec![recipe_with("fam/stem", "")]);
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(("fam/stem".to_string(), Some(String::new())))
        .expect("send");
    s.pending_date = Some(rx);
    s.dating = Some("fam/stem".into());

    assert!(!s.poll_date(), "nothing usable arrived");
    assert!(s.fetched_dates.is_empty(), "and nothing was stored");
    assert!(s.dating.is_none(), "the skeleton still clears");
}

#[test]
fn polling_with_no_lookup_in_flight_is_quiet() {
    let mut s = LibState::default();
    assert!(!s.poll_date());
    assert!(s.pending_date.is_none());
}

#[test]
fn a_lookup_that_has_not_answered_yet_leaves_the_skeleton_up() {
    let mut s = state(vec![recipe_with("fam/stem", "")]);
    let (_tx, rx) = std::sync::mpsc::channel::<(String, Option<String>)>();
    s.pending_date = Some(rx);
    s.dating = Some("fam/stem".into());

    assert!(!s.poll_date(), "nothing has landed");
    assert_eq!(s.dating.as_deref(), Some("fam/stem"), "still in flight");
    assert!(s.pending_date.is_some());
}

#[test]
fn a_failed_lookup_is_never_retried_this_session() {
    let mut s = state(vec![recipe_with("fam/stem", "")]);
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(("fam/stem".to_string(), None)).expect("send");
    s.pending_date = Some(rx);
    s.dating = Some("fam/stem".into());
    s.dated.insert("fam/stem".into());
    assert!(!s.poll_date(), "the lookup failed");

    s.want_date_for("fam/stem");
    assert!(s.dating.is_none(), "the failure is remembered");
    assert!(s.pending_date.is_none());
}
