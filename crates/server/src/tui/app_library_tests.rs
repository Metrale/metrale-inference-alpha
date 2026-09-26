// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the Library actions `App` performs for the reducer: refusals and the download-switch question.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use crossterm::event::{KeyCode, KeyEvent};

fn library() -> App {
    let mut a = App::new(clap::Parser::parse_from(["met", "org/m"]));
    a.on_key(KeyEvent::from(KeyCode::Char('4')));
    a
}

fn key(c: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(c), crossterm::event::KeyModifiers::NONE)
}

fn last_toast(a: &App) -> (&str, bool) {
    let t = a.toasts.last().expect("a toast");
    (t.text.as_str(), t.error)
}

#[test]
fn downloading_with_nothing_selected_refuses_before_it_resolves_a_cache() {
    // 2026-09-26: The selection is checked before the cache root is resolved.
    let mut a = library();
    a.download_selected_model();
    assert_eq!(last_toast(&a), ("no model selected", true));
}

#[test]
fn checking_freshness_with_nothing_selected_refuses_the_same_way() {
    let mut a = library();
    a.check_selected_model();
    assert_eq!(last_toast(&a), ("no model selected", true));
}

#[test]
fn launching_without_a_server_says_so_and_goes_nowhere() {
    // 2026-09-26: With no host attached the launch is refused and the section does not change.
    let mut a = library();
    a.launch_selected_recipe();
    assert_eq!(
        last_toast(&a),
        ("no server attached to this dashboard", true)
    );
    assert_eq!(a.section, Section::Library, "still where the user was");
}

#[test]
fn the_library_download_keys_reach_these_actions() {
    // 2026-09-26: `d` / `u` / `x` are Library keys whose `Outcome`s `App::on_library_key` performs.
    for (key, expected) in [
        ('d', "no model selected"),
        ('u', "no model selected"),
        ('x', "nothing is downloading"),
    ] {
        let mut a = library();
        a.on_key(KeyEvent::from(KeyCode::Char(key)));
        assert_eq!(last_toast(&a).0, expected, "`{key}`");
    }
}

#[test]
fn cancelling_when_nothing_is_downloading_is_not_an_error() {
    // 2026-09-26: A mis-press, not a failure, so the toast is not an error.
    let mut a = library();
    a.on_key(KeyEvent::from(KeyCode::Char('x')));
    assert_eq!(last_toast(&a), ("nothing is downloading", false));
}

/// 2026-09-26: A non-affirmative answer to the download-switch question keeps the running download.
#[test]
fn a_second_download_opens_the_question_instead_of_refusing() {
    let mut a = library();
    let root = std::env::temp_dir().join("metrale-switch");
    std::fs::create_dir_all(&root).ok();
    a.download.start("org/first", root);
    a.download_switch = None;
    a.download_switch = Some(("org/first".to_string(), "org/second".to_string()));
    assert!(a.download_switch.is_some(), "the question is open");

    let consumed = a.answer_download_switch(key('n'));
    assert!(consumed, "the question owns the keyboard");
    assert!(a.download_switch.is_none(), "answered");
    assert!(a.pending_start.is_none(), "nothing was queued");
    assert!(a.download.job.is_some(), "the running download survives");
}

/// 2026-09-26: The affirmative queues the wanted model in `pending_start` rather than starting it.
#[test]
fn the_affirmative_queues_the_second_download_it_does_not_race_it() {
    let mut a = library();
    let root = std::env::temp_dir().join("metrale-switch2");
    std::fs::create_dir_all(&root).ok();
    a.download.start("org/first", root);
    a.download_switch = Some(("org/first".to_string(), "org/second".to_string()));

    assert!(a.answer_download_switch(key('x')));
    assert_eq!(
        a.pending_start.as_deref(),
        Some("org/second"),
        "queued, not started"
    );
    assert!(
        a.download
            .job
            .as_ref()
            .is_some_and(|j| j.repo == "org/first"),
        "B must not have displaced A"
    );
}
