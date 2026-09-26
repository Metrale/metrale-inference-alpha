// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for what [`App::work_in_flight`] reports about model loads.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use crossterm::event::{KeyCode, KeyEvent};

fn app() -> App {
    App::new(clap::Parser::parse_from(["met", "org/m"]))
}

fn press(a: &mut App, c: char) {
    a.on_key(KeyEvent::from(KeyCode::Char(c)));
}

/// 2026-09-26: A boot load in progress makes the first `q` ask; once `ready`, `q` quits.
#[test]
fn q_asks_first_while_the_boot_load_is_still_running() {
    let mut a = app(); // 2026-09-26: argv names a model and `progress.ready` is still false
    assert_eq!(a.work_in_flight(), Some("a model is still loading"));
    press(&mut a, 'q');
    assert!(a.confirm_quit, "the first press asks");
    assert!(!a.should_quit);

    // 2026-09-26: Once serving, the same press quits without a prompt.
    let mut a = app();
    a.progress.ready = true;
    press(&mut a, 'q');
    assert!(a.should_quit);
    assert!(!a.confirm_quit);
}

/// 2026-09-26: `met serve` with no model has nothing loading, so `q` quits without the loading prompt.
#[test]
fn an_awaiting_model_boot_quits_without_the_loading_prompt() {
    let mut a = App::new(clap::Parser::parse_from(["met"]));
    assert!(a.work_in_flight().is_none());
    press(&mut a, 'q');
    assert!(a.should_quit);
}
