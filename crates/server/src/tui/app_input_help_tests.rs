// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests that keys reach the Help section through `App::on_key`, the real entry point.
//!
//! `HelpState`'s own tests call `HelpState::on_key` directly, so they cannot
//! see whether `App` routes a key there.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::tests::{app, press, tap};
use super::*;
use crossterm::event::KeyCode;

/// 2026-09-26: The report title accepts typing through `App::on_key`.
///
/// `in_input()` is true while the title is being edited, so this checks that
/// `on_input_key` hands the key to `HelpState` and not to the `?` modal.
#[test]
fn typing_a_report_title_reaches_the_composer_and_does_not_look_frozen() {
    let mut a = app();
    a.jump(Section::Help);
    a.help.sub = crate::tui::help_state::HelpSub::Report;

    // 2026-09-26: Enter on the Title row starts editing.
    tap(&mut a, KeyCode::Enter);
    assert!(
        a.help.is_editing(),
        "Enter on Title must start editing, else the field can never be filled"
    );

    for c in "gpu oom".chars() {
        press(&mut a, c);
    }
    assert_eq!(
        a.help.title, "gpu oom",
        "every typed character must reach the title buffer"
    );

    tap(&mut a, KeyCode::Backspace);
    assert_eq!(a.help.title, "gpu oo");

    // 2026-09-26: Esc leaves edit mode and keeps the draft title, unlike a filter box's Esc.
    tap(&mut a, KeyCode::Esc);
    assert!(!a.help.is_editing());
    assert_eq!(a.help.title, "gpu oo");
}
