// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the Suite list's scroll keys, which clamp the selection at both ends.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use crossterm::event::{KeyCode, KeyEvent};

use super::*;
use crate::tui::app::BenchSub;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::from(code)
}

/// 2026-09-26: A state with no executor and no target.
fn state() -> BenchState {
    let mut s = BenchState::default();
    s.select(0);
    s
}

fn press(s: &mut BenchState, code: KeyCode) {
    assert!(
        matches!(s.on_key(key(code), BenchSub::Suite), Outcome::None),
        "a scroll key never toasts"
    );
}

#[test]
fn g_and_home_jump_to_the_first_benchmark_and_clamp_there() {
    let n = metrale_bench::registry::all().len();
    let mut s = state();
    s.select(n - 1);
    press(&mut s, KeyCode::Char('g'));
    assert_eq!(s.selected, 0);
    press(&mut s, KeyCode::Char('k'));
    assert_eq!(s.selected, 0, "k at the top must not bank a press");

    let mut s = state();
    s.select(n - 1);
    press(&mut s, KeyCode::Home);
    assert_eq!(s.selected, 0, "Home is g");
}

#[test]
fn shift_g_and_end_jump_to_the_last_benchmark_and_clamp_there() {
    let n = metrale_bench::registry::all().len();
    let mut s = state();
    press(&mut s, KeyCode::Char('G'));
    assert_eq!(s.selected, n - 1);
    press(&mut s, KeyCode::Char('j'));
    assert_eq!(s.selected, n - 1, "j at the bottom must not bank a press");

    let mut s = state();
    press(&mut s, KeyCode::End);
    assert_eq!(s.selected, n - 1, "End is G");
}

#[test]
fn page_keys_move_by_the_renderer_published_viewport() {
    let n = metrale_bench::registry::all().len();
    assert!(n > 4, "paging is only meaningful past one 80x24 screen");
    let mut s = state();
    s.suite_page.set(4);
    press(&mut s, KeyCode::PageDown);
    assert_eq!(s.selected, 4, "one page is what one frame held");
    press(&mut s, KeyCode::PageUp);
    assert_eq!(s.selected, 0);
}

#[test]
fn page_keys_clamp_at_both_ends() {
    let n = metrale_bench::registry::all().len();
    let mut s = state();
    s.suite_page.set(n + 3);
    press(&mut s, KeyCode::PageDown);
    assert_eq!(s.selected, n - 1, "PageDown stops at the last entry");
    press(&mut s, KeyCode::PageDown);
    assert_eq!(s.selected, n - 1, "a press at the end is not banked");
    press(&mut s, KeyCode::PageUp);
    assert_eq!(s.selected, 0, "PageUp stops at the first entry");
    press(&mut s, KeyCode::PageUp);
    assert_eq!(s.selected, 0, "a press at the top is not banked");
}

#[test]
fn paging_before_the_first_render_still_moves() {
    // 2026-09-26: `suite_page` starts at zero; the keys use at least one entry per page.
    let mut s = state();
    assert_eq!(s.suite_page.get(), 0);
    press(&mut s, KeyCode::PageDown);
    assert_eq!(s.selected, 1);
}

#[test]
fn the_scroll_keys_stay_in_the_list() {
    let mut s = state();
    for code in [
        KeyCode::Char('G'),
        KeyCode::PageUp,
        KeyCode::PageDown,
        KeyCode::Char('g'),
        KeyCode::End,
        KeyCode::Home,
    ] {
        press(&mut s, code);
        assert_eq!(s.view, View::List, "{code:?} left the list");
    }
}
