// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for scrolling from the keyboard and the wheel: ceilings, boundaries and empty panes.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use crate::tui::app::{BenchSub, Focus};
use crossterm::event::{KeyCode, KeyEvent};

fn app() -> App {
    App::new(clap::Parser::parse_from(["met", "org/m"]))
}

fn press(a: &mut App, c: char) {
    a.on_key(KeyEvent::from(KeyCode::Char(c)));
}

fn tap(a: &mut App, code: KeyCode) {
    a.on_key(KeyEvent::from(code));
}

/// 2026-09-26: Main ▸ Overview with `lines` rows of scrollback above the fold.
fn log_pane(lines: usize) -> App {
    let mut a = app();
    a.section = Section::Main;
    a.main_sub = MainSub::Overview;
    a.log_scroll_max.set(lines);
    a
}

#[test]
fn the_keyboard_stops_at_the_oldest_line_just_as_the_wheel_does() {
    let mut a = log_pane(4);
    for _ in 0..20 {
        press(&mut a, 'k');
    }
    assert_eq!(a.log_scroll, Some(4), "clamped at the oldest line");
    for _ in 0..4 {
        press(&mut a, 'j');
    }
    assert_eq!(a.log_scroll, None, "and four presses bring it back");
}

#[test]
fn the_arrows_and_the_vi_keys_are_the_same_binding() {
    let mut a = log_pane(10);
    tap(&mut a, KeyCode::Up);
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.log_scroll, Some(2));
    tap(&mut a, KeyCode::Down);
    assert_eq!(a.log_scroll, Some(1));
    press(&mut a, 'k');
    assert_eq!(a.log_scroll, Some(2));
    press(&mut a, 'j');
    assert_eq!(a.log_scroll, Some(1));
}

#[test]
fn a_log_shorter_than_the_viewport_cannot_be_scrolled_at_all() {
    // 2026-09-26: Nothing above the fold, so the offset stays at follow.
    let mut a = log_pane(0);
    for _ in 0..5 {
        press(&mut a, 'k');
    }
    assert_eq!(a.log_scroll, None, "still following the newest line");
    press(&mut a, 'j');
    assert_eq!(a.log_scroll, None);
}

#[test]
fn end_and_capital_g_return_to_following_from_any_depth() {
    for jump_key in [KeyCode::Char('G'), KeyCode::End] {
        let mut a = log_pane(200);
        for _ in 0..30 {
            press(&mut a, 'k');
        }
        assert_eq!(a.log_scroll, Some(30));
        tap(&mut a, jump_key);
        assert_eq!(a.log_scroll, None, "{jump_key:?} snaps to the tip");
        tap(&mut a, jump_key);
        assert_eq!(a.log_scroll, None, "and is idempotent there");
    }
}

#[test]
fn the_kernel_table_clamps_at_both_ends() {
    let mut a = app();
    a.section = Section::Main;
    a.main_sub = MainSub::Kernels;
    a.kernel_scroll_max.set(3);
    for _ in 0..10 {
        press(&mut a, 'j');
    }
    assert_eq!(a.kernel_scroll, 3, "cannot scroll past the last row");
    for _ in 0..10 {
        press(&mut a, 'k');
    }
    assert_eq!(a.kernel_scroll, 0, "nor above the first");

    press(&mut a, 'j');
    press(&mut a, 'g');
    assert_eq!(a.kernel_scroll, 0, "`g` is the way home");
}

#[test]
fn a_kernel_table_that_fits_on_screen_does_not_move() {
    let mut a = app();
    a.section = Section::Main;
    a.main_sub = MainSub::Kernels;
    for _ in 0..5 {
        press(&mut a, 'j');
    }
    assert_eq!(a.kernel_scroll, 0);
}

#[test]
fn a_growing_log_does_not_yank_a_reader_who_scrolled_up() {
    // 2026-09-26: The offset counts backwards from the newest line, so lines arriving underneath
    // leave the reader parked. Only an explicit key resumes following.
    let mut a = log_pane(10);
    for _ in 0..3 {
        press(&mut a, 'k');
    }
    assert_eq!(a.log_scroll, Some(3));
    a.log_scroll_max.set(400);
    a.on_tick();
    press(&mut a, 'z'); // 2026-09-26: unbound on Main
    assert_eq!(a.log_scroll, Some(3), "still parked");
    tap(&mut a, KeyCode::End);
    assert_eq!(a.log_scroll, None);
}

#[test]
fn typing_while_scrolled_back_does_not_yank_the_chat_transcript() {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, '6');
    press(&mut a, 'i');
    a.chat_scroll_max.set(10);
    tap(&mut a, KeyCode::PageUp);
    assert_eq!(a.chat.scroll, Some(10));
    for c in "still reading".chars() {
        press(&mut a, c);
    }
    assert_eq!(a.chat.scroll, Some(10), "typing is not navigation");
    // 2026-09-26: Sending resumes following.
    tap(&mut a, KeyCode::Enter);
    assert_eq!(a.chat.scroll, None);
}

#[test]
fn the_chat_wheel_respects_the_ceiling_and_collapses_to_follow() {
    let mut a = app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Chat;
    a.chat_scroll_max.set(2);
    for _ in 0..10 {
        a.scroll(-3);
    }
    assert_eq!(a.chat.scroll, Some(2));
    a.chat_scroll_max.set(0);
    a.scroll(-3);
    assert_eq!(a.chat.scroll, None, "an empty transcript follows the tip");
}

#[test]
fn the_ops_wheel_moves_ops_output_and_never_the_main_log() {
    // 2026-09-26: Ops renders `ops.output`, not the Main log ring.
    let mut a = app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Ops;
    a.log_scroll_max.set(20);
    a.ops.scroll_max.set(20);
    a.scroll(-3);
    assert_eq!(a.ops.scroll_up, 3, "the pane under the wheel moved");
    assert_eq!(a.log_scroll, None, "the Main log did not");
}

#[test]
fn the_wheel_moves_the_benchmark_selection_and_stops_at_the_ends() {
    let n = metrale_bench::registry::all().len();
    let mut a = app();
    a.section = Section::Benchmarks;
    a.bench_sub = BenchSub::Suite;
    for _ in 0..(n + 5) {
        a.scroll(1);
    }
    assert_eq!(
        a.bench.selected,
        n.saturating_sub(1),
        "the last benchmark, not past it"
    );
    for _ in 0..(n + 5) {
        a.scroll(-1);
    }
    assert_eq!(a.bench.selected, 0);
}

#[test]
fn sections_with_nothing_to_scroll_ignore_the_wheel_without_panicking() {
    let mut a = app();
    a.log_scroll_max.set(10);
    a.scroll(-3);
    let parked = a.log_scroll;
    for s in [Section::Stats, Section::Network, Section::Library] {
        a.section = s;
        a.scroll(3);
        a.scroll(-3);
    }
    assert_eq!(a.log_scroll, parked, "and touch nobody else's offset");
}

#[test]
fn scrolling_does_not_disturb_focus_or_the_section() {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, 'i');
    a.log_scroll_max.set(10);
    a.scroll(-3);
    assert!(a.focus == Focus::Input);
    assert_eq!(a.section, Section::Terminal);
}

#[test]
fn lowercase_g_and_home_jump_the_log_to_its_oldest_line() {
    for jump_key in [KeyCode::Char('g'), KeyCode::Home] {
        let mut a = log_pane(200);
        tap(&mut a, jump_key);
        assert_eq!(a.log_scroll, Some(200), "{jump_key:?} parks at the oldest");
    }
    // 2026-09-26: A log that fits has no top to jump to, so it keeps following.
    let mut a = log_pane(0);
    press(&mut a, 'g');
    assert_eq!(a.log_scroll, None);
}

#[test]
fn capital_g_and_end_jump_the_kernel_table_to_its_last_row() {
    let mut a = app();
    a.section = Section::Main;
    a.main_sub = MainSub::Kernels;
    a.kernel_scroll_max.set(7);
    for jump_key in [KeyCode::Char('G'), KeyCode::End] {
        a.kernel_scroll = 0;
        tap(&mut a, jump_key);
        assert_eq!(a.kernel_scroll, 7, "{jump_key:?} parks at the bottom");
    }
    tap(&mut a, KeyCode::Home);
    assert_eq!(a.kernel_scroll, 0, "Home is the same way back as `g`");
}

#[test]
fn chat_g_and_home_jump_to_the_oldest_row_in_both_focus_states() {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, '6'); // 2026-09-26: the second press cycles Ops to Chat, content focus
    a.chat_scroll_max.set(40);
    press(&mut a, 'g');
    assert_eq!(a.chat.scroll, Some(40), "content-focus g parks at the top");
    tap(&mut a, KeyCode::End);
    assert_eq!(a.chat.scroll, None);

    press(&mut a, 'i'); // 2026-09-26: input focus: `g` is text, Home is the jump
    tap(&mut a, KeyCode::Home);
    assert_eq!(a.chat.scroll, Some(40));
    press(&mut a, 'g');
    assert_eq!(a.chat.input, "g", "a bare g while typing stays a letter");
    assert_eq!(a.chat.scroll, Some(40), "and moves nothing");
}

#[test]
fn an_empty_chat_ignores_the_jump_rather_than_banking_it() {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, '6');
    press(&mut a, 'g');
    assert_eq!(a.chat.scroll, None, "nothing above the fold to park at");
}

#[test]
fn help_scroll_keys_move_the_key_list_and_anything_else_closes_it() {
    let mut a = app();
    press(&mut a, '?');
    assert!(a.help_open);
    a.help_scroll_max.set(2);
    press(&mut a, 'j');
    press(&mut a, 'j');
    press(&mut a, 'j');
    assert!(a.help_open, "scroll keys do not dismiss");
    assert_eq!(a.help_scroll, 2, "and clamp at the ceiling");
    press(&mut a, 'k');
    assert_eq!(a.help_scroll, 1);
    press(&mut a, 'G');
    assert_eq!(a.help_scroll, 2);
    press(&mut a, 'g');
    assert_eq!(a.help_scroll, 0);

    press(&mut a, 'G');
    let section = a.section;
    press(&mut a, '4');
    assert!(!a.help_open, "a non-scroll key closes it");
    assert_eq!(a.section, section, "and is swallowed, not acted on");
    assert_eq!(a.help_scroll, 0, "the next open starts at the top");
}

/// 2026-09-26: Both chat keyboard paths (input-focus arrows, content-focus letters) stop at the ceiling.
#[test]
fn the_chat_keys_clamp_against_the_same_ceiling_as_the_wheel() {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, '6');
    press(&mut a, 'i');
    a.chat_scroll_max.set(3);
    tap(&mut a, KeyCode::PageUp);
    assert_eq!(
        a.chat.scroll,
        Some(3),
        "PageUp lands on the ceiling, not 10"
    );

    tap(&mut a, KeyCode::Esc); // 2026-09-26: back to content focus
    for _ in 0..5 {
        press(&mut a, 'k');
    }
    assert_eq!(a.chat.scroll, Some(3), "content-focus k cannot bank either");
    press(&mut a, 'j');
    assert_eq!(a.chat.scroll, Some(2), "one press back means one row back");
}
