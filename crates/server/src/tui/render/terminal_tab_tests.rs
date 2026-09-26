// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the Terminal section's Ops and Chat panes: the tab
//! strip, the title chips, the input boxes, and the hints that change with
//! focus and width. The transcript rows are tested in `chat_lines_tests`.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::super::harness::{has, screen};
use super::chat_hints;
use crate::tui::app::{App, Focus, Section, TermSub};

fn term(sub: TermSub) -> App {
    let mut a = crate::tui::render::tests::app();
    a.section = Section::Terminal;
    a.term_sub = sub;
    a
}

#[test]
fn the_tab_strip_marks_which_of_the_two_panes_is_showing() {
    let ops = screen(&term(TermSub::Ops), 120, 40);
    assert!(has(&ops, "Ops"), "{ops:#?}");
    assert!(has(&ops, "Chat"));
    assert!(has(&ops, "(6 toggles)"), "and says how to swap them");
    assert!(has(&ops, "OPS ─ 0 lines"), "Ops is the pane on screen");

    let chat = screen(&term(TermSub::Chat), 120, 40);
    assert!(has(&chat, "CHAT ─"), "{chat:#?}");
    assert!(!has(&chat, "OPS ─"));
}

#[test]
fn an_unfocused_ops_input_says_how_to_take_the_keyboard() {
    let rows = screen(&term(TermSub::Ops), 120, 40);
    assert!(has(&rows, "(Enter to focus · /help)"), "{rows:#?}");
}

#[test]
fn a_focused_ops_input_ghosts_the_rest_of_the_command_it_can_finish() {
    let mut a = term(TermSub::Ops);
    a.focus = Focus::Input;
    a.ops.input = "/ker".into();
    let rows = screen(&a, 120, 40);
    assert!(
        has(&rows, "/kernels"),
        "the completion is shown:\n{rows:#?}"
    );
    assert!(has(&rows, "⇥ accept"), "and how to take it:\n{rows:#?}");

    // 2026-09-26: A complete command has no completion (`commands::complete`
    // skips an exact match), so the input shows the cursor, not a ghost.
    a.ops.input = "/quit".into();
    let done = screen(&a, 120, 40);
    assert!(!has(&done, "⇥ accept"), "{done:#?}");
    assert!(has(&done, "❯ /quit▏"), "{done:#?}");
}

#[test]
fn the_ops_pane_keeps_the_newest_output_when_it_overflows() {
    let mut a = term(TermSub::Ops);
    a.ops.output = (0..200).map(|i| format!("line-{i}")).collect();
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "OPS ─ 200 lines"), "{rows:#?}");
    assert!(
        has(&rows, "line-199"),
        "the newest is on screen:\n{rows:#?}"
    );
    assert!(!has(&rows, "line-0 "), "the oldest is not:\n{rows:#?}");
}

#[test]
fn an_echoed_command_is_marked_apart_from_the_output_it_produced() {
    let mut a = term(TermSub::Ops);
    a.ops.output = vec!["❯ /gpu".into(), "metrale 57.2 GB".into()];
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "❯ /gpu"), "{rows:#?}");
    assert!(has(&rows, "metrale 57.2 GB"));
}

#[test]
fn the_chat_hints_shrink_with_the_pane_but_never_vanish() {
    assert!(chat_hints(true, true).contains("Ctrl+T"));
    assert!(chat_hints(true, true).contains("newline"));
    // 2026-09-26: The narrow forms drop the thinking toggles.
    assert_eq!(chat_hints(true, false), "─ ⏎ send · Esc cancel ─");
    assert_eq!(chat_hints(false, false), "─ ⏎ focus ─");
    assert!(chat_hints(false, true).contains("t thinking"));
    for wide in [true, false] {
        for focused in [true, false] {
            assert!(chat_hints(focused, wide).contains('⏎'));
        }
    }
}

#[test]
fn the_chat_title_names_the_model_and_the_thinking_state_it_will_ask_for() {
    let a = term(TermSub::Chat);
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "CHAT ─ nvidia/Qwen3.6-27B-NVFP4"), "{rows:#?}");
    assert!(has(&rows, "thinking auto"), "{rows:#?}");
}

#[test]
fn the_chat_title_reports_streaming_and_how_far_back_it_is_scrolled() {
    let mut a = term(TermSub::Chat);
    a.chat.streaming = true;
    assert!(has(&screen(&a, 120, 40), "streaming"));

    // 2026-09-26: Scrolled back, the title shows the offset instead of
    // "streaming"; the two are exclusive.
    a.chat.scroll = Some(12);
    let scrolled = screen(&a, 120, 40);
    assert!(has(&scrolled, "↑12 ─ End follows"), "{scrolled:#?}");
    assert!(!has(&scrolled, " streaming ─"), "{scrolled:#?}");
}

#[test]
fn the_chat_input_box_grows_with_its_content_and_stops_at_five_rows() {
    let mut a = term(TermSub::Chat);
    a.focus = Focus::Input;
    a.chat.input = (1..=10)
        .map(|i| format!("L{i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "L1"), "{rows:#?}");
    assert!(
        has(&rows, "L5"),
        "five rows of input are visible:\n{rows:#?}"
    );
    assert!(
        !has(&rows, "L6"),
        "the box stops growing rather than eating the transcript:\n{rows:#?}"
    );
}

#[test]
fn a_focused_chat_input_carries_the_cursor() {
    let mut a = term(TermSub::Chat);
    a.chat.input = "why".into();
    assert!(!has(&screen(&a, 120, 40), "why▏"));
    a.focus = Focus::Input;
    assert!(has(&screen(&a, 120, 40), "why▏"));
}

#[test]
fn both_terminal_panes_survive_narrow_and_short_terminals() {
    for sub in [TermSub::Ops, TermSub::Chat] {
        let mut a = term(sub);
        a.ops.output = (0..40).map(|i| format!("line-{i}")).collect();
        a.chat.input = "one\ntwo\nthree".into();
        for (w, h) in [(20u16, 4u16), (20, 8), (20, 40), (40, 12), (200, 3)] {
            let rows = screen(&a, w, h);
            assert_eq!(rows.len(), h as usize, "{w}x{h} drew a partial frame");
        }
    }
}

/// 2026-09-26: The Ops pane scrolls back to its oldest line, and the title
/// says how to return.
#[test]
fn the_ops_pane_renders_scrollback_and_names_the_way_back_down() {
    let mut a = term(TermSub::Ops);
    a.ops.output = (0..100).map(|i| format!("line-{i:03}")).collect();

    // 2026-09-26: Following: the newest line is on screen, the oldest is not.
    let rows = screen(&a, 80, 24);
    assert!(has(&rows, "line-099"), "{rows:#?}");
    assert!(!has(&rows, "line-000"));
    assert!(
        a.ops.scroll_max.get() > 0,
        "the ceiling was published for the reducer"
    );

    // 2026-09-26: Parked at the top: the oldest line is on screen, and the
    // title gives the offset in the Chat pane's words.
    a.ops.scroll_up = a.ops.scroll_max.get();
    let up = a.ops.scroll_up;
    let rows = screen(&a, 80, 24);
    assert!(has(&rows, "line-000"), "{rows:#?}");
    assert!(!has(&rows, "line-099"));
    assert!(has(&rows, &format!("↑{up} ─ End follows")), "{rows:#?}");
}

/// 2026-09-26: An offset past the ceiling is clamped rather than blanking the
/// pane; `commands::execute` trims the output to `OUTPUT_CAP` under a parked
/// reader.
#[test]
fn a_stale_ops_offset_clamps_to_the_oldest_line() {
    let mut a = term(TermSub::Ops);
    a.ops.output = (0..30).map(|i| format!("line-{i:02}")).collect();
    a.ops.scroll_up = usize::MAX;
    let rows = screen(&a, 80, 24);
    assert!(has(&rows, "line-00"), "{rows:#?}");
}
