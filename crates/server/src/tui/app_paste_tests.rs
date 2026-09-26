// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for bracketed-paste routing (`App::on_paste`).
//!
//! A paste arrives as one `Event::Paste`, so a pasted newline must neither
//! send a chat message nor run an Ops line.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;

fn app() -> App {
    App::new(clap::Parser::parse_from(["met", "org/m"]))
}

fn chat_input() -> App {
    let mut a = app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Chat;
    a.focus = Focus::Input;
    a
}

#[test]
fn a_multiline_paste_into_chat_keeps_its_lines_and_sends_nothing() {
    let mut a = chat_input();
    a.on_paste("first line\r\nsecond line\rthird\tline".into());
    assert_eq!(a.chat.input, "first line\nsecond line\nthird    line");
    assert!(a.chat.transcript.is_empty(), "nothing was sent");
    assert!(!a.chat.streaming);
}

#[test]
fn a_multiline_paste_into_ops_flattens_and_executes_nothing() {
    let mut a = app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Ops;
    a.focus = Focus::Input;
    a.on_paste("/status\n/quit\n".into());
    assert_eq!(
        a.ops.input, "/status /quit ",
        "one editable line, not a script"
    );
    assert!(a.ops.output.is_empty(), "an Ops line runs on Enter only");
    assert!(!a.should_quit, "the pasted /quit did NOT run");
}

#[test]
fn single_line_fields_get_the_paste_flattened() {
    let mut a = app();
    a.log_filter_editing = true;
    a.on_paste("weight\nloader".into());
    assert_eq!(a.log_filter, "weight loader");

    let mut a = app();
    a.section = Section::Library;
    a.lib.filter_editing = true;
    a.on_paste("qwen\n3.6".into());
    assert_eq!(a.lib.filter, "qwen 3.6");

    let mut a = app();
    a.section = Section::Library;
    a.lib.editing = true;
    a.on_paste("/models/custom-template.jinja".into());
    assert_eq!(a.lib.edit_buffer, "/models/custom-template.jinja");

    let mut a = app();
    a.section = Section::Benchmarks;
    a.bench.select(0);
    a.bench.row = 0;
    a.bench.editing = true;
    a.bench.edit.get_mut(0).expect("row 0").clear();
    a.on_paste("http://10.0.0.2:8888\n".into());
    assert_eq!(a.bench.edit[0], "http://10.0.0.2:8888 ");
}

#[test]
fn control_characters_never_survive_a_paste() {
    let mut a = chat_input();
    a.on_paste("a\u{1b}b\u{7}c".into());
    assert_eq!(a.chat.input, "abc", "a paste is data, never key chords");
}

#[test]
fn a_paste_with_nothing_focused_is_dropped_not_replayed() {
    let mut a = app();
    assert!(a.section == Section::Main);
    a.on_paste("qqqq\n/quit\n".into());
    assert!(
        !a.should_quit,
        "replaying it through the bindings would quit"
    );
    assert_eq!(a.log_filter, "");
    assert_eq!(a.chat.input, "");
    assert_eq!(a.ops.input, "");
}

#[test]
fn a_paste_while_a_picker_modal_is_open_is_dropped() {
    // 2026-09-26: With a picker open, the edit buffer underneath is hidden, so the paste is dropped.
    let mut a = app();
    a.section = Section::Library;
    a.lib.editing = true;
    a.lib.modal = Some(crate::tui::lib_modal::ConfigModal::Options {
        key: "kv-cache-dtype".into(),
        options: vec!["fp8".into(), "fp16".into()],
        selected: 0,
    });
    a.on_paste("stray".into());
    assert_eq!(a.lib.edit_buffer, "", "the buffer underneath is untouched");
}
