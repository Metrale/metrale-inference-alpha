// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for text entry, driven through [`App::on_key`] rather than the buffers directly.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use crate::tui::app::MainSub;
use crate::tui::chat_thinking::{ThinkingRequest, ThinkingView};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

pub(super) fn app() -> App {
    App::new(clap::Parser::parse_from(["met", "org/m"]))
}

pub(super) fn press(a: &mut App, c: char) {
    a.on_key(KeyEvent::from(KeyCode::Char(c)));
}

pub(super) fn tap(a: &mut App, code: KeyCode) {
    a.on_key(KeyEvent::from(code));
}

fn chord(a: &mut App, c: char, m: KeyModifiers) {
    a.on_key(KeyEvent::new(KeyCode::Char(c), m));
}

fn type_str(a: &mut App, s: &str) {
    for c in s.chars() {
        press(a, c);
    }
}

/// 2026-09-26: Terminal ▸ Ops with the input line focused.
fn ops() -> App {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, 'i');
    a
}

/// 2026-09-26: Terminal ▸ Chat with the input line focused.
fn chat_input() -> App {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, '6');
    press(&mut a, 'i');
    a
}

/// 2026-09-26: Terminal ▸ Chat with the transcript focused.
fn chat_content() -> App {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, '6');
    a
}

#[test]
fn the_ops_line_types_and_deletes_and_runs() {
    let mut a = ops();
    type_str(&mut a, "/help");
    assert_eq!(a.ops.input, "/help");
    tap(&mut a, KeyCode::Backspace);
    // 2026-09-26: Derived rather than spelled out: the partial word is flagged by the spell checker.
    assert_eq!(a.ops.input, &"/help"[.."/help".len() - 1]);
    press(&mut a, 'p');
    tap(&mut a, KeyCode::Enter);
    assert_eq!(a.ops.input, "", "the line is consumed by running it");
    assert_eq!(a.ops.history, ["/help"]);
    assert!(a.ops.output.iter().any(|l| l.contains("/status")));
}

#[test]
fn backspacing_an_empty_ops_line_is_not_an_error() {
    let mut a = ops();
    for _ in 0..3 {
        tap(&mut a, KeyCode::Backspace);
    }
    assert_eq!(a.ops.input, "");
}

#[test]
fn a_blank_ops_line_is_not_recorded_or_run() {
    let mut a = ops();
    tap(&mut a, KeyCode::Enter);
    type_str(&mut a, "   ");
    tap(&mut a, KeyCode::Enter);
    assert!(a.ops.history.is_empty());
    assert!(a.ops.output.is_empty());
}

#[test]
fn up_walks_back_through_the_ops_history() {
    let mut a = ops();
    for line in ["/status", "/cache", "/help"] {
        type_str(&mut a, line);
        tap(&mut a, KeyCode::Enter);
    }
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.ops.input, "/help", "the most recent first");
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.ops.input, "/cache");
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.ops.input, "/status");
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.ops.input, "/status", "and stops at the oldest");
}

#[test]
fn up_on_an_empty_history_leaves_the_line_alone() {
    let mut a = ops();
    type_str(&mut a, "half typed");
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.ops.input, "half typed");
}

#[test]
fn the_ops_line_ignores_the_keys_it_has_no_binding_for() {
    let mut a = ops();
    type_str(&mut a, "abc");
    for code in [
        KeyCode::Down,
        KeyCode::Left,
        KeyCode::Right,
        KeyCode::Home,
        KeyCode::End,
        KeyCode::Delete,
        KeyCode::PageUp,
        KeyCode::F(2),
        KeyCode::Tab,
    ] {
        tap(&mut a, code);
    }
    assert_eq!(a.ops.input, "abc", "nothing above edits the line");
    assert!(a.focus == Focus::Input, "and none of them drops focus");
}

#[test]
fn the_chat_line_sends_on_enter_and_clears() {
    let mut a = chat_input();
    type_str(&mut a, "hello");
    tap(&mut a, KeyCode::Enter);
    assert_eq!(a.chat.input, "");
    assert_eq!(a.chat.transcript.len(), 2, "the turn and its placeholder");
    assert_eq!(a.chat.transcript[0].text, "hello");
    // 2026-09-26: The test app has no runtime handle, so `ChatState::send` writes a refusal and makes no request.
    assert!(a.chat.transcript[1].text.contains("chat unavailable"));
    assert!(!a.chat.streaming);
}

#[test]
fn an_empty_chat_line_sends_nothing() {
    let mut a = chat_input();
    tap(&mut a, KeyCode::Enter);
    type_str(&mut a, "   ");
    tap(&mut a, KeyCode::Enter);
    assert!(a.chat.transcript.is_empty());
}

#[test]
fn a_trailing_backslash_continues_onto_a_new_line_instead_of_sending() {
    // 2026-09-26: Ctrl+⏎ is indistinguishable from ⏎ in legacy terminal protocols, so a trailing
    // backslash is the way to write a second line.
    let mut a = chat_input();
    type_str(&mut a, "first\\");
    tap(&mut a, KeyCode::Enter);
    assert_eq!(a.chat.input, "first\n");
    assert!(a.chat.transcript.is_empty(), "nothing was sent");
    type_str(&mut a, "second");
    tap(&mut a, KeyCode::Enter);
    assert_eq!(a.chat.transcript[0].text, "first\nsecond");
}

#[test]
fn esc_leaves_the_chat_line_and_the_transcript_survives() {
    let mut a = chat_input();
    type_str(&mut a, "draft");
    tap(&mut a, KeyCode::Esc);
    assert!(a.focus == Focus::Content);
    assert_eq!(a.chat.input, "draft", "Esc is not a discard here");
    press(&mut a, 'i');
    assert!(a.focus == Focus::Input);
}

#[test]
fn the_chat_line_scrolls_the_transcript_without_losing_focus() {
    // 2026-09-26: In the chat line Up/Down scroll the transcript; in the Ops line they walk history.
    let mut a = chat_input();
    a.chat_scroll_max.set(11);
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.chat.scroll, Some(1));
    tap(&mut a, KeyCode::PageUp);
    assert_eq!(a.chat.scroll, Some(11));
    tap(&mut a, KeyCode::Down);
    assert_eq!(a.chat.scroll, Some(10));
    tap(&mut a, KeyCode::PageDown);
    assert_eq!(a.chat.scroll, None, "back at the tip, following again");
    assert!(a.focus == Focus::Input);
    assert_eq!(a.chat.input, "", "none of that typed anything");

    tap(&mut a, KeyCode::PageUp);
    tap(&mut a, KeyCode::End);
    assert_eq!(a.chat.scroll, None, "End snaps to the live tip");
}

#[test]
fn backspace_deletes_a_whole_character_not_a_byte() {
    // 2026-09-26: Backspace is `String::pop`, which removes one scalar, so the buffer stays valid UTF-8.
    for (typed, after_one) in [
        ("héllo", "héll"),
        ("日本語", "日本"),
        ("ok👍", "ok"),
        ("naïve", "naïv"),
    ] {
        let mut a = chat_input();
        type_str(&mut a, typed);
        assert_eq!(a.chat.input, typed);
        tap(&mut a, KeyCode::Backspace);
        assert_eq!(a.chat.input, after_one, "deleting from {typed:?}");
        assert_eq!(
            a.chat.input.chars().count() + 1,
            typed.chars().count(),
            "exactly one character, whatever its byte length"
        );
    }
}

#[test]
fn a_four_byte_emoji_goes_in_one_press_not_four() {
    let mut a = chat_input();
    press(&mut a, '👍');
    assert_eq!(a.chat.input.len(), 4, "four bytes on the wire");
    tap(&mut a, KeyCode::Backspace);
    assert_eq!(a.chat.input, "", "and one press to remove all of them");
}

#[test]
fn a_multi_scalar_grapheme_still_deletes_one_scalar_at_a_time() {
    // 2026-09-26: Known limit: the buffers delete by scalar, so a ZWJ sequence or a combining mark
    // takes as many presses as it has scalars. The buffer stays valid UTF-8 and shrinks on each press.
    let mut a = chat_input();
    type_str(&mut a, "e\u{301}"); // 2026-09-26: e + combining acute
    assert_eq!(a.chat.input.chars().count(), 2);
    tap(&mut a, KeyCode::Backspace);
    assert_eq!(a.chat.input, "e", "the mark went, the letter stayed");
    tap(&mut a, KeyCode::Backspace);
    assert_eq!(a.chat.input, "");
}

#[test]
fn the_log_filter_takes_unicode_too() {
    let mut a = app();
    press(&mut a, 'f');
    type_str(&mut a, "модель");
    assert_eq!(a.log_filter, "модель");
    tap(&mut a, KeyCode::Backspace);
    assert_eq!(a.log_filter, "модел");
}

/// 2026-09-26: What the next request would carry for `chat_template_kwargs.enable_thinking`;
/// `None` means the key is absent from the body.
fn on_the_wire(a: &App) -> Option<bool> {
    a.chat.think_req.enable_thinking()
}

#[test]
fn ctrl_t_cycles_the_thinking_request_auto_off_on_auto() {
    let mut a = chat_input();
    assert_eq!(a.chat.think_req, ThinkingRequest::Auto);
    assert_eq!(on_the_wire(&a), None, "Auto sends no key at all");

    chord(&mut a, 't', KeyModifiers::CONTROL);
    assert_eq!(a.chat.think_req, ThinkingRequest::Off);
    assert_eq!(on_the_wire(&a), Some(false));

    chord(&mut a, 't', KeyModifiers::CONTROL);
    assert_eq!(a.chat.think_req, ThinkingRequest::On);
    assert_eq!(on_the_wire(&a), Some(true));

    chord(&mut a, 't', KeyModifiers::CONTROL);
    assert_eq!(a.chat.think_req, ThinkingRequest::Auto);
    assert_eq!(
        on_the_wire(&a),
        None,
        "Auto must OMIT the key, not send a guess at the model's default"
    );
    assert_eq!(a.chat.input, "", "the chord never typed a `t`");
}

#[test]
fn a_bare_t_in_the_chat_line_is_a_letter() {
    let mut a = chat_input();
    type_str(&mut a, "tot");
    assert_eq!(a.chat.input, "tot");
    assert_eq!(a.chat.think_req, ThinkingRequest::Auto, "untouched");
}

#[test]
fn a_bare_t_on_the_transcript_cycles_the_request() {
    // 2026-09-26: With the transcript focused, bare letters are free, so the toggles work unchorded.
    let mut a = chat_content();
    press(&mut a, 't');
    assert_eq!(a.chat.think_req, ThinkingRequest::Off);
    assert_eq!(a.chat.input, "", "and typed nothing");
}

#[test]
fn alt_t_and_shift_t_move_the_display_only() {
    let mut a = chat_input();
    chord(&mut a, 't', KeyModifiers::ALT);
    assert_eq!(a.chat.think_view, ThinkingView::Expanded);
    chord(&mut a, 't', KeyModifiers::ALT);
    assert_eq!(a.chat.think_view, ThinkingView::Hidden);
    chord(&mut a, 't', KeyModifiers::ALT);
    assert_eq!(a.chat.think_view, ThinkingView::Collapsed);
    assert_eq!(
        on_the_wire(&a),
        None,
        "a view choice never reaches the request"
    );

    let mut a = chat_content();
    press(&mut a, 'T');
    assert_eq!(a.chat.think_view, ThinkingView::Expanded);
    assert_eq!(a.chat.think_req, ThinkingRequest::Auto);
}

#[test]
fn every_thinking_toggle_says_what_it_did() {
    let mut a = chat_input();
    chord(&mut a, 't', KeyModifiers::CONTROL);
    let said = &a.toasts.last().expect("a toast").text;
    assert!(said.contains("thinking off"), "got {said:?}");
    assert!(
        said.contains("next message"),
        "and says when it takes effect: {said:?}"
    );

    chord(&mut a, 't', KeyModifiers::ALT);
    let said = &a.toasts.last().expect("a toast").text;
    assert!(said.contains("reasoning expanded"), "got {said:?}");
}

#[test]
fn changing_the_request_forgets_what_the_last_reply_did() {
    // 2026-09-26: `observed_thinking` describes the previous request, so a change clears it.
    let mut a = chat_input();
    a.chat.observed_thinking = Some(true);
    chord(&mut a, 't', KeyModifiers::CONTROL);
    assert_eq!(a.chat.observed_thinking, None);
}

#[test]
fn the_thinking_state_survives_leaving_and_re_entering_the_section() {
    let mut a = chat_input();
    chord(&mut a, 't', KeyModifiers::CONTROL);
    tap(&mut a, KeyCode::Esc);
    press(&mut a, '1');
    press(&mut a, '6');
    press(&mut a, '6');
    assert_eq!(a.chat.think_req, ThinkingRequest::Off);
    assert_eq!(on_the_wire(&a), Some(false));
}

#[test]
fn the_log_filter_outranks_every_other_buffer() {
    // 2026-09-26: `f` works from either Main subsection, and while the filter is open the section's keys do not fire.
    let mut a = app();
    a.main_sub = MainSub::Kernels;
    a.kernel_scroll_max.set(50);
    press(&mut a, 'f');
    type_str(&mut a, "jjj");
    assert_eq!(a.log_filter, "jjj");
    assert_eq!(a.kernel_scroll, 0, "`j` did not also scroll the table");
}

/// 2026-09-26: Terminal ▸ Chat with a two-turn conversation on screen.
fn chat_session(focused: bool) -> App {
    use crate::tui::chat::{ChatMessage, Role};
    let mut a = app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Chat;
    a.focus = if focused {
        Focus::Input
    } else {
        Focus::Content
    };
    a.chat
        .transcript
        .push(ChatMessage::new(Role::User, "hello".into()));
    a.chat
        .transcript
        .push(ChatMessage::new(Role::Model, "hi".into()));
    a
}

#[test]
fn ctrl_n_asks_before_discarding_a_conversation() {
    let mut a = chat_session(false);
    chord(&mut a, 'n', KeyModifiers::CONTROL);
    assert!(a.confirm_chat_clear, "the question is on screen");
    assert_eq!(a.chat.transcript.len(), 2, "nothing destroyed yet");

    // 2026-09-26: Any key but `y`, `Y` or Ctrl+N keeps the conversation, including a plain `n`.
    press(&mut a, 'n');
    assert!(!a.confirm_chat_clear);
    assert_eq!(a.chat.transcript.len(), 2, "kept");

    chord(&mut a, 'n', KeyModifiers::CONTROL);
    press(&mut a, 'y');
    assert!(a.chat.transcript.is_empty(), "cleared");
    assert!(
        a.toasts
            .iter()
            .any(|t| t.text.contains("2 turns discarded")),
        "the receipt names what was lost"
    );
}

#[test]
fn a_second_ctrl_n_is_the_same_affirmative_the_quit_prompt_taught() {
    let mut a = chat_session(false);
    chord(&mut a, 'n', KeyModifiers::CONTROL);
    chord(&mut a, 'n', KeyModifiers::CONTROL);
    assert!(a.chat.transcript.is_empty());
    assert!(!a.confirm_chat_clear);
}

#[test]
fn ctrl_n_on_an_empty_chat_says_so_instead_of_asking() {
    let mut a = chat_session(false);
    a.chat.transcript.clear();
    chord(&mut a, 'n', KeyModifiers::CONTROL);
    assert!(!a.confirm_chat_clear);
    assert!(a.toasts.iter().any(|t| t.text.contains("already empty")));
}

#[test]
fn ctrl_n_works_while_typing_and_a_plain_n_still_types() {
    let mut a = chat_session(true);
    type_str(&mut a, "never");
    assert_eq!(a.chat.input, "never", "bare letters stay text");
    chord(&mut a, 'n', KeyModifiers::CONTROL);
    assert!(a.confirm_chat_clear, "the chord reaches the handler");
    press(&mut a, 'y');
    assert!(a.chat.transcript.is_empty());
    assert_eq!(a.chat.input, "never", "the unsent draft survives the reset");
}
