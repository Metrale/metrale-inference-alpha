// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Text-entry routing for the [`App`] reducer: which text buffer owns a
//! keystroke or a paste.
//!
//! Owner: server tui.
//! Invariants: a paste goes only into a text buffer; with none focused it is
//! dropped, never replayed as keys.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::app::{App, Focus, Section, TermSub};

/// 2026-09-26: Minimal single-line editor, used for the Main log filter: Esc clears
/// and leaves, Enter leaves, Backspace deletes, a character appends.
pub(super) fn edit_line(buf: &mut String, key: KeyEvent, editing: &mut bool) {
    match key.code {
        KeyCode::Esc => {
            buf.clear();
            *editing = false;
        }
        KeyCode::Enter => *editing = false,
        KeyCode::Backspace => {
            buf.pop();
        }
        KeyCode::Char(c) => buf.push(c),
        _ => {}
    }
}

impl App {
    pub(super) fn on_input_key(&mut self, key: KeyEvent) {
        if self.log_filter_editing {
            edit_line(&mut self.log_filter, key, &mut self.log_filter_editing);
            return;
        }
        if self.section == Section::Library {
            self.on_library_key(key);
            return;
        }
        if self.section == Section::Benchmarks {
            self.on_bench_key(key);
            return;
        }
        if self.section == Section::Help {
            // 2026-09-26: `on_help_key` (the section's keys), not `on_help_overlay_key`
            // (the `?` modal's scroll), whose catch-all arm would swallow text.
            self.on_help_key(key);
            return;
        }
        match self.term_sub {
            TermSub::Ops => match key.code {
                KeyCode::Esc => self.focus = Focus::Content,
                KeyCode::Enter => {
                    let line = std::mem::take(&mut self.ops.input);
                    if !line.trim().is_empty() {
                        self.ops.history.push(line.clone());
                        self.ops.history_pos = None;
                        super::commands::execute(&line, self);
                    }
                }
                // 2026-09-26: Tab accepts the ghost-text completion
                // (`commands::complete`).
                KeyCode::Tab => {
                    if let Some(ghost) = super::commands::complete(&self.ops.input) {
                        self.ops.input = ghost.to_string();
                    }
                }
                KeyCode::Up => {
                    let h = &self.ops.history;
                    if !h.is_empty() {
                        let pos = match self.ops.history_pos {
                            None => h.len() - 1,
                            Some(p) => p.saturating_sub(1),
                        };
                        self.ops.history_pos = Some(pos);
                        self.ops.input = h[pos].clone();
                    }
                }
                // 2026-09-26: Down walks history forward; past the newest entry the
                // line returns to empty, as in readline.
                KeyCode::Down => {
                    if let Some(p) = self.ops.history_pos {
                        if p + 1 < self.ops.history.len() {
                            self.ops.history_pos = Some(p + 1);
                            self.ops.input = self.ops.history[p + 1].clone();
                        } else {
                            self.ops.history_pos = None;
                            self.ops.input.clear();
                        }
                    }
                }
                // 2026-09-26: Up/Down are spent on history here, so PageUp/PageDown
                // scroll while typing.
                KeyCode::PageUp => self.scroll(-10),
                KeyCode::PageDown => self.scroll(10),
                KeyCode::Backspace => {
                    self.ops.input.pop();
                }
                KeyCode::Char(c) => self.ops.input.push(c),
                _ => {}
            },
            TermSub::Chat => match key.code {
                KeyCode::Esc => {
                    self.chat.cancel();
                    self.focus = Focus::Content;
                }
                // 2026-09-26: Enter sends; a trailing backslash continues onto a new
                // line (legacy terminal protocols cannot tell Ctrl+Enter from Enter).
                KeyCode::Enter => {
                    if let Some(stripped) = self.chat.input.strip_suffix('\\') {
                        self.chat.input = format!("{stripped}\n");
                    } else {
                        self.chat.send(self.args.port);
                    }
                }
                KeyCode::Backspace => {
                    self.chat.input.pop();
                }
                // 2026-09-26: Up/Down scroll the transcript while the input holds
                // focus; Chat has no input history to spend them on.
                KeyCode::Up => self.chat_scroll(1),
                KeyCode::Down => self.chat_scroll(-1),
                KeyCode::PageUp => self.chat_scroll(10),
                KeyCode::PageDown => self.chat_scroll(-10),
                KeyCode::End => self.chat.follow(),
                // 2026-09-26: Home is not text, so it jumps to the top even while the
                // input is focused, pairing with End.
                KeyCode::Home => self.chat_jump_top(),
                // 2026-09-26: The thinking toggles in their chorded forms (Ctrl+T,
                // Alt+T), matched before the catch-all because they arrive as
                // `Char('t')` with a modifier. A bare `t` is typed.
                KeyCode::Char('t') => match self.chat.on_view_key(key, true) {
                    Some(said) => self.toast(said, false),
                    None => self.chat.input.push('t'),
                },
                // 2026-09-26: Ctrl+N, matched before the catch-all for the same reason.
                KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.request_chat_clear();
                }
                KeyCode::Char(c) => self.chat.input.push(c),
                _ => {}
            },
        }
    }

    /// 2026-09-26: Chat keys when the transcript, not the input box, has focus. Bare
    /// letters are free here, so the toggles also take `t` and `T` unchorded.
    pub(super) fn on_chat_content_key(&mut self, key: KeyEvent) {
        if key.code == KeyCode::Char('n') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.request_chat_clear();
            return;
        }
        // 2026-09-26: `g`/Home need the renderer-published ceiling, which `ChatState`
        // cannot see; `G`/End are handled inside it.
        if matches!(key.code, KeyCode::Char('g') | KeyCode::Home) {
            self.chat_jump_top();
            return;
        }
        if let Some(said) = self.chat.on_content_key(key) {
            self.toast(said, false);
        }
        // 2026-09-26: `on_content_key` moves the offset inside `ChatState`, which
        // cannot see the renderer-published ceiling, so the clamp is here.
        self.clamp_chat_scroll();
    }

    /// 2026-09-26: `Ctrl+N`: start a new chat session, after a confirmation when the
    /// transcript is non-empty or a reply is streaming; otherwise only a toast.
    pub(super) fn request_chat_clear(&mut self) {
        if self.chat.transcript.is_empty() && !self.chat.streaming {
            self.toast("chat is already empty", false);
        } else {
            self.confirm_chat_clear = true;
        }
    }

    /// 2026-09-26: Answer the clear-chat prompt. Always consumes the key, like
    /// `answer_quit_prompt`, so dismissing the prompt navigates nowhere.
    ///
    /// Only `y`/`Y` or `Ctrl+N` again clears, as `q` again confirms the quit
    /// prompt. Any other key, a bare `n` included, cancels.
    pub(super) fn answer_chat_clear(&mut self, key: KeyEvent) -> bool {
        self.confirm_chat_clear = false;
        let again = key.code == KeyCode::Char('n') && key.modifiers.contains(KeyModifiers::CONTROL);
        if again || matches!(key.code, KeyCode::Char('y') | KeyCode::Char('Y')) {
            let turns = self.chat.transcript.len();
            self.chat.reset();
            self.toast(format!("chat cleared — {turns} turns discarded"), false);
        }
        true
    }
}

/// 2026-09-26: A paste flattened for a single-line buffer: line breaks and tabs
/// become spaces, and other control characters are dropped.
fn paste_single_line(text: &str) -> String {
    text.chars()
        .filter_map(|c| match c {
            '\n' | '\r' | '\t' => Some(' '),
            c if c.is_control() => None,
            c => Some(c),
        })
        .collect()
}

/// 2026-09-26: A paste with its line structure kept, for the chat input: CRLF and a
/// lone CR become `\n`, a tab becomes four spaces, and every other control
/// character is dropped.
fn paste_multiline(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push('\n');
            }
            '\n' => out.push('\n'),
            '\t' => out.push_str("    "),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

impl App {
    /// 2026-09-26: Route a bracketed paste into the buffer that owns the keyboard:
    /// the log filter, the Library filter or edit field, the Benchmarks edit row,
    /// or the Terminal input. The Help section takes no paste.
    ///
    /// Chat keeps newlines; every single-line field gets them flattened to
    /// spaces. With nothing focused the paste is dropped, not replayed through
    /// the key bindings.
    pub(super) fn on_paste(&mut self, text: String) {
        if self.log_filter_editing {
            self.log_filter.push_str(&paste_single_line(&text));
            return;
        }
        if self.section == Section::Library {
            if self.lib.filter_editing {
                self.lib.filter.push_str(&paste_single_line(&text));
            } else if self.lib.editing && self.lib.modal.is_none() {
                self.lib.edit_buffer.push_str(&paste_single_line(&text));
            }
            // 2026-09-26: A picker modal navigates; there is nothing to paste into.
            return;
        }
        if self.section == Section::Benchmarks {
            if self.bench.is_editing() {
                let row = self.bench.row;
                if let Some(buf) = self.bench.edit.get_mut(row) {
                    buf.push_str(&paste_single_line(&text));
                }
            }
            return;
        }
        if self.section == Section::Terminal && self.focus == Focus::Input {
            match self.term_sub {
                // 2026-09-26: Flattened, not executed: an Ops line runs on Enter only.
                TermSub::Ops => self.ops.input.push_str(&paste_single_line(&text)),
                TermSub::Chat => self.chat.input.push_str(&paste_multiline(&text)),
            }
        }
    }
}

#[cfg(test)]
#[path = "app_input_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "app_input_help_tests.rs"]
mod help_tests;

#[cfg(test)]
#[path = "app_paste_tests.rs"]
mod paste_tests;
