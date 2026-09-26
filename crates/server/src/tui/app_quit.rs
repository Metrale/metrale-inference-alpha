// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What `q` costs, and when it asks for a second press.
//!
//! `q` stops the server: it sets `should_quit`, the event loop exits with
//! `Exit::Quit` and calls `shutdown::request`, as Ctrl+C does. The prompt is
//! shown only when work is in flight, because a prompt that is usually
//! pointless trains the user to dismiss it without reading.
//!
//! Owner: server tui.
//! Invariants:
//! - While `confirm_quit` is set, `App::on_key` hands every key but Ctrl+C to
//!   the prompt, and only `q`, `y` or `Y` there sets `should_quit`.

use crossterm::event::{KeyCode, KeyEvent};

use super::app::App;

impl App {
    /// 2026-09-26: What `q` would throw away, named for the prompt; `None` when nothing is in flight.
    /// A download the user has already asked to cancel does not count.
    pub fn work_in_flight(&self) -> Option<&'static str> {
        if self.bench.is_running() {
            return Some("a benchmark is running");
        }
        if self.download.job.as_ref().is_some_and(|j| !j.cancelling) {
            return Some("a model download is in progress");
        }
        if self.chat.streaming {
            return Some("a chat reply is still streaming");
        }
        // 2026-09-26: A model load, from the Library or at boot. The boot arm also needs no live
        // model: a failed swap resets `progress` (so `ready` is false) while the old model keeps serving.
        if self.lib.launch_in_flight()
            || (!self.progress.ready
                && !self.awaiting_model
                && self.host.as_ref().and_then(|h| h.live_model()).is_none())
        {
            return Some("a model is still loading");
        }
        // 2026-09-26: An issue report mid-authorisation or mid-submit, or a draft with text, ends with the process.
        if self.help.report_in_flight() {
            return Some("an issue report is being submitted");
        }
        if self.help.has_draft() {
            return Some("an unsubmitted issue report has text in it");
        }
        None
    }

    /// 2026-09-26: `q` pressed with no text field claiming it.
    pub(super) fn on_quit_key(&mut self) {
        if self.work_in_flight().is_some() {
            self.confirm_quit = true;
        } else {
            self.should_quit = true;
        }
    }

    /// 2026-09-26: Answer an open prompt. Always returns `true`: the prompt consumes every key, so
    /// dismissing it never also acts on the section underneath.
    ///
    /// Only `q`, `y` or `Y` confirms. Every other key cancels, because the
    /// safe reading of an ambiguous keystroke is "do not stop the server".
    pub(super) fn answer_quit_prompt(&mut self, key: KeyEvent) -> bool {
        self.confirm_quit = false;
        if matches!(
            key.code,
            KeyCode::Char('q') | KeyCode::Char('y') | KeyCode::Char('Y')
        ) {
            self.should_quit = true;
        }
        true
    }
}

#[cfg(test)]
#[path = "app_quit_tests.rs"]
mod tests;
