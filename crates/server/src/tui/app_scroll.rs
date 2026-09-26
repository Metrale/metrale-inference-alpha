// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Where the mouse wheel goes, per section, and how far.
//!
//! The ceilings (`App::{log,kernel,chat,help}_scroll_max`, `OpsState::scroll_max`)
//! are written by the renderer on every frame and read here: the limit depends
//! on the viewport height and on the filtered, wrapped content, and only the
//! renderer knows both. They are `Cell` because `render::draw` takes `&App`.
//!
//! Owner: server tui.
//! Invariants:
//! - Each section routes the wheel to what its own keys move, and a section
//!   with nothing to scroll (Stats, Network) ignores it.

use crossterm::event::{KeyCode, KeyEvent};

use super::app::{App, MainSub, TermSub};
use super::section::Section;

impl App {
    /// 2026-09-26: Scroll the current view by `rows` (positive = further down).
    pub fn scroll(&mut self, rows: i32) {
        match self.section {
            Section::Main => match self.main_sub {
                // 2026-09-26: The log pane counts backwards from the newest line; `scroll_log` inverts `rows`.
                MainSub::Overview => self.scroll_log(rows),
                MainSub::Kernels => {
                    let max = self.kernel_scroll_max.get() as i32;
                    self.kernel_scroll =
                        (self.kernel_scroll as i32 + rows).clamp(0, max.max(0)) as usize;
                }
            },
            // 2026-09-26: Lists move their selection, as their arrow keys do, so wheel and cursor stay in step.
            Section::Library => self.lib.move_selection(rows as isize),
            Section::Benchmarks => {
                let n = metrale_bench::registry::all().len();
                if n > 0 {
                    let cur = self.bench.selected as i32;
                    let next = (cur + rows).clamp(0, n as i32 - 1);
                    self.bench.select(next as usize);
                }
            }
            Section::Terminal => match self.term_sub {
                // 2026-09-26: The Ops pane's own offset, counted up from the newest output line.
                TermSub::Ops => {
                    let max = self.ops.scroll_max.get() as i32;
                    self.ops.scroll_up =
                        (self.ops.scroll_up as i32 - rows).clamp(0, max.max(0)) as usize;
                }
                TermSub::Chat => self.chat_scroll(-rows),
            },
            // 2026-09-26: The report preview is the only Help surface with a scroll offset.
            Section::Help => self.help.scroll_preview(rows),
            // 2026-09-26: Nothing scrollable.
            Section::Stats | Section::Network => {}
        }
    }

    /// 2026-09-26: Keys while the help modal is open.
    ///
    /// `j`/`k`/`g`/`G` and the arrows, Home and End move the key list, which
    /// `render::overlay::draw_help` scrolls when the terminal is shorter than
    /// the table. Any other key closes the modal and is not passed on.
    pub(super) fn on_help_overlay_key(&mut self, key: KeyEvent) {
        let max = self.help_scroll_max.get();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.help_scroll = (self.help_scroll + 1).min(max);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.help_scroll = self.help_scroll.saturating_sub(1);
            }
            KeyCode::Char('g') | KeyCode::Home => self.help_scroll = 0,
            KeyCode::Char('G') | KeyCode::End => self.help_scroll = max,
            _ => {
                self.help_open = false;
                // 2026-09-26: The next open starts at the top of the list.
                self.help_scroll = 0;
            }
        }
    }

    /// 2026-09-26: Ops keys when the output pane, not the input line, has focus.
    ///
    /// Line and page moves go through [`App::scroll`], the entry point the wheel uses.
    pub(super) fn on_ops_content_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.scroll(-1),
            KeyCode::Down | KeyCode::Char('j') => self.scroll(1),
            KeyCode::PageUp => self.scroll(-10),
            KeyCode::PageDown => self.scroll(10),
            KeyCode::Char('g') | KeyCode::Home => {
                self.ops.scroll_up = self.ops.scroll_max.get();
            }
            KeyCode::Char('G') | KeyCode::End => self.ops.scroll_up = 0,
            _ => {}
        }
    }

    /// 2026-09-26: Chat transcript scroll with the ceiling applied (positive rows = toward older turns).
    ///
    /// Used by the wheel and by the keys while the chat input has focus; with
    /// the transcript focused, `ChatState::on_content_key` moves the offset and
    /// [`Self::clamp_chat_scroll`] applies the ceiling.
    pub(super) fn chat_scroll(&mut self, rows: i32) {
        self.chat.scroll_by(rows);
        self.clamp_chat_scroll();
    }

    /// 2026-09-26: Pull the chat offset back inside the renderer-published ceiling.
    /// Separate from [`Self::chat_scroll`] because `ChatState::on_content_key`
    /// moves the offset itself and only the `App` can see the ceiling.
    pub(super) fn clamp_chat_scroll(&mut self) {
        let max = self.chat_scroll_max.get();
        if self.chat.scroll.is_some_and(|n| n > max) {
            self.chat.scroll = if max == 0 { None } else { Some(max) };
        }
    }

    /// 2026-09-26: Jump the chat transcript to its oldest row (`g`/Home).
    /// `G`/End, which follows the newest row, is `ChatState::follow`. This one
    /// is here because it needs the renderer-published ceiling.
    pub(super) fn chat_jump_top(&mut self) {
        let max = self.chat_scroll_max.get();
        self.chat.scroll = (max > 0).then_some(max);
    }

    /// 2026-09-26: The Main ▸ Overview log pane.
    ///
    /// The offset counts backwards from the newest line, so wheel-up (negative
    /// `rows`) increases it, up to `log_scroll_max`; 0 becomes `None` (follow).
    fn scroll_log(&mut self, rows: i32) {
        let max = self.log_scroll_max.get() as i32;
        let next = (self.log_scroll.unwrap_or(0) as i32 - rows).clamp(0, max.max(0));
        self.log_scroll = if next <= 0 { None } else { Some(next as usize) };
    }
}

#[cfg(test)]
#[path = "app_scroll_tests.rs"]
mod tests;
