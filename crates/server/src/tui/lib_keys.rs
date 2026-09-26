// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Library key handling for the List, Cards and Config views.
//!
//! Owner: server tui.
//! Invariants:
//! - Of the keys that reach `on_key`, the filter gets every one while it is
//!   being edited; in Config an open modal, then an open text edit, gets
//!   every one before the view's own bindings.

use crossterm::event::{KeyCode, KeyEvent};

use super::lib_state::{LibState, View};

/// 2026-09-26: What the section wants the app to do about a keypress; `App` performs every variant but `None`.
#[derive(Debug, Default, PartialEq, Eq)]
pub enum Outcome {
    #[default]
    None,
    /// 2026-09-26: Show a toast; `error` marks it as a failure.
    Toast { text: String, error: bool },
    /// 2026-09-26: Start the configured recipe.
    Launch,
    /// 2026-09-26: Download, resume, or update the selected model.
    Download,
    /// 2026-09-26: Stop the running download; a second press abandons it (`DownloadState::cancel`).
    CancelDownload,
    /// 2026-09-26: Check whether the selected model is behind the Hub.
    CheckFresh,
}

impl LibState {
    /// 2026-09-26: True while a text field, the filter box or a modal owns the keyboard.
    pub fn is_editing(&self) -> bool {
        self.editing || self.filter_editing || self.modal.is_some()
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Outcome {
        if self.filter_editing {
            return self.filter_key(key);
        }
        match self.view {
            View::List => self.list_key(key),
            View::Cards => self.cards_key(key),
            View::Config => self.config_key(key),
        }
    }

    fn filter_key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            KeyCode::Esc => {
                self.filter.clear();
                self.filter_editing = false;
            }
            KeyCode::Enter => self.filter_editing = false,
            KeyCode::Backspace => {
                self.filter.pop();
            }
            KeyCode::Char(c) => self.filter.push(c),
            _ => return Outcome::None,
        }
        // 2026-09-26: The selection indexes the filtered list, so every filter keystroke resets it.
        self.selected = 0;
        Outcome::None
    }

    fn list_key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.move_selection(1),
            KeyCode::Up | KeyCode::Char('k') => self.move_selection(-1),
            KeyCode::Char('/') => self.filter_editing = true,
            // 2026-09-26: `x`, not `c`, cancels a download: `c` cancels a run in the Benchmarks section.
            KeyCode::Char('d') => return Outcome::Download,
            KeyCode::Char('x') => return Outcome::CancelDownload,
            KeyCode::Char('u') => return Outcome::CheckFresh,
            KeyCode::Char('r') => {
                if self.fetching {
                    return Outcome::None;
                }
                self.refresh();
                // 2026-09-26: `r` also asks for a local cache re-scan; the event loop drains `mark_dirty`.
                self.mark_dirty = true;
                // 2026-09-26: `refresh` does nothing without an artifact store root.
                return if self.fetching {
                    Outcome::Toast {
                        text: "fetching recipes…".into(),
                        error: false,
                    }
                } else {
                    Outcome::Toast {
                        text: "no artifact store — recipes cannot be fetched or cached".into(),
                        error: true,
                    }
                };
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if let Err(e) = self.open_cards() {
                    return Outcome::Toast {
                        text: e,
                        error: true,
                    };
                }
            }
            _ => {}
        }
        Outcome::None
    }

    /// 2026-09-26: The recipe cards for one model: j/k moves, Enter opens Config, Esc goes back to the list.
    fn cards_key(&mut self, key: KeyEvent) -> Outcome {
        let n = self.cards().len();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') if n > 0 => {
                self.card = (self.card + 1).min(n - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => self.card = self.card.saturating_sub(1),
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if let Err(e) = self.open_config() {
                    return Outcome::Toast {
                        text: e,
                        error: true,
                    };
                }
            }
            // 2026-09-26: The list's download keys also work here.
            KeyCode::Char('d') => return Outcome::Download,
            KeyCode::Char('x') => return Outcome::CancelDownload,
            KeyCode::Char('u') => return Outcome::CheckFresh,
            KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => self.view = View::List,
            _ => {}
        }
        Outcome::None
    }

    fn config_key(&mut self, key: KeyEvent) -> Outcome {
        if self.modal.is_some() {
            return self.modal_key(key);
        }
        if self.editing {
            return self.edit_key(key);
        }
        let rows = self.config_rows().len();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') if rows > 0 => {
                self.row = (self.row + 1).min(rows - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => self.row = self.row.saturating_sub(1),
            KeyCode::Enter => return self.open_value_editor(),
            KeyCode::Char('a') => return self.open_add_modal(),
            KeyCode::Char('b') => return self.open_borrow_modal(),
            KeyCode::Char('x') => return self.toggle_removed(),
            KeyCode::Char('d') => {
                self.reset_overrides();
                return Outcome::Toast {
                    text: "restored the recipe's own values".into(),
                    error: false,
                };
            }
            KeyCode::Char('s') => return Outcome::Launch,
            KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => {
                self.view = View::Cards;
                self.error = None;
            }
            _ => {}
        }
        Outcome::None
    }

    fn edit_key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            KeyCode::Enter => self.commit_edit(),
            KeyCode::Esc => {
                // 2026-09-26: Cancel keeps the committed value and any error from an earlier commit.
                self.cancel_edit();
            }
            KeyCode::Backspace => {
                self.edit_buffer.pop();
            }
            KeyCode::Char(c) => self.edit_buffer.push(c),
            _ => {}
        }
        Outcome::None
    }
}

#[cfg(test)]
#[path = "lib_keys_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "lib_keys_more_tests.rs"]
mod more_tests;
