// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Config form's pickers: which one is open, and its key handling.
//!
//! Owner: server tui.
//! Invariants:
//! - A picker's cursor (the Preview's scroll) stays within `0..len`, or 0
//!   when the list is empty.

use crossterm::event::{KeyCode, KeyEvent};

use crate::tui::lib_fields::FieldSpec;

/// 2026-09-26: Columns of help text in the add-picker's side panel.
///
/// Shared by the scroll clamp here and the renderer's wrap in
/// `render/library/modal.rs`, so both count the same wrapped lines.
pub(crate) const HELP_PANEL_TEXT_W: usize = 30;
use crate::tui::lib_keys::Outcome;
use crate::tui::lib_state::LibState;

/// 2026-09-26: A picker drawn over the form; while one is open `LibState::config_key` sends it every key.
#[derive(Clone, Debug)]
pub enum ConfigModal {
    /// 2026-09-26: The closed value set for `key`.
    Options {
        key: String,
        options: Vec<String>,
        selected: usize,
    },
    /// 2026-09-26: Every serve flag the form does not already carry.
    Add {
        fields: Vec<&'static FieldSpec>,
        selected: usize,
        /// 2026-09-26: First visible wrapped line of the highlighted flag's full help.
        ///
        /// `J`/`K` move it; any cursor move resets it to 0.
        help_scroll: usize,
    },
    /// 2026-09-26: The recipes whose parameters can be applied over this form.
    Borrow {
        donors: Vec<crate::recipe::Recipe>,
        selected: usize,
    },
    /// 2026-09-26: What applying the chosen donor would change, before Enter applies it.
    ///
    /// Carries the donor list so Esc returns to it with the cursor on `donor`.
    Preview {
        donors: Vec<crate::recipe::Recipe>,
        donor: usize,
        changes: Vec<crate::tui::lib_borrow::BorrowChange>,
        scroll: usize,
    },
}

impl ConfigModal {
    fn len(&self) -> usize {
        match self {
            ConfigModal::Options { options, .. } => options.len(),
            ConfigModal::Add { fields, .. } => fields.len(),
            ConfigModal::Borrow { donors, .. } => donors.len(),
            ConfigModal::Preview { changes, .. } => changes.len(),
        }
    }

    fn selected(&self) -> usize {
        match self {
            ConfigModal::Options { selected, .. }
            | ConfigModal::Add { selected, .. }
            | ConfigModal::Borrow { selected, .. } => *selected,
            // 2026-09-26: The preview has no cursor; j/k drive its `scroll` instead.
            ConfigModal::Preview { scroll, .. } => *scroll,
        }
    }

    fn set_selected(&mut self, i: usize) {
        let n = self.len();
        let clamped = i.min(n.saturating_sub(1));
        match self {
            ConfigModal::Options { selected, .. } | ConfigModal::Borrow { selected, .. } => {
                *selected = clamped;
            }
            ConfigModal::Add {
                selected,
                help_scroll,
                ..
            } => {
                *selected = clamped;
                *help_scroll = 0;
            }
            ConfigModal::Preview { scroll, .. } => *scroll = clamped,
        }
    }

    fn move_selection(&mut self, delta: isize) {
        let next = (self.selected() as isize + delta).max(0) as usize;
        self.set_selected(next);
    }

    /// 2026-09-26: Scroll the add-picker's help panel; a no-op on the other pickers.
    fn scroll_help(&mut self, delta: isize) {
        let ConfigModal::Add {
            fields,
            selected,
            help_scroll,
        } = self
        else {
            return;
        };
        let Some(spec) = fields.get(*selected) else {
            return;
        };
        // 2026-09-26: Clamped to the last wrapped line, so extra `J` presses are not banked.
        let max = crate::tui::format::wrap_help(&spec.help_full, HELP_PANEL_TEXT_W)
            .len()
            .saturating_sub(1);
        let next = (*help_scroll as isize + delta).clamp(0, max as isize);
        *help_scroll = next as usize;
    }
}

impl LibState {
    /// 2026-09-26: Keys while a picker is open: j/k move, g/G jump, J/K scroll the help, Enter selects,
    /// Esc cancels.
    pub fn modal_key(&mut self, key: KeyEvent) -> Outcome {
        let Some(modal) = self.modal.as_mut() else {
            return Outcome::None;
        };
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => modal.move_selection(1),
            KeyCode::Up | KeyCode::Char('k') => modal.move_selection(-1),
            KeyCode::Char('g') => modal.set_selected(0),
            KeyCode::Char('G') => modal.set_selected(usize::MAX),
            KeyCode::Char('J') => modal.scroll_help(1),
            KeyCode::Char('K') => modal.scroll_help(-1),
            KeyCode::Enter => return self.modal_pick(),
            KeyCode::Esc => self.close_modal(),
            _ => {}
        }
        Outcome::None
    }

    /// 2026-09-26: Esc: close the picker, except the borrow preview, which steps back to its donor list.
    fn close_modal(&mut self) {
        self.modal = match self.modal.take() {
            Some(ConfigModal::Preview { donors, donor, .. }) => Some(ConfigModal::Borrow {
                donors,
                selected: donor,
            }),
            _ => None,
        };
    }

    fn modal_pick(&mut self) -> Outcome {
        match self.modal.take() {
            Some(ConfigModal::Options {
                key,
                options,
                selected,
            }) => {
                let Some(value) = options.get(selected) else {
                    return Outcome::None;
                };
                // 2026-09-26: On failure `try_set` sets `error` and commits nothing.
                if self.try_set(&key.clone(), value) {
                    self.select_row(&key);
                }
                Outcome::None
            }
            Some(ConfigModal::Add {
                fields, selected, ..
            }) => match fields.get(selected) {
                Some(spec) => self.add_field(spec),
                None => Outcome::None,
            },
            Some(ConfigModal::Borrow { donors, selected }) => self.pick_donor(donors, selected),
            Some(ConfigModal::Preview {
                donors,
                donor,
                changes,
                ..
            }) => match donors.get(donor) {
                Some(d) => self.apply_borrow(&d.clone(), &changes),
                None => Outcome::None,
            },
            None => Outcome::None,
        }
    }

    fn add_field(&mut self, spec: &'static FieldSpec) -> Outcome {
        match (&spec.default, spec.options.is_empty()) {
            // 2026-09-26: clap declares a default: add the row at it.
            (Some(default), _) => {
                if self.try_set(&spec.key, default) {
                    self.select_row(&spec.key);
                    Outcome::Toast {
                        text: format!("{} added at its default, {default}", spec.key),
                        error: false,
                    }
                } else {
                    Outcome::None
                }
            }
            // 2026-09-26: No default but a closed set: open the value picker.
            (None, false) => {
                self.modal = Some(ConfigModal::Options {
                    key: spec.key.clone(),
                    options: spec.options.clone(),
                    selected: 0,
                });
                Outcome::None
            }
            // 2026-09-26: No default, free text: ask for the value; `commit_edit` creates the row.
            (None, true) => {
                self.pending_add = Some(spec.key.clone());
                self.edit_buffer.clear();
                self.editing = true;
                Outcome::None
            }
        }
    }
}

#[cfg(test)]
#[path = "lib_modal_tests.rs"]
mod tests;
