// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Config form's editing model: rows, the option picker, and adding, removing and restoring settings.
//!
//! The form's fields live on [`LibState`]; this file holds the behaviour.
//! - Pick: Enter on a field with a closed value set (`lib_fields`) opens a
//!   picker instead of a text buffer.
//! - Add: `a` lists every serve flag the form does not already carry.
//! - Remove: `x` un-pins a recipe setting, so the flag is not passed and the
//!   server's default applies. The row stays listed, and `x` or Enter
//!   restores it.
//!
//! Owner: server tui.
//! Invariants:
//! - An edit, removal or restore is committed only after
//!   `Recipe::serve_args_edited` accepts the whole edited config; on failure
//!   `error` is set and the overrides and removals are unchanged. Un-adding an
//!   added row and `reset_overrides` do not validate.

use crate::tui::lib_fields::{self, FieldSpec};
use crate::tui::lib_keys::Outcome;
use crate::tui::lib_modal::ConfigModal;
use crate::tui::lib_state::{LibState, problem_line};

/// 2026-09-26: One row of the form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigRow {
    pub key: String,
    /// 2026-09-26: The override if edited, the recipe value otherwise.
    pub value: String,
    pub changed: bool,
    pub removed: bool,
    /// 2026-09-26: Not in the recipe's `defaults:`; added in this form.
    pub added: bool,
}

/// 2026-09-26: The closed value set for a form key, or `None` for free text.
fn field_options(key: &str) -> Option<Vec<String>> {
    let spec = lib_fields::spec_for_key(key)?;
    (!spec.options.is_empty()).then(|| spec.options.clone())
}

impl LibState {
    /// 2026-09-26: Every recipe key in key order (`Recipe::defaults` is a `BTreeMap`, not the file's order), removed ones included, then the added settings, also in key order.
    pub fn config_rows(&self) -> Vec<ConfigRow> {
        let Some(recipe) = self.config_recipe() else {
            return Vec::new();
        };
        let mut rows: Vec<ConfigRow> = recipe
            .defaults
            .iter()
            .map(|(key, value)| {
                let edited = self.overrides.get(key);
                ConfigRow {
                    key: key.clone(),
                    value: edited.unwrap_or(value).clone(),
                    changed: edited.is_some(),
                    removed: self.removed.contains(key),
                    added: false,
                }
            })
            .collect();
        rows.extend(
            self.overrides
                .iter()
                .filter(|(key, _)| !recipe.defaults.contains_key(*key))
                .map(|(key, value)| ConfigRow {
                    key: key.clone(),
                    value: value.clone(),
                    changed: true,
                    removed: false,
                    added: true,
                }),
        );
        rows
    }

    /// 2026-09-26: Enter on the current row: restore it if removed, open the picker for a closed-set field,
    /// else the text editor seeded with the current value.
    pub fn open_value_editor(&mut self) -> Outcome {
        let Some(row) = self.config_rows().into_iter().nth(self.row) else {
            return Outcome::None;
        };
        if row.removed {
            return self.restore_key(&row.key);
        }
        match field_options(&row.key) {
            Some(options) => {
                let selected = options.iter().position(|o| *o == row.value).unwrap_or(0);
                self.modal = Some(ConfigModal::Options {
                    key: row.key,
                    options,
                    selected,
                });
            }
            None => {
                self.edit_buffer = row.value;
                self.editing = true;
            }
        }
        Outcome::None
    }

    /// 2026-09-26: `a`: list every serve flag not already on the form.
    ///
    /// Rows are matched through the flag they render to, so a recipe's
    /// `max_model_len` also hides `max_seq_len`.
    pub fn open_add_modal(&mut self) -> Outcome {
        if self.config_recipe().is_none() {
            return Outcome::None;
        }
        let present: Vec<String> = self
            .config_rows()
            .iter()
            .filter_map(|r| crate::recipe::schema::flag_for(&r.key))
            .collect();
        let fields: Vec<&'static FieldSpec> = lib_fields::serve_fields()
            .iter()
            .filter(|s| !present.contains(&s.flag))
            .collect();
        if fields.is_empty() {
            return Outcome::Toast {
                text: "every serve flag is already on the form".into(),
                error: false,
            };
        }
        self.modal = Some(ConfigModal::Add {
            fields,
            selected: 0,
            help_scroll: 0,
        });
        Outcome::None
    }

    /// 2026-09-26: `x`: remove the setting under the cursor, restore a removed one, or drop an added one.
    pub fn toggle_removed(&mut self) -> Outcome {
        let Some(row) = self.config_rows().into_iter().nth(self.row) else {
            return Outcome::None;
        };
        if row.added {
            self.overrides.remove(&row.key);
            self.error = None;
            self.row = self.row.min(self.config_rows().len().saturating_sub(1));
            return Outcome::Toast {
                text: format!("{} removed", row.key),
                error: false,
            };
        }
        if row.removed {
            return self.restore_key(&row.key);
        }
        let Some(recipe) = self.config_recipe().cloned() else {
            return Outcome::None;
        };
        let mut overrides = self.overrides.clone();
        overrides.remove(&row.key);
        let mut removed = self.removed.clone();
        removed.insert(row.key.clone());
        match recipe.serve_args_edited(&overrides, &removed) {
            Ok(_) => {
                self.overrides = overrides;
                self.removed = removed;
                self.error = None;
                let text = match lib_fields::spec_for_key(&row.key).and_then(|s| s.default.clone())
                {
                    Some(d) => format!("{} removed — server default {d} applies", row.key),
                    None => format!("{} removed — the flag is not passed", row.key),
                };
                Outcome::Toast { text, error: false }
            }
            Err(e) => {
                self.error = Some(problem_line(&format!("{e:#}")));
                Outcome::None
            }
        }
    }

    fn restore_key(&mut self, key: &str) -> Outcome {
        let Some(recipe) = self.config_recipe().cloned() else {
            return Outcome::None;
        };
        let mut removed = self.removed.clone();
        removed.remove(key);
        // 2026-09-26: Validated too: an override on another row may be legal only without this flag.
        match recipe.serve_args_edited(&self.overrides, &removed) {
            Ok(_) => {
                self.removed = removed;
                self.error = None;
                Outcome::Toast {
                    text: format!("{key} restored"),
                    error: false,
                }
            }
            Err(e) => {
                self.error = Some(problem_line(&format!("{e:#}")));
                Outcome::None
            }
        }
    }

    /// 2026-09-26: Commit the edit buffer through `LibState::try_set`, which validates the whole config.
    ///
    /// The whole config, because `validate_serve_args` checks flags against
    /// each other (for example `--ep-size` against `--world-size`). An empty
    /// buffer is refused; for a pending add it abandons the add.
    pub fn commit_edit(&mut self) {
        self.editing = false;
        let raw = self.edit_buffer.trim().to_string();
        if let Some(key) = self.pending_add.take() {
            if raw.is_empty() {
                self.error = Some(format!("{key} was not added — no value given"));
                return;
            }
            if self.try_set(&key, &raw) {
                self.select_row(&key);
            }
            return;
        }
        let Some(row) = self.config_rows().into_iter().nth(self.row) else {
            return;
        };
        if raw.is_empty() {
            self.error = Some(format!("{} must not be empty", row.key));
            return;
        }
        self.try_set(&row.key, &raw);
    }

    /// 2026-09-26: Leave editing with nothing committed, dropping a pending add.
    pub fn cancel_edit(&mut self) {
        self.editing = false;
        self.edit_buffer.clear();
        self.pending_add = None;
    }

    /// 2026-09-26: Commit `overrides ∪ {key: raw}` if `serve_args_edited` accepts it; else set `error`.
    ///
    /// Returns whether the value was committed.
    pub(super) fn try_set(&mut self, key: &str, raw: &str) -> bool {
        let Some(recipe) = self.config_recipe().cloned() else {
            return false;
        };
        let mut candidate = self.overrides.clone();
        candidate.insert(key.to_string(), raw.to_string());
        match recipe.serve_args_edited(&candidate, &self.removed) {
            Ok(_) => {
                self.overrides = candidate;
                self.error = None;
                true
            }
            Err(e) => {
                self.error = Some(problem_line(&format!("{e:#}")));
                false
            }
        }
    }

    /// 2026-09-26: Put the cursor on `key`'s row, if it is on the form.
    pub(super) fn select_row(&mut self, key: &str) {
        if let Some(i) = self.config_rows().iter().position(|r| r.key == key) {
            self.row = i;
        }
    }

    /// 2026-09-26: Back to the recipe: drop every override, removal, pending add, borrow record and error.
    pub fn reset_overrides(&mut self) {
        self.overrides.clear();
        self.removed.clear();
        self.pending_add = None;
        self.borrowed = None;
        self.error = None;
    }

    /// 2026-09-26: The argv this form would launch, from the current overrides and removals.
    pub fn preview_argv(&self) -> Option<Vec<String>> {
        self.config_recipe()?
            .argv_edited(&self.overrides, &self.removed)
            .ok()
    }
}

#[cfg(test)]
#[path = "lib_config_tests.rs"]
mod tests;
