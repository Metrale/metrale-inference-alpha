// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `b` on the Config form: apply another recipe's `defaults:` over the form, after a preview.
//!
//! The donors are `lib_start::ranked_donors`, the list the starting-point
//! cards are built from, minus the form's own recipe. Enter on a donor opens a
//! preview of the rows that would change; a second Enter applies exactly those
//! rows.
//!
//! Owner: server tui.
//! Invariants:
//! - A borrow changes the form only in `apply_borrow`, and only when the whole
//!   edited config passes `Recipe::serve_args_edited`; on failure the settings
//!   are untouched and `error` is set.
//! - `apply_borrow` applies the `BorrowChange` list the preview showed.

use std::collections::BTreeMap;

use crate::recipe::{Recipe, schema};
use crate::tui::lib_keys::Outcome;
use crate::tui::lib_modal::ConfigModal;
use crate::tui::lib_state::{LibState, problem_line};

/// 2026-09-26: One row of the preview: what applying the donor does to `key`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BorrowChange {
    /// 2026-09-26: The form's spelling of the key (see `form_key`).
    ///
    /// A donor's `max_model_len` lands on an existing `max_seq_len` row, since
    /// `schema::flag_for` renders both as `--max-seq-len`; inserting the
    /// donor's spelling as well would put the flag on the launch line twice.
    pub key: String,
    /// 2026-09-26: The current effective value, or the word "removed" or "not set".
    pub from: String,
    pub to: String,
}

/// 2026-09-26: The rows applying `donor` would change, in the key order of the donor's `defaults` (a `BTreeMap`); unchanged keys are left out.
fn changes_from(
    recipe: &Recipe,
    overrides: &BTreeMap<String, String>,
    removed: &std::collections::BTreeSet<String>,
    donor: &Recipe,
) -> Vec<BorrowChange> {
    donor
        .defaults
        .iter()
        .filter_map(|(donor_key, to)| {
            let key = form_key(recipe, overrides, donor_key);
            if removed.contains(&key) {
                return Some(BorrowChange {
                    key,
                    from: "removed".into(),
                    to: to.clone(),
                });
            }
            let current = overrides.get(&key).or_else(|| recipe.defaults.get(&key));
            match current {
                Some(v) if v == to => None,
                Some(v) => Some(BorrowChange {
                    key,
                    from: v.clone(),
                    to: to.clone(),
                }),
                None => Some(BorrowChange {
                    key,
                    from: "not set".into(),
                    to: to.clone(),
                }),
            }
        })
        .collect()
}

/// 2026-09-26: The form key a donor key lands on: an existing row whose flag matches, else the donor's spelling.
fn form_key(recipe: &Recipe, overrides: &BTreeMap<String, String>, donor_key: &str) -> String {
    let Some(flag) = schema::flag_for(donor_key) else {
        return donor_key.to_string();
    };
    recipe
        .defaults
        .keys()
        .chain(overrides.keys())
        .find(|k| schema::flag_for(k).as_deref() == Some(flag.as_str()))
        .cloned()
        .unwrap_or_else(|| donor_key.to_string())
}

impl LibState {
    /// 2026-09-26: `b` on the Config form: open the donor list, excluding the form's own recipe.
    pub fn open_borrow_modal(&mut self) -> Outcome {
        let Some(recipe) = self.config_recipe() else {
            return Outcome::None;
        };
        let current_id = recipe.id.clone();
        let Some(entry) = self.current() else {
            return Outcome::None;
        };
        let model_type = entry
            .local
            .as_ref()
            .map(|l| l.model_type.as_str())
            .unwrap_or_default();
        let donors: Vec<Recipe> =
            super::lib_start::ranked_donors(&self.index.recipes, &entry.model, model_type)
                .into_iter()
                .filter(|d| d.id != current_id)
                .cloned()
                .collect();
        if donors.is_empty() {
            return Outcome::Toast {
                text: "no other recipe to borrow from".into(),
                error: true,
            };
        }
        self.modal = Some(ConfigModal::Borrow {
            donors,
            selected: 0,
        });
        Outcome::None
    }

    /// 2026-09-26: Enter on a donor: open the preview, or toast when nothing would change. Changes no setting.
    pub(super) fn pick_donor(&mut self, donors: Vec<Recipe>, selected: usize) -> Outcome {
        let (Some(donor), Some(recipe)) = (donors.get(selected), self.config_recipe()) else {
            return Outcome::None;
        };
        let changes = changes_from(recipe, &self.overrides, &self.removed, donor);
        if changes.is_empty() {
            return Outcome::Toast {
                text: format!("the form already matches {}", donor.id),
                error: false,
            };
        }
        self.modal = Some(ConfigModal::Preview {
            donors,
            donor: selected,
            changes,
            scroll: 0,
        });
        Outcome::None
    }

    /// 2026-09-26: Enter on the preview: apply the changes it showed, if the whole edited config validates.
    ///
    /// On success the form records the donor and its model in `borrowed`; on
    /// failure it sets `error` and leaves the overrides untouched.
    pub(super) fn apply_borrow(&mut self, donor: &Recipe, changes: &[BorrowChange]) -> Outcome {
        let Some(recipe) = self.config_recipe().cloned() else {
            return Outcome::None;
        };
        let mut overrides = self.overrides.clone();
        let mut removed = self.removed.clone();
        for change in changes {
            // 2026-09-26: A borrowed key is no longer removed; the donor gives it a value.
            removed.remove(&change.key);
            if recipe.defaults.get(&change.key) == Some(&change.to) {
                overrides.remove(&change.key);
            } else {
                overrides.insert(change.key.clone(), change.to.clone());
            }
        }
        match recipe.serve_args_edited(&overrides, &removed) {
            Ok(_) => {
                self.overrides = overrides;
                self.removed = removed;
                self.borrowed = Some(format!("{} (measured on {})", donor.id, donor.model));
                self.error = None;
                Outcome::Toast {
                    text: format!(
                        "{} settings borrowed from {} — d restores the recipe",
                        changes.len(),
                        donor.id
                    ),
                    error: false,
                }
            }
            Err(e) => {
                self.error = Some(problem_line(&format!("{e:#}")));
                Outcome::None
            }
        }
    }
}

#[cfg(test)]
#[path = "lib_borrow_tests.rs"]
mod tests;
