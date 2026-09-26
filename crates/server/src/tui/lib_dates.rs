// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A recipe's "updated" date: its `metadata.updated`, else the file's last GitHub commit.
//!
//! The GitHub lookup is lazy: only the recipe whose details are on screen,
//! only when it has no date of its own, and only once per id, failures
//! included. It runs on a worker thread (`recipe::fetch::updated_in_background`);
//! the render thread only `try_recv`s.
//!
//! Owner: server tui.
//! Invariants:
//! - At most one lookup is in flight (`pending_date`), and an id enters
//!   `dated` before its lookup starts and never leaves it.
//! - A recipe's own `metadata.updated` wins over a fetched date, and a
//!   starting-point card shows no date.

use super::lib_state::{LibState, View};

use crate::recipe::fetch;

impl LibState {
    /// 2026-09-26: The `updated` row's value for one recipe, used by both detail panes.
    ///
    /// In order: empty for a starting point; `metadata.updated`; a date in
    /// `fetched_dates`; a skeleton while this id's lookup is in flight; else
    /// empty, which the panes render as no row.
    ///
    /// Fetched dates live in their own map because `LibState::poll` replaces
    /// `index` wholesale on a refresh.
    pub fn date_text(&self, recipe: &crate::recipe::Recipe) -> String {
        // 2026-09-26: A starting point carries the donor's id, so a lookup would date the donor's file.
        if recipe.starting_point.is_some() {
            return String::new();
        }
        if !recipe.updated.is_empty() {
            return recipe.updated.clone();
        }
        if let Some(d) = self.fetched_dates.get(&recipe.id) {
            return d.clone();
        }
        if self.dating.as_deref() == Some(recipe.id.as_str()) {
            // 2026-09-26: Ten cells, the width of a `YYYY-MM-DD` date.
            return "░░░░░░░░░░".to_string();
        }
        String::new()
    }

    /// 2026-09-26: The recipe whose details are on screen: the row's `primary()` in List, the selected card
    /// in Cards and Config; never a starting point.
    pub fn visible_recipe_id(&self) -> Option<String> {
        match self.view {
            View::List => self
                .current()
                .and_then(|e| e.primary())
                .map(|r| r.id.clone()),
            View::Cards | View::Config => self
                .selected_card()
                .filter(|r| r.starting_point.is_none())
                .map(|r| r.id.clone()),
        }
    }

    /// 2026-09-26: Start a GitHub date lookup for `id`, if no lookup is in flight, `id` was never looked up,
    /// and the index holds `id` with an empty `metadata.updated`.
    ///
    /// The event loop calls it every tick while the Library is polled.
    pub fn want_date_for(&mut self, id: &str) {
        if self.pending_date.is_some() || self.dated.contains(id) {
            return;
        }
        let needs = self
            .index
            .recipes
            .iter()
            .any(|r| r.id == id && r.updated.is_empty());
        if !needs {
            return;
        }
        tracing::debug!("dating recipe {id} from GitHub commit history");
        self.dated.insert(id.to_string());
        self.dating = Some(id.to_string());
        self.pending_date = Some(fetch::updated_in_background(id));
    }

    /// 2026-09-26: Collect a finished date lookup into `fetched_dates`; true when a non-empty date arrived.
    pub fn poll_date(&mut self) -> bool {
        let Some(rx) = &self.pending_date else {
            return false;
        };
        match rx.try_recv() {
            Ok((id, date)) => {
                self.pending_date = None;
                self.dating = None;
                // 2026-09-26: Keyed by the id the worker echoed back, not the current selection.
                if let Some(d) = date.filter(|d| !d.is_empty()) {
                    self.fetched_dates.insert(id, d);
                    return true;
                }
                false
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => false,
            // 2026-09-26: The worker died without sending; the id stays in `dated`, so it is not retried.
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.pending_date = None;
                self.dating = None;
                false
            }
        }
    }
}

#[cfg(test)]
#[path = "lib_dates_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "lib_dates_more_tests.rs"]
mod more_tests;
