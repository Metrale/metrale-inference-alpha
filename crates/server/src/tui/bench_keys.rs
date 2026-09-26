// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Keyboard handling for the Benchmarks section: the Suite views (List, Variants,
//! Params, Run) and the History subsection.
//!
//! Owner: server tui.
//! Invariants:
//! - In Suite ▸ Params, while a preflight is open, every key goes to it.

use crossterm::event::{KeyCode, KeyEvent};

use super::app::BenchSub;
use super::bench_state::{BenchState, View};

/// 2026-09-26: What the section wants the app to do afterwards.
pub enum Outcome {
    None,
    /// 2026-09-26: Show a toast: a refused start or a refused preflight acceptance.
    Toast {
        text: String,
        error: bool,
    },
}

impl BenchState {
    pub fn on_key(&mut self, key: KeyEvent, sub: BenchSub) -> Outcome {
        if sub == BenchSub::History {
            return self.history_key(key);
        }
        match self.view {
            View::List => self.list_key(key),
            View::Variants => {
                self.variants_key(key);
                Outcome::None
            }
            View::Params => self.params_key(key),
            View::Run => self.run_key(key),
        }
    }

    fn list_key(&mut self, key: KeyEvent) -> Outcome {
        let n = metrale_bench::registry::all().len();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') if n > 0 => {
                self.select((self.selected + 1).min(n - 1));
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.select(self.selected.saturating_sub(1));
            }
            // 2026-09-26: Enter on the running benchmark opens its run. Otherwise `enter_selected` opens
            // the variant step when the benchmark has variants, and the form when it does not.
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if self.is_running() && self.running_id == self.descriptor().map(|d| d.id) {
                    self.view = View::Run;
                } else {
                    self.enter_selected();
                }
            }
            // 2026-09-26: The last run's frame stays reachable after navigating away.
            KeyCode::Char('v') if self.frame.is_some() => self.view = View::Run,
            // 2026-09-26: The list keeps the selection in view, so first and last are top and bottom.
            KeyCode::Char('g') | KeyCode::Home if n > 0 => self.select(0),
            KeyCode::Char('G') | KeyCode::End if n > 0 => self.select(n - 1),
            // 2026-09-26: Paged by the renderer-published `suite_page` and clamped at both ends.
            KeyCode::PageDown if n > 0 => {
                let page = self.suite_page.get().max(1);
                self.select((self.selected + page).min(n - 1));
            }
            KeyCode::PageUp => {
                let page = self.suite_page.get().max(1);
                self.select(self.selected.saturating_sub(page));
            }
            _ => {}
        }
        Outcome::None
    }

    fn params_key(&mut self, key: KeyEvent) -> Outcome {
        // 2026-09-26: The preflight owns the keyboard while it is open, so the hidden form cannot change.
        if self.preflight.is_some() {
            return self.preflight_key(key);
        }
        if self.confirm_open {
            return self.confirm_key(key);
        }
        if self.editing {
            return self.edit_key(key);
        }
        let rows = self.row_count();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') if rows > 0 => {
                self.row = (self.row + 1).min(rows - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => self.row = self.row.saturating_sub(1),
            KeyCode::Enter => self.editing = true,
            // 2026-09-26: Back retraces the way in: the variant step when there is one, else the list.
            KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => {
                self.view = if self.variants.is_empty() {
                    View::List
                } else {
                    View::Variants
                };
            }
            // 2026-09-26: Reset the form to the schema's defaults (`select` reloads them).
            KeyCode::Char('d') => {
                let selected = self.selected;
                self.select(selected);
            }
            // 2026-09-26: Toggle the pre-run coherence probe between `Probe` and `Skip`.
            KeyCode::Char('p') => {
                self.coherence = match self.coherence {
                    metrale_bench::CoherencePolicy::Probe => metrale_bench::CoherencePolicy::Skip,
                    metrale_bench::CoherencePolicy::Skip => metrale_bench::CoherencePolicy::Probe,
                };
            }
            KeyCode::Char('s') => return self.request_start(),
            _ => {}
        }
        Outcome::None
    }

    /// 2026-09-26: A benchmark whose descriptor sets `needs_confirmation` asks first; the prompt is
    /// answered with `y`/`Y`, not by pressing `s` again.
    fn request_start(&mut self) -> Outcome {
        let needs_confirmation = self.descriptor().is_some_and(|d| d.needs_confirmation);
        if needs_confirmation && !self.confirm_open {
            self.confirm_open = true;
            return Outcome::None;
        }
        self.confirm_open = false;
        match self.begin_start() {
            Ok(()) => Outcome::None,
            Err(e) => Outcome::Toast {
                text: e,
                error: true,
            },
        }
    }

    /// 2026-09-26: While checking, only Esc does anything. Once checked, `p`, `P` or Enter proceeds
    /// and any other key cancels.
    fn preflight_key(&mut self, key: KeyEvent) -> Outcome {
        let checking = self
            .preflight
            .as_ref()
            .is_some_and(crate::tui::bench_preflight::Preflight::is_checking);
        if checking {
            if key.code == KeyCode::Esc {
                self.cancel_preflight();
            }
            return Outcome::None;
        }
        match key.code {
            KeyCode::Char('p') | KeyCode::Char('P') | KeyCode::Enter => {
                match self.accept_preflight() {
                    Ok(()) => Outcome::None,
                    Err(e) => Outcome::Toast {
                        text: e,
                        error: true,
                    },
                }
            }
            _ => {
                self.cancel_preflight();
                Outcome::None
            }
        }
    }

    fn confirm_key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => self.request_start(),
            _ => {
                self.confirm_open = false;
                Outcome::None
            }
        }
    }

    fn edit_key(&mut self, key: KeyEvent) -> Outcome {
        let row = self.row;
        match key.code {
            KeyCode::Enter => {
                self.commit_row(row);
                self.editing = false;
            }
            KeyCode::Esc => {
                // 2026-09-26: Restore the committed value, so a cancelled edit leaves no half-typed string.
                self.reset_row_buffer(row);
                self.editing = false;
            }
            KeyCode::Backspace => {
                if let Some(buf) = self.edit.get_mut(row) {
                    buf.pop();
                }
            }
            KeyCode::Char(c) => {
                if let Some(buf) = self.edit.get_mut(row) {
                    buf.push(c);
                }
            }
            _ => {}
        }
        Outcome::None
    }

    fn reset_row_buffer(&mut self, row: usize) {
        let current = match self.specs.get(row) {
            Some(spec) => self
                .values
                .get(spec.key)
                .map(|v| v.to_edit_string())
                .unwrap_or_else(|| spec.default.to_edit_string()),
            None if row == self.specs.len() => self.target.base_url.clone(),
            _ => self.target.model.clone(),
        };
        if let Some(buf) = self.edit.get_mut(row) {
            *buf = current;
        }
    }

    fn run_key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            // 2026-09-26: No toast: `cancel` sets the pane's own status line.
            KeyCode::Char('c') => self.cancel(),
            KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => self.view = View::List,
            // 2026-09-26: Clamped to the renderer-published `table_scroll_max`.
            KeyCode::Down | KeyCode::Char('j') => {
                self.table_scroll = (self.table_scroll + 1).min(self.table_scroll_max.get());
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.table_scroll = self.table_scroll.saturating_sub(1);
            }
            KeyCode::Char('g') | KeyCode::Home => self.table_scroll = 0,
            KeyCode::Char('G') | KeyCode::End => {
                self.table_scroll = self.table_scroll_max.get();
            }
            _ => {}
        }
        Outcome::None
    }

    fn history_key(&mut self, key: KeyEvent) -> Outcome {
        let n = self.history.len();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') if n > 0 => {
                self.history_row = (self.history_row + 1).min(n - 1);
                // 2026-09-26: The table offset belonged to the previous run, so it resets.
                self.history_table_scroll = 0;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.history_row = self.history_row.saturating_sub(1);
                self.history_table_scroll = 0;
            }
            // 2026-09-26: j/k select the run here, so PgUp/PgDn scroll the stored table, 5 rows at a time.
            KeyCode::PageDown => {
                self.history_table_scroll =
                    (self.history_table_scroll + 5).min(self.history_table_scroll_max.get());
            }
            KeyCode::PageUp => {
                self.history_table_scroll = self.history_table_scroll.saturating_sub(5);
            }
            // 2026-09-26: Write a card for the selected run's benchmark. It is rendered from the committed
            // gate record, not the history entry: a `RunRecord` has no hardware or commit sha to print.
            KeyCode::Char('c') if n > 0 => return self.export_card(),
            _ => {}
        }
        Outcome::None
    }
}

#[cfg(test)]
#[path = "bench_keys_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "bench_keys_more_tests.rs"]
mod more_tests;
