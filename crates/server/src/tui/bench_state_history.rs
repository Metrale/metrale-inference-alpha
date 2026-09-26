// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: History-pane members of [`BenchState`]: loading the run history, the elapsed-time
//! text and the result-card export.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::BenchState;

impl BenchState {
    /// 2026-09-26: Populate the History pane, once until the next finished run clears `history_loaded`.
    ///
    /// `history::load_all` sorts newest first across all benchmarks, not grouped by benchmark.
    pub fn load_history(&mut self) {
        if self.history_loaded {
            return;
        }
        self.history_loaded = true;
        self.history = match &self.executor {
            Some(executor) => metrale_bench::history::load_all(executor.artifacts()),
            None => Vec::new(),
        };
        self.history_row = self.history_row.min(self.history.len().saturating_sub(1));
    }

    pub fn elapsed_text(&self) -> String {
        let secs = self.started.map(|s| s.elapsed().as_secs()).unwrap_or(0);
        format!(
            "{:02}:{:02}:{:02}",
            secs / 3600,
            (secs / 60) % 60,
            secs % 60
        )
    }
}

impl BenchState {
    /// 2026-09-26: `c` in History: write a result card for the selected run's benchmark.
    ///
    /// Reports through `status`, never a toast.
    pub fn export_card(&mut self) -> crate::tui::bench_keys::Outcome {
        let Some(run) = self.history.get(self.history_row) else {
            return crate::tui::bench_keys::Outcome::None;
        };
        self.status =
            match crate::cli::bench_card::render_card_for_benchmark(&run.benchmark_id, None) {
                Ok(path) => format!("card written to {}", path.display()),
                Err(e) => format!("no card: {e}"),
            };
        crate::tui::bench_keys::Outcome::None
    }
}
