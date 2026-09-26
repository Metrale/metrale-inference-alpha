// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Library's background scan of the local HuggingFace cache, and the attach-state queries.
//!
//! The scan runs on a `tui::worker` thread (`data::library::scan_in_background`);
//! the render thread only `try_recv`s.
//!
//! Owner: server tui.
//! Invariants:
//! - At most one scan is in flight (`pending_scan`).

use super::lib_state::LibState;
use crate::tui::data::library::LibraryEntry;

impl LibState {
    /// 2026-09-26: Whether the recipe store has been attached (`root` is set).
    pub fn attached(&self) -> bool {
        self.root.is_some()
    }

    /// 2026-09-26: Whether attaching was tried and found impossible.
    pub fn recipes_unavailable(&self) -> bool {
        self.recipes_unavailable
    }

    /// 2026-09-26: Whether a background scan is running; `events_rules::tick_work` starts no scan while one is.
    pub fn scan_in_flight(&self) -> bool {
        self.pending_scan.is_some()
    }

    /// 2026-09-26: Start a background scan of the local HF cache; ignored while one is in flight.
    pub fn start_scan(&mut self, cache_dir: Option<&std::path::Path>) {
        if self.pending_scan.is_some() {
            return;
        }
        self.pending_scan = Some(crate::tui::data::library::scan_in_background(cache_dir));
    }

    /// 2026-09-26: Collect a finished scan's entries for the caller to store; `None` while running or if the
    /// scanner died.
    pub fn poll_scan(&mut self) -> Option<Vec<LibraryEntry>> {
        let rx = self.pending_scan.as_ref()?;
        match rx.try_recv() {
            Ok(found) => {
                self.pending_scan = None;
                Some(found)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.pending_scan = None;
                None
            }
        }
    }
}

#[cfg(test)]
#[path = "lib_scan_tests.rs"]
mod tests;
