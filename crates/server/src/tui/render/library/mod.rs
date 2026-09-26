// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Library section: the model list, one model's recipe cards,
//! and the config form, with any open picker drawn over them.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

pub mod cards;
pub mod config;
pub mod list;
mod list_detail;
mod modal;

use ratatui::Frame;
use ratatui::layout::Rect;

use crate::tui::app::App;
use crate::tui::lib_state::View;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    match app.lib.view {
        View::List => list::draw(f, app, area),
        View::Cards => cards::draw(f, app, area),
        View::Config => config::draw(f, app, area),
    }
    // 2026-09-26: Over the pane but under the app-level overlays, which
    // `render::draw` paints after this returns.
    modal::draw(f, app, area);
}

#[cfg(test)]
#[path = "library_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "start_tests.rs"]
mod start_tests;
