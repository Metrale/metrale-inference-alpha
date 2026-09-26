// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Plain data carried by [`super::app::App`]: the toast record and the Ops REPL state.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use std::time::Instant;

pub struct Toast {
    pub text: String,
    pub error: bool,
    pub at: Instant,
}

#[derive(Default)]
pub struct OpsState {
    pub input: String,
    pub history: Vec<String>,
    pub history_pos: Option<usize>,
    pub output: Vec<String>,
    /// 2026-09-26: Rows scrolled up from the newest output line; 0 follows the newest.
    pub scroll_up: usize,
    /// 2026-09-26: Largest useful `scroll_up`, set by the Ops renderer on every frame.
    pub scroll_max: std::cell::Cell<usize>,
}
