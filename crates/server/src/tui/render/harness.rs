// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Shared `TestBackend` helpers for the render tests: `screen`
//! draws the whole dashboard, `draw_into` one closure; both return one
//! trailing-trimmed `String` per row.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use ratatui::Terminal;
use ratatui::backend::TestBackend;

use crate::tui::app::App;

/// 2026-09-26: `render::draw` at `w`×`h`, one `String` per row, trailing
/// blanks trimmed.
pub(super) fn screen(app: &App, w: u16, h: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("backend");
    terminal
        .draw(|f| crate::tui::render::draw(f, app))
        .expect("draw");
    let buf = terminal.backend().buffer();
    (0..h)
        .map(|y| {
            (0..w)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}

pub(super) fn has(rows: &[String], needle: &str) -> bool {
    rows.iter().any(|r| r.contains(needle))
}

/// 2026-09-26: Render one closure over the whole `w`×`h` area, for helpers
/// that take a `Rect` rather than an `App`. Rows as in [`screen`].
pub(super) fn draw_into(
    w: u16,
    h: u16,
    f: impl FnOnce(&mut ratatui::Frame, ratatui::layout::Rect),
) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("backend");
    terminal
        .draw(|frame| {
            let area = frame.area();
            f(frame, area);
        })
        .expect("draw");
    let buf = terminal.backend().buffer();
    (0..h)
        .map(|y| {
            (0..w)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}
