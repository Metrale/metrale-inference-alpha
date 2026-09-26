// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Dragging a selection across the dashboard, and reading its
//! text back out of the rendered frame. `TerminalGuard` enables mouse
//! capture, so the terminal forwards drags here instead of drawing its own
//! highlight.
//!
//! A selection is linear, in reading order: from the start cell to the end
//! cell, with whole lines in between. A drag across a pane border therefore
//! picks up cells of the pane beside it; [`extract`] trims trailing whitespace
//! per line.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

/// 2026-09-26: An in-progress or finished drag, in terminal cell
/// coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    /// 2026-09-26: Where the button went down. Fixed for the life of the
    /// drag.
    pub anchor: (u16, u16),
    /// 2026-09-26: Where the pointer is now.
    pub cursor: (u16, u16),
}

impl Selection {
    pub fn new(at: (u16, u16)) -> Self {
        Self {
            anchor: at,
            cursor: at,
        }
    }

    /// 2026-09-26: `(start, end)` in reading order, whichever way the drag
    /// went.
    pub fn ordered(&self) -> ((u16, u16), (u16, u16)) {
        let (ax, ay) = self.anchor;
        let (cx, cy) = self.cursor;
        if (ay, ax) <= (cy, cx) {
            (self.anchor, self.cursor)
        } else {
            (self.cursor, self.anchor)
        }
    }

    /// 2026-09-26: Whether the cell is inside the selection.
    pub fn contains(&self, x: u16, y: u16) -> bool {
        let ((sx, sy), (ex, ey)) = self.ordered();
        if y < sy || y > ey {
            return false;
        }
        // 2026-09-26: On one line, a column range. Otherwise the first line
        // runs to the right edge and the last starts at the left edge.
        if sy == ey {
            return x >= sx && x <= ex;
        }
        if y == sy {
            return x >= sx;
        }
        if y == ey {
            return x <= ex;
        }
        true
    }

    /// 2026-09-26: Whether the pointer moved. The mouse-up handler copies only
    /// a drag, so a plain click copies nothing (`events.rs`).
    pub fn is_drag(&self) -> bool {
        self.anchor != self.cursor
    }
}

/// 2026-09-26: The text under a selection, read out of the rendered frame.
///
/// Trailing whitespace is trimmed per line, blank lines inside the selection
/// are kept and trailing blank lines are dropped. Returns an empty string
/// when nothing is covered.
pub fn extract(buf: &Buffer, area: Rect, sel: &Selection) -> String {
    let mut lines: Vec<String> = Vec::new();
    let ((_, sy), (_, ey)) = sel.ordered();
    for y in sy..=ey {
        if y < area.y || y >= area.y.saturating_add(area.height) {
            continue;
        }
        let mut line = String::new();
        for x in area.x..area.x.saturating_add(area.width) {
            if !sel.contains(x, y) {
                continue;
            }
            line.push_str(buf[(x, y)].symbol());
        }
        lines.push(line.trim_end().to_string());
    }
    // 2026-09-26: Trailing blank lines come from dragging past the content.
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}

#[cfg(test)]
#[path = "selection_tests.rs"]
mod tests;
