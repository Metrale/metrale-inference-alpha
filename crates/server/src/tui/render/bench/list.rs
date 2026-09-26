// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The benchmark suite list and its detail pane.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::panel;
use super::super::wrap;
use super::metadata_lines;
use crate::tui::app::App;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(area);
    draw_list(f, app, cols[0]);
    draw_detail(f, app, cols[1]);
}

fn draw_list(f: &mut Frame, app: &App, area: Rect) {
    let all = metrale_bench::registry::all();
    let n = all.len();
    let block = panel("SUITE ─".into(), true);
    let inner = block.inner(area);

    // 2026-09-26: The scroll offset counts entries, not lines. Each entry
    // takes `ROOMY` rows (with a blank separator) when every entry fits that
    // way, else `COMPACT`; the offset keeps the selection on screen.
    const ROOMY: usize = 4;
    const COMPACT: usize = 3;
    let rows_per_entry = if n * ROOMY <= inner.height as usize {
        ROOMY
    } else {
        COMPACT
    };
    let visible = (inner.height as usize / rows_per_entry).max(1);
    // 2026-09-26: Clamped so the last page is full.
    let offset = app
        .bench
        .selected
        .saturating_sub(visible.saturating_sub(1))
        .min(n.saturating_sub(visible));
    // 2026-09-26: Published for PgUp/PgDn in `bench_keys`; only this function
    // knows how many entries one page holds.
    app.bench.suite_page.set(visible);

    // 2026-09-26: A "first-last of n" position on the bottom border, only when
    // the list is clipped.
    let block = if n > visible {
        let last = (offset + visible).min(n);
        block.title_bottom(Span::styled(
            format!("─ {}-{last} of {n} ─", offset + 1),
            theme::dim(),
        ))
    } else {
        block
    };
    f.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    for (i, descriptor) in all.iter().enumerate().skip(offset) {
        let selected = i == app.bench.selected;
        let running = app.bench.running_id == Some(descriptor.id) && app.bench.is_running();
        let marker = if running {
            Span::styled(
                theme::SPINNER[(app.tick as usize / 2) % theme::SPINNER.len()],
                theme::brand_cyan(),
            )
        } else if selected {
            Span::styled("▌", theme::brand_purple())
        } else {
            Span::raw(" ")
        };
        let name_style = if selected {
            theme::text().add_modifier(Modifier::BOLD)
        } else {
            theme::text2()
        };
        let label = format!(" {}", descriptor.name);
        let mut line = Line::from(vec![marker, Span::styled(label, name_style)]);
        if selected {
            line = line.style(theme::selected());
        }
        lines.push(line);
        lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled(descriptor.summary.to_string(), theme::dim()),
        ]));
        lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled(descriptor.duration_hint.to_string(), theme::dim()),
            // 2026-09-26: A benchmark with `needs_confirmation` is flagged in
            // the list, before it is opened.
            if descriptor.needs_confirmation {
                Span::styled("  ⚠ runs shell", theme::warn())
            } else {
                Span::raw("")
            },
        ]));
        if rows_per_entry == ROOMY {
            lines.push(Line::default());
        }
    }
    f.render_widget(Paragraph::new(lines), inner);
}

#[cfg(test)]
#[path = "list_tests.rs"]
mod tests;

fn draw_detail(f: &mut Frame, app: &App, area: Rect) {
    let Some(descriptor) = app.bench.descriptor() else {
        return;
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(11)])
        .split(area);

    let block = panel(format!("{} ─", descriptor.name.to_uppercase()), false);
    let inner = block.inner(rows[0]);
    f.render_widget(block, rows[0]);
    let width = inner.width.saturating_sub(2) as usize;
    let mut lines = wrap(descriptor.detail, width, theme::text2());
    lines.push(Line::default());
    lines.push(Line::from(vec![
        Span::styled(" Parameters  ", theme::dim()),
        Span::styled(
            format!("{} editable", app.bench.specs.len()),
            theme::text2(),
        ),
    ]));
    // 2026-09-26: `updated` is when the benchmark's measurement last changed
    // (see `BenchmarkDescriptor::updated`).
    lines.push(Line::from(vec![
        Span::styled(" Updated     ", theme::dim()),
        Span::styled(descriptor.updated.to_string(), theme::text2()),
    ]));
    lines.push(Line::from(vec![
        Span::styled(" Target      ", theme::dim()),
        Span::styled(app.bench.target.base_url.clone(), theme::brand_cyan()),
        Span::styled(format!("  {}", app.bench.target.model), theme::text2()),
    ]));
    lines.push(Line::default());
    lines.push(Line::from(Span::styled(
        " ⏎ configure and start",
        theme::brand_cyan(),
    )));
    f.render_widget(Paragraph::new(lines), inner);

    // 2026-09-26: Plugin authorship and support links.
    let meta_block = panel("PLUGIN ─".into(), false);
    let meta_inner = meta_block.inner(rows[1]);
    f.render_widget(meta_block, rows[1]);
    f.render_widget(
        Paragraph::new(metadata_lines(app.bench.plugin_metadata())),
        meta_inner,
    );
}
