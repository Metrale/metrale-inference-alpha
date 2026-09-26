// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Benchmarks section rendering: Suite (list, variants, parameters, live run) and History.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.
//!
//! One file per view; this one dispatches and holds the pieces they share.
//! The live run and History both hold a `BenchmarkResult`, so they draw it
//! with the same stats, table and verdict functions.

mod history;
mod list;
mod params;
mod run;
mod variants;

use ratatui::Frame;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Paragraph, Row, Table};

use super::panel;
use crate::tui::app::{App, BenchSub};
use crate::tui::bench_state::View;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    match app.bench_sub {
        BenchSub::History => history::draw(f, app, area),
        BenchSub::Suite => match app.bench.view {
            View::List => list::draw(f, app, area),
            View::Variants => variants::draw(f, app, area),
            View::Params => params::draw(f, app, area),
            View::Run => run::draw(f, app, area),
        },
    }
}

/// 2026-09-26: The `OFFICIAL` / `COMMUNITY` badge: brand green for official
/// plugins, the warning style otherwise.
pub(super) fn origin_badge(meta: &metrale_bench::PluginMetadata) -> Span<'static> {
    if meta.official {
        Span::styled(
            " OFFICIAL ",
            theme::brand_green().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        )
    } else {
        Span::styled(
            " COMMUNITY ",
            theme::warn().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        )
    }
}

/// 2026-09-26: The origin badge and version, then the plugin's non-empty
/// authorship and support fields as label/value rows.
pub(super) fn metadata_lines(meta: &metrale_bench::PluginMetadata) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(vec![
        origin_badge(meta),
        Span::styled(format!("  v{}", meta.version), theme::text2()),
    ])];
    for (label, value) in meta.rows() {
        lines.push(Line::from(vec![
            Span::styled(format!(" {label:<13}"), theme::dim()),
            Span::styled(value.to_string(), theme::text2()),
        ]));
    }
    lines
}

/// 2026-09-26: The verdict banner. `Info` is cyan, not green: a benchmark
/// that measured without gating has not passed anything.
pub(super) fn verdict_line(verdict: &metrale_bench::Verdict) -> Line<'static> {
    use metrale_bench::VerdictKind as K;
    let (label, style) = match verdict.kind {
        K::Pass => (" PASS ", theme::brand_green()),
        K::Fail => (" FAIL ", theme::error()),
        K::Info => (" INFO ", theme::brand_cyan()),
    };
    Line::from(vec![
        Span::styled(
            label,
            style.add_modifier(Modifier::REVERSED | Modifier::BOLD),
        ),
        Span::styled(format!("  {}", verdict.reason), theme::text()),
    ])
}

/// 2026-09-26: The stat tile row above a results table; draws nothing for no
/// stats.
pub(super) fn draw_stats(f: &mut Frame, stats: &[metrale_bench::Stat], area: Rect) {
    if stats.is_empty() {
        return;
    }
    let widths: Vec<Constraint> =
        std::iter::repeat_n(Constraint::Ratio(1, stats.len() as u32), stats.len()).collect();
    let cols = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Horizontal)
        .constraints(widths)
        .split(area);
    for (stat, cell) in stats.iter().zip(cols.iter()) {
        let block = panel(format!("{} ─", stat.label.to_uppercase()), false);
        let inner = block.inner(*cell);
        f.render_widget(block, *cell);
        let line = Line::from(vec![
            Span::styled(
                format!(" {}", stat.value),
                theme::cell_style(stat.style).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" {}", stat.unit), theme::text2()),
        ]);
        f.render_widget(Paragraph::new(line), inner);
    }
}

/// 2026-09-26: A benchmark's results table, for the live run and History.
/// Returns the scroll ceiling it clamped to, which the caller stores for its
/// key handler; only this function knows the viewport height.
pub(super) fn draw_table(
    f: &mut Frame,
    table: &metrale_bench::ResultTable,
    scroll: usize,
    area: Rect,
) -> usize {
    let block = panel(
        format!("{} ─ {} rows ─", table.title, table.rows.len()),
        false,
    );
    let inner = block.inner(area);
    f.render_widget(block, area);
    let header = Row::new(
        table
            .columns
            .iter()
            .map(|c| Cell::from(c.title.clone()).style(theme::dim()))
            .collect::<Vec<_>>(),
    );
    // 2026-09-26: Rows scroll; the header row stays.
    let visible = inner.height.saturating_sub(1) as usize;
    let max_scroll = table.rows.len().saturating_sub(visible);
    let rows: Vec<Row> = table
        .rows
        .iter()
        .skip(scroll.min(max_scroll))
        .take(visible)
        .map(|cells| {
            Row::new(
                cells
                    .iter()
                    .map(|c| Cell::from(c.text.clone()).style(theme::cell_style(c.style)))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let widths: Vec<Constraint> = table
        .columns
        .iter()
        .map(|c| Constraint::Length(c.width))
        .collect();
    f.render_widget(
        Table::new(rows, widths)
            .header(header)
            .column_spacing(1)
            .style(theme::text()),
        inner,
    );
    max_scroll
}

#[cfg(test)]
#[path = "bench_tests.rs"]
mod tests;
