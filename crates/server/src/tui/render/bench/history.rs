// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: History view: runs recorded under the store's `runs/` directory, a list and the selected run's result.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.
//!
//! A stored run holds the same `BenchmarkResult` the live pane drew, so its
//! detail uses the shared stats, table and verdict helpers. The list draws a
//! rule, labelled with the new basis (`basis_of`), wherever the model, the
//! serve overrides or the parameters change between adjacent runs.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::panel;
use super::{draw_stats, draw_table, verdict_line};
use crate::tui::app::App;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    if app.bench.history.is_empty() {
        let block = panel("HISTORY ─".into(), false);
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(
            Paragraph::new(vec![
                Line::default(),
                Line::from(Span::styled("  No runs recorded yet.", theme::text2())),
                Line::from(Span::styled(
                    "  Every completed run is written to ~/.metrale/runs and appears here.",
                    theme::dim(),
                )),
            ]),
            inner,
        );
        return;
    }

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(34), Constraint::Min(30)])
        .split(area);
    draw_list(f, app, cols[0]);
    draw_detail(f, app, cols[1]);
}

fn draw_list(f: &mut Frame, app: &App, area: Rect) {
    let block = panel(format!("RUNS ─ {} ─", app.bench.history.len()), true);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let visible = inner.height as usize;
    let offset = app
        .bench
        .history_row
        .saturating_sub(visible.saturating_sub(1));
    let mut lines: Vec<Line> = Vec::new();
    let mut prev_basis: Option<String> = None;
    for (i, entry) in app
        .bench
        .history
        .iter()
        .enumerate()
        .skip(offset)
        .take(visible)
    {
        // 2026-09-26: A rule where the basis changes. `serve_overrides` is
        // empty for a run this process did not serve (a `--url` attach, and
        // every TUI run: `bench_state` records `Default::default()`), so two
        // attached runs against differently configured servers share the
        // `unpinned` basis and get no rule between them.
        let basis = basis_of(&entry.target_model, &entry.params, &entry.serve_overrides);
        if let Some(prev) = &prev_basis
            && *prev != basis
            && !lines.is_empty()
        {
            lines.push(Line::from(Span::styled(
                format!(" ┄┄ {basis} ┄┄"),
                theme::dim(),
            )));
        }
        prev_basis = Some(basis);
        let selected = i == app.bench.history_row;
        let mark = match entry.frame.verdict.as_ref().map(|v| v.kind) {
            Some(metrale_bench::VerdictKind::Pass) => Span::styled("✓", theme::brand_green()),
            Some(metrale_bench::VerdictKind::Fail) => Span::styled("✗", theme::error()),
            _ => Span::styled("·", theme::dim()),
        };
        let mut line = Line::from(vec![
            Span::styled(if selected { "▌" } else { " " }, theme::brand_purple()),
            mark,
            Span::styled(
                format!(" {:<20}", entry.benchmark_id),
                if selected {
                    theme::text().add_modifier(Modifier::BOLD)
                } else {
                    theme::text2()
                },
            ),
            Span::styled(entry.age_text(), theme::dim()),
        ]);
        if selected {
            line = line.style(theme::selected());
        }
        lines.push(line);
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// 2026-09-26: A run's comparison basis: the last path segment of the model,
/// the serve regime (`regime_of`), and a short hash of every parameter. Takes
/// the fields rather than a `RunRecord`, so tests need not build one.
pub(super) fn basis_of(
    target_model: &str,
    params: &std::collections::BTreeMap<String, String>,
    serve_overrides: &std::collections::BTreeMap<String, String>,
) -> String {
    let model = target_model.rsplit('/').next().unwrap_or(target_model);
    // 2026-09-26: Every parameter goes into the hash, not a chosen subset.
    let params = joined(params);
    format!(
        "{model} · {} · {}",
        regime_of(serve_overrides),
        short_hash(&params)
    )
}

/// 2026-09-26: The serve regime as a label: up to two override keys by name
/// (then `+n` for the rest), and a short hash of the keys and values, so the
/// same keys with different values still differ. Values are not spelled out
/// because the list is 34 columns wide.
fn regime_of(serve_overrides: &std::collections::BTreeMap<String, String>) -> String {
    if serve_overrides.is_empty() {
        // 2026-09-26: Empty means no regime was recorded (see `draw_list`),
        // not that the server had no overrides.
        return "unpinned".into();
    }
    let keys: Vec<&str> = serve_overrides.keys().map(String::as_str).collect();
    let named = match keys.len() {
        1..=2 => keys.join("+"),
        n => format!("{}+{}", keys[..2].join("+"), n - 2),
    };
    format!("{named}:{}", short_hash(&joined(serve_overrides)))
}

/// 2026-09-26: `k=v` pairs joined with `,`, in `BTreeMap` key order, so the
/// string does not depend on insertion order.
fn joined(map: &std::collections::BTreeMap<String, String>) -> String {
    map.iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// 2026-09-26: A 4-hex-digit tag: 64-bit FNV-1a folded to 16 bits. Not a
/// security hash, and distinct inputs can share a tag.
fn short_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:04x}", (h ^ (h >> 32)) as u16)
}

fn draw_detail(f: &mut Frame, app: &App, area: Rect) {
    let Some(entry) = app.bench.history.get(app.bench.history_row) else {
        return;
    };
    let frame = &entry.frame;
    let has_stats = !frame.summary.is_empty();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(if has_stats { 3 } else { 0 }),
            Constraint::Min(6),
            Constraint::Length(2),
        ])
        .split(area);
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                format!(" {} ", entry.benchmark_id),
                theme::text().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "· {} · {:.0}s · {}",
                    frame.phase,
                    frame.elapsed.as_secs_f64(),
                    entry.age_text()
                ),
                theme::text2(),
            ),
        ])),
        rows[0],
    );
    if has_stats {
        draw_stats(f, &frame.summary, rows[1]);
    }
    if let Some(table) = &frame.table {
        // 2026-09-26: History keeps its own table scroll, separate from the
        // live run's.
        let max = draw_table(f, table, app.bench.history_table_scroll, rows[2]);
        app.bench.history_table_scroll_max.set(max);
    }
    if let Some(verdict) = &frame.verdict {
        f.render_widget(Paragraph::new(verdict_line(verdict)), rows[3]);
    }
}
