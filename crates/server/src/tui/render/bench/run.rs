// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The live run pane: header and progress, stat tiles, results table, verdict, log.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::{gradient_bar, panel};
use super::{draw_stats, draw_table, verdict_line};
use crate::tui::app::App;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let has_table = app
        .bench
        .frame
        .as_ref()
        .is_some_and(|fr| fr.table.is_some());
    let has_stats = app
        .bench
        .frame
        .as_ref()
        .is_some_and(|fr| !fr.summary.is_empty());
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(if has_stats { 3 } else { 0 }),
            Constraint::Min(if has_table { 8 } else { 0 }),
            Constraint::Length(2),
            Constraint::Length(8),
        ])
        .split(area);

    draw_header(f, app, rows[0]);
    if let Some(frame) = &app.bench.frame {
        if has_stats {
            draw_stats(f, &frame.summary, rows[1]);
        }
        if let Some(table) = &frame.table {
            let max = draw_table(f, table, app.bench.table_scroll, rows[2]);
            app.bench.table_scroll_max.set(max);
        }
        if let Some(verdict) = &frame.verdict {
            f.render_widget(Paragraph::new(verdict_line(verdict)), rows[3]);
        }
    }
    draw_log(f, app, rows[4]);
}

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    // 2026-09-26: The three header rows are placed by hand; on a short
    // terminal the slot can be under three rows high, so a row below
    // `area.bottom()` is skipped rather than drawn outside the slot.
    let row = |y: u16| -> Option<Rect> {
        (y < area.bottom()).then_some(Rect {
            y,
            height: 1,
            ..area
        })
    };
    let name = app.bench.descriptor().map(|d| d.name).unwrap_or("");
    let running = app.bench.is_running();
    let spinner = if running {
        theme::SPINNER[(app.tick as usize / 2) % theme::SPINNER.len()]
    } else {
        "●"
    };
    let spinner_style = if running {
        theme::brand_cyan()
    } else {
        theme::brand_green()
    };
    let head = Line::from(vec![
        Span::styled(format!(" {spinner} "), spinner_style),
        Span::styled(name.to_string(), theme::text().add_modifier(Modifier::BOLD)),
        Span::styled(format!("  {}", app.bench.status), theme::text2()),
        Span::styled(format!("   {}  ", app.bench.elapsed_text()), theme::dim()),
        Span::styled(app.bench.target.base_url.clone(), theme::dim()),
    ]);
    if let Some(r) = row(area.y) {
        f.render_widget(Paragraph::new(head), r);
    }

    // 2026-09-26: With no known total the bar is a caption ("working…" or
    // "idle"), not a bar.
    let Some(bar_area) = row(area.y + 1).map(|r| Rect {
        x: r.x + 1,
        width: r.width.saturating_sub(2),
        ..r
    }) else {
        return;
    };
    match app.bench.progress {
        Some((done, total)) if total > 0 => {
            let frac = done as f64 / total as f64;
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(10), Constraint::Length(14)])
                .split(bar_area);
            f.render_widget(Paragraph::new(gradient_bar(frac, cols[0].width)), cols[0]);
            f.render_widget(
                Paragraph::new(Span::styled(
                    format!(" {done}/{total}  {:.0}%", frac * 100.0),
                    theme::text2(),
                )),
                cols[1],
            );
        }
        _ => f.render_widget(
            Paragraph::new(Span::styled(
                if running { "working…" } else { "idle" },
                theme::dim(),
            )),
            bar_area,
        ),
    }
    if let Some(r) = row(area.y + 2) {
        f.render_widget(
            Paragraph::new(Span::styled(
                if running {
                    " c cancel · j/k scroll table · Esc back to suite"
                } else {
                    " Esc back to suite · j/k scroll table"
                },
                theme::dim(),
            )),
            r,
        );
    }
}

fn draw_log(f: &mut Frame, app: &App, area: Rect) {
    let block = panel(format!("LOG ─ {} lines ─", app.bench.log.len()), false);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // 2026-09-26: Log entries wrap rather than truncate; continuations are
    // indented further so a wrapped entry reads as one entry.
    let visible = inner.height as usize;
    let width = inner.width.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::new();
    for entry in app.bench.log.iter().rev() {
        use metrale_bench::LogLevel as L;
        let style = match entry.level {
            L::Error => theme::error(),
            L::Warn => theme::warn(),
            L::Info => theme::text2(),
            L::Debug => theme::dim(),
        };
        let mut wrapped = super::super::wrap(&entry.text, width.saturating_sub(1), style);
        // 2026-09-26: First line indented one column, continuations three.
        for (i, line) in wrapped.iter_mut().enumerate() {
            let pad = if i == 0 { " " } else { "   " };
            line.spans.insert(0, Span::styled(pad, style));
        }
        // 2026-09-26: Built newest-first, so each entry goes in front.
        wrapped.extend(std::mem::take(&mut lines));
        lines = wrapped;
        if lines.len() >= visible {
            break;
        }
    }
    // 2026-09-26: Keep the newest lines when the tail overflows the pane.
    if lines.len() > visible {
        lines.drain(..lines.len() - visible);
    }
    f.render_widget(Paragraph::new(lines), inner);
}
