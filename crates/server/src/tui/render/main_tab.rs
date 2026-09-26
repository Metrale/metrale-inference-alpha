// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Main ▸ Overview: the startup checklist and weight-load panel
//! while loading, the READY strip once serving, then the badge chips and the
//! live log pane. Main ▸ Kernels is `main_tab_kernels.rs`.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Sparkline};

use super::{gradient_bar, panel};
use crate::tui::app::App;
use crate::tui::progress::PhaseState;
use crate::tui::{log_ring, logo, theme};

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    // 2026-09-26: `progress.ready` alone does not mean a model is loaded, so
    // `awaiting_model` keeps the loading layout.
    let loading = !app.progress.ready || app.awaiting_model;
    let top_h = if loading { 15 } else { 3 };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(top_h),
            Constraint::Length(2),
            Constraint::Min(5),
        ])
        .split(area);
    if loading {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(34), Constraint::Min(30)])
            .split(rows[0]);
        draw_phases(f, app, cols[0]);
        draw_weight_load(f, app, cols[1]);
    } else {
        draw_ready_strip(f, app, rows[0]);
    }
    draw_chips(f, app, rows[1]);
    draw_logs(f, app, rows[2]);
}

fn draw_phases(f: &mut Frame, app: &App, area: Rect) {
    let (done, total, secs) = app.progress.phase_counts();
    // 2026-09-26: With no model, no count or clock, and every phase is drawn
    // pending.
    let block = panel(
        if app.awaiting_model {
            "STARTUP ─ awaiting a model ─".to_string()
        } else {
            format!("STARTUP ─ {done}/{total} ── {secs:.1}s ─")
        },
        false,
    );
    let mut lines = Vec::new();
    for p in &app.progress.phases {
        let state = if app.awaiting_model {
            PhaseState::Pending
        } else {
            p.state
        };
        let (glyph, gstyle, label_style) = match state {
            PhaseState::Done => ("✓", theme::brand_green(), theme::text2()),
            PhaseState::Running => (
                theme::SPINNER[(app.tick as usize) % theme::SPINNER.len()],
                theme::brand_cyan(),
                theme::text(),
            ),
            PhaseState::Pending => ("○", theme::dim(), theme::dim()),
        };
        let secs = match state {
            PhaseState::Done if p.secs > 0.005 => format!("{:>7.1}s", p.secs),
            PhaseState::Running => {
                format!(
                    "{:>7.1}s",
                    p.started.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0)
                )
            }
            _ => "        ".into(),
        };
        lines.push(Line::from(vec![
            Span::styled(format!(" {glyph} "), gstyle),
            Span::styled(format!("{:<18}", p.name), label_style),
            Span::styled(secs, theme::text2()),
        ]));
    }
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_weight_load(f: &mut Frame, app: &App, area: Rect) {
    let p = &app.progress;
    let title = if p.shard_name.is_empty() {
        "WEIGHT LOAD ─".to_string()
    } else {
        format!("WEIGHT LOAD ─ {} ─", middle_truncate(&p.shard_name, 34))
    };
    let block = panel(title, false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    // 2026-09-26: With no model there is no load to draw a bar for.
    if app.awaiting_model {
        f.render_widget(
            Paragraph::new(vec![
                Line::default(),
                Line::from(Span::styled(
                    "  no model loaded — open the Library (4) to choose one",
                    theme::dim(),
                )),
            ]),
            inner,
        );
        return;
    }
    let bar_w = inner.width.saturating_sub(22);

    let mut lines: Vec<Line> = vec![Line::default()];
    // 2026-09-26: OVERALL, the only gradient bar.
    let overall = p.displayed_overall();
    let mut l = vec![Span::styled(" OVERALL    ", theme::text2())];
    l.extend(gradient_bar(overall, bar_w).spans);
    l.push(Span::styled(
        format!(" {:>3.0}%", overall * 100.0),
        theme::text().add_modifier(Modifier::BOLD),
    ));
    lines.push(Line::from(l));
    if p.disk_gb > 0.0 {
        lines.push(Line::from(Span::styled(
            format!(
                "            {:.1} / {:.1} GB preflight",
                p.disk_gb * overall,
                p.disk_gb
            ),
            theme::dim(),
        )));
    }
    lines.push(Line::default());
    // 2026-09-26: SHARD, plain cyan.
    if p.shard_total > 0 {
        let frac = p.shard_target();
        let filled = (frac * bar_w as f64) as usize;
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_w as usize - filled);
        lines.push(Line::from(vec![
            Span::styled(
                format!(" SHARD {:>2}/{:<3}", p.shard, p.shard_total),
                theme::text2(),
            ),
            Span::styled(bar, theme::brand_cyan()),
        ]));
    }
    if p.layer_total > 0 {
        lines.push(Line::from(Span::styled(
            format!(
                " LAYERS {:>2}/{:<3} GPU used {:.1} · free {:.1} GB",
                p.layer, p.layer_total, p.gpu_used_gb, p.gpu_free_gb
            ),
            theme::text2(),
        )));
    }
    lines.push(Line::default());
    if let Some((rate, eta)) = p.rate_eta() {
        // 2026-09-26: Once the load is over (`load_secs`), show how long it
        // took instead of the ETA, which `rate_eta` then reports as 0.
        let tail = match p.load_secs() {
            Some(s) => format!("loaded in {}:{:02}", (s as u64) / 60, (s as u64) % 60),
            None => format!("eta {}:{:02}", (eta as u64) / 60, (eta as u64) % 60),
        };
        lines.push(Line::from(Span::styled(
            format!(" {rate:.2} GB/s · {tail}"),
            theme::brand_cyan(),
        )));
    }
    f.render_widget(Paragraph::new(lines), inner);
    // 2026-09-26: MEM sparkline along the bottom of the panel interior.
    if !p.mem_history.is_empty() && inner.height >= 8 {
        let mem_chart_area = Rect {
            y: inner.y + inner.height - 1,
            height: 1,
            x: inner.x + 1,
            width: inner.width - 2,
        };
        f.render_widget(
            Sparkline::default()
                .data(&p.mem_history)
                .style(theme::brand_cyan()),
            mem_chart_area,
        );
    }
}

fn draw_ready_strip(f: &mut Frame, app: &App, area: Rect) {
    let block = panel("READY ─".into(), false);
    let model = super::live_model_name(app);
    let line = Line::from(vec![
        Span::styled(" ✓ ", theme::brand_green().add_modifier(Modifier::BOLD)),
        Span::styled(model, theme::text().add_modifier(Modifier::BOLD)),
        Span::styled(
            format!(
                " · loaded in {:.1}s · listening ",
                app.progress.ready_in_secs
            ),
            theme::text2(),
        ),
        Span::styled(
            format!("{}:{}", app.args.bind, app.args.port),
            theme::brand_green(),
        ),
    ]);
    f.render_widget(Paragraph::new(line).block(block), area);
}

fn draw_chips(f: &mut Frame, app: &App, area: Rect) {
    // 2026-09-26: The chips come from the host's argv (`ModelHost::args`)
    // before the boot argv. While `app.awaiting_model`, the field the status
    // pill also reads, `logo::badges` returns only the Library hint and the
    // port.
    let live_args = app.host.as_ref().and_then(|h| h.args());
    let chips = logo::badges(live_args.as_ref().unwrap_or(&app.args), app.awaiting_model);
    let mut spans: Vec<Span> = Vec::new();
    for b in &chips {
        let tint = match b.tint {
            logo::BadgeTint::Model => theme::brand_purple(),
            logo::BadgeTint::Quant => theme::brand_cyan(),
            logo::BadgeTint::Role => theme::brand_green(),
            logo::BadgeTint::Neutral => theme::text2(),
        };
        spans.push(Span::styled(
            "▐",
            Style::default().fg(theme::BG_RAISED.color()),
        ));
        // 2026-09-26: First word in the badge's tint, the rest `text2`.
        let mut words = b.text.splitn(2, ' ');
        let first = words.next().unwrap_or_default().to_string();
        let rest = words.next().map(|r| format!(" {r}")).unwrap_or_default();
        spans.push(Span::styled(first, tint.bg(theme::BG_RAISED.color())));
        spans.push(Span::styled(
            rest,
            theme::text2().bg(theme::BG_RAISED.color()),
        ));
        spans.push(Span::styled(
            "▌",
            Style::default().fg(theme::BG_RAISED.color()),
        ));
        spans.push(Span::raw(" "));
    }
    f.render_widget(Paragraph::new(Line::from(spans)).wrapped(), area);
}

fn draw_logs(f: &mut Frame, app: &App, area: Rect) {
    let follow = app.log_scroll.is_none();
    let right = if follow {
        "⏵ follow".to_string()
    } else {
        format!("⏸ {}↑", app.log_scroll.unwrap_or(0))
    };
    let filter_part = if app.log_filter_editing {
        format!(" filter: {}▏", app.log_filter)
    } else if !app.log_filter.is_empty() {
        format!(" filter: {}", app.log_filter)
    } else {
        String::new()
    };
    let block = panel(format!("LOGS ── {right}{filter_part} ─"), follow);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // 2026-09-26: Fetch past the current offset (512 lines when scrolled, 64
    // when following), so the ceiling published below stays ahead of the
    // scroll without copying the whole `log_ring::CAP`-line ring each frame.
    let headroom = if app.log_scroll.is_some() { 512 } else { 64 };
    let want = inner.height as usize + app.log_scroll.unwrap_or(0) + headroom;
    let mut lines: Vec<Line> = log_ring::tail(want)
        .into_iter()
        .filter(|l| {
            app.log_filter.is_empty()
                || l.message
                    .to_lowercase()
                    .contains(&app.log_filter.to_lowercase())
                || l.target.contains(&app.log_filter)
        })
        .map(|l| {
            let ts = chrono_lite(l.at);
            let target_short: String = l
                .target
                .split("::")
                .next()
                .unwrap_or("")
                .chars()
                .take(14)
                .collect();
            Line::from(vec![
                Span::styled(format!("{ts} "), theme::dim()),
                Span::styled(format!("{:<5} ", l.level), theme::level_style(l.level)),
                Span::styled(format!("{target_short:<14} "), theme::dim()),
                Span::styled(l.message, theme::text()),
            ])
        })
        .collect();
    // 2026-09-26: The scroll ceiling: every fetched line that passed the
    // filter, less what fits on screen. Recorded before the scroll truncates.
    app.log_scroll_max
        .set(lines.len().saturating_sub(inner.height as usize));
    // 2026-09-26: Scrolled up: drop that many lines from the end.
    if let Some(up) = app.log_scroll {
        let keep = lines.len().saturating_sub(up);
        lines.truncate(keep);
    }
    // 2026-09-26: Wrap before choosing what fits: the skip below counts
    // rows, so the newest lines stay at the bottom. A `Paragraph` `Wrap` would
    // not: its input would be counted in entries, not rows.
    let width = inner.width as usize;
    let rows: Vec<Line> = lines
        .into_iter()
        .flat_map(|line| wrap_line(line, width))
        .collect();
    let visible = rows.len().saturating_sub(inner.height as usize);
    let shown: Vec<Line> = rows.into_iter().skip(visible).collect();
    f.render_widget(Paragraph::new(shown), inner);
}

/// 2026-09-26: Break one composed log line into as many rows as it needs.
/// Only the message (the last span) is wrapped; continuation rows are indented
/// by the width of the three prefix spans (timestamp, level, target), which
/// keep their styles on the first row.
fn wrap_line(line: Line<'static>, width: usize) -> Vec<Line<'static>> {
    if width == 0 {
        return vec![line];
    }
    let prefix: usize = line
        .spans
        .iter()
        .take(3)
        .map(|s| s.content.chars().count())
        .sum();
    let msg = line
        .spans
        .last()
        .map(|s| s.content.to_string())
        .unwrap_or_default();
    if prefix + msg.chars().count() <= width {
        return vec![line];
    }
    let style = line.spans.last().map(|s| s.style).unwrap_or_default();
    let avail = width.saturating_sub(prefix).max(8);
    let mut chunks = super::wrap(&msg, avail, style);
    let mut out = Vec::with_capacity(chunks.len());
    let head = chunks.remove(0);
    let mut first: Vec<Span<'static>> = line.spans.iter().take(3).cloned().collect();
    first.extend(head.spans);
    out.push(Line::from(first));
    for c in chunks {
        let mut row = vec![Span::raw(" ".repeat(prefix))];
        row.extend(c.spans);
        out.push(Line::from(row));
    }
    out
}

fn middle_truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let half = (max.saturating_sub(1)) / 2;
    let start: String = s.chars().take(half).collect();
    let end: String = s
        .chars()
        .rev()
        .take(half)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{start}…{end}")
}

/// 2026-09-26: `t` as UTC `hh:mm:ss`, computed from the Unix epoch.
fn chrono_lite(t: std::time::SystemTime) -> String {
    let secs = t
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    format!("{h:02}:{m:02}:{s:02}")
}

trait Wrapped {
    fn wrapped(self) -> Self;
}
impl Wrapped for Paragraph<'_> {
    fn wrapped(self) -> Self {
        self.wrap(ratatui::widgets::Wrap { trim: false })
    }
}

#[cfg(test)]
#[path = "main_tab_tests.rs"]
mod tests;
