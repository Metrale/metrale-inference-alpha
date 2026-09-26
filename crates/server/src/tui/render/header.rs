// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The header: logo, status pill, mini-strip, download chip and
//! thermal alert.
//!
//! Owner: server tui.
//! Invariants:
//! - The logo wave, the status pill and the mini-strip all decide "no model"
//!   from `app.awaiting_model`.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::live_model_name;
use crate::tui::app::App;
use crate::tui::{logo, theme};

pub(crate) fn status_pill(app: &App) -> Span<'static> {
    // 2026-09-26: Three states: while `awaiting_model` (no load started) the
    // pill says NO MODEL, not LOADING.
    let (label, bg) = if app.awaiting_model {
        (" ○ NO MODEL ", theme::TEXT_DIM)
    } else if app.progress.ready {
        (" ● SERVING ", theme::GREEN)
    } else {
        (" ● LOADING ", theme::WARN)
    };
    Span::styled(
        label,
        Style::default()
            .bg(bg.color())
            .fg(theme::BG_BASE.color())
            .add_modifier(Modifier::BOLD),
    )
}

/// 2026-09-26: The download chip, shown in the header so a transfer is visible
/// from every section. `None` when no download job exists.
///
/// Fields drop by width: the rate below 96 columns, the name below 60, and
/// everything but the glyph below 48.
pub(crate) fn download_chip(app: &App, width: u16) -> Option<Vec<Span<'static>>> {
    let job = app.download.job.as_ref()?;
    // 2026-09-26: The same `(tick / 4) % 2` pulse as the Library row's dot
    // (`library/list.rs`); at the 100 ms `events::TICK` that is a 0.8 s
    // period.
    let pulse = if (app.tick / 4).is_multiple_of(2) {
        Modifier::BOLD
    } else {
        Modifier::DIM
    };
    // 2026-09-26: While stopping, the glyph is steady warn colour; under
    // `NO_COLOR` the halted pulse is the only signal.
    let (glyph_style, steady) = if job.cancelling {
        (theme::warn(), true)
    } else {
        (theme::brand_cyan(), false)
    };
    let glyph = Span::styled(
        "\u{2193} ",
        if steady {
            glyph_style
        } else {
            glyph_style.add_modifier(pulse)
        },
    );
    if width < 48 {
        return Some(vec![glyph, Span::raw(" ")]);
    }
    // 2026-09-26: With no `fraction()` (no sizes known) the chip shows bytes
    // moved, or "resolving…" before the first byte.
    let measure = match job.fraction() {
        Some(f) => crate::tui::format::percent(f),
        None if job.done > 0 => crate::tui::format::bytes(job.done),
        None => "resolving\u{2026}".to_string(),
    };
    let mut out = vec![glyph];
    if job.cancelling {
        out.push(Span::styled("stopping", theme::warn()));
        out.push(Span::raw(" "));
        return Some(out);
    }
    if width >= 60 {
        // 2026-09-26: The repo tail after the last `/`, capped at 24 chars
        // from 96 columns and at 16 below.
        let tail = job.repo.rsplit('/').next().unwrap_or(&job.repo);
        let cap = if width >= 96 { 24 } else { 16 };
        let name: String = if tail.chars().count() > cap {
            tail.chars().take(cap - 1).collect::<String>() + "\u{2026}"
        } else {
            tail.to_string()
        };
        out.push(Span::styled(format!("{name} \u{b7} "), theme::text2()));
    }
    out.push(Span::styled(measure, theme::text2()));
    if width >= 96 && job.rate_bps > 0.0 {
        out.push(Span::styled(
            format!(" \u{b7} {}", crate::tui::format::rate(job.rate_bps)),
            theme::text2(),
        ));
    }
    out.push(Span::raw(" "));
    Some(out)
}

/// 2026-09-26: Header indicator for thermal throttling. `None` for both `Ok`
/// and `Unknown`; the Stats section (`stats_thermal.rs`) tells those apart.
/// `ThermalSnapshot::alert` returns `Thrashing` ahead of `Throttling`.
fn thermal_alert_span(app: &App) -> Option<Span<'static>> {
    use crate::tui::data::thermal::ThermalAlert;
    match app.thermal.snapshot().alert() {
        ThermalAlert::Unknown | ThermalAlert::Ok => None,
        ThermalAlert::Throttling => Some(Span::styled(
            " \u{26a0} THROTTLING ",
            theme::warn().add_modifier(ratatui::style::Modifier::BOLD),
        )),
        ThermalAlert::Thrashing => Some(Span::styled(
            " \u{26a0} THERMAL THRASH ",
            theme::error().add_modifier(ratatui::style::Modifier::BOLD),
        )),
    }
}

pub(crate) fn draw_header(f: &mut Frame, app: &App, area: Rect, tall: bool) {
    // 2026-09-26: The chevron wave stops only in the pill's SERVING state:
    // `progress.ready` and not `awaiting_model`.
    let wave = if app.progress.ready && !app.awaiting_model {
        None
    } else {
        Some((app.tick / 3) as usize % 3)
    };
    let up = app.started.elapsed().as_secs();
    let uptime = fmt_uptime(up);
    // 2026-09-26: In the one-line header the chip sits left of the pill; the
    // tall header puts it on row 2.
    let chip = download_chip(app, area.width);
    let mut right_spans = Vec::new();
    if !tall && let Some(c) = chip.as_ref() {
        right_spans.extend(c.iter().cloned());
        right_spans.push(Span::raw(" "));
    }
    if let Some(alert) = thermal_alert_span(app) {
        right_spans.push(alert);
        right_spans.push(Span::raw(" "));
    }
    right_spans.push(status_pill(app));
    right_spans.push(Span::styled(format!("  {uptime} "), theme::text2()));
    let right = Line::from(right_spans);
    if tall {
        let lines = logo::three_line(wave);
        for (i, line) in lines.into_iter().enumerate() {
            let row = Rect {
                y: area.y + i as u16,
                height: 1,
                ..area
            };
            f.render_widget(Paragraph::new(line), row);
        }
        // 2026-09-26: Right cluster on row 0, `header_line` on row 1.
        f.render_widget(
            Paragraph::new(right).alignment(ratatui::layout::Alignment::Right),
            Rect {
                y: area.y,
                height: 1,
                ..area
            },
        );
        let sub = Line::from(Span::styled(header_line(app), theme::text2()));
        f.render_widget(
            Paragraph::new(sub).alignment(ratatui::layout::Alignment::Right),
            Rect {
                y: area.y + 1,
                height: 1,
                ..area
            },
        );
        // 2026-09-26: The chip takes row 2, the one right-hand row left free.
        if let Some(c) = chip {
            f.render_widget(
                Paragraph::new(Line::from(c)).alignment(ratatui::layout::Alignment::Right),
                Rect {
                    y: area.y + 2,
                    height: 1,
                    ..area
                },
            );
        }
    } else {
        f.render_widget(Paragraph::new(logo::one_line(wave)), area);
        f.render_widget(
            Paragraph::new(right).alignment(ratatui::layout::Alignment::Right),
            area,
        );
    }
}

/// 2026-09-26: The header's mini-strip: model, KV dtype and port once a load
/// has started; the way to the Library and the port while
/// `app.awaiting_model`.
pub(crate) fn header_line(app: &App) -> String {
    if app.awaiting_model {
        // 2026-09-26: The pill beside this already says NO MODEL. Library is
        // the fourth entry of `Section::ALL`.
        return format!("press 4 for Library · :{} ", app.args.port);
    }
    // 2026-09-26: The host's argv (`ModelHost::args`) before the boot argv.
    let live = app.host.as_ref().and_then(|h| h.args());
    let a = live.as_ref().unwrap_or(&app.args);
    format!(
        "{} · kv {} · :{} ",
        live_model_name(app),
        // 2026-09-26: An omitted --kv-cache-dtype is resolved against the
        // model's behaviour default at load (`serve_phases/kv_cache.rs`), so the
        // argv can only say "auto".
        a.kv_cache_dtype.as_deref().unwrap_or("auto"),
        a.port
    )
}

/// 2026-09-26: `up H:MM:SS`, or `up Nd HH:MM` from one day.
pub(super) fn fmt_uptime(secs: u64) -> String {
    let (d, h, m, s) = (secs / 86_400, secs / 3_600 % 24, secs / 60 % 60, secs % 60);
    if d > 0 {
        format!("up {d}d {h:02}:{m:02}")
    } else {
        format!("up {h}:{m:02}:{s:02}")
    }
}

#[cfg(test)]
#[path = "header_tests.rs"]
mod tests;
