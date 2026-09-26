// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The detail pane for the selected list row: the primary recipe's
//! summary, the on-disk facts, the download bar, and the row's key hints.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::{gradient_bar, panel, wrap};
use crate::tui::app::App;
use crate::tui::data::catalogue::Entry;
use crate::tui::theme;

pub(super) fn draw_detail(f: &mut Frame, app: &App, area: Rect) {
    let Some(entry) = app.lib.current() else {
        let block = panel("MODEL ─".into(), false);
        f.render_widget(block, area);
        return;
    };
    let block = panel(format!("{} ─", entry.model.to_uppercase()), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let width = inner.width.saturating_sub(2) as usize;

    let mut lines: Vec<Line> = Vec::new();
    match entry.primary() {
        Some(recipe) => {
            lines.extend(wrap(&recipe.description, width, theme::text2()));
            lines.push(Line::from(""));
            for (label, value) in [
                ("recipe", recipe.id.clone()),
                ("maintainer", recipe.maintainer.clone()),
                ("updated", app.lib.date_text(recipe)),
                ("quantization", recipe.quantization.clone()),
                ("kv cache", recipe.kv_dtype.clone()),
                ("container", recipe.container.clone()),
                (
                    "nodes",
                    if recipe.min_nodes > 1 {
                        format!("{} (multi-node)", recipe.min_nodes)
                    } else {
                        "1".into()
                    },
                ),
            ] {
                if value.is_empty() {
                    continue;
                }
                lines.push(kv(label, &value));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                match entry.recipes.len() {
                    1 => format!(" SETTINGS  {} editable", recipe.defaults.len()),
                    n => format!(" {n} RECIPES  ⏎ to choose"),
                },
                theme::dim(),
            )));
            // 2026-09-26: The first six defaults, then a count of the rest.
            for (key, value) in recipe.defaults.iter().take(6) {
                lines.push(kv(key, value));
            }
            if recipe.defaults.len() > 6 {
                lines.push(Line::from(Span::styled(
                    format!("   … {} more", recipe.defaults.len() - 6),
                    theme::dim(),
                )));
            }
        }
        None => {
            lines.push(Line::from(Span::styled(
                "No recipe covers this checkpoint. ⏎ offers starting points —",
                theme::text2(),
            )));
            lines.push(Line::from(Span::styled(
                "published recipes re-aimed at this model, or a blank config.",
                theme::text2(),
            )));
            lines.push(Line::from(Span::styled(
                "None of them is measured on this model; review before launch.",
                theme::warn(),
            )));
        }
    }

    if let Some(local) = &entry.local {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(" ON DISK", theme::dim())));
        lines.push(kv("size", &entry.size_text()));
        lines.push(kv("architecture", &local.model_type));
        lines.push(kv("layers", &local.layers.to_string()));
        lines.push(kv(
            "kernels",
            if local.optimized {
                "optimized"
            } else {
                "generic"
            },
        ));
    } else {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            " weights are not in the local cache",
            theme::warn(),
        )));
    }

    // 2026-09-26: This model's download in full: gradient bar, bytes, rate
    // and current file.
    if let Some(job) = app.download.job.as_ref().filter(|j| j.repo == entry.model) {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(" DOWNLOADING", theme::dim())));
        let bar_w = inner.width.saturating_sub(12).clamp(8, 36);
        let mut bar = vec![Span::raw("  ")];
        match job.fraction() {
            Some(frac) => {
                bar.extend(gradient_bar(frac, bar_w).spans);
                bar.push(Span::styled(
                    format!(" {:>3.0}%", frac * 100.0),
                    theme::text().add_modifier(Modifier::BOLD),
                ));
            }
            // 2026-09-26: No sizes known: a spinner instead of a bar.
            None => {
                let phase = (app.tick as usize / 2) % theme::SPINNER.len();
                bar.push(Span::styled(theme::SPINNER[phase], theme::brand_cyan()));
                bar.push(Span::styled(" resolving…", theme::text2()));
            }
        }
        lines.push(Line::from(bar));
        if job.total > 0 {
            let mut detail = vec![Span::styled(
                format!(
                    "  {} / {}",
                    crate::tui::format::bytes(job.done),
                    crate::tui::format::bytes(job.total)
                ),
                theme::text2(),
            )];
            // 2026-09-26: No rate while stopping.
            if job.rate_bps > 0.0 && !job.cancelling {
                detail.push(Span::styled(
                    format!("  {}", crate::tui::format::rate(job.rate_bps)),
                    theme::dim(),
                ));
            }
            lines.push(Line::from(detail));
        }
        if let Some((i, of, name)) = &job.file {
            lines.push(Line::from(Span::styled(
                format!("  file {i}/{of}  {name}"),
                theme::dim(),
            )));
        }
        if job.cancelling {
            lines.push(Line::from(Span::styled("  stopping…", theme::warn())));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        detail_footer(app, entry),
        theme::brand_cyan(),
    )));
    f.render_widget(Paragraph::new(lines), inner);
}

fn kv(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {label:<14}"), theme::dim()),
        Span::styled(value.to_string(), theme::text()),
    ])
}
/// 2026-09-26: The keys that apply to this row: `x` while it downloads, else
/// `d` worded for the row's state (update, resume or download).
fn detail_footer(app: &App, entry: &Entry) -> String {
    if app.download.is_downloading(&entry.model) {
        return " x stop the download  ·  ⏎ choose a recipe".into();
    }
    let stale = app
        .download
        .freshness
        .get(&entry.model)
        .is_some_and(|f| f.is_stale());
    match (
        entry.runnable_now(),
        entry.has_recipe(),
        entry.local.is_some(),
    ) {
        // 2026-09-26: Runnable: `d update` only for `Freshness::Stale`.
        (true, _, _) if stale => " d update  ·  ⏎ choose a recipe".into(),
        (true, _, _) => " ⏎ choose a recipe  ·  u check for updates".into(),
        (_, true, true) => " d resume the download  ·  ⏎ choose a recipe".into(),
        (_, true, false) => " d download the weights  ·  ⏎ choose a recipe".into(),
        _ => " d download the weights  ·  ⏎ pick a starting point".into(),
    }
}
