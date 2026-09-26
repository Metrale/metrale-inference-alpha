// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Library's model list (recipes joined with local weights),
//! beside its detail pane (`list_detail.rs`).
//!
//! Owner: server tui.
//! Invariants:
//! - `draw_list` publishes the search field's rect to `App::lib_search_click`
//!   whenever the list pane has an inner row.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::{panel, wrap};
use crate::tui::app::App;
use crate::tui::data::catalogue::Entry;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
        .split(area);
    draw_list(f, app, cols[0]);
    super::list_detail::draw_detail(f, app, cols[1]);
}

/// 2026-09-26: The row's badges, in order: the weights state as a word, the
/// primary recipe if any (`recipe` or `vllm`), `optimized` when a kernel target
/// resolves for the local weights, and `update` for a confirmed stale revision.
/// Recipe and optimized are independent facts.
///
/// The weights state comes first, so clipping drops the subtitle before it.
/// Under `NO_COLOR` the four states are four words, and only "not downloaded"
/// is plain dim text rather than a reversed block.
fn badges(app: &App, entry: &Entry) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    // 2026-09-26: Static while downloading: the row's dot already pulses and
    // the third line carries the bar.
    let (label, style) = if app.download.is_downloading(&entry.model) {
        (
            " downloading ",
            theme::brand_cyan().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        )
    } else if entry.has_weights() {
        (
            " on disk ",
            theme::brand_green().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        )
    } else if entry.local.is_some() {
        // 2026-09-26: A local entry without complete weights.
        (
            " partial ",
            theme::warn().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        )
    } else {
        (" not downloaded ", theme::dim())
    };
    out.push(Span::styled(label, style));
    out.push(Span::raw(" "));
    if let Some(r) = entry.primary() {
        let (label, style) = if r.is_metrale() {
            (" recipe ", theme::brand_purple())
        } else {
            // 2026-09-26: Listed, but it cannot be launched from here.
            (" vllm ", theme::dim())
        };
        out.push(Span::styled(
            label,
            style.add_modifier(Modifier::REVERSED | Modifier::BOLD),
        ));
        out.push(Span::raw(" "));
    }
    if entry.optimized() {
        out.push(Span::styled(
            " optimized ",
            theme::brand_cyan().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        ));
        out.push(Span::raw(" "));
    }
    // 2026-09-26: Last, so it is the first thing clipped, and only for
    // `Freshness::Stale`; an unreachable Hub is `Unknown` and draws nothing.
    match app.download.freshness.get(&entry.model) {
        Some(f) if f.is_stale() => out.push(Span::styled(
            " update ",
            theme::warn().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        )),
        _ if app.download.checking.as_deref() == Some(entry.model.as_str()) => {
            // 2026-09-26: A skeleton where the badge will go while the check
            // runs.
            out.push(Span::styled(" ░░░░░░ ", theme::dim()));
        }
        _ => {}
    }
    out
}

/// 2026-09-26: The search field: one row at the top of the list, drawn empty,
/// set or editing, so the filter `/` opens is visible before it is used.
///
/// Publishes the rect it drew through `App::lib_search_click`, which
/// `events::on_mouse` tests, so the hit-tester has no copy of this layout.
/// While editing, the row is `theme::selected` (reversed under `NO_COLOR`).
fn draw_search_field(f: &mut Frame, app: &App, row: Rect, total: usize, shown: usize) {
    app.lib_search_click.set(Some(row));
    let mut spans = vec![Span::styled(" ⌕ ", theme::brand_cyan())];
    if app.lib.filter_editing {
        spans.push(Span::styled(app.lib.filter.clone(), theme::text()));
        spans.push(Span::styled("▏", theme::brand_cyan()));
    } else if !app.lib.filter.is_empty() {
        spans.push(Span::styled(app.lib.filter.clone(), theme::text()));
        spans.push(Span::styled(
            format!("  — {shown} of {total} · / edits"),
            theme::dim(),
        ));
    } else {
        spans.push(Span::styled("search models — / or click", theme::dim()));
    }
    let mut line = Line::from(spans);
    if app.lib.filter_editing {
        line = line.style(theme::selected());
    }
    f.render_widget(Paragraph::new(line), row);
}

fn draw_list(f: &mut Frame, app: &App, area: Rect) {
    let rows = app.lib.visible();
    // 2026-09-26: The recipe index's fetch state is in the title.
    let status = if app.lib.fetching {
        format!(
            " {} fetching ─",
            theme::SPINNER[(app.tick as usize / 2) % theme::SPINNER.len()]
        )
    } else {
        format!(" recipes {} ─", app.lib.index.status_text())
    };
    let block = panel(format!("MODELS ─ {} ─{status} ─", rows.len()), true);
    let mut inner = block.inner(area);
    f.render_widget(block, area);

    // 2026-09-26: The search field owns the first inner row.
    if inner.height >= 1 {
        let field = Rect { height: 1, ..inner };
        draw_search_field(f, app, field, app.lib.rows.len(), rows.len());
        inner.y += 1;
        inner.height -= 1;
    }

    // 2026-09-26: The cause of an offline recipe index, wrapped above the
    // rows.
    let mut header: Vec<Line> = Vec::new();
    if let Some(detail) = app.lib.index.offline_detail() {
        header.extend(wrap(
            &detail,
            inner.width.saturating_sub(2) as usize,
            theme::warn(),
        ));
        header.push(Line::from(""));
    }

    if rows.is_empty() {
        let hint = if app.lib.filter.is_empty() {
            "no models or recipes yet — press r to fetch recipes"
        } else {
            "nothing matches this search"
        };
        header.push(Line::from(Span::styled(format!(" {hint}"), theme::dim())));
        f.render_widget(Paragraph::new(header), inner);
        return;
    }

    // 2026-09-26: Three lines per row; keep the selection on screen.
    let per_row = 3usize;
    let visible = (inner.height as usize / per_row).max(1);
    let first = app.lib.selected.saturating_sub(visible.saturating_sub(1));

    let mut lines: Vec<Line> = header;
    for (i, entry) in rows.iter().enumerate().skip(first).take(visible) {
        let selected = i == app.lib.selected;
        let bar = if selected {
            Span::styled("▌", theme::brand_purple())
        } else {
            Span::raw(" ")
        };
        // 2026-09-26: `↓` downloading, `✓` complete weights, `◐` partial,
        // `·` none.
        let mark = if app.download.is_downloading(&entry.model) {
            Span::styled("↓ ", theme::brand_cyan())
        } else if entry.has_weights() {
            Span::styled("✓ ", theme::brand_green())
        } else if entry.local.is_some() {
            Span::styled("◐ ", theme::warn())
        } else {
            Span::styled("· ", theme::dim())
        };
        let name_style = if selected {
            theme::text().add_modifier(Modifier::BOLD)
        } else {
            theme::text()
        };
        let mut head_spans = vec![bar, mark, Span::styled(entry.model.clone(), name_style)];
        // 2026-09-26: The downloading row gets a dot after the name,
        // alternating bold and dim on `(tick / 4) % 2`, the header chip's
        // cadence.
        if let Some(job) = app.download.job.as_ref().filter(|j| j.repo == entry.model) {
            // 2026-09-26: Cyan while moving, like the header chip. While
            // stopping it is steady warn colour; under `NO_COLOR` the halted
            // pulse is the only signal.
            let glow = if job.cancelling {
                theme::warn().add_modifier(Modifier::DIM)
            } else if (app.tick / 4).is_multiple_of(2) {
                theme::brand_cyan().add_modifier(Modifier::BOLD)
            } else {
                theme::brand_cyan().add_modifier(Modifier::DIM)
            };
            head_spans.push(Span::styled(" ●", glow));
        }
        let mut head = Line::from(head_spans);
        if selected {
            head = head.style(theme::selected());
        }
        lines.push(head);

        let mut second = vec![Span::raw("   ")];
        second.extend(badges(app, entry));
        let subtitle = entry.subtitle();
        if !subtitle.is_empty() {
            second.push(Span::styled(format!(" {subtitle}"), theme::dim()));
        }
        lines.push(Line::from(second));
        // 2026-09-26: Line three is the model's size and recipe count, or its
        // progress bar while it downloads; `per_row` stays 3 either way.
        lines.push(match progress_line(app, &entry.model, inner.width) {
            Some(l) => l,
            None => Line::from(vec![
                Span::raw("   "),
                Span::styled(entry.size_text(), theme::text2()),
                Span::styled(
                    match entry.recipes.len() {
                        // 2026-09-26: Enter opens starting points
                        // (`lib_start`).
                        0 => "  ·  no recipe — ⏎ starting points".to_string(),
                        1 => format!("  ·  {}", entry.recipes[0].id),
                        n => format!("  ·  {n} recipes"),
                    },
                    theme::dim(),
                ),
            ]),
        });
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// 2026-09-26: The progress line for `model`, if it is the download job's
/// repo. As the pane narrows, the file counter goes below 88 columns, the rate
/// below 70 and the byte pair below 60; the spinner, bar and percentage stay.
fn progress_line(app: &App, model: &str, width: u16) -> Option<Line<'static>> {
    let job = app.download.job.as_ref().filter(|j| j.repo == model)?;
    let mut spans = vec![Span::raw("   ")];

    // 2026-09-26: The spinner is always drawn, so the line moves while the
    // bar and percentage cannot yet.
    let phase = (app.tick as usize / 2) % theme::SPINNER.len();
    spans.push(Span::styled(theme::SPINNER[phase], theme::brand_cyan()));
    spans.push(Span::raw(" "));

    match job.fraction() {
        Some(f) => {
            const CELLS: usize = 12;
            // 2026-09-26: At least one cell once any bytes have moved.
            let exact = (f * CELLS as f64).round() as usize;
            let filled = if job.done > 0 { exact.max(1) } else { exact };
            spans.push(Span::styled(
                "▓".repeat(filled.min(CELLS)),
                theme::brand_cyan(),
            ));
            spans.push(Span::styled(
                "░".repeat(CELLS.saturating_sub(filled)),
                theme::dim(),
            ));
            // 2026-09-26: `format::percent`, which the header chip uses too;
            // the width here only aligns the column.
            spans.push(Span::styled(
                format!("  {:>5}", crate::tui::format::percent(f)),
                theme::text(),
            ));
        }
        // 2026-09-26: No sizes known, so no fraction to draw.
        None => spans.push(Span::styled("fetching", theme::text())),
    }

    if job.cancelling {
        spans.push(Span::styled("  stopping…", theme::warn()));
        return Some(Line::from(spans));
    }
    if width >= 60 && job.total > 0 {
        spans.push(Span::styled(
            format!("  {} / {}", gb(job.done), gb(job.total)),
            theme::text2(),
        ));
    }
    if width >= 70 && job.rate_bps > 0.0 {
        spans.push(Span::styled(
            format!("  {}", crate::tui::format::rate(job.rate_bps)),
            theme::dim(),
        ));
    }
    if width >= 88
        && let Some((i, of, _)) = &job.file
    {
        spans.push(Span::styled(format!("  file {i}/{of}"), theme::dim()));
    }
    Some(Line::from(spans))
}

/// 2026-09-26: `format::bytes`, the formatter `Entry::size_text` also uses
/// (through `data::library::human_size`), so the size does not change when the
/// download finishes.
fn gb(bytes: u64) -> String {
    crate::tui::format::bytes(bytes)
}
