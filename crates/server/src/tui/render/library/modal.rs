// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The config form's pickers, one per `ConfigModal`: a flag's
//! options, add-a-setting (with a help panel), the borrow donors, and the
//! borrow preview.
//!
//! Owner: server tui.
//! Invariants:
//! - Every box renders `Clear` over its rect before drawing into it.
//! - A box too small for its border (and, for the preview, its two header
//!   rows) is not drawn.
//!
//! The lists scroll with a `▌` cursor bar, and the current value carries a `✓`,
//! so both survive `NO_COLOR`.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};

use crate::tui::app::App;
use crate::tui::lib_modal::ConfigModal;
use crate::tui::theme;

pub(super) fn draw(f: &mut Frame, app: &App, area: Rect) {
    let Some(modal) = &app.lib.modal else {
        return;
    };
    match modal {
        ConfigModal::Options {
            key,
            options,
            selected,
        } => {
            // 2026-09-26: The row's current value in the form gets the ✓; the
            // cursor is separate.
            let current = app
                .lib
                .config_rows()
                .into_iter()
                .find(|r| r.key == *key)
                .map(|r| r.value);
            let rows: Vec<(String, bool)> = options
                .iter()
                .map(|o| (o.clone(), current.as_deref() == Some(o)))
                .collect();
            let flag = key.replace('_', "-").to_uppercase();
            draw_list(f, area, &format!("─ {flag} ─"), &rows, *selected, 40);
        }
        ConfigModal::Add {
            fields,
            selected,
            help_scroll,
        } => {
            let rows: Vec<(String, bool)> = fields
                .iter()
                .map(|s| {
                    // 2026-09-26: Key, then the first line of its clap help.
                    (format!("{:<26} {}", s.key, s.help), false)
                })
                .collect();
            draw_add(
                f,
                area,
                &rows,
                *selected,
                fields.get(*selected),
                *help_scroll,
            );
        }
        ConfigModal::Borrow { donors, selected } => {
            // 2026-09-26: Each row names the model the donor recipe is for.
            let rows: Vec<(String, bool)> = donors
                .iter()
                .map(|d| (format!("{:<40} measured on {}", d.id, d.model), false))
                .collect();
            // 2026-09-26: Wider than the other pickers: a row holds a recipe
            // id and a model id.
            draw_list(f, area, "─ BORROW PARAMETERS FROM ─", &rows, *selected, 100);
        }
        ConfigModal::Preview {
            donors,
            donor,
            changes,
            scroll,
        } => {
            if let Some(d) = donors.get(*donor) {
                draw_preview(f, area, d, changes, *scroll);
            }
        }
    }
}

/// 2026-09-26: The borrow preview: the rows applying the donor would change,
/// current value beside incoming. Only Enter here applies the borrow
/// (`LibState::modal_pick` → `apply_borrow`).
fn draw_preview(
    f: &mut Frame,
    area: Rect,
    donor: &crate::recipe::Recipe,
    changes: &[crate::tui::lib_borrow::BorrowChange],
    scroll: usize,
) {
    let w = 76.min(area.width.saturating_sub(4));
    // 2026-09-26: 2 border rows + 2 header rows + the change rows; capped to
    // the area, and the rest scrolls.
    let h = ((changes.len() + 4) as u16).min(area.height.saturating_sub(2));
    if w < 10 || h < 5 {
        return;
    }
    let visible = (h - 4) as usize;
    let top = scroll.min(changes.len().saturating_sub(visible));
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);

    let inner_w = w.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::with_capacity(visible + 2);
    // 2026-09-26: The first header line says the values were set for the
    // donor's model; `theme::warn` is bold under `NO_COLOR`.
    lines.push(Line::from(Span::styled(
        clip(
            &format!("copied from {} — not measured on this model", donor.model),
            inner_w,
        ),
        theme::warn(),
    )));
    lines.push(Line::from(Span::styled(
        clip(
            &format!(
                "{} changes; settings the donor does not name keep their values",
                changes.len()
            ),
            inner_w,
        ),
        theme::dim(),
    )));
    for change in changes.iter().skip(top).take(visible) {
        // 2026-09-26: `from → to`; the arrow carries the direction without
        // colour.
        let budget = inner_w.saturating_sub(28 + change.to.len() + 3);
        lines.push(Line::from(vec![
            Span::styled(format!(" {:<26} ", clip(&change.key, 26)), theme::text2()),
            Span::styled(clip(&change.from, budget.max(4)), theme::dim()),
            Span::styled(" → ", theme::dim()),
            Span::styled(change.to.clone(), theme::brand_green()),
        ]));
    }

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::border(true))
        .title(Span::styled(
            format!("─ BORROW: {} ─", donor.id.to_uppercase()),
            theme::title(true),
        ))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    let block = if changes.len() > visible {
        block.title_bottom(Span::styled(
            format!("─ {}/{} ─", top + 1, changes.len()),
            theme::dim(),
        ))
    } else {
        block
    };
    f.render_widget(Paragraph::new(lines).block(block), modal);
}

/// 2026-09-26: The add-picker: the flag list and, when the area is wide
/// enough, a side panel with the highlighted flag's full help, wrapped and
/// scrollable. The list row shows only the help's first line.
fn draw_add(
    f: &mut Frame,
    area: Rect,
    rows: &[(String, bool)],
    selected: usize,
    spec: Option<&&'static crate::tui::lib_fields::FieldSpec>,
    help_scroll: usize,
) {
    // 2026-09-26: Panel text width + 2 border columns + 2 padding columns,
    // from `HELP_PANEL_TEXT_W`, the width `ConfigModal::scroll_help` wraps at.
    const PANEL_W: u16 = (crate::tui::lib_modal::HELP_PANEL_TEXT_W + 4) as u16;
    // 2026-09-26: With fewer than 50 columns left for the list, the panel is
    // dropped and the picker is the single centred list.
    let avail = area.width.saturating_sub(4);
    let (Some(spec), true) = (spec, avail >= 50 + PANEL_W) else {
        draw_list(f, area, "─ ADD A SETTING ─", rows, selected, 76);
        return;
    };
    let list_w = (avail - PANEL_W).min(76);
    let w = list_w + PANEL_W;
    let h = ((rows.len() + 2) as u16).min(area.height.saturating_sub(2));
    if h < 3 {
        return;
    }
    let x = area.x + (area.width - w) / 2;
    let y = area.y + (area.height - h) / 2;
    draw_list_at(
        f,
        Rect {
            x,
            y,
            width: list_w,
            height: h,
        },
        "─ ADD A SETTING ─",
        rows,
        selected,
    );
    draw_help_panel(
        f,
        Rect {
            x: x + list_w,
            y,
            width: PANEL_W,
            height: h,
        },
        spec,
        help_scroll,
    );
}

/// 2026-09-26: The full clap help for one flag, wrapped to the panel and
/// scrolled to `help_scroll`. The key handler clamps it to the last line; it is
/// clamped again here to the panel's height.
fn draw_help_panel(
    f: &mut Frame,
    panel: Rect,
    spec: &crate::tui::lib_fields::FieldSpec,
    help_scroll: usize,
) {
    let wrapped =
        crate::tui::format::wrap_help(&spec.help_full, crate::tui::lib_modal::HELP_PANEL_TEXT_W);
    let visible = panel.height.saturating_sub(2) as usize;
    let top = help_scroll.min(wrapped.len().saturating_sub(visible));
    let mut lines: Vec<Line> = wrapped
        .iter()
        .skip(top)
        .take(visible)
        .map(|l| Line::from(Span::styled(format!(" {l}"), theme::text2())))
        .collect();
    if lines.is_empty() {
        // 2026-09-26: Say so rather than draw an empty box.
        lines.push(Line::from(Span::styled(" no help text", theme::dim())));
    }
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::border(true))
        .title(Span::styled(
            clip(
                &format!("─ {} ─", spec.key.to_uppercase()),
                panel.width.saturating_sub(2) as usize,
            ),
            theme::title(true),
        ))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    // 2026-09-26: The `J/K` binding and position in the bottom border, only
    // when the help is longer than the panel.
    let block = if wrapped.len() > visible {
        block.title_bottom(Span::styled(
            format!("─ J/K {}/{} ─", top + 1, wrapped.len()),
            theme::dim(),
        ))
    } else {
        block
    };
    f.render_widget(Clear, panel);
    f.render_widget(Paragraph::new(lines).block(block), panel);
}

/// 2026-09-26: `s` cut to `width` chars, the last one replaced by `…` when it
/// is cut, so a clipped value does not read as the whole value.
fn clip(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.to_string();
    }
    let mut out: String = s.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// 2026-09-26: One scrolling selection list, centred in `area`. `rows` are
/// (label, is_current).
fn draw_list(
    f: &mut Frame,
    area: Rect,
    title: &str,
    rows: &[(String, bool)],
    selected: usize,
    want_w: u16,
) {
    let w = want_w.min(area.width.saturating_sub(4));
    // 2026-09-26: +2 border rows; capped to the area, and the rest scrolls.
    let h = ((rows.len() + 2) as u16).min(area.height.saturating_sub(2));
    if w < 10 || h < 3 {
        return;
    }
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    draw_list_at(f, modal, title, rows, selected);
}

/// 2026-09-26: [`draw_list`]'s body at an exact rect, for `draw_add`, which
/// places the list beside its help panel.
fn draw_list_at(f: &mut Frame, modal: Rect, title: &str, rows: &[(String, bool)], selected: usize) {
    let visible = modal.height.saturating_sub(2) as usize;
    // 2026-09-26: Scroll so the cursor stays inside the window.
    let top = selected.saturating_sub(visible.saturating_sub(1));
    f.render_widget(Clear, modal);

    let inner_w = modal.width.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::with_capacity(visible);
    for (i, (label, is_current)) in rows.iter().enumerate().skip(top).take(visible) {
        let cursor = i == selected;
        let marker = if cursor { "▌" } else { " " };
        // 2026-09-26: ✓ before the text, not a colour on it, so it shows under
        // `NO_COLOR` and under the selection style.
        let mark = if *is_current { "✓ " } else { "  " };
        let text = clip(label, inner_w.saturating_sub(3));
        let mut line = Line::from(vec![
            Span::styled(marker, theme::brand_purple()),
            Span::styled(mark, theme::brand_green().add_modifier(Modifier::BOLD)),
            Span::styled(
                text,
                if cursor {
                    theme::text()
                } else {
                    theme::text2()
                },
            ),
        ]);
        if cursor {
            line = line.style(theme::selected());
        }
        lines.push(line);
    }

    // 2026-09-26: The position, in the bottom border, only when the list is
    // clipped.
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::border(true))
        .title(Span::styled(title.to_string(), theme::title(true)))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    let block = if rows.len() > visible {
        block.title_bottom(Span::styled(
            format!("─ {}/{} ─", selected + 1, rows.len()),
            theme::dim(),
        ))
    } else {
        block
    };
    f.render_widget(Paragraph::new(lines).block(block), modal);
}

#[cfg(test)]
#[path = "modal_tests.rs"]
mod tests;
