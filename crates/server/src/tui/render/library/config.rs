// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The config form: one recipe's settings, editable before launch,
//! over the command line it would run.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::{panel, wrap};
use crate::tui::app::App;
use crate::tui::lib_fields;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let Some(recipe) = app.lib.config_recipe() else {
        f.render_widget(panel("SETTINGS ─".into(), true), area);
        return;
    };
    let edited = app.lib.overrides.len() + app.lib.removed.len();
    let title = if edited == 0 {
        format!("{} ─ SETTINGS ─", recipe.id.to_uppercase())
    } else {
        format!(
            "{} ─ SETTINGS ─ {edited} changed ─",
            recipe.id.to_uppercase()
        )
    };
    let block = panel(title, true);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let width = inner.width.saturating_sub(4) as usize;

    // 2026-09-26: Head (model and provenance), body (the rows) and tail (the
    // command). Head and tail are always drawn; only the body scrolls.
    let mut head: Vec<Line> = Vec::new();
    head.push(Line::from(vec![
        Span::styled("  model  ", theme::dim()),
        Span::styled(recipe.model.clone(), theme::text()),
    ]));
    // 2026-09-26: A starting point stays marked on the screen `s` launches
    // from. `theme::warn` is bold under `NO_COLOR`, and the sentence says it
    // in words.
    if let Some(provenance) = &recipe.starting_point {
        head.push(Line::from(Span::styled(
            format!("  starting point — {provenance}; unverified on this model"),
            theme::warn(),
        )));
    }
    // 2026-09-26: Borrowed values get their own line, beside the
    // starting-point line, not instead of it.
    if let Some(borrowed) = &app.lib.borrowed {
        head.push(Line::from(Span::styled(
            format!("  borrowed — values from {borrowed}; not a measurement for this model"),
            theme::warn(),
        )));
    }
    head.push(Line::from(""));

    // 2026-09-26: The row region. `anchor_end` is the line index just past
    // the selected row and its attached error.
    let mut body: Vec<Line> = Vec::new();
    let mut anchor_end = 0usize;
    for (i, row) in app.lib.config_rows().into_iter().enumerate() {
        let selected = i == app.lib.row;
        let editing = selected && app.lib.editing && app.lib.pending_add.is_none();
        let marker = if selected { "▌" } else { " " };
        // 2026-09-26: Row state is a gutter glyph, not colour alone: `✗`
        // removed, `+` added, `•` changed.
        let (change_mark, mark_style) = if row.removed {
            ("✗", theme::dim())
        } else if row.added {
            ("+", theme::brand_green())
        } else if row.changed {
            ("•", theme::brand_green())
        } else {
            (" ", theme::dim())
        };
        let value_style = if editing {
            theme::brand_cyan().add_modifier(Modifier::BOLD)
        } else if row.removed {
            theme::dim().add_modifier(Modifier::DIM)
        } else if row.changed {
            theme::brand_green()
        } else {
            theme::text()
        };
        let shown = if editing {
            format!("{}▏", app.lib.edit_buffer)
        } else if row.removed {
            // 2026-09-26: A removed flag is not passed, so the value column
            // shows the flag's default from `lib_fields::spec_for_key`, if any.
            match lib_fields::spec_for_key(&row.key).and_then(|s| s.default.clone()) {
                Some(d) => format!("removed — server default {d}"),
                None => "removed — flag not passed".to_string(),
            }
        } else {
            row.value.clone()
        };
        let key_style = if row.removed {
            theme::dim().add_modifier(Modifier::DIM | Modifier::CROSSED_OUT)
        } else {
            theme::text2()
        };
        let mut line = Line::from(vec![
            Span::styled(marker, theme::brand_purple()),
            Span::styled(change_mark, mark_style),
            Span::styled(format!(" {:<26}", row.key), key_style),
            Span::styled(shown, value_style),
        ]);
        if selected {
            line = line.style(theme::selected());
        }
        body.push(line);

        // 2026-09-26: The error is drawn under the selected row.
        if let Some(err) = app.lib.error.as_ref().filter(|_| selected && !editing) {
            body.extend(wrap(&format!("  {err}"), width, theme::error()));
        }
        if selected {
            anchor_end = body.len();
        }
    }
    // 2026-09-26: A setting being added exists only in `pending_add`: a
    // synthetic row at the bottom, drawn like an edited row, that
    // `cancel_edit` drops.
    if let (Some(key), true) = (&app.lib.pending_add, app.lib.editing) {
        body.push(
            Line::from(vec![
                Span::styled("▌", theme::brand_purple()),
                Span::styled("+", theme::brand_green()),
                Span::styled(format!(" {key:<26}"), theme::text2()),
                Span::styled(
                    format!("{}▏", app.lib.edit_buffer),
                    theme::brand_cyan().add_modifier(Modifier::BOLD),
                ),
            ])
            .style(theme::selected()),
        );
        anchor_end = body.len();
    }

    let mut tail: Vec<Line> = vec![Line::from("")];
    match app.lib.preview_argv() {
        Some(argv) => {
            tail.push(Line::from(Span::styled(" COMMAND", theme::dim())));
            // 2026-09-26: `preview_argv` is built from the live overrides and
            // removals.
            let rendered = argv.iter().skip(1).cloned().collect::<Vec<_>>().join(" ");
            tail.extend(wrap(&format!("met {rendered}"), width, theme::text2()));
        }
        None => tail.push(Line::from(Span::styled(
            " this recipe cannot be launched from here",
            theme::warn(),
        ))),
    }

    // 2026-09-26: The body scrolls just far enough that `anchor_end` is on
    // screen.
    let body_h = (inner.height as usize)
        .saturating_sub(head.len() + tail.len())
        .max(1);
    let off = anchor_end.saturating_sub(body_h);
    let mut lines = head;
    lines.extend(body.into_iter().skip(off).take(body_h));
    lines.extend(tail);
    f.render_widget(Paragraph::new(lines), inner);
}
