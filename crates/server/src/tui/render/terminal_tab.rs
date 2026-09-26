// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Terminal section: the Ops command line and the Chat view,
//! switched by `TermSub`. Ops draws a purple `❯` prompt with ghost-text
//! completion; Chat draws the transcript through `chat_lines::message_lines`.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use super::{chat_lines, panel};
use crate::tui::app::{App, Focus, TermSub};
use crate::tui::{commands, theme};

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(4)])
        .split(area);
    draw_tabs(f, app, rows[0]);
    match app.term_sub {
        TermSub::Ops => draw_ops(f, app, rows[1]),
        TermSub::Chat => draw_chat(f, app, rows[1]),
    }
}

fn draw_tabs(f: &mut Frame, app: &App, area: Rect) {
    let tab = |name: &str, active: bool| {
        if active {
            Span::styled(
                format!(" {name} "),
                theme::brand_cyan().add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )
        } else {
            Span::styled(format!(" {name} "), theme::text2())
        }
    };
    let line = Line::from(vec![
        tab("Ops", app.term_sub == TermSub::Ops),
        Span::styled("─", theme::dim()),
        tab("Chat", app.term_sub == TermSub::Chat),
        Span::styled("   (6 toggles)", theme::dim()),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn draw_ops(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(area);
    // 2026-09-26: The scroll offset counts up from the newest line. The
    // renderer publishes its ceiling in `scroll_max`, and the title shows the
    // offset in the same words as the Chat pane.
    let total = app.ops.output.len();
    let visible = rows[0].height.saturating_sub(2) as usize;
    let max = total.saturating_sub(visible);
    app.ops.scroll_max.set(max);
    let up = app.ops.scroll_up.min(max);
    let out_block = panel(
        if up > 0 {
            format!("OPS ─ {total} lines ─ ↑{up} ─ End follows ─")
        } else {
            format!("OPS ─ {total} lines ─")
        },
        false,
    );
    let inner = out_block.inner(rows[0]);
    f.render_widget(out_block, rows[0]);
    let end = total - up;
    let lines: Vec<Line> = app
        .ops
        .output
        .iter()
        .take(end)
        .skip(end.saturating_sub(visible))
        .map(|l| {
            if let Some(cmd) = l.strip_prefix("❯ ") {
                Line::from(vec![
                    Span::styled("❯ ", theme::brand_purple().add_modifier(Modifier::BOLD)),
                    Span::styled(cmd.to_string(), theme::text().add_modifier(Modifier::BOLD)),
                ])
            } else {
                Line::from(Span::styled(l.clone(), theme::text2()))
            }
        })
        .collect();
    f.render_widget(Paragraph::new(lines), inner);
    let focused = app.focus == Focus::Input;
    let in_block = panel("─".into(), focused);
    let in_inner = in_block.inner(rows[1]);
    f.render_widget(in_block, rows[1]);
    let mut spans = vec![
        Span::styled("❯ ", theme::brand_purple().add_modifier(Modifier::BOLD)),
        Span::styled(app.ops.input.clone(), theme::text()),
    ];
    if focused {
        if let Some(ghost) = commands::complete(&app.ops.input) {
            let rest = &ghost[app.ops.input.len()..];
            spans.push(Span::styled(rest.to_string(), theme::dim()));
            spans.push(Span::styled("  ⇥ accept", theme::dim()));
        } else {
            spans.push(Span::styled("▏", theme::brand_cyan()));
        }
    } else {
        spans.push(Span::styled("  (Enter to focus · /help)", theme::dim()));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), in_inner);
}

/// 2026-09-26: The input pane's key hints. They differ by focus because a bare
/// letter is typed text while the input box has focus.
fn chat_hints(focused: bool, wide: bool) -> String {
    match (focused, wide) {
        (true, true) => {
            "─ ⏎ send · \\+⏎ newline · Ctrl+T thinking · Alt+T reasoning · Ctrl+N new · Esc cancel ─"
                .into()
        }
        (true, false) => "─ ⏎ send · Esc cancel ─".into(),
        (false, true) => "─ ⏎ focus · t thinking · T reasoning · Ctrl+N new chat ─".into(),
        (false, false) => "─ ⏎ focus ─".into(),
    }
}

fn draw_chat(f: &mut Frame, app: &App, area: Rect) {
    let input_h = (app.chat.input.lines().count().clamp(1, 5) + 2) as u16;
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(input_h)])
        .split(area);
    // 2026-09-26: The thinking request is shown in the title. In `Auto` the
    // wide chip adds what the last reply was observed to do, never a guess.
    let wide = rows[0].width >= 76;
    let block = panel(
        format!(
            "CHAT ─ {} ─ {} ─{}",
            super::live_model_name(app),
            app.chat.think_req.chip(app.chat.observed_thinking, wide),
            match (app.chat.streaming, app.chat.scroll) {
                (_, Some(n)) => format!(" ↑{n} ─ End follows ─"),
                (true, None) => " streaming ─".to_string(),
                (false, None) => String::new(),
            }
        ),
        false,
    );
    let inner = block.inner(rows[0]);
    f.render_widget(block, rows[0]);
    // 2026-09-26: Body width is the pane minus the 2-column gutter and the
    // 1-column rule.
    let body_w = inner.width.saturating_sub(3) as usize;
    let tip = app.chat.transcript.len().saturating_sub(1);
    let mut lines: Vec<Line> = Vec::new();
    for (i, m) in app.chat.transcript.iter().enumerate() {
        let is_tip = app.chat.streaming && i == tip;
        lines.extend(chat_lines::message_lines(
            m,
            is_tip,
            app.chat.think_view,
            app.tick,
            body_w,
        ));
        lines.push(Line::default());
    }
    // 2026-09-26: `lines` is already in display rows (`wrap_rows`), so the
    // tail slice is exact and the Paragraph needs no `Wrap`.
    let h = inner.height as usize;
    let max_off = lines.len().saturating_sub(h);
    // 2026-09-26: `max_off` is the scroll ceiling: scrolled back that far, the
    // oldest line is at the top.
    app.chat_scroll_max.set(max_off);
    let off = match app.chat.scroll {
        None => max_off,
        Some(n) => max_off.saturating_sub(n),
    };
    let shown: Vec<Line> = lines.into_iter().skip(off).take(h).collect();
    f.render_widget(Paragraph::new(shown), inner);
    let focused = app.focus == Focus::Input;
    let in_block = panel(chat_hints(focused, wide), focused);
    let in_inner = in_block.inner(rows[1]);
    f.render_widget(in_block, rows[1]);
    let mut text = app.chat.input.clone();
    if focused {
        text.push('▏');
    }
    f.render_widget(
        Paragraph::new(text)
            .style(theme::text())
            .wrap(Wrap { trim: false }),
        in_inner,
    );
}

#[cfg(test)]
#[path = "terminal_tab_tests.rs"]
mod tests;
