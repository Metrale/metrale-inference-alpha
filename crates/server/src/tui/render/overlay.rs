// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What `render::draw` paints over a section's pane: toasts, the
//! key map modal, and the download-switch, chat-clear and quit prompts.
//!
//! Owner: server tui.
//! Invariants:
//! - Each overlay renders `Clear` over its rect before drawing into it.
//! - A toast that does not fit inside the content area is skipped, not
//!   clipped.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};

use super::wrap;
use crate::tui::app::App;
use crate::tui::theme;

/// 2026-09-26: The newest three toasts, top right of `content`, each in a
/// rounded box whose border is `theme::error` for a failure and
/// `theme::brand_green` for a success.
pub(super) fn draw_toasts(f: &mut Frame, app: &App, content: Rect) {
    let width = 56.min(content.width.saturating_sub(2));
    let inner_w = width.saturating_sub(2) as usize;
    let mut y = content.y + 1;
    for t in app.toasts.iter().rev().take(3) {
        // 2026-09-26: Errors wrap to at most 3 lines, because a
        // `DownloadError::hint` ends with the fix; successes are one ellipsised
        // line.
        let text_w = inner_w.saturating_sub(2);
        let body: Vec<Line> = if t.error {
            wrap(&t.text, text_w, theme::text())
                .into_iter()
                .take(3)
                .collect()
        } else {
            vec![Line::from(Span::styled(
                truncate_toast(&t.text, text_w),
                theme::text(),
            ))]
        };
        let height = body.len() as u16 + 2;
        let area = Rect {
            x: content.x + content.width.saturating_sub(width + 1),
            y,
            width,
            height,
        };
        if area.bottom() > content.bottom() || width < 6 {
            // 2026-09-26: No room for the whole box: skip it rather than clip
            // the border.
            continue;
        }
        y += height + 1;
        let accent = if t.error {
            theme::error()
        } else {
            theme::brand_green()
        };
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(accent.add_modifier(Modifier::BOLD))
            .style(Style::default().bg(theme::BG_RAISED.color()));
        let inner = block.inner(area);
        // 2026-09-26: `Clear` over the whole box, border cells included.
        f.render_widget(Clear, area);
        f.render_widget(block, area);
        // 2026-09-26: An outcome glyph as well as the border colour: under
        // `NO_COLOR` both accents are `Color::Reset` (`theme::C::color`), so
        // the glyph is what tells success from failure.
        let mark = if t.error {
            Span::styled("\u{2717} ", theme::error().add_modifier(Modifier::BOLD))
        } else {
            Span::styled(
                "\u{2713} ",
                theme::brand_green().add_modifier(Modifier::BOLD),
            )
        };
        let mut lines: Vec<Line> = Vec::with_capacity(body.len());
        for (n, mut l) in body.into_iter().enumerate() {
            // 2026-09-26: Continuation lines indent under the glyph.
            l.spans.insert(
                0,
                if n == 0 {
                    mark.clone()
                } else {
                    Span::raw("  ")
                },
            );
            lines.push(l);
        }
        f.render_widget(
            Paragraph::new(lines).style(Style::default().bg(theme::BG_RAISED.color())),
            inner,
        );
    }
}

/// 2026-09-26: `text` cut to `width` chars, the last one replaced by `…` when
/// it is cut.
pub(super) fn truncate_toast(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let n = text.chars().count();
    if n <= width {
        return text.to_string();
    }
    let head: String = text.chars().take(width.saturating_sub(1)).collect();
    format!("{head}…")
}

/// 2026-09-26: The key map; `draw_help` sizes and scrolls its modal by
/// `KEYS.len()`.
pub(super) const KEYS: [(&str, &str); 23] = [
    ("1-7", "jump to section (repeat cycles its subsections)"),
    (
        "Tab / Shift+Tab",
        "walk every sidebar row, subsections included",
    ),
    ("j/k ↑/↓", "move / scroll"),
    ("g / G", "top / bottom (follow)"),
    ("f", "log filter (Main)"),
    ("/", "search (Library)"),
    ("←/→ + Enter", "select node / detail (Network)"),
    ("Enter", "focus input (Terminal) / edit field (Benchmarks)"),
    ("s / c", "start / cancel the configured benchmark"),
    (
        "d",
        "Library: download / resume / update the selected model",
    ),
    (
        "a / x",
        "Config form: add a setting / remove it (server default applies)",
    ),
    (
        "b",
        "Config form: borrow parameters from another recipe (previewed first)",
    ),
    ("x", "Library: stop the running download"),
    ("u", "Library: check the selected model for updates"),
    ("t / Ctrl+T", "Chat: ask for thinking — auto / off / on"),
    ("T / Alt+T", "Chat: reasoning collapsed / expanded / hidden"),
    (
        "Ctrl+N",
        "Chat: clear the conversation (confirms if not empty)",
    ),
    ("Esc", "back / cancel"),
    ("Ctrl+C", "clean shutdown (drain + exit)"),
    // 2026-09-26: `q` stops the server, not only the TUI: it sets
    // `should_quit`, and the event loop then calls `shutdown::request`. With
    // `App::work_in_flight` set it asks first.
    ("q", "shut down the server (drain + exit; confirms if busy)"),
    // 2026-09-26: A slash command, not a key, listed because it is the one
    // command that leaves the dashboard with the server still running.
    (
        "/detach",
        "Terminal: leave the TUI, keep serving with plain logs",
    ),
    ("7", "Help: guide + report an issue to GitHub"),
    ("?", "this help"),
];

pub(super) fn draw_help(f: &mut Frame, app: &App, area: Rect) {
    let w = 64.min(area.width.saturating_sub(4));
    // 2026-09-26: Sized to the list when the terminal is tall enough, and
    // scrolled, not clipped, when it is not.
    let h = ((KEYS.len() + 2) as u16).min(area.height.saturating_sub(2));
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);
    let visible = (h.saturating_sub(2) as usize).max(1);
    // 2026-09-26: The scroll ceiling, published for
    // `App::on_help_overlay_key`.
    let max = KEYS.len().saturating_sub(visible);
    app.help_scroll_max.set(max);
    let off = app.help_scroll.min(max);
    let mut lines = Vec::with_capacity(visible);
    for (k, d) in KEYS.iter().skip(off).take(visible) {
        lines.push(Line::from(vec![
            Span::styled(format!("  {k:<16}"), theme::brand_cyan()),
            Span::styled(d.to_string(), theme::text2()),
        ]));
    }
    let mut block = Block::bordered()
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(theme::border(false))
        .title(Span::styled("─ KEYS ─", theme::text2()))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    // 2026-09-26: Position in the bottom border, only when the list is
    // clipped, as the `library/modal.rs` lists do.
    if max > 0 {
        block = block.title_bottom(Span::styled(
            format!(
                "─ j/k scroll · {}-{} of {} ─",
                off + 1,
                (off + visible).min(KEYS.len()),
                KEYS.len()
            ),
            theme::text2(),
        ));
    }
    f.render_widget(Paragraph::new(lines).block(block), modal);
}

/// 2026-09-26: The "one download at a time" question, drawn while
/// `App::download_switch` is set. Same frame as [`draw_quit_confirm`]: a
/// rounded warn border and a key list, and only an affirmative acts
/// (`App::answer_download_switch`).
///
/// "Stopping keeps its bytes" holds for this downloader: a cancel leaves the
/// `.part` sibling, and a resume sends `Range:` from its length
/// (`model_download/hf.rs`).
pub(super) fn draw_download_switch(f: &mut Frame, app: &App, area: Rect) {
    let Some((running, wanted)) = app.download_switch.as_ref() else {
        return;
    };
    let job = app.download.job.as_ref();
    // 2026-09-26: Read live each frame: the transfer keeps moving while the
    // question is on screen.
    let progress = match job {
        Some(j) if j.cancelling => " is stopping — waiting for the current chunk.".to_string(),
        Some(j) => match j.fraction() {
            Some(fr) if j.rate_bps > 0.0 => format!(
                " is still downloading — {}, {}.",
                crate::tui::format::percent(fr),
                crate::tui::format::rate(j.rate_bps)
            ),
            Some(fr) => format!(
                " is still downloading — {}.",
                crate::tui::format::percent(fr)
            ),
            None => format!(
                " is still downloading — {} so far.",
                crate::tui::format::bytes(j.done)
            ),
        },
        None => " has just finished.".to_string(),
    };
    let mut lines = vec![
        Line::from(vec![
            Span::styled(format!("  {running}"), theme::warn()),
            Span::styled(progress, theme::warn()),
        ]),
        Line::from(Span::styled(
            "  A second pull would share the same disk and halve both.",
            theme::text(),
        )),
    ];
    // 2026-09-26: Only claim the bytes are kept when there are bytes.
    if let Some(j) = job.filter(|j| j.done > 0) {
        lines.push(Line::from(Span::styled(
            format!(
                "  Stopping keeps its {} on disk; d resumes it later.",
                crate::tui::format::bytes(j.done)
            ),
            theme::text2(),
        )));
    }
    lines.push(Line::from(""));
    // 2026-09-26: Bold keys, not colour alone: under `NO_COLOR` cyan is
    // `Color::Reset`.
    let key = theme::brand_cyan().add_modifier(Modifier::BOLD);
    lines.push(Line::from(vec![
        Span::styled("  x / y", key),
        Span::styled(format!("  stop it, start {wanted}"), theme::text2()),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  any other key", key),
        Span::styled("  keep the current download", theme::text2()),
    ]));
    let w = 72.min(area.width.saturating_sub(4));
    let h = ((lines.len() + 2) as u16).min(area.height.saturating_sub(2));
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::warn())
        .title(Span::styled(
            "\u{2500} ONE DOWNLOAD AT A TIME \u{2500}",
            theme::warn(),
        ))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    f.render_widget(Paragraph::new(lines).block(block), modal);
}

/// 2026-09-26: Ask before `Ctrl+N` discards the conversation, drawn while
/// `App::confirm_chat_clear` is set. Same frame as [`draw_quit_confirm`], and
/// only an affirmative clears (`App::answer_chat_clear`). The first line names
/// the turn count and whether a reply is still streaming.
pub(super) fn draw_chat_clear_confirm(f: &mut Frame, app: &App, area: Rect) {
    if !app.confirm_chat_clear {
        return;
    }
    let turns = app.chat.transcript.len();
    let what = if app.chat.streaming {
        format!("  {turns} turns, one still streaming — it will be cancelled.")
    } else {
        format!("  {turns} turns will be discarded.")
    };
    let lines = vec![
        Line::from(Span::styled(what, theme::warn())),
        Line::from(Span::styled(
            "  The model keeps no memory of them once cleared.",
            theme::text(),
        )),
        Line::from(""),
        Line::from(vec![
            // 2026-09-26: Bold keys, for `NO_COLOR`.
            Span::styled(
                "  y / Ctrl+N",
                theme::brand_cyan().add_modifier(Modifier::BOLD),
            ),
            Span::styled("  clear it", theme::text2()),
            Span::styled(
                "     any other key",
                theme::brand_cyan().add_modifier(Modifier::BOLD),
            ),
            Span::styled("  keep it", theme::text2()),
        ]),
    ];
    let w = 62.min(area.width.saturating_sub(4));
    let h = ((lines.len() + 2) as u16).min(area.height.saturating_sub(2));
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::warn())
        .title(Span::styled("─ CLEAR THE CONVERSATION? ─", theme::warn()))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    f.render_widget(Paragraph::new(lines).block(block), modal);
}

/// 2026-09-26: Ask before `q` stops a server with work in flight, drawn while
/// `App::confirm_quit` is set and `App::work_in_flight` names something; the
/// first line is that name.
pub(super) fn draw_quit_confirm(f: &mut Frame, app: &App, area: Rect) {
    let Some(what) = app.work_in_flight() else {
        return;
    };
    let lines = vec![
        Line::from(Span::styled(format!("  {what}."), theme::warn())),
        Line::from(Span::styled(
            "  Quitting drains it and stops the server.",
            theme::text(),
        )),
        Line::from(""),
        Line::from(vec![
            // 2026-09-26: Bold keys, for `NO_COLOR`.
            Span::styled("  q / y", theme::brand_cyan().add_modifier(Modifier::BOLD)),
            Span::styled("  quit anyway", theme::text2()),
            Span::styled(
                "     any other key",
                theme::brand_cyan().add_modifier(Modifier::BOLD),
            ),
            Span::styled("  stay", theme::text2()),
        ]),
    ];
    let w = 62.min(area.width.saturating_sub(4));
    let h = ((lines.len() + 2) as u16).min(area.height.saturating_sub(2));
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::warn())
        .title(Span::styled(
            "\u{2500} STOP THE SERVER? \u{2500}",
            theme::warn(),
        ))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    f.render_widget(Paragraph::new(lines).block(block), modal);
}
