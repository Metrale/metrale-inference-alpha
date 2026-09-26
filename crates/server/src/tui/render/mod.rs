// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Frame layout: header, sidebar, the active section's pane,
//! footer, then the overlays; `draw` renders `App` state into a `Frame`.
//!
//! Owner: server tui.
//! Invariants:
//! - `Chrome::of` is the one source of the header height and sidebar width,
//!   used by `draw` here and by `events::on_mouse`.
//! - Every frame resets `App::lib_search_click` before any pane draws.

mod bench;
mod chat_lines;
mod header;
mod help_tab;
mod hints;
mod library;
mod main_tab;
mod main_tab_kernels;
mod network_tab;
mod overlay;
mod stats_tab;
mod stats_thermal;
mod terminal_tab;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect, Size};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph};

use super::app::{App, Focus, MainSub, Section};
use super::theme;

/// 2026-09-26: Where the header ends and the sidebar ends, for a terminal of
/// this size. `events::on_mouse` maps a click to a sidebar row with the same
/// value, so the renderer and the hit-tester cannot disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chrome {
    /// 2026-09-26: Header rows above the sidebar's first row.
    pub header_h: u16,
    /// 2026-09-26: Sidebar columns.
    pub sidebar_w: u16,
}

impl Chrome {
    /// 2026-09-26: The chrome a terminal this size gets.
    pub fn of(size: Size) -> Self {
        Self {
            // 2026-09-26: The three-row header carries the logo block; below
            // 28 rows it is a one-line strip.
            header_h: if size.height >= 28 { 3 } else { 1 },
            // 2026-09-26: The wide sidebar carries labels; the narrow one is
            // icons only.
            sidebar_w: if size.width >= 96 { 18 } else { 4 },
        }
    }

    /// 2026-09-26: Is the header drawing the logo block rather than the
    /// one-line strip?
    pub fn tall_header(&self) -> bool {
        self.header_h > 1
    }

    /// 2026-09-26: Is the sidebar drawing labels, and so the active section's
    /// subsection rows, which shift every row below them?
    pub fn full_sidebar(&self) -> bool {
        self.sidebar_w >= 18
    }
}

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    // 2026-09-26: Clear resets every cell's symbol. The base block below only
    // sets a style (ratatui's `Block::render` calls `Buffer::set_style`), which
    // keeps whatever glyph the buffer already holds.
    f.render_widget(ratatui::widgets::Clear, area);
    f.render_widget(
        Block::default().style(Style::default().bg(theme::BG_BASE.color())),
        area,
    );
    // 2026-09-26: A published hit-target resets each frame, so a rect from a
    // frame no longer on screen cannot catch a click.
    app.lib_search_click.set(None);
    let chrome = Chrome::of(area.as_size());
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(chrome.header_h),
            Constraint::Min(5),
            Constraint::Length(1),
        ])
        .split(area);
    header::draw_header(f, app, rows[0], chrome.tall_header());

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(chrome.sidebar_w), Constraint::Min(20)])
        .split(rows[1]);
    draw_sidebar(f, app, cols[0], chrome.full_sidebar());

    // 2026-09-26: The content area always wears a 1-cell ring, so nothing
    // shifts when a benchmark starts. It is the dim border while
    // `app.bench.glow` is off and pulses `theme::glow` while it is on, in every
    // section.
    let content = draw_glow_ring(f, app, cols[1]);

    match app.section {
        Section::Main => match app.main_sub {
            MainSub::Overview => main_tab::draw(f, app, content),
            MainSub::Kernels => main_tab_kernels::draw(f, app, content),
        },
        Section::Stats => stats_tab::draw(f, app, content),
        Section::Network => network_tab::draw(f, app, content),
        Section::Library => library::draw(f, app, content),
        Section::Benchmarks => bench::draw(f, app, content),
        Section::Terminal => terminal_tab::draw(f, app, content),
        Section::Help => help_tab::draw(f, app, content),
    }

    draw_footer(f, app, rows[2]);
    overlay::draw_toasts(f, app, content);
    if app.help_open {
        overlay::draw_help(f, app, area);
    }
    // 2026-09-26: Stacking order, bottom to top: the help modal, the download
    // question, the chat-clear question, the quit prompt. A question outranks
    // the reference, and the quit prompt outranks every other question.
    overlay::draw_download_switch(f, app, area);
    overlay::draw_chat_clear_confirm(f, app, area);
    if app.confirm_quit {
        overlay::draw_quit_confirm(f, app, area);
    }
    // 2026-09-26: Last, over every overlay: the copy is read back out of this
    // finished frame, so the highlight has to cover what it will read.
    draw_selection(f, app);
}

/// 2026-09-26: Paint the drag highlight onto the finished frame. It reverses
/// the cells rather than setting a colour, so it shows over any background.
fn draw_selection(f: &mut Frame, app: &App) {
    let Some(sel) = app.selection.filter(|s| s.is_drag()) else {
        return;
    };
    let area = f.area();
    let buf = f.buffer_mut();
    let ((_, sy), (_, ey)) = sel.ordered();
    for y in sy..=ey.min(area.height.saturating_sub(1)) {
        for x in area.x..area.x.saturating_add(area.width) {
            if sel.contains(x, y) {
                buf[(x, y)].modifier |= Modifier::REVERSED;
            }
        }
    }
}

/// 2026-09-26: Paint the content ring and return the area inside it.
fn draw_glow_ring(f: &mut Frame, app: &App, area: Rect) -> Rect {
    let style = if app.bench.glow {
        Style::default().fg(theme::glow(app.tick))
    } else {
        theme::border(false)
    };
    let mut block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(style);
    if app.bench.glow {
        block = block.title(Span::styled(
            format!(
                "─ ⏱ {} ─",
                app.bench
                    .descriptor()
                    .map(|d| d.name)
                    .unwrap_or("benchmark")
            ),
            Style::default()
                .fg(theme::glow(app.tick))
                .add_modifier(Modifier::BOLD),
        ));
    }
    let inner = block.inner(area);
    f.render_widget(block, area);
    inner
}

fn draw_sidebar(f: &mut Frame, app: &App, area: Rect, full: bool) {
    let mut lines: Vec<Line> = Vec::new();
    for s in Section::ALL {
        let selected = app.section == s;
        let bar = if selected {
            Span::styled("▌", theme::brand_purple())
        } else {
            Span::raw(" ")
        };
        let icon_style = if selected {
            theme::text()
        } else {
            theme::text2()
        };
        let mut spans = vec![bar, Span::styled(format!("{} ", s.icon()), icon_style)];
        if full {
            let label_style = if selected {
                theme::text().add_modifier(Modifier::BOLD)
            } else {
                theme::text2()
            };
            spans.push(Span::styled(s.label().to_string(), label_style));
            // 2026-09-26: Main's dot is the startup lamp only: warn colour
            // until `progress.ready`, then green. Unresolved kernels are
            // bannered in Main ▸ Kernels and announced by a toast instead.
            if s == Section::Main {
                let lamp = if app.progress.ready {
                    theme::brand_green()
                } else {
                    theme::warn()
                };
                spans.push(Span::styled("  ●", lamp));
            }
        }
        let mut line = Line::from(spans);
        if selected {
            line = line.style(theme::selected());
        }
        lines.push(line);
        // 2026-09-26: Subsections under the active section (full mode).
        if full && selected {
            let subs = s.subs();
            let active_sub = app.sub_index(s);
            for (i, name) in subs.iter().enumerate() {
                let active = i == active_sub;
                let glyph = if i + 1 == subs.len() { "└" } else { "├" };
                let style = if active {
                    theme::brand_cyan()
                } else {
                    theme::dim()
                };
                lines.push(Line::from(vec![
                    Span::styled(format!("   {glyph} "), theme::dim()),
                    Span::styled(name.to_string(), style),
                ]));
            }
        }
    }
    f.render_widget(Paragraph::new(lines), area);
    // 2026-09-26: 1-col rule on the right edge. `Layout` returns a zero-width
    // rect when the terminal is narrower than the constraints, and
    // `area.width - 1` below would then underflow.
    if area.width == 0 {
        return;
    }
    for y in area.y..area.y + area.height {
        f.render_widget(
            Paragraph::new(Span::styled("│", theme::dim())),
            Rect {
                x: area.x + area.width - 1,
                y,
                width: 1,
                height: 1,
            },
        );
    }
}

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let mode = if app.help_open {
        (" HELP ", theme::TEXT_2)
    } else if app.focus == Focus::Input || app.log_filter_editing || app.lib.is_editing() {
        (" INPUT ", theme::CYAN)
    } else {
        (" NORMAL ", theme::BORDER_DIM)
    };
    let hints = match app.section {
        Section::Main => "j/k scroll · f filter · ⇥ Overview↔Kernels · 1-7 jump · ? help · q quit",
        Section::Stats => "⇥ cycle · 1-7 jump · ? help · q quit",
        Section::Network => "←/→ node · ⇥ cycle · 1-7 jump · ? help",
        Section::Library => hints::library_hints(app),
        Section::Benchmarks => hints::bench_hints(app),
        // 2026-09-26: `/detach` is the one command that leaves the dashboard
        // with the server still running, and this is the tab it is typed into.
        Section::Terminal => {
            "⏎ input · Esc back · ↑/↓ scroll · ⇥ Ops↔Chat · /detach leave · ? help"
        }
        Section::Help => hints::help_hints(app),
    };
    let line = Line::from(vec![
        Span::styled(
            mode.0,
            Style::default()
                .bg(mode.1.color())
                .fg(theme::BG_BASE.color()),
        ),
        Span::styled(format!("  {hints}"), theme::dim()),
    ]);
    f.render_widget(
        Paragraph::new(line).style(Style::default().bg(theme::BG_PANEL.color())),
        area,
    );
}

/// 2026-09-26: Shared rounded-panel block.
pub(super) fn panel(title: String, focused: bool) -> Block<'static> {
    Block::bordered()
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(theme::border(focused))
        .title(Span::styled(format!("─ {title} "), theme::title(focused)))
        .style(Style::default().bg(theme::BG_PANEL.color()))
}

/// 2026-09-26: A gradient bar as a styled line: `█` filled, `▓` at the edge of
/// a partial fill, `░` track, coloured per cell by `theme::gradient_at`.
pub(super) fn gradient_bar(frac: f64, width: u16) -> Line<'static> {
    let width = width.max(1) as usize;
    let filled = ((frac.clamp(0.0, 1.0)) * width as f64).round() as usize;
    let mut spans = Vec::with_capacity(width);
    for i in 0..width {
        if i < filled {
            let t = i as f64 / (width.saturating_sub(1)).max(1) as f64;
            let ch = if i + 1 == filled && filled < width {
                "▓"
            } else {
                "█"
            };
            spans.push(Span::styled(ch, Style::default().fg(theme::gradient_at(t))));
        } else {
            spans.push(Span::styled(
                "░",
                Style::default().fg(theme::GAUGE_TRACK.color()),
            ));
        }
    }
    Line::from(spans)
}

#[cfg(test)]
#[path = "harness.rs"]
mod harness;

#[cfg(test)]
#[path = "render_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "chrome_tests.rs"]
mod chrome_tests;

#[cfg(test)]
#[path = "download_render_tests.rs"]
mod download_tests;

/// 2026-09-26: The model the host reports as live, else the argv's
/// `model_name`, else its `model`, else empty. The argv is only what the
/// dashboard started with, so the host is asked first.
pub(crate) fn live_model_name(app: &App) -> String {
    app.host
        .as_ref()
        .and_then(|h| h.live_model())
        .or_else(|| app.args.model_name.clone())
        .or_else(|| app.args.model.clone())
        .unwrap_or_default()
}

/// 2026-09-26: Wrap `text` to `width` columns as styled lines, using
/// `format::wrap_words` (which measures bytes).
pub(crate) fn wrap(text: &str, width: usize, style: ratatui::style::Style) -> Vec<Line<'static>> {
    crate::tui::format::wrap_words(text, width)
        .into_iter()
        .map(|l| Line::from(Span::styled(l, style)))
        .collect()
}

#[cfg(test)]
#[path = "selection_render_tests.rs"]
mod selection_tests;
