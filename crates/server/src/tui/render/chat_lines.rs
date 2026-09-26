// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: One chat transcript entry as display rows.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.
//!
//! Rows are wrapped in display columns because `terminal_tab` slices its
//! viewport on them without wrapping again.
//!
//! ```text
//! ❯ what the user asked                     purple chevron, full-strength text
//! ⬢ ┆ ⠹ thinking 4.2s                       cyan spinner, dim dashed rule
//!   ┆ muted reasoning, streaming live       TEXT_2 behind the dashed rule
//!   ▏the answer                             solid cyan rule, full-strength text
//!   ttft 412ms · 247 think + 312 tok        dim footer
//! ```
//!
//! Reasoning uses dimmer text and a dashed rule against the answer's solid
//! one, and collapses to one summary line once the answer starts.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::tui::chat::{ChatMessage, Role};
use crate::tui::chat_thinking::{ThinkingView, dur};
use crate::tui::theme;

/// 2026-09-26: Rows of a live reasoning trace the collapsed view shows (the
/// newest ones); the expanded view shows all of it.
const PREVIEW_ROWS: usize = 6;

/// 2026-09-26: The answer's rule: solid, brand cyan.
fn rule_answer() -> Span<'static> {
    Span::styled("▏", theme::brand_cyan())
}

/// 2026-09-26: The reasoning rule: dashed and dim, not cyan, so it is told
/// apart from the answer's rule.
fn rule_think() -> Span<'static> {
    Span::styled("┆", theme::dim())
}

fn model_body() -> Style {
    theme::text().bg(theme::BG_PANEL.color())
}

/// 2026-09-26: Word-wrap `text` into rows of at most `width` display columns
/// (`unicode-width`, not bytes). A word wider than `width` is split across
/// rows.
pub(super) fn wrap_rows(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    let mut rows = Vec::new();
    for logical in text.split('\n') {
        let (mut cur, mut cur_w) = (String::new(), 0usize);
        for word in logical.split_inclusive(' ') {
            let w = UnicodeWidthStr::width(word);
            if cur_w + w > width && !cur.is_empty() {
                rows.push(std::mem::take(&mut cur));
                cur_w = 0;
            }
            if w > width {
                for ch in word.chars() {
                    let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
                    if cur_w + cw > width {
                        rows.push(std::mem::take(&mut cur));
                        cur_w = 0;
                    }
                    cur.push(ch);
                    cur_w += cw;
                }
            } else {
                cur.push_str(word);
                cur_w += w;
            }
        }
        rows.push(cur);
    }
    rows
}

/// 2026-09-26: Truncate to `width` display columns, ending the cut with `…`.
/// Used for the reasoning header, which is one row.
fn clip(s: &str, width: usize) -> String {
    if UnicodeWidthStr::width(s) <= width {
        return s.to_string();
    }
    let mut out = String::new();
    let mut w = 0usize;
    for ch in s.chars() {
        let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
        if w + cw + 1 > width {
            break;
        }
        out.push(ch);
        w += cw;
    }
    out.push('…');
    out
}

/// 2026-09-26: The reply footer. `ttft` is the time to the first reasoning or
/// answer token; `answer` is the time to the first answer token. Reasoning tokens are
/// shown apart from answer tokens, and the ms/tok figure is computed over both
/// (`chat_stream::Clocks::done`).
fn footer(m: &ChatMessage, wide: bool) -> Option<String> {
    if m.ttft_ms.is_none() && m.tok_per_s.is_none() {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    match m.ttft_ms {
        Some(v) => parts.push(format!("ttft {}", dur(v / 1000.0))),
        None => parts.push("ttft —".into()),
    }
    // 2026-09-26: `answer` only on a wide pane and when there was reasoning.
    if wide
        && !m.reasoning.is_empty()
        && let Some(v) = m.answer_ttft_ms
    {
        parts.push(format!("answer {}", dur(v / 1000.0)));
    }
    if m.reasoning.tokens > 0 {
        parts.push(format!("{} think + {} tok", m.reasoning.tokens, m.tokens));
    } else {
        parts.push(format!("{} tok", m.tokens));
    }
    if let Some(tps) = m.tok_per_s.filter(|t| *t > 0.0) {
        parts.push(format!("{:.1} ms/tok", 1000.0 / tps));
    }
    Some(parts.join(" · "))
}

/// 2026-09-26: The reasoning block: a header, then a body per `view`.
fn thinking_rows(
    rows: &mut Vec<Vec<Span<'static>>>,
    m: &ChatMessage,
    live: bool,
    view: ThinkingView,
    tick: u64,
    body_w: usize,
) {
    // 2026-09-26: No reasoning, or a hidden view: no rows at all.
    if m.reasoning.is_empty() || view == ThinkingView::Hidden {
        return;
    }
    let secs = m.reasoning.seconds().unwrap_or(0.0);
    // 2026-09-26: Only the streaming tip with no answer text yet is live and
    // gets the spinner.
    let (lead, lead_style, rest) = if live {
        (
            format!(" {} ", theme::SPINNER[(tick % 10) as usize]),
            theme::brand_cyan(),
            format!("thinking {}", dur(secs)),
        )
    } else {
        let caret = if view == ThinkingView::Expanded {
            " ▾ "
        } else {
            " ▸ "
        };
        let n = m.reasoning.tokens;
        // 2026-09-26: A shorter wording below 34 columns, so a narrow pane
        // still gets words rather than an ellipsis.
        let rest = if body_w >= 34 {
            format!("thought for {} · {n} tokens", dur(secs))
        } else {
            format!("thought {} · {n} tok", dur(secs))
        };
        (caret.to_string(), theme::dim(), rest)
    };
    let room = body_w.saturating_sub(UnicodeWidthStr::width(lead.as_str()) + 1);
    rows.push(vec![
        rule_think(),
        Span::styled(lead, lead_style),
        Span::styled(clip(&rest, room), theme::text2()),
    ]);
    // 2026-09-26: Expanded shows all of it; collapsed shows the last
    // `PREVIEW_ROWS` while live and no body after.
    let limit = match (view, live) {
        (ThinkingView::Expanded, _) => usize::MAX,
        (_, true) => PREVIEW_ROWS,
        (_, false) => return,
    };
    // 2026-09-26: One column is kept between `┆` and the text.
    let wrapped = wrap_rows(&m.reasoning.text, body_w.saturating_sub(1));
    let skip = wrapped.len().saturating_sub(limit);
    for r in wrapped.into_iter().skip(skip) {
        rows.push(vec![
            rule_think(),
            Span::styled(format!(" {r}"), theme::text2()),
        ]);
    }
}

fn answer_rows(rows: &mut Vec<Vec<Span<'static>>>, m: &ChatMessage, is_tip: bool, body_w: usize) {
    if m.is_answerless() {
        // 2026-09-26: Say that no answer came rather than stop silently.
        let msg = if m.reasoning.is_empty() {
            "(no answer — the reply produced nothing)"
        } else {
            "(no answer — the model stopped after thinking)"
        };
        rows.push(vec![rule_answer(), Span::styled(msg, theme::warn())]);
        return;
    }
    // 2026-09-26: While reasoning streams there is no answer row; the cursor
    // goes on the last reasoning row instead.
    if m.text.is_empty() && is_tip && !rows.is_empty() {
        return;
    }
    for r in wrap_rows(&m.text, body_w) {
        rows.push(vec![rule_answer(), Span::styled(r, model_body())]);
    }
}

/// 2026-09-26: One transcript entry as display rows, gutter included.
/// `is_tip` marks the live streaming message; only it gets a spinner or a
/// cursor.
pub(super) fn message_lines(
    m: &ChatMessage,
    is_tip: bool,
    view: ThinkingView,
    tick: u64,
    body_w: usize,
) -> Vec<Line<'static>> {
    let mut rows: Vec<Vec<Span<'static>>> = Vec::new();
    match m.role {
        Role::User => {
            for r in wrap_rows(&m.text, body_w) {
                rows.push(vec![Span::raw(""), Span::styled(r, theme::text())]);
            }
        }
        Role::Model => {
            // 2026-09-26: Live means the tip with no answer text yet. Reasoning
            // that arrives after the answer started still joins the block,
            // but the header no longer spins.
            let live = is_tip && m.text.is_empty();
            thinking_rows(&mut rows, m, live, view, tick, body_w);
            answer_rows(&mut rows, m, is_tip, body_w);
            if is_tip && let Some(last) = rows.last_mut() {
                last.push(Span::styled("▍", theme::brand_cyan()));
            }
            if let Some(f) = footer(m, body_w >= 60) {
                for r in wrap_rows(&f, body_w) {
                    rows.push(vec![Span::raw(""), Span::styled(r, theme::dim())]);
                }
            }
        }
    }
    let (glyph, gstyle) = match m.role {
        Role::User => ("❯ ", theme::brand_purple().add_modifier(Modifier::BOLD)),
        Role::Model => ("⬢ ", theme::brand_cyan()),
    };
    rows.into_iter()
        .enumerate()
        .map(|(i, mut spans)| {
            let mut line = vec![if i == 0 {
                Span::styled(glyph, gstyle)
            } else {
                Span::styled("  ", Style::default())
            }];
            line.append(&mut spans);
            Line::from(line)
        })
        .collect()
}

#[cfg(test)]
#[path = "chat_lines_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "chat_wide_tests.rs"]
mod wide_tests;
