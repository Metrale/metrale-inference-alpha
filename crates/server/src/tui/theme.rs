// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The TUI palette and styles. Each colour [`C`] has a 24-bit
//! value and a pinned 256-colour index (all 16 or above, so the terminal's own
//! 16-colour palette is not used). [`depth`] picks which is drawn: 24-bit when
//! `COLORTERM` says so, the index otherwise, and `Color::Reset` under
//! `NO_COLOR`.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use ratatui::style::{Color, Modifier, Style};

/// 2026-09-26: How much colour this terminal is given.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Depth {
    /// 2026-09-26: `NO_COLOR` is set. `warn`, `selected` and `border` carry
    /// their signal with a modifier instead of a hue.
    None,
    /// 2026-09-26: The pinned 256-colour indices.
    Ansi256,
    True,
}

/// 2026-09-26: Resolve the depth from the values of `NO_COLOR` and
/// `COLORTERM`, passed in so the precedence is tested without setting env
/// variables.
///
/// `NO_COLOR` outranks `COLORTERM`, and it is not a boolean: any non-empty
/// value, `0` included, means no colour; only an empty value is ignored.
pub fn depth_of(no_color: Option<&str>, colorterm: Option<&str>) -> Depth {
    if no_color.is_some_and(|v| !v.is_empty()) {
        return Depth::None;
    }
    match colorterm {
        Some(v) if v.contains("truecolor") || v.contains("24bit") => Depth::True,
        _ => Depth::Ansi256,
    }
}

/// 2026-09-26: This process's colour depth, read from the environment on
/// every call. It is not cached, so one test's environment cannot fix the
/// answer for the rest of the test binary.
pub fn depth() -> Depth {
    depth_of(
        std::env::var("NO_COLOR").ok().as_deref(),
        std::env::var("COLORTERM").ok().as_deref(),
    )
}

/// 2026-09-26: Whether the terminal advertises 24-bit colour.
fn truecolor() -> bool {
    depth() == Depth::True
}

/// 2026-09-26: A themed colour: 24-bit value, then the 256-colour index.
#[derive(Clone, Copy)]
pub struct C(pub u8, pub u8, pub u8, pub u8);

impl C {
    pub fn color(self) -> Color {
        match depth() {
            // 2026-09-26: `Reset`, not a black or white guess: the terminal's
            // default is legible against the background the user chose.
            Depth::None => Color::Reset,
            Depth::Ansi256 => Color::Indexed(self.3),
            Depth::True => Color::Rgb(self.0, self.1, self.2),
        }
    }
}

pub const PURPLE: C = C(0xBE, 0x9D, 0xF8, 141);
pub const CYAN: C = C(0x49, 0xC3, 0xDB, 80);
pub const GREEN: C = C(0x12, 0xB9, 0x81, 36);
pub const BG_BASE: C = C(0x0F, 0x11, 0x17, 232);
pub const BG_PANEL: C = C(0x15, 0x18, 0x23, 233);
pub const BG_RAISED: C = C(0x1E, 0x22, 0x30, 235);
pub const BG_SELECTION: C = C(0x2B, 0x26, 0x40, 237);
pub const BORDER_DIM: C = C(0x2A, 0x2F, 0x3F, 237);
pub const TEXT: C = C(0xE6, 0xE9, 0xF0, 254);
pub const TEXT_2: C = C(0x93, 0x97, 0xA0, 246);
pub const TEXT_DIM: C = C(0x56, 0x5B, 0x68, 240);
pub const WARN: C = C(0xE5, 0xC0, 0x7B, 179);
pub const ERROR: C = C(0xF7, 0x76, 0x8E, 204);
pub const GAUGE_TRACK: C = C(0x25, 0x2A, 0x38, 236);

pub fn text() -> Style {
    Style::default().fg(TEXT.color())
}
pub fn text2() -> Style {
    Style::default().fg(TEXT_2.color())
}
pub fn dim() -> Style {
    Style::default().fg(TEXT_DIM.color())
}
pub fn brand_purple() -> Style {
    Style::default().fg(PURPLE.color())
}
pub fn brand_cyan() -> Style {
    Style::default().fg(CYAN.color())
}
pub fn brand_green() -> Style {
    Style::default().fg(GREEN.color())
}
pub fn warn() -> Style {
    let s = Style::default().fg(WARN.color());
    // 2026-09-26: `error()` is bold in every mode; without colour, bold is
    // what separates a warning from an info line.
    if depth() == Depth::None {
        s.add_modifier(Modifier::BOLD)
    } else {
        s
    }
}

/// 2026-09-26: The style of the selected row: the selection background, or
/// reverse video under `NO_COLOR`, where a background would be
/// `Color::Reset`.
pub fn selected() -> Style {
    if depth() == Depth::None {
        Style::default().add_modifier(Modifier::REVERSED)
    } else {
        Style::default().bg(BG_SELECTION.color())
    }
}
pub fn error() -> Style {
    Style::default()
        .fg(ERROR.color())
        .add_modifier(Modifier::BOLD)
}

/// 2026-09-26: Panel border style; `focused` turns it brand cyan.
pub fn border(focused: bool) -> Style {
    if !focused {
        return Style::default().fg(BORDER_DIM.color());
    }
    // 2026-09-26: Without colour, focus is shown in bold instead.
    if depth() == Depth::None {
        Style::default().add_modifier(Modifier::BOLD)
    } else {
        brand_cyan()
    }
}

/// 2026-09-26: Panel title style; bold cyan when the panel has focus.
pub fn title(focused: bool) -> Style {
    if focused {
        brand_cyan().add_modifier(Modifier::BOLD)
    } else {
        text2()
    }
}

/// 2026-09-26: Style for a log level.
pub fn level_style(level: tracing::Level) -> Style {
    match level {
        tracing::Level::ERROR => error(),
        tracing::Level::WARN => warn(),
        tracing::Level::INFO => brand_cyan(),
        _ => dim(),
    }
}

/// 2026-09-26: The progress gradient at `t ∈ [0,1]`: purple → cyan on
/// [0,0.5), cyan → green on [0.5,1]. In 256-colour mode, three hard bands.
pub fn gradient_at(t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    // 2026-09-26: Without colour the bar still fills, in the terminal's
    // default colour.
    if depth() == Depth::None {
        return Color::Reset;
    }
    if !truecolor() {
        return if t < 0.34 {
            Color::Indexed(PURPLE.3)
        } else if t < 0.67 {
            Color::Indexed(CYAN.3)
        } else {
            Color::Indexed(GREEN.3)
        };
    }
    let lerp = |a: u8, b: u8, f: f64| (a as f64 + (b as f64 - a as f64) * f).round() as u8;
    let (from, to, f) = if t < 0.5 {
        (PURPLE, CYAN, t * 2.0)
    } else {
        (CYAN, GREEN, (t - 0.5) * 2.0)
    };
    Color::Rgb(
        lerp(from.0, to.0, f),
        lerp(from.1, to.1, f),
        lerp(from.2, to.2, f),
    )
}

/// 2026-09-26: Gauge fill colour override when nearly full: ≥97% error,
/// ≥90% warn.
pub fn pressure_color(frac: f64) -> Option<Color> {
    if frac >= 0.97 {
        Some(ERROR.color())
    } else if frac >= 0.90 {
        Some(WARN.color())
    } else {
        None
    }
}

/// 2026-09-26: The benchmark run glow: brand cyan pulsing between the dim
/// border colour and full cyan every 16 ticks (1.6 s at the 100 ms tick). In
/// 256-colour mode it alternates between the two indices at the same cadence.
pub fn glow(tick: u64) -> Color {
    // 2026-09-26: The pulse is a hue animation only, so without colour it is
    // a steady default.
    if depth() == Depth::None {
        return Color::Reset;
    }
    let phase = (tick % 16) as f64 / 16.0;
    // 2026-09-26: 0.35 -> 1 -> 0.35 over the period, never fully dark.
    let t = 0.35 + 0.65 * (1.0 - (phase * std::f64::consts::TAU).cos()) / 2.0;
    if !truecolor() {
        return if t > 0.6 {
            Color::Indexed(CYAN.3)
        } else {
            Color::Indexed(BORDER_DIM.3)
        };
    }
    let mix = |lit: u8, dark: u8| (dark as f64 + (lit as f64 - dark as f64) * t).round() as u8;
    Color::Rgb(
        mix(CYAN.0, BORDER_DIM.0),
        mix(CYAN.1, BORDER_DIM.1),
        mix(CYAN.2, BORDER_DIM.2),
    )
}

/// 2026-09-26: Map a benchmark's semantic cell style onto the palette; the
/// only place `metrale_bench::CellStyle` becomes a colour.
pub fn cell_style(style: metrale_bench::CellStyle) -> Style {
    use metrale_bench::CellStyle as S;
    match style {
        S::Neutral => text(),
        S::Dim => dim(),
        S::Accent => brand_cyan(),
        S::Good => brand_green(),
        S::Warn => warn(),
        S::Bad => error(),
    }
}

/// 2026-09-26: Braille spinner frames.
pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[cfg(test)]
#[path = "theme_tests.rs"]
mod tests;
