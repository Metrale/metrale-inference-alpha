// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Thinking preferences for the Chat pane, as two separate
//! controls: [`ThinkingRequest`] sets what the request body asks for, and
//! [`ThinkingView`] sets how an arriving reasoning trace is drawn and changes
//! nothing on the wire. Also the per-reply [`Reasoning`] clocks.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use std::time::Instant;

/// 2026-09-26: What the request asks the server to do about thinking.
///
/// [`ThinkingRequest::Auto`] leaves `chat_template_kwargs` out of the body, so
/// the server's own precedence decides: `--disable-thinking`, then
/// `--default-chat-template-kwargs`, then the model's MODEL.toml
/// `[behavior].thinking_default`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThinkingRequest {
    /// 2026-09-26: Send no thinking key; the server decides.
    #[default]
    Auto,
    /// 2026-09-26: `chat_template_kwargs: {"enable_thinking": false}`.
    Off,
    /// 2026-09-26: `chat_template_kwargs: {"enable_thinking": true}`.
    On,
}

impl ThinkingRequest {
    /// 2026-09-26: Auto → Off → On → Auto.
    pub fn next(self) -> Self {
        match self {
            Self::Auto => Self::Off,
            Self::Off => Self::On,
            Self::On => Self::Auto,
        }
    }

    /// 2026-09-26: The value for `chat_template_kwargs.enable_thinking`, or
    /// `None` to leave the key out.
    pub fn enable_thinking(self) -> Option<bool> {
        match self {
            Self::Auto => None,
            Self::Off => Some(false),
            Self::On => Some(true),
        }
    }

    /// 2026-09-26: The state chip for the chat pane. Only the wide `Auto` chip
    /// shows `observed` (whether the last completed reply thought); `None`
    /// adds no suffix.
    pub fn chip(self, observed: Option<bool>, wide: bool) -> String {
        let resolved = match observed {
            Some(true) => " (thinking)",
            Some(false) => " (no thinking)",
            None => "",
        };
        match self {
            Self::Auto if wide => format!("thinking auto{resolved}"),
            Self::Auto => "think auto".to_string(),
            Self::Off if wide => "thinking off".to_string(),
            Self::Off => "think off".to_string(),
            Self::On if wide => "thinking on".to_string(),
            Self::On => "think on".to_string(),
        }
    }
}

/// 2026-09-26: How a reasoning trace is rendered once it arrives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThinkingView {
    /// 2026-09-26: A live tail while it streams, then the header line only.
    #[default]
    Collapsed,
    /// 2026-09-26: The whole trace, while streaming and after.
    Expanded,
    /// 2026-09-26: Not drawn at all.
    Hidden,
}

impl ThinkingView {
    pub fn next(self) -> Self {
        match self {
            Self::Collapsed => Self::Expanded,
            Self::Expanded => Self::Hidden,
            Self::Hidden => Self::Collapsed,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Collapsed => "collapsed",
            Self::Expanded => "expanded",
            Self::Hidden => "hidden",
        }
    }
}

/// 2026-09-26: The reasoning half of one model reply.
#[derive(Default)]
pub struct Reasoning {
    pub text: String,
    /// 2026-09-26: Reasoning deltas received, replaced by the `Done` delta's
    /// `reasoning_tokens` when that is above zero.
    pub tokens: usize,
    /// 2026-09-26: The thinking span in ms. `seal` sets it from `started` when
    /// the first answer token arrives; a `Done` delta that carries `think_ms`
    /// replaces it with the streaming task's span.
    pub think_ms: Option<f64>,
    /// 2026-09-26: When the first reasoning delta reached the TUI; drives the
    /// live timer until `think_ms` is set.
    started: Option<Instant>,
}

impl Reasoning {
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// 2026-09-26: Note that reasoning has begun. Only the first call starts the
    /// clock.
    pub fn begin(&mut self) {
        self.started.get_or_insert_with(Instant::now);
    }

    /// 2026-09-26: Freeze the live timer when the answer starts: set `think_ms`
    /// from `started`, unless it is already set or reasoning never began.
    pub fn seal(&mut self) {
        if self.think_ms.is_none()
            && let Some(t) = self.started
        {
            self.think_ms = Some(t.elapsed().as_secs_f64() * 1000.0);
        }
    }

    /// 2026-09-26: Seconds to show beside the header: `think_ms` once set,
    /// otherwise the time since `begin`. `None` before `begin`.
    pub fn seconds(&self) -> Option<f64> {
        self.think_ms
            .map(|ms| ms / 1000.0)
            .or_else(|| self.started.map(|t| t.elapsed().as_secs_f64()))
    }
}

/// 2026-09-26: `412ms` below 1 s, `18.2s` below 60 s, then `1m 04s`.
pub fn dur(secs: f64) -> String {
    if secs < 1.0 {
        format!("{:.0}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{secs:.1}s")
    } else {
        format!("{}m {:02.0}s", (secs / 60.0).floor(), secs % 60.0)
    }
}

#[cfg(test)]
#[path = "chat_thinking_more_tests.rs"]
mod chip_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_omits_the_key_entirely() {
        assert_eq!(ThinkingRequest::Auto.enable_thinking(), None);
        assert_eq!(ThinkingRequest::Off.enable_thinking(), Some(false));
        assert_eq!(ThinkingRequest::On.enable_thinking(), Some(true));
    }

    #[test]
    fn both_cycles_return_home_in_three() {
        let mut r = ThinkingRequest::default();
        for _ in 0..3 {
            r = r.next();
        }
        assert_eq!(r, ThinkingRequest::Auto);
        assert_eq!(ThinkingRequest::Auto.next(), ThinkingRequest::Off);
        let mut v = ThinkingView::default();
        for _ in 0..3 {
            v = v.next();
        }
        assert_eq!(v, ThinkingView::Collapsed);
    }

    #[test]
    fn auto_claims_nothing_until_it_has_been_observed() {
        assert_eq!(ThinkingRequest::Auto.chip(None, true), "thinking auto");
        assert_eq!(
            ThinkingRequest::Auto.chip(Some(true), true),
            "thinking auto (thinking)"
        );
        assert_eq!(
            ThinkingRequest::Auto.chip(Some(false), true),
            "thinking auto (no thinking)"
        );
        // 2026-09-26: `Off` and `On` ignore `observed`.
        assert_eq!(ThinkingRequest::Off.chip(Some(true), true), "thinking off");
    }

    #[test]
    fn durations_stay_short_enough_for_eighty_columns() {
        assert_eq!(dur(0.412), "412ms");
        assert_eq!(dur(18.24), "18.2s");
        assert_eq!(dur(64.0), "1m 04s");
    }

    #[test]
    fn a_reasoning_clock_starts_once_and_prefers_the_measurement() {
        let mut r = Reasoning::default();
        assert!(r.seconds().is_none(), "nothing to time yet");
        r.begin();
        let first = r.started;
        r.begin();
        assert_eq!(first, r.started, "the second delta does not restart it");
        assert!(r.seconds().is_some());
        r.think_ms = Some(18_200.0);
        assert_eq!(r.seconds(), Some(18.2), "the measurement wins");
    }
}
