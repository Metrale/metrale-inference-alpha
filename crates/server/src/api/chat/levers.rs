// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`ChatLevers`]: the chat request path's per-model and
//! per-deployment switches, resolved once when [`crate::AppState`] is built
//! and read from `state.chat`.
//!
//! Owner: server (chat API).
//! Invariants: none beyond the types.

use crate::tool_parser::PromptLevers;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatLevers {
    /// 2026-09-26: Prompt-rendering decisions, handed to
    /// `ToolCallParser::system_prompt` and read by the chat template step.
    pub prompt: PromptLevers,
    /// 2026-09-26: `METRALE_BASH_WANDER_WATCHDOG=1`: append the steering hint
    /// from `hint_injector::bash_wander_hint` (enough tool calls in the
    /// conversation, none of them productive) to the latest tool result. Off
    /// unless the value is exactly `1`.
    pub bash_wander: bool,
    /// 2026-09-26: `METRALE_CHAT_PHASE_TIMING=1`: log `CHAT_PHASE` timing lines
    /// from `chat_completions_inner` and `prepare_chat_prompt`. Off unless the
    /// value is exactly `1`.
    pub phase_timing: bool,
    /// 2026-09-26: MODEL.toml `[behavior] disable_cwd_hint_injection`: do not
    /// append the `<environment>working_directory: …</environment>` hint to
    /// the system message of a tools-active request.
    pub disable_cwd_hint_injection: bool,
    /// 2026-09-26: `METRALE_INTHINK_TOOL_LEAK_OPENERS=N`: a streaming request
    /// is cancelled once its reasoning has shown `N` tool-call openers
    /// (`api/chat_stream/handle_token.rs`). Default 1; 0 never cancels.
    pub in_think_leak_openers: u32,
}

impl Default for ChatLevers {
    /// 2026-09-26: [`ChatLevers::OFF`]. Not derived: a derived `Default` would
    /// set `in_think_leak_openers` to 0, which never cancels.
    fn default() -> Self {
        Self::OFF
    }
}

impl ChatLevers {
    /// 2026-09-26: What `resolve(false, false)` gives with none of its env
    /// vars set. Request-path tests build against it.
    pub const OFF: Self = Self {
        prompt: PromptLevers::OFF,
        bash_wander: false,
        phase_timing: false,
        disable_cwd_hint_injection: false,
        in_think_leak_openers: 1,
    };

    /// 2026-09-26: Resolve from the environment and this model's `[behavior]`
    /// values. Called when `AppState` is built (`serve_load.rs`).
    pub fn resolve(tscg: bool, disable_cwd_hint_injection: bool) -> Self {
        Self {
            prompt: PromptLevers::new(tscg),
            bash_wander: std::env::var("METRALE_BASH_WANDER_WATCHDOG").as_deref() == Ok("1"),
            phase_timing: std::env::var("METRALE_CHAT_PHASE_TIMING").as_deref() == Ok("1"),
            disable_cwd_hint_injection,
            in_think_leak_openers: in_think_leak_openers_from(
                std::env::var("METRALE_INTHINK_TOOL_LEAK_OPENERS")
                    .ok()
                    .as_deref(),
            ),
        }
    }
}

/// 2026-09-26: Parse `METRALE_INTHINK_TOOL_LEAK_OPENERS`. Unset gives the
/// default; a value that is not a `u32` logs a warning and gives the default.
fn in_think_leak_openers_from(env: Option<&str>) -> u32 {
    match env {
        None => ChatLevers::OFF.in_think_leak_openers,
        Some(v) => match v.trim().parse::<u32>() {
            Ok(n) => n,
            Err(_) => {
                tracing::warn!(
                    value = %v,
                    "METRALE_INTHINK_TOOL_LEAK_OPENERS is set but not a u32; using default"
                );
                ChatLevers::OFF.in_think_leak_openers
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_is_the_default() {
        assert_eq!(ChatLevers::default(), ChatLevers::OFF);
    }

    #[test]
    fn leak_opener_threshold_parses_and_defaults() {
        assert_eq!(in_think_leak_openers_from(None), 1);
        assert_eq!(in_think_leak_openers_from(Some("0")), 0);
        assert_eq!(in_think_leak_openers_from(Some("3")), 3);
        // 2026-09-26: Not 0, which would disable the cancel.
        assert_eq!(in_think_leak_openers_from(Some("many")), 1);
    }

    #[test]
    fn the_model_behavior_reaches_the_renderer() {
        // 2026-09-26: `resolve` takes `tscg` and `disable_cwd_hint_injection`
        // from its caller (MODEL.toml `[behavior]`), not from the environment.
        assert!(ChatLevers::resolve(true, false).prompt.tscg);
        assert!(!ChatLevers::resolve(false, false).prompt.tscg);
        assert!(ChatLevers::resolve(false, true).disable_cwd_hint_injection);
    }
}
