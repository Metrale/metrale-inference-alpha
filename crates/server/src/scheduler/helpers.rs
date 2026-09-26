// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Helpers: BF16 conversion, sampling defaults, loop-detection
//! constants (the detectors are in `helpers/detectors.rs`), env parsers,
//! and the pure predicates behind the length limits (the limits travel on
//! `SchedCtx`; see `scheduler::limits`).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

/// 2026-09-25: Convert two little-endian BF16 bytes to f32.
#[inline]
pub fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
}

/// 2026-09-25: Would the next decode step reach the served context
/// ceiling? `position` is `SequenceState::seq_len`. `max_seq_len == 0`
/// means no ceiling, and this is then always false.
#[inline]
pub fn seqlen_force_stop(position: usize, max_seq_len: usize) -> bool {
    max_seq_len != 0 && position + 1 >= max_seq_len
}

/// 2026-09-25: Has this sequence hit a hard ceiling: completion budget
/// exhausted (`remaining == 0`) or [`seqlen_force_stop`]?
#[inline]
pub fn hard_ceiling_hit(remaining: usize, position: usize, max_seq_len: usize) -> bool {
    remaining == 0 || seqlen_force_stop(position, max_seq_len)
}

/// 2026-09-25: Whether thinking holds back a sampled EOS: inside `<think>`
/// and no hard ceiling hit. At a ceiling the EOS is not held back by
/// thinking.
#[inline]
pub fn eos_suppressed_by_thinking(inside_thinking: bool, hard_ceiling_hit: bool) -> bool {
    inside_thinking && !hard_ceiling_hit
}

// 2026-09-25: sampling defaults for the `SamplingParams` the scheduler
// builds.
pub const DEFAULT_LZ_PENALTY: f32 = 0.0;
pub const DEFAULT_DRY_MULTIPLIER: f32 = 0.0;
pub const DEFAULT_DRY_BASE: f32 = 1.75;
pub const DEFAULT_DRY_ALLOWED_LENGTH: u32 = 3;

/// 2026-09-25: Thinking-loop watchdog parameters. When the model's
/// `WatchdogParams::enable_think_loop_watchdog` is set, `decode_logits_step`
/// checks every `THINK_LOOP_CHECK_STRIDE` thinking tokens once
/// `THINK_LOOP_MIN_TOKENS` have been spent inside `<think>`: if the output
/// ends in a pattern of `THINK_LOOP_PERIOD_MIN..=THINK_LOOP_PERIOD_MAX`
/// tokens repeated back to back `WatchdogParams::think_loop_min_repeats`
/// times (3 when MODEL.toml leaves it unset), unless the request overrides
/// these, it sets `force_end_thinking`.
pub const THINK_LOOP_MIN_TOKENS: u32 = 48;
pub const THINK_LOOP_CHECK_STRIDE: u32 = 8;
pub const THINK_LOOP_PERIOD_MIN: usize = 4;
pub const THINK_LOOP_PERIOD_MAX: usize = 20;
pub const THINK_LOOP_MIN_REPEATS: usize = 3;
/// 2026-09-25: Carried as `WatchdogParams::think_loop_scan_window`, but
/// `detect_token_loop` does not read it (`_scan_window`).
pub const THINK_LOOP_SCAN_WINDOW: usize = 160;

/// 2026-09-25: Content-loop watchdog parameters. Outside a tool body, every
/// `CONTENT_LOOP_CHECK_STRIDE` content tokens once `CONTENT_LOOP_MIN_TOKENS`
/// have been emitted, the content-phase handlers (`decode_logits_content.rs`,
/// `emit_step/token.rs`) check whether the output ends in a pattern of
/// `CONTENT_LOOP_PERIOD_MIN..=CONTENT_LOOP_PERIOD_MAX` tokens repeated
/// `CONTENT_LOOP_MIN_REPEATS` times back to back, unless the request or
/// operator overrides these; a hit rolls back or ends the response.
///
/// Armed by `SchedLevers::loop_watchdog()`, which serve sets from
/// `--content-loop-watchdog`, then `METRALE_CONTENT_LOOP_WATCHDOG`, then
/// MODEL.toml `[behavior].enable_loop_watchdog` (false when unset); the
/// dashboard's `/watchdog` command can change it mid-run.
pub const CONTENT_LOOP_MIN_TOKENS: u32 = 48;
pub const CONTENT_LOOP_CHECK_STRIDE: u32 = 16;
pub const CONTENT_LOOP_PERIOD_MIN: usize = 2;
pub const CONTENT_LOOP_PERIOD_MAX: usize = 64;
pub const CONTENT_LOOP_MIN_REPEATS: usize = 3;
pub const CONTENT_LOOP_SCAN_WINDOW: usize = 280;
/// 2026-09-25: Repeat threshold of the digit-normalized content-loop
/// detector when the request sets none. It is one more than
/// `CONTENT_LOOP_MIN_REPEATS`, so a three-item numbered list, whose lines
/// normalize to the same pattern, does not fire.
pub const CONTENT_LOOP_NORM_MIN_REPEATS: usize = 4;
/// 2026-09-25: Stands in for each run of numeric tokens in the normalized
/// tail (`detect_content_token_loop_normalized_with`). It must not be a
/// real token id.
pub const NUMERIC_SENTINEL: u32 = u32::MAX;

/// 2026-09-25: Parse of `METRALE_DISABLE_WATCHDOGS`: true for `1` or `true`
/// (trimmed, case-insensitive), false otherwise. `SchedLevers` resolves
/// `disable_watchdogs` with it.
pub(crate) fn parse_disable_watchdogs(env: Option<&str>) -> bool {
    match env {
        Some(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true")
        }
        None => false,
    }
}

/// 2026-09-25: Parse of `METRALE_DISABLE_FORCED_TOKEN` into "forced-token
/// fast path enabled": unset → true; `1` or `true` (trimmed,
/// case-insensitive) → false; anything else → true.
///
/// `SchedLevers::forced_token_fastpath` holds the result, and
/// `ForcedTokenFastPath` (`logit_processors/forced_token.rs`) reads it:
/// when the grammar admits exactly one next token, that token is emitted
/// without sampling.
pub(crate) fn parse_forced_token_fastpath(env: Option<&str>) -> bool {
    match env {
        Some(v) => {
            let v = v.trim();
            !(v == "1" || v.eq_ignore_ascii_case("true"))
        }
        None => true,
    }
}

/// 2026-09-25: The default-on rule: unset → true; `0` or `false` (trimmed,
/// case-insensitive) → false; anything else → true. `SchedLevers` resolves
/// `tool_response_stop`, `tool_eos_escape` and `grammar_budget_close` with
/// it.
pub(crate) fn parse_flag_default_on(env: Option<&str>) -> bool {
    match env.map(str::trim) {
        Some(v) => !(v == "0" || v.eq_ignore_ascii_case("false")),
        None => true,
    }
}

mod detectors;
mod watchdog;
pub use detectors::*;
pub use watchdog::*;

#[cfg(test)]
mod content_loop_override_tests;
#[cfg(test)]
mod hard_limit_tests;
#[cfg(test)]
mod inter_tool_prose_tests;
#[cfg(test)]
#[path = "helpers_tests.rs"]
mod thinking_loop_tests;
