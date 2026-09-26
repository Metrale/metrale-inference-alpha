// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: `[behavior]` defaults that the build-time MODEL.toml parse and
// `ModelBehavior::default()` both read.
//
// Owner: kernels crate.
// Invariants:
// - `lib.rs` compiles this file as a module and `build_parse_behavior.rs`
//   `include!`s it, so both sides use the same constants. `//!` cannot appear
//   at an `include!` site, which is why this header uses `//`.

/// 2026-09-25: Default `[behavior].max_inter_tool_prose`: the cap on free-text
/// tokens between tool calls on a tool request. 0 is reserved: the runtime
/// resolver (`resolve_max_inter_tool_prose`) maps it to "guard disabled". A
/// model can pin a tighter bound in its MODEL.toml, as
/// `kernels/strix/qwen3.6-35b-a3b` does with 384.
pub const DEFAULT_MAX_INTER_TOOL_PROSE: u32 = 3072;

/// 2026-09-25: Default `[behavior].max_thinking_budget`: the anchor E of the
/// `reasoning_effort` ladder and the budget of a thinking-on request that
/// names none.
pub const DEFAULT_MAX_THINKING_BUDGET: u32 = 256;

/// 2026-09-25: Default `[behavior].effort_capped_at_ceiling`: whether the
/// qualitative `reasoning_effort` levels are clamped at E
/// (`max_thinking_budget`, or `--max-thinking-budget`).
///
/// `false` keeps high = 2E and xhigh = 4E, above the ceiling; at the default
/// E that is 512 and 1024, which
/// `effort_ladder_at_default_ceiling_matches_the_historical_absolutes` pins.
/// `true` clamps every level at E, for a model whose MODEL.toml records that
/// a larger budget scores worse (`kernels/gb10/qwen3.5-397b-a17b` sets it).
/// The clamp applies only to the effort ladder; an explicit client token
/// budget is not clamped.
pub const DEFAULT_EFFORT_CAPPED_AT_CEILING: bool = false;
