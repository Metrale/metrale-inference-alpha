// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Row threshold and switches for the grouped-GEMM MoE decode arm.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - `moe_grouped_decode_for` is false at every width while
//!   `METRALE_NO_MOE_GROUPED_DECODE` is set, even when the arm is forced.

/// 2026-09-25: Minimum rows for the grouped-GEMM MoE decode arm. The SSM stack
/// (`qwen3_ssm::trait_decode_multi_seq`) and the attention layers' multi-seq FFN
/// both decide through `moe_grouped_decode_for`, so they share this width. The
/// arm reads each routed expert once for all rows but first sorts and permutes
/// the rows, a fixed cost that the threshold keeps off narrow batches.
pub fn moe_grouped_decode_min_rows() -> usize {
    16
}

/// 2026-09-25: False when `METRALE_NO_MOE_GROUPED_DECODE` is set to any value,
/// `0` included. Read once per process.
pub fn moe_grouped_decode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_MOE_GROUPED_DECODE").is_none())
}

/// 2026-09-25: `METRALE_MOE_GROUPED_DECODE=1` runs the arm below
/// `moe_grouped_decode_min_rows()` too, for measuring narrow widths. Read once
/// per process.
pub fn moe_grouped_decode_forced() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_MOE_GROUPED_DECODE").as_deref() == Ok("1"))
}

/// 2026-09-25: Whether the arm runs for `n` rows, given the two switches. It
/// reads no env, so tests do not depend on the latched `OnceLock`s above.
pub fn moe_grouped_decode_decide(n: usize, enabled: bool, forced: bool) -> bool {
    enabled && (n >= moe_grouped_decode_min_rows() || forced)
}

/// 2026-09-25: Whether the arm runs for `n` rows under this process's switches.
pub fn moe_grouped_decode_for(n: usize) -> bool {
    moe_grouped_decode_decide(n, moe_grouped_decode_enabled(), moe_grouped_decode_forced())
}

#[cfg(test)]
#[path = "moe_grouped_decode_tests.rs"]
mod moe_grouped_decode_tests;
