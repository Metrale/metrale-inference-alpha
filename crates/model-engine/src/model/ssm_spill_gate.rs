// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The spill-side cost gate: the minimum victim depth, in tokens, that the
//! SSM tier spills rather than drops (`METRALE_SSM_SPILL_MIN_TOKENS`).
//!
//! Owner: model-engine SSM snapshot pool.
//! Invariants:
//! - [`spill_min_tokens`] is 0 (gate disabled) or at least the fault-in gate
//!   `fault_in_min_tokens()` (`trait_impl/ssm_fault_in.rs`).

use std::sync::atomic::{AtomicBool, Ordering};

const SPILL_COST_MS: usize = 45;

/// 2026-09-25: Spill gate used when `METRALE_SSM_SPILL_MIN_TOKENS` is unset or does
/// not parse as a `usize`.
const DEFAULT_SPILL_MIN_TOKENS: usize = 1024;

/// 2026-09-25: Latch so the clamp warning is logged once per process.
static CLAMP_WARNED: AtomicBool = AtomicBool::new(false);

/// 2026-09-25: The effective spill gate: the parsed value clamped by
/// [`clamp_spill_to_fault`] against the fault-in gate, read from the env on every call.
pub(in crate::model) fn spill_min_tokens() -> usize {
    let raw = parse_spill_min_tokens(std::env::var("METRALE_SSM_SPILL_MIN_TOKENS").ok());
    let fault = super::trait_impl::ssm_fault_in::fault_in_min_tokens();
    let eff = clamp_spill_to_fault(raw, fault);
    if eff != raw && !CLAMP_WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            "METRALE_SSM_SPILL_MIN_TOKENS={raw} is below METRALE_SSM_FAULT_MIN_TOKENS={fault}; \
             clamping the spill gate to {eff}. Spilling a snapshot the fault-in gate would \
             then REFUSE to read back is a guaranteed pure loss — the spill cost is paid and \
             the benefit can never be claimed."
        );
    }
    eff
}

/// 2026-09-25: Parse `METRALE_SSM_SPILL_MIN_TOKENS`: unset or unparseable gives
/// [`DEFAULT_SPILL_MIN_TOKENS`]; `0` disables the gate.
pub(in crate::model) fn parse_spill_min_tokens(raw: Option<String>) -> usize {
    raw.and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SPILL_MIN_TOKENS)
}

/// 2026-09-25: `max(spill, fault)`, except that a spill gate of 0 (disabled) stays 0.
/// It clamps rather than returning an error.
pub(in crate::model) fn clamp_spill_to_fault(spill: usize, fault: usize) -> usize {
    if spill == 0 { 0 } else { spill.max(fault) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_spill_min_defaults_when_unset_or_garbage() {
        assert_eq!(parse_spill_min_tokens(None), DEFAULT_SPILL_MIN_TOKENS);
        assert_eq!(
            parse_spill_min_tokens(Some("twelve".into())),
            DEFAULT_SPILL_MIN_TOKENS
        );
        assert_eq!(
            parse_spill_min_tokens(Some("".into())),
            DEFAULT_SPILL_MIN_TOKENS
        );
    }

    #[test]
    fn parse_spill_min_parses_explicit_values() {
        assert_eq!(parse_spill_min_tokens(Some("0".into())), 0);
        assert_eq!(parse_spill_min_tokens(Some("2048".into())), 2048);
    }

    /// 2026-09-25: Never spill a snapshot the fault-in gate would refuse to read back.
    #[test]
    fn spill_min_clamped_to_fault_min() {
        assert_eq!(clamp_spill_to_fault(64, 256), 256);
        assert_eq!(clamp_spill_to_fault(256, 256), 256);
        assert_eq!(clamp_spill_to_fault(1024, 256), 1024);
        assert_eq!(clamp_spill_to_fault(0, 256), 0);
    }

    /// 2026-09-25: The default spill gate is already at or above the default fault-in
    /// gate.
    #[test]
    fn shipped_defaults_satisfy_the_invariant() {
        let fault = super::super::trait_impl::ssm_fault_in::DEFAULT_FAULT_MIN_TOKENS;
        assert_eq!(
            clamp_spill_to_fault(DEFAULT_SPILL_MIN_TOKENS, fault),
            DEFAULT_SPILL_MIN_TOKENS,
            "DEFAULT_SPILL_MIN_TOKENS must already be >= the fault-in gate"
        );
        // 2026-09-25: A const assert, since a runtime assert on a constant trips
        // `clippy::assertions_on_constants`.
        const _: () = assert!(SPILL_COST_MS > 0);
    }
}
