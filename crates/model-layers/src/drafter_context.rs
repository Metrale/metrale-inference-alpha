// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Resolution of the MTP drafter context from the environment: whether
//! the drafter KV is batch-prefilled over the prompt (`prefill`) and whether it is
//! carried across the turns of a session (`carry`).
//!
//! Carry depends on prefill. In the model's `trait_impl/speculative.rs` a carried
//! drafter KV is adopted (`try_carry_drafter`) only inside
//! `if !self.mtp_prefill_hidden.is_null()`, and the model allocates that capture
//! only when `drafter.prefill` is set (`layers::mtp_drafter_prefill_enabled`), so
//! the two are resolved together.
//! A configured carry is armed only in single-sequence MTP
//! (`mtp_carry::carry_armed`).
//!
//! | variable (exactly `1`) | result |
//! |---|---|
//! | neither | `DrafterContext::BOTH` |
//! | `METRALE_NO_MTP_DRAFTER_CONTEXT` | `DrafterContext::OFF` |
//! | `METRALE_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE` | prefill on, carry off; warned at startup |
//!
//! Any other value, `0` included, changes nothing.
//!
//! Owner: model-layers (MTP drafter).
//! Invariants:
//! - [`resolve`] never returns `carry` without `prefill`.
//! - The kill switch wins over the prefill-only variable.

/// 2026-09-25: Kill switch for both halves; only the exact value `1` counts.
pub const DISABLE_ENV: &str = "METRALE_NO_MTP_DRAFTER_CONTEXT";

/// 2026-09-25: Prefill without carry; only the exact value `1` counts, and
/// [`DISABLE_ENV`] wins over it.
pub const PREFILL_ONLY_ENV: &str = "METRALE_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE";

/// 2026-09-25: Which halves of the drafter context are configured. [`resolve`]
/// never returns `carry` without `prefill`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrafterContext {
    /// 2026-09-25: Capture prompt hiddens and batch-prefill the drafter KV.
    pub prefill: bool,
    /// 2026-09-25: Carry the drafter KV across turns of a session; it runs only
    /// when `mtp_carry::carry_armed` also holds.
    pub carry: bool,
}

impl Default for DrafterContext {
    /// 2026-09-25: [`BOTH`](DrafterContext::BOTH). Written out because a derived
    /// `Default` would be [`OFF`](DrafterContext::OFF).
    fn default() -> Self {
        Self::BOTH
    }
}

impl DrafterContext {
    /// 2026-09-25: Both halves on: what [`resolve`] returns when neither variable is `1`.
    pub const BOTH: Self = Self {
        prefill: true,
        carry: true,
    };
    /// 2026-09-25: Both halves off: what [`resolve`] returns for the kill switch.
    pub const OFF: Self = Self {
        prefill: false,
        carry: false,
    };
}

/// 2026-09-25: Resolve the two variables' values with no environment access, so
/// the rule is unit-tested (SBIO). `disable` wins over `prefill_only`.
pub fn resolve(disable: Option<&str>, prefill_only: Option<&str>) -> DrafterContext {
    if disable == Some("1") {
        return DrafterContext::OFF;
    }
    if prefill_only == Some("1") {
        // 2026-09-25: The only branch that returns prefill without carry.
        return DrafterContext {
            prefill: true,
            carry: false,
        };
    }
    DrafterContext::BOTH
}

/// 2026-09-25: Resolve the policy from the environment. `ModelLevers::from_env`
/// calls it, and the result is `ModelLevers::drafter`. The outcome is logged once
/// per process (`REPORTED`).
pub fn resolve_from_env() -> DrafterContext {
    {
        let disable = std::env::var(DISABLE_ENV).ok();
        let prefill_only = std::env::var(PREFILL_ONLY_ENV).ok();
        let cfg = resolve(disable.as_deref(), prefill_only.as_deref());

        REPORTED.call_once(|| report(cfg));
        cfg
    }
}

/// 2026-09-25: Makes `report` run once per process, on the first [`resolve_from_env`].
static REPORTED: std::sync::Once = std::sync::Once::new();

fn report(cfg: DrafterContext) {
    // 2026-09-25: Report the armed carry (`mtp_carry::carry_armed`, the predicate
    // the runtime gates on), not the configured one: the MTP dispatch cap can
    // force the carry off under this config.
    let carry_armed = crate::mtp_carry::carry_armed(cfg);
    tracing::info!(
        "MTP drafter context: prefill={} carry={} ({}). Disable both with {}=1.",
        on_off(cfg.prefill),
        on_off(carry_armed),
        if cfg == DrafterContext::BOTH {
            "default"
        } else {
            "overridden by environment"
        },
        DISABLE_ENV,
    );
    if cfg.carry && !carry_armed {
        tracing::warn!(
            "MTP cross-turn carry is CONFIGURED ON but INERT: the MTP \
             dispatch cap is {} (>1), and the carry slot is single-sequence \
             by design, so it is force-disabled. Set METRALE_MTP_MAX_SEQS=1 \
             to arm it; leave it unset to keep multi-sequence MTP.",
            crate::speculative::mtp_max_seqs(),
        );
    }
    if cfg.prefill && !cfg.carry {
        tracing::warn!(
            "{PREFILL_ONLY_ENV}=1: drafter prefill is ON with cross-turn carry \
             OFF. This is a MEASUREMENT ARM, not a deployment — a warm turn \
             rebuilds the drafter for a measured 1136 ms against ~211 ms/turn \
             of decode saving (net -927 ms/turn, spent on TTFT).",
        );
    }
}

fn on_off(b: bool) -> &'static str {
    if b { "ON" } else { "OFF" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_environment_enables_both() {
        assert_eq!(resolve(None, None), DrafterContext::BOTH);
    }

    #[test]
    fn kill_switch_disables_both() {
        assert_eq!(resolve(Some("1"), None), DrafterContext::OFF);
    }

    #[test]
    fn prefill_only_arm_disables_carry_only() {
        assert_eq!(
            resolve(None, Some("1")),
            DrafterContext {
                prefill: true,
                carry: false
            }
        );
    }

    #[test]
    fn kill_switch_beats_the_research_arm() {
        assert_eq!(resolve(Some("1"), Some("1")), DrafterContext::OFF);
    }

    #[test]
    fn only_exactly_one_switches_anything() {
        for v in ["0", "", "true", "yes", "2", "1 "] {
            assert_eq!(
                resolve(Some(v), None),
                DrafterContext::BOTH,
                "{DISABLE_ENV}={v:?} must not disable",
            );
            assert_eq!(
                resolve(None, Some(v)),
                DrafterContext::BOTH,
                "{PREFILL_ONLY_ENV}={v:?} must not change anything",
            );
        }
    }
}
