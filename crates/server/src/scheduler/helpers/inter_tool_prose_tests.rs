// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the inter-tool prose budget's resolution and
//! precedence.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::{WatchdogParams, resolve_max_inter_tool_prose};

#[test]
fn default_prose_budget_is_plan_friendly() {
    const {
        assert!(
            super::MAX_INTER_TOOL_PROSE >= 2048,
            "inter-tool prose budget must fit a typical plan/analysis turn"
        );
    };
}

#[test]
fn resolved_default_budget_is_plan_friendly() {
    // 2026-09-25: The decode paths compare against the value
    // `from_behavior` resolves (`sched.watchdog.max_inter_tool_prose`), not
    // the constant, so assert the resolved default too.
    let p = WatchdogParams::from_behavior(&metrale_kernels::ModelBehavior::default(), None, None);
    assert!(
        p.max_inter_tool_prose >= 2048,
        "resolved inter-tool prose budget must fit a plan/analysis turn \
         (got {}) — the build-time and lib defaults have drifted",
        p.max_inter_tool_prose
    );
}

#[test]
fn prose_budget_precedence_is_cli_env_toml() {
    assert_eq!(resolve_max_inter_tool_prose(384, None, None), 384);
    assert_eq!(resolve_max_inter_tool_prose(384, Some(8192), None), 8192);
    assert_eq!(
        resolve_max_inter_tool_prose(384, Some(8192), Some(4096)),
        4096
    );
    assert_eq!(resolve_max_inter_tool_prose(384, None, Some(4096)), 4096);
}

#[test]
fn prose_budget_zero_disables_instead_of_instant_firing() {
    assert_eq!(resolve_max_inter_tool_prose(0, None, None), u32::MAX);
    assert_eq!(resolve_max_inter_tool_prose(384, Some(0), None), u32::MAX);
    assert_eq!(
        resolve_max_inter_tool_prose(384, Some(8192), Some(0)),
        u32::MAX
    );
}
