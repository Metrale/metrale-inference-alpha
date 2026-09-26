// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the pure hard-limit decision cores
//! `seqlen_force_stop`, `hard_ceiling_hit` and `eos_suppressed_by_thinking`.
//! Their call sites in the decode paths are not exercised here.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
use super::{eos_suppressed_by_thinking, hard_ceiling_hit, seqlen_force_stop};

#[test]
fn seqlen_guard_disabled_when_unset() {
    assert!(!seqlen_force_stop(0, 0));
    assert!(!seqlen_force_stop(8191, 0));
    assert!(!seqlen_force_stop(usize::MAX, 0));
}

#[test]
fn seqlen_guard_fires_one_token_before_ceiling() {
    assert!(!seqlen_force_stop(8190, 8192), "room for one more token");
    assert!(seqlen_force_stop(8191, 8192), "next token would be at 8191");
    assert!(seqlen_force_stop(8192, 8192), "already at the ceiling");
    assert!(
        seqlen_force_stop(9000, 8192),
        "past the ceiling never continues"
    );
}

#[test]
fn hard_ceiling_hit_on_budget_or_seqlen() {
    assert!(
        hard_ceiling_hit(0, 10, 8192),
        "remaining==0 is a hard ceiling"
    );
    assert!(
        hard_ceiling_hit(500, 8191, 8192),
        "seq-len ceiling is a hard ceiling"
    );
    assert!(
        !hard_ceiling_hit(500, 10, 8192),
        "budget + room left → no ceiling"
    );
    assert!(
        !hard_ceiling_hit(500, 10, 0),
        "max_seq_len unset → only budget matters"
    );
}

#[test]
fn eos_reachable_at_hard_ceiling_even_inside_thinking() {
    assert!(
        eos_suppressed_by_thinking(true, false),
        "inside thinking, no ceiling → suppress (unchanged baseline)"
    );
    assert!(
        !eos_suppressed_by_thinking(true, true),
        "inside thinking, hard ceiling → EOS must fire (the fix)"
    );
    assert!(
        !eos_suppressed_by_thinking(false, false),
        "outside thinking → never suppressed"
    );
}
