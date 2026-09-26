// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `advance_envelope_streak`: parameter-value tokens
//! never trip the envelope cap, other tokens trip it one past
//! `MAX_TOOL_BODY_TOKENS`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
use super::tool_param::{MAX_TOOL_BODY_TOKENS, advance_envelope_streak};

#[test]
fn parameter_value_content_is_exempt_at_any_size() {
    let mut streak = 0u32;
    for _ in 0..6000 {
        let (s, exceeded) = advance_envelope_streak(true, streak);
        streak = s;
        assert!(
            !exceeded,
            "parameter-value content must never trip the envelope cap"
        );
    }
    assert_eq!(
        streak, 0,
        "value content must not advance the envelope streak"
    );
}

#[test]
fn never_closing_envelope_still_trips_cap() {
    let mut streak = 0u32;
    let mut tripped = false;
    for _ in 0..(MAX_TOOL_BODY_TOKENS + 5) {
        let (s, exceeded) = advance_envelope_streak(false, streak);
        streak = s;
        if exceeded {
            tripped = true;
            break;
        }
    }
    assert!(
        tripped,
        "a never-closing envelope emitting >cap non-value tokens must trip"
    );
    assert_eq!(
        streak,
        MAX_TOOL_BODY_TOKENS + 1,
        "fires exactly one token past the cap"
    );
}

#[test]
fn exact_cap_boundary() {
    assert_eq!(
        advance_envelope_streak(false, MAX_TOOL_BODY_TOKENS - 1),
        (MAX_TOOL_BODY_TOKENS, false)
    );
    assert_eq!(
        advance_envelope_streak(false, MAX_TOOL_BODY_TOKENS),
        (MAX_TOOL_BODY_TOKENS + 1, true)
    );
}

#[test]
fn saturates_without_panic() {
    let (s, exceeded) = advance_envelope_streak(false, u32::MAX);
    assert_eq!(s, u32::MAX);
    assert!(exceeded);
}
