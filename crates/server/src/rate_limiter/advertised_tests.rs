// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the limits the rate limiter advertises in its headers, as distinct from what it enforces.
//!
//! Owner: server rate limiter.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn an_unenforced_axis_is_not_advertised_as_a_limit_of_one() {
    // 2026-09-26: `tpm == 0` leaves the token axis unenforced, so its floored
    // burst of 1 must not be advertised.
    let cfg = RateLimitConfig {
        rpm: 3,
        tpm: 0,
        burst_rpm: 3,
        burst_tpm: 1,
    };
    let d = RateLimiter::with_config(cfg).admit("k", 8192);
    assert_eq!(
        d.requests.limit, 3,
        "the enforced axis reports its real limit"
    );
    assert!(
        d.tokens.limit > 1_000_000,
        "the unenforced axis reports effectively unlimited, not 1: {}",
        d.tokens.limit
    );
    assert!(
        d.tokens.remaining > 1_000_000,
        "remaining must agree with the advertised limit: {}",
        d.tokens.remaining
    );
}

#[test]
fn both_axes_report_their_real_limits_when_both_are_enforced() {
    let cfg = RateLimitConfig {
        rpm: 5,
        tpm: 900,
        burst_rpm: 5,
        burst_tpm: 900,
    };
    let d = RateLimiter::with_config(cfg).admit("k", 1);
    assert_eq!(d.requests.limit, 5);
    assert_eq!(d.tokens.limit, 900);
}
