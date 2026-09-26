// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for [`super`] (strict `METRALE_*` parsing).
//!
//! Owner: server config.
//! Invariants: every case calls `parse_min` or `RateLimitConfig::from_raw`
//! and none calls `set_var`, because the environment is process-global.

use super::parse_min;

#[test]
fn a_typo_in_the_rate_limit_is_refused_instead_of_disabling_the_limit() {
    let err = crate::rate_limiter::RateLimitConfig::from_raw(Some("1oo"), None, None, None)
        .expect_err("a malformed rate limit must not be accepted");
    assert!(
        err.contains("METRALE_RATE_LIMIT_RPM"),
        "the message must name the key the operator has to fix: {err}"
    );
    assert!(
        err.contains("1oo"),
        "the message must quote the value it rejected: {err}"
    );
    assert!(
        err.contains("fix:"),
        "a diagnostic without a fix is half of one: {err}"
    );
}

#[test]
fn the_old_silent_fallback_would_have_produced_an_unlimited_server() {
    // 2026-09-26: A lenient parse that defaults turns `1oo` into 0, which
    // disables the limit (`RateLimitConfig::is_enabled`); `from_raw` refuses it.
    let silently: u64 = "1oo".parse().ok().unwrap_or(0);
    assert_eq!(silently, 0, "0 is the value that means NO rate limit");
    assert!(crate::rate_limiter::RateLimitConfig::from_raw(Some("1oo"), None, None, None).is_err());
}

#[test]
fn a_valid_rate_limit_still_parses_and_burst_still_defaults_to_the_rate() {
    let cfg = crate::rate_limiter::RateLimitConfig::from_raw(Some("100"), Some("5000"), None, None)
        .expect("well-formed values must be accepted");
    assert_eq!(cfg.rpm, 100);
    assert_eq!(cfg.tpm, 5000);
    // 2026-09-26: An unset burst takes the sustained rate (`unwrap_or(rpm)` in
    // `from_raw`), which is why `parse_min` returns an `Option`.
    assert_eq!(cfg.burst_rpm, 100);
    assert_eq!(cfg.burst_tpm, 5000);
}

#[test]
fn every_rate_limit_key_is_checked_not_just_the_first() {
    for (i, name) in [
        "METRALE_RATE_LIMIT_RPM",
        "METRALE_RATE_LIMIT_TPM",
        "METRALE_RATE_LIMIT_BURST_RPM",
        "METRALE_RATE_LIMIT_BURST_TPM",
    ]
    .iter()
    .enumerate()
    {
        let mut raw: [Option<&str>; 4] = [None; 4];
        raw[i] = Some("nope");
        let err = crate::rate_limiter::RateLimitConfig::from_raw(raw[0], raw[1], raw[2], raw[3])
            .expect_err(&format!("{name} must be validated"));
        assert!(err.contains(name), "wrong key named for {name}: {err}");
    }
}

#[test]
fn unset_and_blank_mean_unset_not_an_error() {
    assert_eq!(parse_min::<u64>("K", None, 0, "m"), Ok(None));
    assert_eq!(parse_min::<u64>("K", Some(""), 0, "m"), Ok(None));
    assert_eq!(parse_min::<u64>("K", Some("   "), 0, "m"), Ok(None));
}

#[test]
fn surrounding_whitespace_is_tolerated() {
    assert_eq!(parse_min::<u64>("K", Some(" 42 "), 0, "m"), Ok(Some(42)));
}

#[test]
fn a_value_below_the_minimum_is_refused_and_the_minimum_is_named() {
    let err = parse_min::<usize>("METRALE_STORE_MAX_ENTRIES", Some("0"), 1, "entries kept")
        .expect_err("0 is below the stated minimum of 1");
    assert!(err.contains("METRALE_STORE_MAX_ENTRIES"), "{err}");
    assert!(err.contains(">= 1"), "must name the bound: {err}");
    assert!(err.contains("fix:"), "{err}");
}

#[test]
fn a_negative_value_is_refused_for_an_unsigned_setting() {
    assert!(parse_min::<u64>("METRALE_STORE_TTL_SECONDS", Some("-1"), 1, "seconds").is_err());
}

#[test]
fn a_duration_with_a_unit_suffix_is_refused_rather_than_read_as_the_default() {
    let err = parse_min::<u64>("METRALE_STORE_TTL_SECONDS", Some("1h"), 1, "seconds")
        .expect_err("`1h` is not a number of seconds");
    assert!(err.contains("1h"), "{err}");
    assert!(
        err.contains("whole number"),
        "must say what form is expected: {err}"
    );
}

#[test]
fn the_message_carries_what_why_and_fix() {
    let err = parse_min::<u64>("METRALE_X", Some("bad"), 0, "what X controls").unwrap_err();
    assert!(err.contains("METRALE_X=\"bad\""), "{err}");
    assert!(err.contains("why: what X controls"), "{err}");
    assert!(err.contains("fix:"), "{err}");
}
