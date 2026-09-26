// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the dashboard's number, enum and text-wrap formatting in `format.rs`.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: `data::library::human_size` prints the same text as `bytes`, and `GB` is 1024³.
#[test]
fn one_file_is_one_size_wherever_it_is_shown() {
    assert_eq!(bytes(20_000_000_000), "18.6 GB");
    assert_eq!(
        bytes(20_000_000_000),
        crate::tui::data::library::human_size(20_000_000_000),
    );
}

#[test]
fn the_unit_turns_over_at_a_gibibyte_not_a_gigabyte() {
    assert_eq!(bytes(1024 * 1024 * 1024 - 1), "1023 MB");
    assert_eq!(bytes(1024 * 1024 * 1024), "1.0 GB");
    assert_eq!(bytes(1_000_000_000), "953 MB");
}

#[test]
fn zero_is_a_size_not_an_error() {
    assert_eq!(bytes(0), "0 MB");
}

#[test]
fn a_rate_divides_into_the_size_printed_beside_it() {
    assert_eq!(rate(1024.0 * 1024.0 * 1024.0), "1.0 GB/s");
    assert_eq!(bytes(1024 * 1024 * 1024), "1.0 GB");
    assert_eq!(rate(96_000_000.0), "92 MB/s");
    // 2026-09-26: `bytes` truncates below a gibibyte and `rate` rounds, so the last digit differs.
    assert_eq!(bytes(96_000_000), "91 MB");
}

#[test]
fn a_rate_says_what_unit_it_is_in() {
    assert_eq!(rate(3_145_728.0), "3.0 MB/s");
    assert_eq!(rate(2048.0), "2.0 KB/s");
    for r in [0.0, 512.0, 2048.0, 3e6, 4e9] {
        assert!(rate(r).ends_with("B/s"), "{r} rendered as {}", rate(r));
    }
}

#[test]
fn a_rate_turns_over_on_the_same_boundaries_as_a_size() {
    assert_eq!(rate(1023.0), "1023 B/s");
    assert_eq!(rate(1024.0), "1.0 KB/s");
    // 2026-09-26: Unlike `bytes`, `rate` rounds, so one byte short of a MiB reads `1024 KB/s`.
    assert_eq!(rate(1024.0 * 1024.0 - 1.0), "1024 KB/s");
    assert_eq!(rate(1024.0 * 1024.0), "1.0 MB/s");
    assert_eq!(rate(1024.0 * 1024.0 * 1024.0), "1.0 GB/s");
    assert_eq!(rate(1_000_000.0), "977 KB/s");
}

#[test]
fn the_decimal_appears_only_where_it_says_something() {
    assert_eq!(rate(9.9 * 1024.0 * 1024.0), "9.9 MB/s");
    assert_eq!(rate(10.4 * 1024.0 * 1024.0), "10 MB/s");
}

#[test]
fn an_idle_or_unmeasurable_rate_is_zero_rather_than_nonsense() {
    assert_eq!(rate(0.0), "0 B/s");
    assert_eq!(rate(-1.0), "0 B/s");
    assert_eq!(rate(f64::NAN), "0 B/s");
}

#[test]
fn every_mtp_state_reads_as_english_not_as_a_variant_name() {
    for (mode, want) in [
        (MtpModeSnap::Mtp, "speculative"),
        (MtpModeSnap::Serial, "serial"),
        (MtpModeSnap::Probing, "probing"),
        (MtpModeSnap::Off, "off"),
    ] {
        assert_eq!(mtp_mode_label(mode), want);
        assert_ne!(
            mtp_mode_label(mode),
            format!("{mode:?}"),
            "a label that equals the Debug output has not been written yet"
        );
    }
}

#[test]
fn wrap_help_keeps_paragraph_breaks_that_wrap_words_flattens() {
    let text =
        "First paragraph here.\n\nSecond paragraph, which is deliberately long enough to wrap.";
    let wrapped = wrap_help(text, 20);
    assert_eq!(wrapped[0], "First paragraph");
    assert!(
        wrapped.contains(&String::new()),
        "the blank line survives: {wrapped:?}"
    );
    assert!(
        wrap_words(text, 20).iter().all(|l| !l.is_empty()),
        "wrap_words flattens whitespace, which is its contract"
    );
    assert!(!wrapped.last().unwrap().is_empty());
}

#[test]
fn wrap_words_hard_splits_a_token_wider_than_the_pane() {
    let lines = wrap_words("see https://example.com/very/long/path/indeed", 10);
    assert!(lines.iter().all(|l| l.len() <= 10), "{lines:?}");
    assert_eq!(
        lines.concat().replace(' ', ""),
        "seehttps://example.com/very/long/path/indeed".replace(' ', "")
    );
}
