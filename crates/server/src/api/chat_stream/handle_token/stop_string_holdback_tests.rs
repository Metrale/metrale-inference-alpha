// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `apply_stop_string_holdback`.
//!
//! Owner: server streaming API.
//! Invariants: none beyond the types.

use super::apply_stop_string_holdback;

/// 2026-09-26: A stop string split across two chunks is never sent in part; once it
/// completes, the text before it is sent and the stop string is consumed.
#[test]
fn stop_string_spanning_chunk_boundary_does_not_leak() {
    let stops = vec!["<|im_start|>".to_string()];
    let buffer_len = "<|im_start|>".len() - 1;
    let mut acc = String::new();
    let mut emitted = 0usize;
    let mut triggered = false;

    // 2026-09-26: 6 bytes, all inside the 11-byte hold-back.
    let out = apply_stop_string_holdback(
        "hello ",
        &stops,
        buffer_len,
        &mut acc,
        &mut emitted,
        &mut triggered,
    );
    assert_eq!(out, "");
    assert_eq!(acc, "hello ");
    assert!(!triggered);

    // 2026-09-26: The accumulator is 13 bytes; 13 - 11 = 2, so "he" goes out and the
    // partial stop string stays held back.
    let out = apply_stop_string_holdback(
        "<|im_st",
        &stops,
        buffer_len,
        &mut acc,
        &mut emitted,
        &mut triggered,
    );
    assert_eq!(out, "he");
    assert!(!out.contains("<|im_st"), "partial stop leaked to client");
    assert!(!triggered);

    // 2026-09-26: The stop string completes at byte 6: the accumulator is cut to
    // "hello " and bytes 2..6 go out.
    let out = apply_stop_string_holdback(
        "art|>",
        &stops,
        buffer_len,
        &mut acc,
        &mut emitted,
        &mut triggered,
    );
    assert_eq!(out, "llo ");
    assert_eq!(acc, "hello ");
    assert!(triggered);

    let total = String::new() + "" + "he" + "llo ";
    assert_eq!(total, "hello ");
    assert!(!total.contains("<|im_st"));
    assert!(!total.contains("<|im_start|>"));
}

/// 2026-09-26: With no stop strings, `buffer_len` is 0 and every delta passes through
/// at once.
#[test]
fn no_stop_strings_is_zero_behavior_change() {
    let stops: Vec<String> = Vec::new();
    let buffer_len = 0usize;
    let mut acc = String::new();
    let mut emitted = 0usize;
    let mut triggered = false;

    let out = apply_stop_string_holdback(
        "hello ",
        &stops,
        buffer_len,
        &mut acc,
        &mut emitted,
        &mut triggered,
    );
    assert_eq!(out, "hello ");
    assert!(!triggered);

    let out = apply_stop_string_holdback(
        "world",
        &stops,
        buffer_len,
        &mut acc,
        &mut emitted,
        &mut triggered,
    );
    assert_eq!(out, "world");
    assert!(!triggered);

    let out = apply_stop_string_holdback(
        "<|im_start|>",
        &stops,
        buffer_len,
        &mut acc,
        &mut emitted,
        &mut triggered,
    );
    assert_eq!(out, "<|im_start|>");
    assert!(!triggered);
}

/// 2026-09-26: A hold-back cut inside a multibyte codepoint snaps down to its start.
#[test]
fn utf8_boundary_safety_in_holdback() {
    let stops = vec!["STOP".to_string()];
    let buffer_len = 3usize;
    let mut acc = String::new();
    let mut emitted = 0usize;
    let mut triggered = false;

    // 2026-09-26: "aébc" is 5 bytes; the cut at 5 - 3 = 2 falls inside 'é' (bytes 1..3)
    // and snaps to 1.
    let out = apply_stop_string_holdback(
        "aébc",
        &stops,
        buffer_len,
        &mut acc,
        &mut emitted,
        &mut triggered,
    );
    assert_eq!(out, "a");
    assert!(out.is_char_boundary(out.len()));
    assert!(!triggered);
}
