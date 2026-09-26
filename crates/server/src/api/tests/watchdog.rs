// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Loop-watchdog cases that need a long stream: a repeat separated by
//! kilobytes of unique text, and a repeat whose last copy has more text on its line.
//! `stream_guards`' inline tests cover an already-triggered call, empty text, four
//! identical lines and the buffer trim.
//!
//! Owner: server streaming API.
//! Invariants: none beyond the types.

use crate::api::stream_guards::check_loop_watchdog;

const PHRASE: &str = "I'll create the project files and verify everything works:";

/// 2026-09-26: Feed `text` in slices of `chunk` bytes (each extended to the next char
/// boundary) and report whether the watchdog fired at any point.
fn fires(text: &str, chunk: usize) -> bool {
    let mut scan = String::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + chunk).min(text.len());
        while !text.is_char_boundary(end) {
            end += 1;
        }
        if check_loop_watchdog(&text[start..end], &mut scan, false) {
            return true;
        }
        start = end;
    }
    false
}

/// 2026-09-26: `n` copies of `PHRASE`, each followed by 40 filler lines that name the
/// copy's index, so only the phrase repeats.
fn phrase_with_unique_interstitials(n: usize) -> String {
    let mut feed = String::new();
    for i in 0..n {
        feed.push_str(PHRASE);
        feed.push('\n');
        for j in 0..40 {
            feed.push_str(&format!(
                "segment {i}.{j} of the source dump under review\n"
            ));
        }
    }
    feed
}

/// 2026-09-26: Chunk sizes to sweep. The watchdog takes its needle from the newest line in
/// its buffer, so the result can depend on where chunk boundaries land.
const TOKEN_SIZES: std::ops::RangeInclusive<usize> = 1..=16;

#[test]
fn a_repeat_separated_by_kilobytes_of_unique_prose_still_fires() {
    // 2026-09-26: Four copies of the phrase span about 5.6 KB, which the buffer still
    // holds: it is trimmed only above 10,240 bytes. The filler is unique to each copy,
    // so only the phrase can fire the watchdog.
    let feed = phrase_with_unique_interstitials(5);
    assert!(
        feed.len() > 3_000,
        "the interstitials must exceed the old 3 KB window to be a regression test; got {}",
        feed.len()
    );
    for chunk in TOKEN_SIZES {
        assert!(
            fires(&feed, chunk),
            "5 repeats across large unique interstitials must fire at chunk size {chunk}"
        );
    }
}

#[test]
fn three_repeats_do_not_fire() {
    // 2026-09-26: The watchdog fires at 4 occurrences (`check_loop_watchdog`).
    let feed = phrase_with_unique_interstitials(3);
    for chunk in TOKEN_SIZES {
        assert!(
            !fires(&feed, chunk),
            "three repeats are below the threshold and must not stop the stream \
             (chunk size {chunk})"
        );
    }
}

#[test]
fn a_repeat_whose_last_instance_starts_mid_line_still_fires() {
    // 2026-09-26: The fourth copy has more text after it on the same line, so the
    // finished line does not match the first three; the watchdog has to fire while
    // that line is still arriving.
    let mut feed = String::new();
    for _ in 0..3 {
        feed.push_str(PHRASE);
        feed.push('\n');
        feed.push_str("intermediate prose\n");
    }
    feed.push_str(PHRASE);
    feed.push_str("        let body = vec![];");
    for chunk in TOKEN_SIZES {
        assert!(
            fires(&feed, chunk),
            "substring scan must catch a repeated phrase whose last instance \
             is mid-line (chunk size {chunk})"
        );
    }
}

#[test]
fn ordinary_varied_prose_never_fires() {
    let mut feed = String::new();
    for i in 0..200 {
        feed.push_str(&format!(
            "Step {i}: inspect the {i}th module and record what it exports.\n"
        ));
    }
    assert!(!fires(&feed, 64), "varied prose must not trip the watchdog");
}

#[test]
fn a_repeating_short_line_does_not_fire() {
    // 2026-09-26: Lines of 15 bytes or fewer (trimmed) are never the needle, so repeated
    // short code lines such as `}` do not fire.
    let feed = "}\n".repeat(50);
    assert!(!fires(&feed, 8), "short repeated lines must be ignored");
}
