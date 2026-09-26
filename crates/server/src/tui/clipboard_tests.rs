// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the OSC 52 encoding, the size limit and `copy`.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn the_sequence_is_a_well_formed_osc52() {
    let seq = osc52("hi").expect("non-empty text encodes");
    let s = String::from_utf8(seq).expect("ascii escape sequence");
    assert!(s.starts_with("\x1b]52;c;"), "{s:?}");
    assert!(s.ends_with('\x07'), "{s:?}");
    assert!(s.contains("aGk="), "base64 of 'hi': {s:?}");
}

#[test]
fn utf8_survives_the_encoding() {
    let text = "Qwen3.6-35B — ✓ 0.8% ▓░";
    let s = String::from_utf8(osc52(text).unwrap()).unwrap();
    let b64 = s.trim_start_matches("\x1b]52;c;").trim_end_matches('\x07');
    let back = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .unwrap();
    assert_eq!(String::from_utf8(back).unwrap(), text);
}

#[test]
fn empty_text_is_not_a_copy() {
    assert!(osc52("").is_none());
    assert!(
        copy("").is_err(),
        "an empty selection must not claim success"
    );
}

#[test]
fn an_oversized_selection_is_refused_rather_than_truncated() {
    let huge = "x".repeat(MAX_BYTES);
    assert!(
        too_large(&huge),
        "should exceed the limit once base64-expanded"
    );
    assert!(osc52(&huge).is_none());
    let err = copy(&huge).unwrap_err();
    assert!(err.contains("too large"), "{err}");
}

#[test]
fn a_selection_just_under_the_limit_is_accepted() {
    let ok = "y".repeat(MAX_BYTES / 2);
    assert!(!too_large(&ok));
    assert!(osc52(&ok).is_some());
}

#[test]
fn the_limit_bites_on_encoded_bytes_not_on_characters() {
    // 2026-09-26: The cap is on base64 length, so a 3-byte character counts
    // the same as three ASCII bytes.
    let ascii = MAX_BYTES / 4 * 3;
    assert!(
        !too_large(&"a".repeat(ascii)),
        "the largest ASCII that fits"
    );
    assert!(too_large(&"a".repeat(ascii + 3)), "one base64 group over");
    assert!(too_large(&"日".repeat(ascii / 3 + 1)));
}

#[test]
fn an_empty_selection_is_not_reported_as_too_large() {
    assert!(!too_large(""));
}

#[test]
fn a_successful_copy_reports_characters_not_bytes() {
    // 2026-09-26: The only test in this file that writes the escape to stdout.
    // `write_raw` uses `std::io::stdout()` directly, which the test harness
    // does not capture, so a suite run in a terminal sets its clipboard.
    assert_eq!(
        copy("日本語のテキスト").expect("a normal selection copies"),
        8
    );
}

#[test]
fn an_oversized_copy_explains_itself_in_characters() {
    let err = copy(&"日".repeat(MAX_BYTES)).unwrap_err();
    assert!(err.contains(&format!("{MAX_BYTES} chars")), "{err}");
}

#[test]
fn control_characters_in_the_selection_are_encoded_not_forwarded() {
    let seq = osc52("\x1b]0;pwned\x07 rest").expect("encodes");
    let payload = &seq[b"\x1b]52;c;".len()..seq.len() - 1];
    assert!(
        !payload.contains(&0x1b) && !payload.contains(&0x07),
        "no bare ESC or BEL survives into the payload"
    );
    let back = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .expect("valid base64");
    assert_eq!(
        String::from_utf8(back).expect("utf8"),
        "\x1b]0;pwned\x07 rest"
    );
}

#[test]
fn a_multi_line_selection_is_one_sequence_with_no_embedded_newlines() {
    // 2026-09-26: The `STANDARD` base64 engine does not wrap lines.
    let seq = osc52("line one\nline two\r\nline three").expect("encodes");
    assert_eq!(
        seq.iter().filter(|b| **b == b'\n' || **b == b'\r').count(),
        0
    );
}
