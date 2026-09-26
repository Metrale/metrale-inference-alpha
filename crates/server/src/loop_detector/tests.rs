// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for [`super`], the loop detector.
//!
//! Owner: server chat API.
//! Invariants: none beyond the types.

use super::*;

fn sig_text_only(text: &str) -> Signature {
    Signature::build(text, std::iter::empty())
}

fn sig_with_tool(text: &str, name: &str, args: &str) -> Signature {
    Signature::build(text, std::iter::once((name, args)))
}

#[test]
fn empty_recent_returns_none() {
    assert_eq!(detect(&[]), LoopState::None);
    assert_eq!(
        detect(&[sig_text_only("hello world how are you doing today")]),
        LoopState::None
    );
}

#[test]
fn three_distinct_messages_no_loop() {
    let recent = vec![
        sig_text_only("the quick brown fox jumps over the lazy dog"),
        sig_text_only("once upon a time there was a small village"),
        sig_text_only("rust is a systems programming language designed for safety"),
    ];
    assert_eq!(detect(&recent), LoopState::None);
}

#[test]
fn three_identical_intros_fire_loop() {
    let intro = "I will create a proper echo server for you. Let me write \
                     a clean implementation with axum and serde.";
    let recent = vec![
        sig_text_only(intro),
        sig_text_only(intro),
        sig_text_only(intro),
    ];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Suppress { .. } | LoopState::Hint { .. }),
        "got {v:?}"
    );
}

#[test]
fn identical_tool_args_fire_loop() {
    let cmd = r#"{"command":"mkdir -p /tmp/axum-test-5 && cd /tmp/axum-test-5 && cargo init --name axum_echo_server"}"#;
    let recent = vec![
        sig_with_tool("", "Bash", cmd),
        sig_with_tool("", "Bash", cmd),
        sig_with_tool("", "Bash", cmd),
        sig_with_tool("", "Bash", cmd),
    ];
    let v = detect(&recent);
    assert!(matches!(v, LoopState::Suppress { .. }), "got {v:?}");
}

#[test]
fn different_tool_args_with_same_name_do_not_fire() {
    let recent = vec![
        sig_with_tool("", "Bash", r#"{"command":"ls /a"}"#),
        sig_with_tool("", "Bash", r#"{"command":"echo hi"}"#),
        sig_with_tool("", "Bash", r#"{"command":"pwd"}"#),
    ];
    let v = detect(&recent);
    assert_eq!(v, LoopState::None, "got {v:?}");
}

#[test]
fn slightly_varied_intros_still_fire() {
    let a = "I'll create a proper echo server using axum and serde with full tests.";
    let b = "I'll create a proper echo server using axum and serde with passing tests.";
    let c = "I'll create a proper echo server using axum and serde, including tests.";
    let recent = vec![sig_text_only(a), sig_text_only(b), sig_text_only(c)];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Hint { .. } | LoopState::Suppress { .. }),
        "near-identical paraphrases must trigger detection: {v:?}"
    );
}

#[test]
fn moderate_similarity_3_turns_now_suppress() {
    let common = "Let me create the necessary files for the Axum server with an echo endpoint and passing tests in the project directory now";
    let recent = vec![
        sig_text_only(&format!("{common} immediately")),
        sig_text_only(&format!("{common} carefully")),
        sig_text_only(&format!("{common} step by step")),
        sig_text_only(&format!("{common} as planned")),
    ];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Suppress { .. }),
        "post-recalibration: 4 turns sharing the same long prose stem must Suppress: {v:?}"
    );
}

#[test]
fn moderate_band_with_065_max_suppress() {
    let s = "the quick brown fox jumps over the lazy dog and chases the mouse around the garden";
    let recent = vec![
        sig_text_only(&format!("{s} alpha")),
        sig_text_only(&format!("{s} bravo")),
        sig_text_only(&format!("{s} charlie")),
    ];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Suppress { .. }),
        "highly-similar 3-turn run must Suppress post-recalibration: {v:?}"
    );
}

#[test]
fn empty_signatures_are_skipped_not_counted() {
    let intro =
        "I will create a proper echo server for you. Let me write a clean implementation now.";
    let recent = vec![
        sig_text_only(intro),
        sig_text_only(""),
        sig_text_only(intro),
        sig_text_only(intro),
    ];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Suppress { .. } | LoopState::Hint { .. }),
        "got {v:?}"
    );
}

#[test]
fn one_off_repeat_does_not_trigger_suppress() {
    let intro =
        "I will create a proper echo server for you. Let me write a clean implementation now.";
    let recent = vec![sig_text_only(intro), sig_text_only(intro)];
    let v = detect(&recent);
    assert_eq!(v, LoopState::None, "two-turn repeat is not yet a loop");
}

#[test]
fn p1_5_three_identical_failing_short_calls_detected_without_suppress() {
    let sig = sig_with_tool("", "write", r#"{"p":""}"#);
    assert!(
        sig.is_empty(),
        "short call must be below MIN_CHANNEL_TOKENS for this test to be meaningful"
    );
    let sigs = vec![sig.clone(), sig.clone(), sig];
    assert_eq!(
        detect(&sigs),
        LoopState::None,
        "legacy detect() must stay blind (⇒ suppress NOT set via Suppress verdict)"
    );

    let turn = CallOutcome {
        call_unit: Some("write\u{1f}{\"p\":\"\"}".to_string()),
        failing: true,
        result_unit: None,
    };
    let turns = vec![turn.clone(), turn.clone(), turn];
    assert_eq!(
        detect_exact_failing_repeat(&turns),
        Some(3),
        "fast path must fire on 3 byte-identical failing calls"
    );
}

#[test]
fn p1_5_three_identical_succeeding_short_calls_unchanged_legacy() {
    let turn = CallOutcome {
        call_unit: Some("write\u{1f}{\"p\":\"\"}".to_string()),
        failing: false,
        result_unit: None,
    };
    let turns = vec![turn.clone(), turn.clone(), turn];
    assert_eq!(detect_exact_failing_repeat(&turns), None);

    let sig = sig_with_tool("", "write", r#"{"p":""}"#);
    assert_eq!(
        detect(&[sig.clone(), sig.clone(), sig]),
        LoopState::None,
        "legacy behavior unchanged for short succeeding repeats"
    );
}

#[test]
fn p1_5_two_identical_failing_calls_not_enough() {
    let turn = CallOutcome {
        call_unit: Some("x\u{1f}{}".to_string()),
        failing: true,
        result_unit: None,
    };
    assert_eq!(detect_exact_failing_repeat(&[turn.clone(), turn]), None);
}

#[test]
fn p1_5_differing_units_break_the_run() {
    let a = CallOutcome {
        call_unit: Some("write\u{1f}{\"p\":\"a\"}".to_string()),
        failing: true,
        result_unit: None,
    };
    let b = CallOutcome {
        call_unit: Some("write\u{1f}{\"p\":\"b\"}".to_string()),
        failing: true,
        result_unit: None,
    };
    assert_eq!(detect_exact_failing_repeat(&[a.clone(), a, b]), None);
}

#[test]
fn p1_5_no_tool_call_turn_breaks_the_run() {
    let call = CallOutcome {
        call_unit: Some("x\u{1f}{}".to_string()),
        failing: true,
        result_unit: None,
    };
    let prose = CallOutcome {
        call_unit: None,
        failing: false,
        result_unit: None,
    };
    assert_eq!(
        detect_exact_failing_repeat(&[call.clone(), prose, call]),
        None
    );
}

#[test]
fn p1_5_recent_calls_all_failing_gate() {
    let fail = CallOutcome {
        call_unit: Some("x\u{1f}{}".to_string()),
        failing: true,
        result_unit: None,
    };
    let ok = CallOutcome {
        call_unit: Some("x\u{1f}{}".to_string()),
        failing: false,
        result_unit: None,
    };
    assert!(recent_calls_all_failing(
        &[fail.clone(), fail.clone(), fail.clone()],
        3
    ));
    assert!(!recent_calls_all_failing(
        &[fail.clone(), ok, fail.clone()],
        3
    ));
    assert!(!recent_calls_all_failing(&[fail], 3));
}

#[test]
fn signature_below_min_tokens_is_empty() {
    let s = Signature::build("yes", std::iter::empty());
    assert!(s.is_empty(), "3-token text must yield empty signature");
}

fn outcome(unit: &str, failing: bool, result: &str) -> CallOutcome {
    CallOutcome {
        call_unit: Some(unit.into()),
        failing,
        result_unit: Some(result.into()),
    }
}

#[test]
fn progressing_cycle_detected_when_results_differ() {
    let outcomes = vec![
        outcome(
            "bash\u{1f}cargo check",
            false,
            "error[E0308]: mismatched types in transactions.rs line 41",
        ),
        outcome(
            "bash\u{1f}cargo check",
            false,
            "error[E0433]: unresolved import sqlx::SqlitePool in db.rs",
        ),
        outcome(
            "bash\u{1f}cargo check",
            false,
            "error[E0599]: no method named fetch_all found for Pool",
        ),
    ];
    assert!(
        recent_results_progressing(&outcomes, 3),
        "differing results round-to-round = progress; must not hard-mask"
    );
}

#[test]
fn true_loop_not_progressing_when_results_identical() {
    let outcomes = vec![
        outcome("write\u{1f}{\"f\":1}", false, "Wrote file successfully."),
        outcome("write\u{1f}{\"f\":1}", false, "Wrote file successfully."),
        outcome("write\u{1f}{\"f\":1}", false, "Wrote file successfully."),
    ];
    assert!(
        !recent_results_progressing(&outcomes, 3),
        "identical results = true loop; legacy Suppress must apply"
    );
}

#[test]
fn missing_results_are_conservatively_not_progressing() {
    let outcomes = vec![
        CallOutcome {
            call_unit: Some("x".into()),
            failing: false,
            result_unit: None,
        },
        outcome("x", false, "a result"),
    ];
    assert!(!recent_results_progressing(&outcomes, 2));
}
