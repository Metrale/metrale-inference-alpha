// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for color extraction and the verdict.
//!
//! Owner: bench, video.
//! Invariants: none beyond the types.

use super::*;

const PAL: &[&str] = &["red", "green", "blue", "yellow"];

#[test]
fn a_bare_comma_list_reads_in_order() {
    assert_eq!(
        colors_in_order("red, green, blue, yellow", PAL),
        vec!["red", "green", "blue", "yellow"]
    );
}

#[test]
fn case_and_surrounding_prose_do_not_matter() {
    let reply = "The video shows Red first, then Green, followed by Blue and finally Yellow.";
    assert!(order_matches(
        reply,
        &["red", "green", "blue", "yellow"],
        PAL
    ));
}

/// 2026-09-26: A recap ("the red one came first") does not add a second red.
#[test]
fn a_recap_does_not_duplicate_a_color() {
    let reply = "red, green, blue, yellow — and the red one came first.";
    assert_eq!(
        colors_in_order(reply, PAL),
        vec!["red", "green", "blue", "yellow"]
    );
    assert!(order_matches(
        reply,
        &["red", "green", "blue", "yellow"],
        PAL
    ));
}

/// 2026-09-26: The forward answer does not match the reversed clip.
#[test]
fn the_reversed_sequence_is_a_different_answer() {
    let fwd = "red, green, blue, yellow";
    assert!(order_matches(fwd, &["red", "green", "blue", "yellow"], PAL));
    assert!(
        !order_matches(fwd, &["yellow", "blue", "green", "red"], PAL),
        "forward and reversed must not both match — that would make the pair worthless"
    );
}

/// 2026-09-26: A reply naming only gray finds no palette color.
#[test]
fn the_grey_field_answer_names_no_colors() {
    let reply = "gray, gray, gray, gray, gray, gray";
    assert!(colors_in_order(reply, PAL).is_empty());
    assert!(!order_matches(
        reply,
        &["red", "green", "blue", "yellow"],
        PAL
    ));
}

#[test]
fn color_names_inside_other_words_are_not_evidence() {
    assert!(
        colors_in_order("hundred evergreen blueprints yellowish", PAL).is_empty(),
        "substring hits do not show that the model named a color"
    );
}

#[test]
fn a_partial_sequence_does_not_match() {
    assert!(!order_matches(
        "red, green",
        &["red", "green", "blue", "yellow"],
        PAL
    ));
}

fn ok_order() -> OrderCell {
    OrderCell::Match {
        clip: "c",
        seen: "red, green".into(),
    }
}
fn bad_order() -> OrderCell {
    OrderCell::WrongOrder {
        clip: "c",
        want: "a".into(),
        got: "b".into(),
    }
}
fn ok_count() -> CountCell {
    CountCell::Match {
        id: "x",
        detail: String::new(),
    }
}

#[test]
fn all_legs_passing_with_a_held_control_is_a_pass() {
    assert_eq!(verdict(&[ok_order()], &[ok_count()], true), Verdict::Pass);
}

#[test]
fn any_failing_leg_fails_the_run() {
    assert_eq!(verdict(&[bad_order()], &[ok_count()], true), Verdict::Fail);
    assert_eq!(
        verdict(
            &[OrderCell::NotSeen {
                clip: "c",
                reply: "no idea".into(),
            }],
            &[ok_count()],
            true
        ),
        Verdict::Fail
    );
    assert_eq!(
        verdict(
            &[ok_order()],
            &[CountCell::Mismatch {
                id: "x",
                detail: "wrong geometry".into(),
            }],
            true
        ),
        Verdict::Fail
    );
}

/// 2026-09-26: Every leg passing with a failed control is Vacuous, not Pass.
#[test]
fn a_broken_control_makes_the_run_vacuous_not_green() {
    assert_eq!(
        verdict(&[ok_order()], &[ok_count()], false),
        Verdict::Vacuous
    );
}

/// 2026-09-26: A run where every leg was skipped is Inconclusive, not Pass.
#[test]
fn skipping_everything_is_inconclusive_rather_than_pass() {
    let skipped = vec![OrderCell::Skipped {
        clip: "c",
        why: "no ffmpeg".into(),
    }];
    let counts = vec![CountCell::Skipped {
        id: "x",
        why: "no ffmpeg".into(),
    }];
    assert_eq!(verdict(&skipped, &counts, true), Verdict::Inconclusive);
    assert_eq!(asserted(&skipped, &counts), 0);
}

/// 2026-09-26: A partly skipped run still judges what it measured.
#[test]
fn a_partial_skip_still_judges_the_rest() {
    let order = vec![
        ok_order(),
        OrderCell::Skipped {
            clip: "c",
            why: "no ffmpeg".into(),
        },
    ];
    assert_eq!(asserted(&order, &[]), 1);
    assert_eq!(verdict(&order, &[], true), Verdict::Pass);
}

#[test]
fn a_leg_that_errored_counts_as_asserted_and_fails() {
    let order = vec![OrderCell::Error {
        clip: "c",
        msg: "request reset".into(),
    }];
    let counts = vec![CountCell::Error {
        id: "x",
        msg: "decode failed".into(),
    }];
    assert_eq!(asserted(&order, &counts), 2);
    assert_eq!(passed(&order, &counts), 0);
    assert_eq!(verdict(&order, &counts, true), Verdict::Fail);
}

#[test]
fn every_verdict_has_an_exact_operator_label() {
    assert_eq!(
        [
            Verdict::Pass,
            Verdict::Fail,
            Verdict::Vacuous,
            Verdict::Inconclusive,
        ]
        .map(|verdict| verdict.to_string()),
        ["PASS", "FAIL", "VACUOUS", "INCONCLUSIVE"]
    );
}
