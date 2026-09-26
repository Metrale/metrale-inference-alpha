// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `Preflight::poll` and the concern text each `Report` produces.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use metrale_bench::TargetEndpoint;
use metrale_bench::coherence::Answer;

fn target() -> TargetEndpoint {
    TargetEndpoint::local(8888, "m")
}

/// 2026-09-26: Build a Preflight whose answer is already waiting, without a runtime.
fn resolved(report: Report) -> Preflight {
    let (tx, rx) = channel();
    tx.send(report).expect("send");
    Preflight {
        phase: Phase::Checking,
        rx: Some(rx),
    }
}

#[test]
fn a_clean_check_starts_the_run_without_asking() {
    let mut pre = resolved(Report {
        answers: vec![Answer {
            label: "recall",
            answer: "Paris".into(),
            passed: true,
        }],
        transport_error: None,
        served_instead: None,
        wrong_family: None,
    });
    assert_eq!(pre.poll(&target()), Some(true));
}

#[test]
fn a_concern_stops_to_ask_and_keeps_the_reason() {
    let mut pre = resolved(Report {
        answers: vec![Answer {
            label: "recall",
            answer: String::new(),
            passed: false,
        }],
        transport_error: None,
        served_instead: None,
        wrong_family: None,
    });
    assert_eq!(pre.poll(&target()), Some(false));
    match &pre.phase {
        Phase::Concern(text) => {
            assert!(text.contains("answered nothing"), "{text}");
            assert!(text.contains("still valid"), "it is a warning: {text}");
        }
        other => panic!("expected a concern, got {other:?}"),
    }
    assert!(!pre.is_checking());
}

#[test]
fn waiting_reports_nothing_yet() {
    let (_tx, rx) = channel::<Report>();
    let mut pre = Preflight {
        phase: Phase::Checking,
        rx: Some(rx),
    };
    assert_eq!(pre.poll(&target()), None);
    assert!(pre.is_checking());
}

#[test]
fn a_dropped_check_lets_the_run_proceed_rather_than_stranding_it() {
    let (tx, rx) = channel::<Report>();
    drop(tx);
    let mut pre = Preflight {
        phase: Phase::Checking,
        rx: Some(rx),
    };
    assert_eq!(pre.poll(&target()), Some(true));
}

#[test]
fn polling_after_the_answer_is_harmless() {
    let mut pre = resolved(Report::default());
    assert_eq!(pre.poll(&target()), Some(true));
    assert_eq!(pre.poll(&target()), None, "the receiver is spent");
}

/// 2026-09-26: The concern text a report produces, driven through `poll` rather than `Report::concern`.
fn concern_for(report: Report) -> String {
    let mut pre = resolved(report);
    assert_eq!(
        pre.poll(&target()),
        Some(false),
        "a concern must stop to ask"
    );
    match &pre.phase {
        Phase::Concern(text) => text.clone(),
        other => panic!("expected a concern, got {other:?}"),
    }
}

fn answers(passed: bool) -> Vec<Answer> {
    vec![Answer {
        label: "arithmetic",
        answer: if passed { "4".into() } else { "banana".into() },
        passed,
    }]
}

#[test]
fn an_unreachable_endpoint_names_the_url_and_what_it_said() {
    let text = concern_for(Report {
        transport_error: Some("connection refused".into()),
        ..Report::default()
    });
    assert!(text.contains("http://127.0.0.1:8888"), "{text}");
    assert!(text.contains("connection refused"), "{text}");
}

#[test]
fn a_server_with_nothing_loaded_says_so_and_says_how_to_load_one() {
    // 2026-09-26: An empty served list gets its own wording, not the wrong-model one.
    let text = concern_for(Report {
        served_instead: Some(Vec::new()),
        ..Report::default()
    });
    assert!(text.contains("no model loaded"), "{text}");
    assert!(text.contains("Library"), "names the remedy: {text}");
    assert!(
        !text.contains("is serving m"),
        "and not a served name: {text}"
    );
}

#[test]
fn a_server_holding_a_different_model_says_the_numbers_will_still_come() {
    let text = concern_for(Report {
        served_instead: Some(vec!["org/other".into()]),
        ..Report::default()
    });
    assert!(text.contains("org/other"), "{text}");
    assert!(text.contains("\"m\""), "names what was requested: {text}");
    assert!(text.contains("different model"), "{text}");
}

#[test]
fn a_model_the_gate_is_not_defined_on_is_reported_in_the_gates_own_words() {
    let text = concern_for(Report {
        wrong_family: Some("gate A is defined on the 35B MoE".into()),
        ..Report::default()
    });
    assert_eq!(text, "gate A is defined on the 35B MoE");
}

#[test]
fn a_wrong_answer_quotes_it_back_and_still_calls_the_run_valid() {
    let text = concern_for(Report {
        answers: answers(false),
        ..Report::default()
    });
    assert!(text.contains("arithmetic answered"), "{text}");
    assert!(text.contains("banana"), "quotes what came back: {text}");
    assert!(text.contains("base (non-instruct)"), "{text}");
    assert!(
        text.contains("still valid"),
        "a warning, not a veto: {text}"
    );
}

#[test]
fn every_refusal_reason_reads_differently() {
    let reasons = [
        concern_for(Report {
            transport_error: Some("connection refused".into()),
            ..Report::default()
        }),
        concern_for(Report {
            wrong_family: Some("this gate is defined on the 35B".into()),
            ..Report::default()
        }),
        concern_for(Report {
            served_instead: Some(Vec::new()),
            ..Report::default()
        }),
        concern_for(Report {
            served_instead: Some(vec!["org/other".into()]),
            ..Report::default()
        }),
        concern_for(Report {
            answers: answers(false),
            ..Report::default()
        }),
    ];
    let distinct: std::collections::BTreeSet<&String> = reasons.iter().collect();
    assert_eq!(distinct.len(), reasons.len(), "{reasons:#?}");
    for r in &reasons {
        assert!(!r.is_empty());
    }
}

#[test]
fn the_cause_is_reported_ahead_of_the_symptom_it_explains() {
    // 2026-09-26: `Report::concern` reports the transport error, then the wrong family, then the
    // served name, and only then the answers.
    let everything = Report {
        answers: answers(false),
        transport_error: Some("connection refused".into()),
        wrong_family: Some("wrong family".into()),
        served_instead: Some(vec!["org/other".into()]),
    };
    assert!(concern_for(everything.clone()).contains("connection refused"));

    let no_transport = Report {
        transport_error: None,
        ..everything.clone()
    };
    assert_eq!(concern_for(no_transport), "wrong family");

    let served_only = Report {
        transport_error: None,
        wrong_family: None,
        ..everything
    };
    assert!(concern_for(served_only).contains("org/other"));
}

#[test]
fn the_probe_warns_and_never_vetoes() {
    // 2026-09-26: Every outcome is either "start now" or "ask"; none refuses.
    for report in [
        Report {
            transport_error: Some("connection refused".into()),
            ..Report::default()
        },
        Report {
            wrong_family: Some("wrong family".into()),
            ..Report::default()
        },
        Report {
            served_instead: Some(Vec::new()),
            ..Report::default()
        },
        Report {
            answers: answers(false),
            ..Report::default()
        },
        Report {
            answers: answers(true),
            ..Report::default()
        },
    ] {
        let clean = report.is_clean();
        let mut pre = resolved(report);
        let decided = pre.poll(&target()).expect("an answered check decides");
        assert_eq!(decided, clean, "a clean report starts, a concern asks");
        // 2026-09-26: A concern leaves `Checking`. A clean check keeps it, and `poll_preflight` drops
        // the pre-flight in the same call.
        assert_eq!(pre.is_checking(), clean);
    }
}
