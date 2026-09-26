// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the journey ledger, with no mocks: the graph engine
//! runs in memory and the file cases use a directory under the temp dir.
//!
//! Owner: metrale-governance.
//! Invariants: none beyond the types.

use super::event::{Event, EventKind, Verdict};
use super::ledger::{self, Journey, append, materialize, path_for, read_all};

fn ev(sha: &str, attempt: u32, kind: EventKind) -> Event {
    Event {
        pr: 389,
        head_sha: sha.to_string(),
        run_id: "run-1".to_string(),
        attempt,
        at: 1_786_200_000,
        kind,
    }
}

fn gate(id: &str, verdict: Verdict) -> EventKind {
    EventKind::Gate {
        id: id.to_string(),
        verdict,
        invalidated_by: Vec::new(),
        detail: None,
    }
}

fn tmpdir() -> std::path::PathBuf {
    let base = std::env::temp_dir().join(format!(
        "metrale-governance-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).expect("temp dir");
    base
}

#[test]
fn appending_then_reading_round_trips() {
    let dir = tmpdir();
    let path = path_for(&dir, 389);
    let first = ev("aaa", 0, gate("bfcl-subset", Verdict::Missing));
    let second = ev(
        "bbb",
        0,
        EventKind::State {
            to: "gates".to_string(),
        },
    );
    append(&path, &first).unwrap();
    append(&path, &second).unwrap();

    let journey = read_all(&path).unwrap();
    assert_eq!(journey.events, vec![first, second]);
}

#[test]
fn append_creates_the_directory() {
    let dir = tmpdir();
    let path = path_for(&dir.join("nested").join("deeper"), 7);
    append(&path, &ev("aaa", 0, gate("ttft-warm-gate", Verdict::Pass))).unwrap();
    assert!(path.exists());
}

/// 2026-09-26: The same event three times deduplicates to one.
#[test]
fn replayed_events_collapse() {
    let e = ev("aaa", 0, gate("bfcl-subset", Verdict::Pass));
    let journey = Journey {
        events: vec![e.clone(), e.clone(), e],
    }
    .deduplicated();
    assert_eq!(journey.events.len(), 1);
}

/// 2026-09-26: Two attempts of one run are two events.
#[test]
fn a_second_attempt_is_a_distinct_event() {
    let journey = Journey {
        events: vec![
            ev("aaa", 0, gate("bfcl-subset", Verdict::Fail)),
            ev("aaa", 1, gate("bfcl-subset", Verdict::Pass)),
        ],
    }
    .deduplicated();
    assert_eq!(journey.events.len(), 2, "a re-run must survive dedup");
}

/// 2026-09-26: `at` does not change the identity.
#[test]
fn the_timestamp_does_not_affect_identity() {
    let mut a = ev("aaa", 0, gate("bfcl-subset", Verdict::Pass));
    let mut b = a.clone();
    a.at = 1;
    b.at = 999_999;
    assert_eq!(a.identity(), b.identity());
}

/// 2026-09-26: Different gate ids, and a gate versus a category, at the same
/// commit and attempt have different identities.
#[test]
fn different_kinds_do_not_collide() {
    let a = ev("aaa", 0, gate("bfcl-subset", Verdict::Pass));
    let b = ev("aaa", 0, gate("ttft-warm-gate", Verdict::Pass));
    let c = ev(
        "aaa",
        0,
        EventKind::Category {
            value: "numerics".into(),
            status: "ok".into(),
        },
    );
    assert_ne!(a.identity(), b.identity());
    assert_ne!(a.identity(), c.identity());
}

#[test]
fn gate_identity_includes_verdict_and_diagnostics() {
    let pass = ev("aaa", 0, gate("bfcl-subset", Verdict::Pass));
    let fail = ev("aaa", 0, gate("bfcl-subset", Verdict::Fail));
    let invalidated = ev(
        "aaa",
        0,
        EventKind::Gate {
            id: "bfcl-subset".into(),
            verdict: Verdict::Pass,
            invalidated_by: vec!["kernels/common.cu".into()],
            detail: None,
        },
    );
    let detailed = ev(
        "aaa",
        0,
        EventKind::Gate {
            id: "bfcl-subset".into(),
            verdict: Verdict::Pass,
            invalidated_by: Vec::new(),
            detail: Some("record expired".into()),
        },
    );

    for other in [&fail, &invalidated, &detailed] {
        assert_ne!(pass.identity(), other.identity());
    }
}

/// 2026-09-26: Deduplicating two orderings of the same events gives the same
/// set of identities.
#[test]
fn dedup_is_order_independent() {
    let a = ev("aaa", 0, gate("bfcl-subset", Verdict::Pass));
    let b = ev("bbb", 0, gate("ttft-cold-gate", Verdict::Fail));
    let one = Journey {
        events: vec![a.clone(), b.clone(), a.clone()],
    }
    .deduplicated();
    let two = Journey {
        events: vec![b.clone(), a.clone(), b],
    }
    .deduplicated();

    let mut ids_one: Vec<String> = one.events.iter().map(Event::identity).collect();
    let mut ids_two: Vec<String> = two.events.iter().map(Event::identity).collect();
    ids_one.sort();
    ids_two.sort();
    assert_eq!(ids_one, ids_two);
}

/// 2026-09-26: A malformed line fails the read and the error names its line.
#[test]
fn a_corrupt_line_is_refused() {
    let dir = tmpdir();
    let path = path_for(&dir, 1);
    append(&path, &ev("aaa", 0, gate("bfcl-subset", Verdict::Pass))).unwrap();
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(f, "{{not json").unwrap();
    }
    let err = read_all(&path).unwrap_err().to_string();
    assert!(err.contains("line 2"), "{err}");
}

#[test]
fn blank_lines_are_tolerated() {
    let dir = tmpdir();
    let path = path_for(&dir, 2);
    append(&path, &ev("aaa", 0, gate("bfcl-subset", Verdict::Pass))).unwrap();
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(f).unwrap();
        writeln!(f, " \t ").unwrap();
    }
    assert_eq!(read_all(&path).unwrap().events.len(), 1);
}

/// 2026-09-26: `Missing` and `Fail` serialise as `"missing"` and `"fail"`.
#[test]
fn missing_and_fail_are_distinct_on_the_wire() {
    let m = serde_json::to_string(&ev("a", 0, gate("g", Verdict::Missing))).unwrap();
    let f = serde_json::to_string(&ev("a", 0, gate("g", Verdict::Fail))).unwrap();
    assert!(m.contains("\"missing\""), "{m}");
    assert!(f.contains("\"fail\""), "{f}");
    assert_ne!(m, f);
}

/// 2026-09-26: Two commits and three events give five nodes, and the first
/// commit has an `observed` edge to each of its two events.
#[test]
fn materialize_builds_commit_and_event_nodes() {
    let journey = Journey {
        events: vec![
            ev("aaa", 0, gate("bfcl-subset", Verdict::Missing)),
            ev("aaa", 0, gate("ttft-warm-gate", Verdict::Pass)),
            ev("bbb", 0, gate("bfcl-subset", Verdict::Pass)),
        ],
    };
    let engine = materialize(&journey).unwrap();
    assert_eq!(engine.point_ids().unwrap().len(), 5);

    // 2026-09-26: `aaa` sorts first, so it is node 0.
    let edges = engine.get_edges(0).unwrap();
    assert_eq!(edges.len(), 2, "commit aaa observed two events");
    assert!(edges.iter().all(|e| e.relation == "observed"));
}

/// 2026-09-26: Appending an event for a commit already in the journey leaves
/// node 0 on the same sha.
#[test]
fn commit_ids_are_stable_as_events_are_appended() {
    let first = Journey {
        events: vec![ev("aaa", 0, gate("bfcl-subset", Verdict::Missing))],
    };
    let engine_a = materialize(&first).unwrap();
    let sha_before = engine_a.get_point(0).unwrap().unwrap();

    let mut later = first.clone();
    later
        .events
        .push(ev("aaa", 1, gate("bfcl-subset", Verdict::Pass)));
    let engine_b = materialize(&later).unwrap();
    let sha_after = engine_b.get_point(0).unwrap().unwrap();

    assert_eq!(
        sha_before.payload.get("sha"),
        sha_after.payload.get("sha"),
        "commit node 0 moved when an event was appended"
    );
}

#[test]
fn an_empty_journey_materializes_to_an_empty_graph() {
    let engine = materialize(&Journey::default()).unwrap();
    assert!(engine.point_ids().unwrap().is_empty());
}

/// 2026-09-26: `gate_history` returns only that gate's events, in order.
#[test]
fn gate_history_selects_only_that_gate() {
    let journey = Journey {
        events: vec![
            ev("aaa", 0, gate("bfcl-subset", Verdict::Missing)),
            ev("aaa", 0, gate("ttft-warm-gate", Verdict::Pass)),
            ev("bbb", 1, gate("bfcl-subset", Verdict::Pass)),
        ],
    };
    let hits: Vec<&Event> = journey.gate_history("bfcl-subset").collect();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].head_sha, "aaa");
    assert_eq!(hits[1].head_sha, "bbb");
}

/// 2026-09-26: Two category events that differ only in `status` have
/// different identities.
#[test]
fn an_abstained_category_is_distinguishable_from_an_answer() {
    let ok = ev(
        "aaa",
        0,
        EventKind::Category {
            value: "numerics".into(),
            status: "ok".into(),
        },
    );
    let abstain = ev(
        "aaa",
        0,
        EventKind::Category {
            value: "numerics".into(),
            status: "abstain".into(),
        },
    );
    assert_ne!(ok.identity(), abstain.identity());
}

#[test]
fn the_path_is_one_file_per_pull_request() {
    let root = std::path::Path::new("/repo");
    assert_eq!(
        ledger::path_for(root, 389),
        std::path::Path::new("/repo/governance/pr-389.jsonl")
    );
    assert_ne!(ledger::path_for(root, 389), ledger::path_for(root, 390));
}
