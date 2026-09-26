// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the `serve_resolved` disclosure: its keys, its JSON
//! round trip, the committed fixture records, and that scoring ignores it.
//!
//! Owner: bench gate (records).
//! Invariants: none beyond the types.

use super::super::tests::{SHA, bfcl_baseline, hw, run_record};
use super::super::{GateRecord, check_record, read_record, records_newest_first};
use super::{MTP_GATE, PREFILL_CODISPATCH, SPECULATIVE, W4A4_DOWNCAST, disclosure};
use crate::result::Verdict;
use std::collections::BTreeMap;

fn keys(m: &BTreeMap<String, String>) -> Vec<(&str, &str)> {
    m.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
}

#[test]
fn disclosure_spells_the_regime_and_omits_what_was_not_resolved() {
    assert_eq!(
        keys(&disclosure(Some(true), true, false, false)),
        vec![(MTP_GATE, "force"), (SPECULATIVE, "true")]
    );
    assert_eq!(
        keys(&disclosure(Some(false), true, false, false)),
        vec![(MTP_GATE, "auto"), (SPECULATIVE, "true")]
    );
    // 2026-09-26: No `--mtp-gate`: the key is absent, not `auto`.
    assert_eq!(
        keys(&disclosure(None, false, false, false)),
        vec![(SPECULATIVE, "false")]
    );
    // 2026-09-26: `--prefill-codispatch` is disclosed only when given.
    assert_eq!(
        keys(&disclosure(None, true, true, false)),
        vec![(PREFILL_CODISPATCH, "true"), (SPECULATIVE, "true")]
    );
    // 2026-09-26: `--w4a4-downcast` is disclosed only when on.
    assert_eq!(
        keys(&disclosure(None, true, false, true)),
        vec![(SPECULATIVE, "true"), (W4A4_DOWNCAST, "true")]
    );
}

fn passing_record() -> GateRecord {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-35b-a3b-fp8-bf16head".into()),
    )
    .unwrap()
}

#[test]
fn serve_resolved_round_trips_and_older_records_simply_lack_it() {
    let record = passing_record().with_serve_resolved(disclosure(Some(true), true, false, false));
    let json = serde_json::to_value(&record).unwrap();
    assert_eq!(json["serve_resolved"][MTP_GATE], "force");
    assert_eq!(json["serve_resolved"][SPECULATIVE], "true");
    let back: GateRecord = serde_json::from_value(json).unwrap();
    assert_eq!(back.serve_resolved, record.serve_resolved);

    // 2026-09-26: A record without the field loads with it empty, and an
    // empty map is not serialised.
    let bare = serde_json::to_value(passing_record()).unwrap();
    assert!(bare.get("serve_resolved").is_none());
    let old: GateRecord = serde_json::from_value(bare).unwrap();
    assert!(old.serve_resolved.is_empty());

    // 2026-09-26: Every agentic-webserver record under
    // `test_data/gate-records` parses, and any non-empty disclosure carries
    // `speculative`.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace layout")
        .join("test_data/gate-records");
    let committed = records_newest_first(&root, "agentic-webserver");
    assert!(
        !committed.is_empty(),
        "no fixture agentic-webserver records found"
    );
    for path in committed {
        let record = read_record(&path).unwrap_or_else(|e| panic!("{e:#}"));
        assert!(
            record.serve_resolved.is_empty() || record.serve_resolved.contains_key(SPECULATIVE),
            "{}: a disclosure without the `speculative` key is malformed",
            path.display()
        );
    }
}

/// 2026-09-26: `check_record` gives the same result with and without
/// `serve_resolved`, for a passing and for a failing record.
#[test]
fn serve_resolved_never_reaches_check_record() {
    let baseline = bfcl_baseline();
    let without = passing_record();
    let with = passing_record().with_serve_resolved(disclosure(Some(false), true, false, false));
    assert_eq!(check_record(&with, &baseline), None);
    assert_eq!(
        check_record(&with, &baseline),
        check_record(&without, &baseline)
    );

    let mut failing_without = without;
    failing_without
        .metrics
        .insert("overall_accuracy".into(), 80.0);
    let failing_with =
        failing_without
            .clone()
            .with_serve_resolved(disclosure(Some(true), true, false, false));
    let verdict = check_record(&failing_with, &baseline);
    assert!(
        verdict.is_some(),
        "the floor must bite for the comparison to mean anything"
    );
    assert_eq!(verdict, check_record(&failing_without, &baseline));
}
