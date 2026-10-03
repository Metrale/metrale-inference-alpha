// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the `serve_resolved` disclosure: its keys, its JSON
//! round trip, the committed fixture records, and that scoring ignores it.
//!
//! Owner: bench gate (records).
//! Invariants: none beyond the types.

use super::super::tests::{SHA, bfcl_baseline, hw, run_record};
use super::super::{GateRecord, check_record, read_record, records_newest_first};
use super::{
    ACTIVATION_QUANTIZATION, EXPERT_QUANTIZATION, MTP_GATE, PREFILL_CODISPATCH, SPECULATIVE,
    W4A4_DOWNCAST, WEIGHT_QUANTIZATION, disclosure,
};
use crate::result::Verdict;
use std::collections::BTreeMap;

fn keys(m: &BTreeMap<String, String>) -> Vec<(&str, &str)> {
    m.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
}

#[test]
fn disclosure_spells_the_regime_and_omits_what_was_not_resolved() {
    let d = |g, s, c, w, e| disclosure(g, s, c, w, e, "nvfp4", "adaptive");
    let aq = (ACTIVATION_QUANTIZATION, "adaptive");
    let wq = (WEIGHT_QUANTIZATION, "nvfp4");
    assert_eq!(
        keys(&d(Some(true), true, false, false, None)),
        vec![aq, (MTP_GATE, "force"), (SPECULATIVE, "true"), wq]
    );
    assert_eq!(
        keys(&d(Some(false), true, false, false, None)),
        vec![aq, (MTP_GATE, "auto"), (SPECULATIVE, "true"), wq]
    );
    // 2026-09-26: No `--mtp-gate`: the key is absent, not `auto`.
    assert_eq!(
        keys(&d(None, false, false, false, None)),
        vec![aq, (SPECULATIVE, "false"), wq]
    );
    // 2026-09-26: `--prefill-codispatch` is disclosed only when given.
    assert_eq!(
        keys(&d(None, true, true, false, None)),
        vec![aq, (PREFILL_CODISPATCH, "true"), (SPECULATIVE, "true"), wq]
    );
    // 2026-09-26: `--w4a4-downcast` is disclosed only when on.
    assert_eq!(
        keys(&d(None, true, false, true, None)),
        vec![aq, (SPECULATIVE, "true"), (W4A4_DOWNCAST, "true"), wq]
    );
    // 2026-09-27: `--expert-quantization` names a tier other than `fp8`.
    assert_eq!(
        keys(&d(None, true, false, false, Some("nvfp4-gate-up"))),
        vec![
            aq,
            (EXPERT_QUANTIZATION, "nvfp4-gate-up"),
            (SPECULATIVE, "true"),
            wq
        ]
    );
    // 2026-09-28: `--weight-quantization` is written for either tier, the default included.
    assert_eq!(
        keys(&disclosure(
            None, true, false, false, None, "declared", "declared"
        )),
        vec![
            (ACTIVATION_QUANTIZATION, "declared"),
            (SPECULATIVE, "true"),
            (WEIGHT_QUANTIZATION, "declared")
        ]
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
    let record = passing_record().with_serve_resolved(disclosure(
        Some(true),
        true,
        false,
        false,
        None,
        "nvfp4",
        "adaptive",
    ));
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
    let with = passing_record().with_serve_resolved(disclosure(
        Some(false),
        true,
        false,
        false,
        None,
        "nvfp4",
        "adaptive",
    ));
    assert_eq!(check_record(&with, &baseline), None);
    assert_eq!(
        check_record(&with, &baseline),
        check_record(&without, &baseline)
    );

    let mut failing_without = without;
    failing_without
        .metrics
        .insert("overall_accuracy".into(), 80.0);
    let failing_with = failing_without.clone().with_serve_resolved(disclosure(
        Some(true),
        true,
        false,
        false,
        None,
        "nvfp4",
        "adaptive",
    ));
    let verdict = check_record(&failing_with, &baseline);
    assert!(
        verdict.is_some(),
        "the floor must bite for the comparison to mean anything"
    );
    assert_eq!(verdict, check_record(&failing_without, &baseline));
}

fn live(forward: &str, digest: Option<&str>) -> super::LiveForward {
    super::LiveForward {
        auto_max_batch_size: None,
        moe_expert_tables: None,
        forward: forward.to_string(),
        plan_digest: digest.map(str::to_string),
    }
}

#[test]
fn a_legacy_forward_adds_no_keys_and_a_circuit_adds_its_digest() {
    let mut m = disclosure(None, false, false, false, None, "nvfp4", "adaptive");
    let before = m.clone();
    super::merge_live_forward(&mut m, "legacy", &live("legacy", None)).unwrap();
    assert_eq!(m, before);
    super::merge_live_forward(&mut m, "circuit", &live("circuit", Some("abc"))).unwrap();
    assert_eq!(m.get(super::FORWARD).map(String::as_str), Some("circuit"));
    assert_eq!(m.get(super::PLAN_DIGEST).map(String::as_str), Some("abc"));
}

#[test]
fn a_live_forward_that_contradicts_the_request_is_refused() {
    let mut m = BTreeMap::new();
    let cases = [
        ("circuit", live("legacy", None), "asked for `circuit`"),
        ("legacy", live("circuit", Some("abc")), "asked for `legacy`"),
        ("circuit", live("circuit", None), "no plan digest"),
        ("legacy", live("legacy", Some("abc")), "reports plan digest"),
    ];
    for (requested, l, want) in cases {
        let err = super::merge_live_forward(&mut m, requested, &l).unwrap_err();
        assert!(err.contains(want), "{err} lacks `{want}`");
        assert!(
            m.is_empty(),
            "a refused merge leaves the disclosure untouched"
        );
    }
}

#[test]
fn the_live_forward_report_round_trips_and_tolerates_a_missing_digest() {
    let l = live("circuit", Some("d"));
    let back: super::LiveForward =
        serde_json::from_str(&serde_json::to_string(&l).unwrap()).unwrap();
    assert_eq!(back, l);
    let bare: super::LiveForward = serde_json::from_str(r#"{"forward":"legacy"}"#).unwrap();
    assert_eq!(bare, live("legacy", None));
}

/// 2026-10-01: `--max-batch-size auto` discloses the count it resolved to; an explicit count adds
/// nothing (the rendered serve states it).
#[test]
fn an_auto_slot_count_is_disclosed_and_an_explicit_one_is_not() {
    let mut m = BTreeMap::new();
    super::merge_live_forward(&mut m, "legacy", &live("legacy", None)).unwrap();
    assert!(!m.contains_key(super::MAX_BATCH_SIZE));
    let mut auto = live("legacy", None);
    auto.auto_max_batch_size = Some(91);
    super::merge_live_forward(&mut m, "legacy", &auto).unwrap();
    assert_eq!(
        m.get(super::MAX_BATCH_SIZE).map(String::as_str),
        Some("auto:91")
    );
}

/// 2026-10-02: Skipped MoE expert tables are disclosed; built ones (what every serve before the
/// decision did) add nothing, and an unknown report is refused before anything is written.
#[test]
fn skipped_moe_expert_tables_are_disclosed_and_built_ones_are_not() {
    let with = |t: Option<&str>| super::LiveForward {
        moe_expert_tables: t.map(str::to_string),
        ..live("legacy", None)
    };
    let mut m = BTreeMap::new();
    for t in [None, Some("build")] {
        super::merge_live_forward(&mut m, "legacy", &with(t)).unwrap();
        assert!(m.is_empty(), "{t:?}");
    }
    let mut odd = with(Some("sometimes"));
    odd.auto_max_batch_size = Some(4);
    assert!(super::merge_live_forward(&mut m, "legacy", &odd).is_err());
    assert!(
        m.is_empty(),
        "a refused merge leaves the disclosure untouched"
    );
    super::merge_live_forward(&mut m, "legacy", &with(Some("skip"))).unwrap();
    assert_eq!(
        m.get(super::MOE_EXPERT_TABLES).map(String::as_str),
        Some("skip")
    );
}
