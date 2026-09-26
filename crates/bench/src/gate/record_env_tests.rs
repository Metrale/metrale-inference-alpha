// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for what a gate record discloses about its server's
//! environment: `serve_env`, and the `perf_env` switch defaults checked
//! against each lever's source.
//!
//! Owner: bench gate (records).
//! Invariants: none beyond the types.

use super::super::GateRecord;
use super::super::tests::{SHA, hw, run_record};
use crate::result::Verdict;
use std::collections::BTreeMap;

/// 2026-09-26: `with_serve_env` records the applied lever set and resolves
/// `perf_env` through it first: a declared
/// `METRALE_PREFILL_CODISPATCH_WINDOW_MS=50` that this process lacks is
/// recorded as `50`, not `100`. Without it the field is empty and omitted.
#[test]
fn the_record_discloses_the_applied_serve_env_and_reads_perf_env_through_it() {
    // 2026-09-26: Control: without an applied set the value is this
    // process's, which the test does not export.
    let bare = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.8/qwen3.8-27b-nvfp4-unsloth".into()),
    )
    .unwrap();
    assert!(bare.serve_env.is_empty());
    assert_eq!(
        bare.perf_env
            .get("METRALE_PREFILL_CODISPATCH_WINDOW_MS")
            .map(String::as_str),
        Some("100"),
        "the test environment must not export the control this test flips"
    );
    assert!(
        serde_json::to_value(&bare)
            .unwrap()
            .get("serve_env")
            .is_none()
    );

    let applied: BTreeMap<String, String> = [
        ("METRALE_FP8_ROWWISE", "1"),
        ("METRALE_MTP_DCUT_RATIO", "1.0"),
        ("METRALE_MTP_K_LADDER", "1:3,2:1,4:2,8:2,16:1"),
        ("METRALE_PREFILL_CODISPATCH", "1"),
        ("METRALE_PREFILL_CODISPATCH_WINDOW_MS", "50"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    let record = bare.clone().with_serve_env(applied.clone());
    assert_eq!(record.serve_env, applied);
    assert_eq!(
        record
            .perf_env
            .get("METRALE_PREFILL_CODISPATCH_WINDOW_MS")
            .map(String::as_str),
        Some("50"),
        "the applied set is what the server read, so it is what the record says"
    );
    assert_eq!(
        record
            .perf_env
            .get("METRALE_PREFILL_CODISPATCH_SETTLE_MS")
            .map(String::as_str),
        Some("10"),
        "a control the set does not name still resolves to its default"
    );
    // 2026-09-26: An applied `METRALE_PREFILL_CODISPATCH` is disclosed in
    // `serve_env` only, never in `perf_env`.
    assert!(
        !record.perf_env.contains_key("METRALE_PREFILL_CODISPATCH"),
        "{:?}",
        record.perf_env
    );
    assert_eq!(record.serve_env["METRALE_PREFILL_CODISPATCH"], "1");
    let json = serde_json::to_value(&record).unwrap();
    assert_eq!(json["serve_env"]["METRALE_FP8_ROWWISE"], "1");
    assert_eq!(
        json["serve_env"]["METRALE_MTP_K_LADDER"],
        "1:3,2:1,4:2,8:2,16:1"
    );
    let back: GateRecord = serde_json::from_value(json).unwrap();
    assert_eq!(back.serve_env, applied);
    assert_eq!(back.perf_env, record.perf_env);
}

/// 2026-09-26: A set `METRALE_NO_W4A16_TC` or `METRALE_W4A16_TC_WIDE` is
/// recorded verbatim; `0` is a set value for these presence switches.
#[test]
fn a_set_tc_kill_switch_is_disclosed_verbatim() {
    for var in ["METRALE_NO_W4A16_TC", "METRALE_W4A16_TC_WIDE"] {
        for v in ["1", "0"] {
            let resolved = super::resolve_perf_env(|k| (k == var).then(|| v.to_string()));
            assert_eq!(resolved.get(var).map(String::as_str), Some(v));
        }
    }
}

/// 2026-09-26: The `unset` defaults are right only while `tc_enabled` treats
/// unset or empty `METRALE_NO_W4A16_TC` as on and `wide_rows_enabled` treats
/// only a non-empty `METRALE_W4A16_TC_WIDE` as on. Reads `gemv_tc.rs` so a
/// rule change there fails here.
#[test]
fn tc_kill_switch_default_matches_the_lever() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("crates/model-layers/src/layers/ops/gemv_tc.rs");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let at = src
        .find("pub fn tc_enabled()")
        .expect("the tensor-core lever SSOT is gone; PERF_CONTROLS discloses a dead variable");
    let body: String = src[at..].chars().take(260).collect();
    assert!(
        body.contains("std::env::var_os(\"METRALE_NO_W4A16_TC\").is_none_or(|v| v.is_empty())"),
        "the tc kill-switch rule changed; the record's \"unset\" default assumes unset or \
         empty means the tensor-core path ran: {body}"
    );
    let at = src
        .find("pub fn wide_rows_enabled()")
        .expect("the row-edge widening lever is gone; PERF_CONTROLS discloses a dead variable");
    let body: String = src[at..].chars().take(260).collect();
    assert!(
        body.contains("std::env::var_os(\"METRALE_W4A16_TC_WIDE\").is_some_and(|v| !v.is_empty())"),
        "the widening opt-in rule changed; the record's \"unset\" default assumes unset or \
         empty means the widening did NOT run: {body}"
    );
    let resolved = super::resolve_perf_env(|_| None);
    for var in ["METRALE_NO_W4A16_TC", "METRALE_W4A16_TC_WIDE"] {
        assert_eq!(resolved.get(var).map(String::as_str), Some("unset"));
    }
}

/// 2026-09-26: The `unset` default for `METRALE_NO_MTP_TC` is right only while
/// `mtp_tc_from` treats unset or empty as on and any other value, `0`
/// included, as off. Reads `dense_gemv_tc.rs`; a set value is recorded
/// verbatim.
#[test]
fn tc_mtp_switch_default_matches_the_lever() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("crates/model-layers/src/layers/ops/dense_gemv_tc.rs");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let rule = src
        .find("pub fn mtp_tc_from(")
        .map(|at| src[at..].chars().take(200).collect::<String>())
        .expect("the drafter tc rule is gone; PERF_CONTROLS discloses a dead variable");
    assert!(
        rule.contains("kill.is_none_or(|v| v.is_empty())"),
        "the METRALE_NO_MTP_TC rule changed; the record's \"unset\" default assumes unset or \
         empty means the tensor-core path ran: {rule}"
    );
    let read = src
        .find("pub fn mtp_tc_enabled()")
        .map(|at| src[at..].chars().take(300).collect::<String>())
        .expect("the drafter tc lever is gone");
    assert!(
        read.contains("mtp_tc_from(std::env::var_os(\"METRALE_NO_MTP_TC\")"),
        "the lever no longer reads METRALE_NO_MTP_TC through mtp_tc_from: {read}"
    );
    assert_eq!(
        super::resolve_perf_env(|_| None)
            .get("METRALE_NO_MTP_TC")
            .map(String::as_str),
        Some("unset")
    );
    for v in ["1", "0"] {
        let resolved =
            super::resolve_perf_env(|k| (k == "METRALE_NO_MTP_TC").then(|| v.to_string()));
        assert_eq!(
            resolved.get("METRALE_NO_MTP_TC").map(String::as_str),
            Some(v)
        );
    }
}

/// 2026-09-26: The `unset` default for `METRALE_NO_BORROW_STREAK_LIMIT` is
/// right only while `streak_limit_from` treats unset or empty as on. Reads
/// `borrow_streak.rs`; a set value is recorded verbatim.
#[test]
fn borrow_streak_switch_default_matches_the_lever() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("crates/model-engine/src/model/trait_impl/borrow_streak.rs");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let rule = src
        .find("pub fn streak_limit_from(")
        .map(|at| src[at..].chars().take(200).collect::<String>())
        .expect("the borrow-streak rule is gone; PERF_CONTROLS discloses a dead variable");
    assert!(
        rule.contains("kill.is_none_or(|v| v.is_empty())"),
        "the METRALE_NO_BORROW_STREAK_LIMIT rule changed; the record's \"unset\" default \
         assumes unset or empty means the guard ran: {rule}"
    );
    assert!(
        src.contains("streak_limit_from(std::env::var_os(\"METRALE_NO_BORROW_STREAK_LIMIT\")"),
        "the guard no longer reads METRALE_NO_BORROW_STREAK_LIMIT through streak_limit_from"
    );
    assert_eq!(
        super::resolve_perf_env(|_| None)
            .get("METRALE_NO_BORROW_STREAK_LIMIT")
            .map(String::as_str),
        Some("unset")
    );
    for v in ["1", "0"] {
        let resolved = super::resolve_perf_env(|k| {
            (k == "METRALE_NO_BORROW_STREAK_LIMIT").then(|| v.to_string())
        });
        assert_eq!(
            resolved
                .get("METRALE_NO_BORROW_STREAK_LIMIT")
                .map(String::as_str),
            Some(v)
        );
    }
}
