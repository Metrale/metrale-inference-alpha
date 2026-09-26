// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for serve and param overrides on a gate record: how they
//! are recorded, replayed in `command`, and checked against baseline pins.
//!
//! Owner: bench gate (records).
//! Invariants: none beyond the types.

use super::tests::*;
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;

/// 2026-09-26: A passing run record measured under `overrides`. `from_run`
/// reads the overrides off the run record, as in production.
fn served_with(
    metrics: BTreeMap<String, f64>,
    overrides: &BTreeMap<String, String>,
) -> crate::history::RunRecord {
    let mut r = run_record(metrics, Verdict::pass("ok"));
    r.serve_overrides = overrides.clone();
    r
}

#[test]
fn a_run_with_serve_overrides_records_them_and_stays_replayable() {
    let mut overrides = BTreeMap::new();
    overrides.insert("kv_cache_dtype".to_string(), "fp8".to_string());
    overrides.insert("fp8_kv_calibration_tokens".to_string(), "512".to_string());
    let gate = GateRecord::from_run(
        &served_with(BTreeMap::new(), &overrides),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth".to_string()),
    )
    .unwrap();
    assert_eq!(gate.serve_overrides, overrides);
    assert_eq!(
        gate.command,
        [
            "met",
            "benchmark",
            "run",
            "bfcl-subset",
            "--param",
            "repeats=12",
            "--serve-override",
            "fp8_kv_calibration_tokens=512",
            "--serve-override",
            "kv_cache_dtype=fp8",
            "--pull-request-gate",
        ]
    );
}

/// 2026-09-26: Without overrides the record has an empty map, `command` has no
/// `--serve-override`, and the JSON omits the field.
#[test]
fn a_run_without_overrides_carries_no_override_provenance() {
    let gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth".to_string()),
    )
    .unwrap();
    assert!(gate.serve_overrides.is_empty());
    assert!(!gate.command.join(" ").contains("--serve-override"));
    let json = serde_json::to_string(&gate).unwrap();
    assert!(!json.contains("serve_overrides"), "{json}");
}

/// 2026-09-26: A record without a baseline serve pin fails, naming the pin.
#[test]
fn a_record_missing_a_baseline_serve_pin_fails() {
    let mut baseline = bfcl_baseline();
    baseline
        .hardware
        .get_mut(TEST_HW)
        .unwrap()
        .models
        .get_mut(MODEL)
        .unwrap()
        .serve_overrides
        .insert("ssm_cache_slots".into(), "256".into());

    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth".into()),
    )
    .unwrap();
    let problems = check_record(&gate, &baseline).expect("must fail");
    assert_eq!(
        problems,
        [
            "serve override ssm_cache_slots=256 is pinned on the baseline but missing from the record"
        ]
    );
}

/// 2026-09-26: A record carrying the pin at the pinned value passes; the pin
/// check adds no failure of its own.
#[test]
fn a_record_with_the_baseline_serve_pin_still_scores_metrics() {
    let mut baseline = bfcl_baseline();
    baseline
        .hardware
        .get_mut(TEST_HW)
        .unwrap()
        .models
        .get_mut(MODEL)
        .unwrap()
        .serve_overrides
        .insert("ssm_cache_slots".into(), "256".into());

    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let mut overrides = BTreeMap::new();
    overrides.insert("ssm_cache_slots".to_string(), "256".to_string());
    let gate = GateRecord::from_run(
        &served_with(metrics, &overrides),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth".into()),
    )
    .unwrap();
    assert!(check_record(&gate, &baseline).is_none());
}

#[test]
fn a_record_with_an_unpinned_serve_override_fails() {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let gate = GateRecord::from_run(
        &served_with(
            metrics,
            &BTreeMap::from([("kv_cache_dtype".to_string(), "fp8".to_string())]),
        ),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap();
    let problems = check_record(&gate, &bfcl_baseline()).expect("must fail");
    assert_eq!(
        problems,
        [
            "serve override kv_cache_dtype=fp8 is present on the record but not pinned by the baseline"
        ]
    );
}

/// 2026-09-26: A record served at another value fails, naming both values.
#[test]
fn a_record_with_a_different_pin_value_fails() {
    let mut baseline = bfcl_baseline();
    baseline
        .hardware
        .get_mut(TEST_HW)
        .unwrap()
        .models
        .get_mut(MODEL)
        .unwrap()
        .serve_overrides
        .insert("ssm_cache_slots".into(), "256".into());

    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let mut overrides = BTreeMap::new();
    overrides.insert("ssm_cache_slots".to_string(), "16".to_string());
    let gate = GateRecord::from_run(
        &served_with(metrics, &overrides),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth".into()),
    )
    .unwrap();
    let problems = check_record(&gate, &baseline).expect("must fail");
    assert_eq!(
        problems,
        ["serve override ssm_cache_slots=16 does not match the baseline pin ssm_cache_slots=256"]
    );
}

fn baseline_with_param_pin(key: &str, value: &str) -> crate::gate::GateBaseline {
    let mut baseline = bfcl_baseline();
    baseline
        .hardware
        .get_mut(TEST_HW)
        .unwrap()
        .models
        .get_mut(MODEL)
        .unwrap()
        .param_overrides
        .insert(key.into(), value.into());
    baseline
}

/// 2026-09-26: A record without a baseline param pin fails, naming the pin.
#[test]
fn a_record_missing_a_baseline_param_pin_fails() {
    let baseline = baseline_with_param_pin("osl", "320");
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap();
    let problems = check_record(&gate, &baseline).expect("must fail");
    assert_eq!(
        problems,
        ["param osl=320 is pinned on the baseline but missing from the record"]
    );
}

/// 2026-09-26: A record carrying the param pin passes, and `1, 4, 8, 16` on the
/// record matches the pin `1,4,8,16`.
#[test]
fn a_record_with_the_baseline_param_pin_scores_and_list_rendering_matches() {
    let baseline = baseline_with_param_pin("concurrencies", "1,4,8,16");
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let mut record = run_record(metrics, Verdict::pass("ok"));
    record
        .params
        .insert("concurrencies".into(), "1, 4, 8, 16".into());
    let gate = GateRecord::from_run(&record, hw(), SHA.into(), Vec::new(), None).unwrap();
    assert!(check_record(&gate, &baseline).is_none());
}

/// 2026-09-26: A param at another value fails, naming both; spaces inside an
/// item (`3 20`) are not ignored.
#[test]
fn a_record_with_a_different_param_pin_value_fails() {
    let baseline = baseline_with_param_pin("osl", "320");
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".into(), 90.0);
    let mut record = run_record(metrics, Verdict::pass("ok"));
    record.params.insert("osl".into(), "3 20".into());
    let gate = GateRecord::from_run(&record, hw(), SHA.into(), Vec::new(), None).unwrap();
    let problems = check_record(&gate, &baseline).expect("must fail");
    assert_eq!(
        problems,
        [
            "param osl=3 20 does not match the baseline pin osl=320 — the run measured a \
             different instrument than the one these thresholds describe"
        ]
    );
}

/// 2026-09-26: A multi-GPU run records both the box's GPU count
/// (`hardware.gpu_count`) and the topology overrides it served with, and both
/// survive `command` replay and a JSON round trip.
#[test]
fn a_multi_gpu_run_records_its_width_and_replays_its_topology_flags() {
    let mut overrides = BTreeMap::new();
    overrides.insert("tp_size".to_string(), "2".to_string());
    overrides.insert("ep_size".to_string(), "2".to_string());
    overrides.insert("world_size".to_string(), "2".to_string());
    let mut record = run_record(BTreeMap::new(), Verdict::pass("ok"));
    record.serve_overrides = overrides.clone();
    let mut gate = GateRecord::from_run(
        &record,
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth".to_string()),
    )
    .unwrap();
    gate.hardware = crate::hardware::Hardware {
        gpu: "NVIDIA H100 80GB HBM3".into(),
        driver: "580.126.09".into(),
        gpu_count: Some(8),
        ..Default::default()
    };

    assert_eq!(gate.hardware.gpu_count, Some(8));
    assert_eq!(gate.hardware.gate_key(), "h100");
    assert_eq!(gate.serve_overrides, overrides);
    let replay = gate.command.join(" ");
    for pin in [
        "--serve-override ep_size=2",
        "--serve-override tp_size=2",
        "--serve-override world_size=2",
    ] {
        assert!(replay.contains(pin), "{pin} missing from: {replay}");
    }
    let back: GateRecord = serde_json::from_str(&serde_json::to_string(&gate).unwrap()).unwrap();
    assert_eq!(back.hardware.gpu_count, Some(8));
    assert_eq!(back.serve_overrides, overrides);
    assert_eq!(back.command, gate.command);
}
