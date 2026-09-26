// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of gate records and baselines (`compare`, `check_record`,
//! write and read, `check_gates`), plus the fixtures other gate tests share:
//! `hw`, `run_record`, `baseline_for`, `plant`, `plant_required` and
//! `tempdir`.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::*;
use crate::hardware::Hardware;
use crate::history::{RunRecord, RunSource};
use crate::result::{BenchmarkResult, RunStatus, Verdict};
use std::collections::BTreeMap;

pub(super) const MODEL: &str = "Qwen/Qwen3.6-35B-A3B-FP8";
/// 2026-09-26: The key `hw().gate_key()` yields, and the hardware key of
/// [`baseline_for`].
pub(super) const TEST_HW: &str = "gb10";

/// 2026-09-26: A GB10 fingerprint, whose `gpu` string the SKU table keys as
/// `gb10`, so `gate_key()` runs its real derivation;
/// `an_unknown_fingerprint_never_silently_matches` covers the unknown one.
pub(super) fn hw() -> Hardware {
    Hardware {
        gpu: "NVIDIA GB10".to_string(),
        driver: "580.126.09".to_string(),
        sm_clock_mhz: Some(2405.0),
        gpu_count: Some(1),
        source: "nvidia-smi".to_string(),
    }
}
pub(super) const SHA: &str = "b72dad1893";

pub(super) mod tempdir {
    use std::path::{Path, PathBuf};
    pub struct Dir(PathBuf);
    impl Dir {
        pub fn new() -> Self {
            let n = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default();
            let p = std::env::temp_dir().join(format!(
                "metrale-gate-{n}-{:?}",
                std::thread::current().id()
            ));
            std::fs::create_dir_all(&p).expect("scratch dir");
            Self(p)
        }
        pub fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

pub(super) fn frame(
    status: RunStatus,
    metrics: BTreeMap<String, f64>,
    verdict: Verdict,
) -> BenchmarkResult {
    let mut f = BenchmarkResult::completed("done", std::time::Duration::ZERO);
    f.status = status;
    f.with_metrics(metrics).with_verdict(verdict)
}

pub(super) fn run_record(metrics: BTreeMap<String, f64>, verdict: Verdict) -> RunRecord {
    let mut params = BTreeMap::new();
    params.insert("repeats".to_string(), "12".to_string());
    RunRecord {
        schema: 1,
        run_id: "run-1".to_string(),
        benchmark_id: "bfcl-subset".to_string(),
        benchmark_name: "BFCL (subset)".to_string(),
        recorded_at: 1_785_891_382,
        serve_overrides: Default::default(),
        target_url: "http://127.0.0.1:8888".to_string(),
        target_model: MODEL.to_string(),
        params,
        source: RunSource::Cli,
        metrale_version: "test".to_string(),
        frame: frame(RunStatus::Completed, metrics, verdict),
    }
}

pub(super) use super::fixture_baseline::write_baseline;

pub(super) fn bfcl_baseline() -> GateBaseline {
    let mut metrics = BTreeMap::new();
    metrics.insert(
        "overall_accuracy".to_string(),
        Bound {
            min: Some(83.64),
            ..Bound::default()
        },
    );
    baseline_for(MODEL, metrics)
}

/// 2026-09-26: A schema-2 baseline with one hardware class, [`TEST_HW`], whose
/// default and only model is `model`.
pub(super) fn baseline_for(model: &str, metrics: BTreeMap<String, Bound>) -> GateBaseline {
    let mut models = BTreeMap::new();
    models.insert(
        model.to_string(),
        crate::gate::ModelBaseline {
            recipe: Some("qwen3.6/test-recipe".to_string()),
            label: String::new(),
            note: "MLPerf floor".to_string(),
            metrics,
            serve_overrides: BTreeMap::new(),
            param_overrides: BTreeMap::new(),
            serve_env: BTreeMap::new(),
        },
    );
    let mut hardware = BTreeMap::new();
    hardware.insert(
        TEST_HW.to_string(),
        crate::gate::HardwareBaseline {
            default: model.to_string(),
            models,
        },
    );
    GateBaseline {
        schema: 2,
        hardware,
    }
}

#[test]
fn compare_enforces_min_max_and_noise() {
    let floor = Bound {
        min: Some(83.64),
        noise: Some(0.4),
        ..Bound::default()
    };
    assert!(matches!(compare("x", 83.24, &floor), Comparison::Pass));
    assert!(matches!(
        compare("x", 83.23, &floor),
        Comparison::Fail(reason)
            if reason == "x 83.23 is below the floor 83.64 (noise 0.40)"
    ));

    let ceiling = Bound {
        max: Some(1300.0),
        ..Bound::default()
    };
    assert!(matches!(
        compare("wall", 1300.0, &ceiling),
        Comparison::Pass
    ));
    assert!(matches!(
        compare("wall", 1300.01, &ceiling),
        Comparison::Fail(reason)
            if reason == "wall 1300.01 is above the ceiling 1300.00 (noise 0.00)"
    ));

    // 2026-09-26: A two-sided bound is a range; both ends pass. An exact pin
    // uses it (`an_exact_pin_passes_only_on_the_pinned_value` in
    // coverage_tests).
    let range = Bound {
        min: Some(1.0),
        max: Some(2.0),
        ..Bound::default()
    };
    assert!(matches!(compare("x", 1.0, &range), Comparison::Pass));
    assert!(matches!(compare("x", 2.0, &range), Comparison::Pass));
    assert!(matches!(
        compare("x", 2.01, &range),
        Comparison::Fail(reason)
            if reason == "x 2.01 is outside [1.00, 2.00] (noise 0.00)"
    ));

    let no_side = Bound::default();
    assert!(matches!(
        compare("x", 1.5, &no_side),
        Comparison::Skip(reason) if reason == "x has no bound"
    ));
}

#[test]
fn check_record_refuses_a_cross_checkpoint_comparison() {
    let gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap();
    // 2026-09-26: The baseline knows only another model, so the record's
    // model does not resolve and the record is refused.
    let mut metrics = BTreeMap::new();
    metrics.insert(
        "overall_accuracy".to_string(),
        Bound {
            min: Some(83.64),
            ..Bound::default()
        },
    );
    let baseline = baseline_for("some-other-model", metrics);
    assert_eq!(
        check_record(&gate, &baseline),
        Some(vec![format!(
            "no baseline for model {MODEL:?} on \"gb10\"; it has [some-other-model]"
        )])
    );
}

#[test]
fn check_record_refuses_a_cross_hardware_comparison() {
    // 2026-09-26: A record from a box class the baseline lacks is refused, and
    // the reason names both sides.
    let mut gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap();
    gate.hardware = Hardware {
        gpu: "AMD Instinct MI300X".to_string(),
        ..Hardware::default()
    };
    // 2026-09-26: The SKU table (`hardware::ids`) keys this part as `mi300x`.
    assert_eq!(
        check_record(&gate, &bfcl_baseline()),
        Some(vec![
            "no baseline for hardware \"mi300x\"; this benchmark has entries for [gb10]".into()
        ])
    );
}

#[test]
fn an_unknown_fingerprint_never_silently_matches() {
    // 2026-09-26: `fetch_hardware` returns `Hardware::unknown()` on every error
    // path without surfacing one; its key, "unknown", resolves to no entry.
    let mut gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap();
    gate.hardware = Hardware::unknown();
    assert_eq!(gate.hardware.gate_key(), "unknown");
    assert_eq!(
        check_record(&gate, &bfcl_baseline()),
        Some(vec![
            "no baseline for hardware \"unknown\"; this benchmark has entries for [gb10]".into()
        ])
    );
}

#[test]
fn check_record_scores_every_bound_and_missing_metric() {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap();
    let mut metrics = BTreeMap::new();
    metrics.insert(
        "overall_accuracy".to_string(),
        Bound {
            min: Some(83.64),
            ..Bound::default()
        },
    );
    metrics.insert(
        "samples".to_string(),
        Bound {
            min: Some(995.0),
            ..Bound::default()
        },
    );
    let baseline = baseline_for(MODEL, metrics);
    assert_eq!(
        check_record(&gate, &baseline),
        Some(vec!["samples: missing from the record".into()])
    );

    let passing = bfcl_baseline();
    assert!(check_record(&gate, &passing).is_none());
    let mut below_floor = gate;
    below_floor.metrics.insert("overall_accuracy".into(), 80.0);
    assert_eq!(
        check_record(&below_floor, &passing),
        Some(vec![
            "overall_accuracy 80.00 is below the floor 83.64 (noise 0.00)".into()
        ])
    );
}

#[test]
fn write_and_read_round_trip_through_the_repo_layout() {
    let dir = tempdir::Dir::new();
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap();
    let path = write_record(dir.path(), &gate).unwrap();
    assert_eq!(
        path,
        dir.path()
            .join(".benchmarks/bfcl-subset/2026-08-05-b72dad1893.json")
    );
    let back = read_record(&path).unwrap();
    assert_eq!(
        serde_json::to_value(&back).unwrap(),
        serde_json::to_value(&gate).unwrap()
    );
    assert!(std::fs::read_to_string(path).unwrap().ends_with("\n"));
}

pub(super) fn plant(root: &Path, id: &str, sha: &str, secs: u64, verdict: &str) {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 90.0);
    let record = run_record(metrics, Verdict::pass("ok"));
    let mut gate = GateRecord::from_run(&record, hw(), sha.to_string(), Vec::new(), None).unwrap();
    gate.benchmark_id = id.to_string();
    gate.verdict = Some(verdict.to_string());
    gate.recorded_at = secs;
    write_record(root, &gate).unwrap();
}

/// 2026-09-26: Plant records that satisfy a required gate at `sha`: one
/// record for a plain gate, or [`PLANTED_SHARDS`] shard records under a
/// group's id (a group needs a complete partition). The shard records carry
/// the metric keys `benchmarks/bfcl/report.rs` writes: subset tallies,
/// `shard.index`, `shard.count` and `transport_errors`.
pub(super) fn plant_required(root: &Path, id: &str, sha: &str, secs: u64, verdict: &str) {
    if super::group::find(id).is_none() {
        return plant(root, id, sha, secs, verdict);
    }
    for index in 0..PLANTED_SHARDS {
        plant_shard(
            root,
            id,
            (index, PLANTED_SHARDS),
            sha,
            secs + index as u64,
            verdict,
        );
    }
}

/// 2026-09-26: How many shard records `plant_required` writes for a group.
pub(super) const PLANTED_SHARDS: usize = 3;

/// 2026-09-26: One shard record of `group` at `sha`, with tallies and shard
/// identity.
pub(super) fn plant_shard(
    root: &Path,
    group: &str,
    shard: (usize, usize),
    sha: &str,
    secs: u64,
    verdict: &str,
) {
    std::fs::create_dir_all(gate_dir(root, group)).unwrap();
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 90.0);
    metrics.insert("subset.simple_python.hits".to_string(), 95.0);
    metrics.insert("subset.simple_python.n".to_string(), 100.0);
    metrics.insert("shard.index".to_string(), shard.0 as f64);
    metrics.insert("shard.count".to_string(), shard.1 as f64);
    metrics.insert("transport_errors".to_string(), 0.0);
    let record = run_record(metrics, Verdict::pass("ok"));
    let mut gate = GateRecord::from_run(&record, hw(), sha.to_string(), Vec::new(), None).unwrap();
    gate.benchmark_id = group.to_string();
    gate.verdict = Some(verdict.to_string());
    gate.recorded_at = secs;
    write_record(root, &gate).unwrap();
}

#[test]
fn check_gates_reports_each_required_bench() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(root, id)).unwrap();
        write_baseline(root, id, &bfcl_baseline());
    }
    plant(root, "ssm-state-poisoning-gate", SHA, 1_785_891_382, "PASS");
    plant(root, "ttft-warm-gate", "aaaaaaaaaa", 1_785_891_382, "PASS");
    plant(root, "agentic-webserver", SHA, 1_785_891_382, "FAIL");

    let gates = check_gates(root, SHA);
    let actual: BTreeMap<_, _> = gates
        .iter()
        .map(|(id, status)| {
            let class = match status {
                GateStatus::Pass => "Pass",
                GateStatus::Fail(_) => "Fail",
                GateStatus::Missing(_) => "Missing",
            };
            (id.as_str(), class)
        })
        .collect();
    let mut expected: BTreeMap<_, _> = REQUIRED_GATES.map(|id| (id, "Missing")).into();
    expected.insert("ssm-state-poisoning-gate", "Pass");
    expected.insert("agentic-webserver", "Fail");
    assert_eq!(actual, expected);
    assert!(matches!(
        &gates["ttft-warm-gate"],
        GateStatus::Missing(reason) if reason.contains("aaaaaaaaaa")
    ));
    assert!(matches!(
        &gates["agentic-webserver"],
        GateStatus::Fail(reasons) if reasons == &["run verdict is not PASS: ok"]
    ));
}
