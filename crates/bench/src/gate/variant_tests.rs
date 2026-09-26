// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of model-variant records: two checkpoints of one
//! benchmark get separate files, only the default discharges a required gate,
//! and each is scored against its own baseline entry.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::tests::{MODEL, SHA, TEST_HW, hw, run_record, tempdir, write_baseline};
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;
use std::path::Path;

const DENSE: &str = "unsloth/Qwen3.8-27B-NVFP4";

/// 2026-09-26: A gb10 baseline with two models: `MODEL`, the default, with
/// floor 84.0, and `DENSE` with floor 87.0.
fn two_variant_baseline() -> GateBaseline {
    let bound = |min: f64| Bound {
        min: Some(min),
        ..Bound::default()
    };
    let entry = |recipe: &str, min: f64| ModelBaseline {
        recipe: Some(recipe.to_string()),
        label: String::new(),
        note: String::new(),
        metrics: BTreeMap::from([("overall_accuracy".to_string(), bound(min))]),
        serve_overrides: BTreeMap::new(),
        param_overrides: BTreeMap::new(),
        serve_env: BTreeMap::new(),
    };
    let mut models = BTreeMap::new();
    models.insert(MODEL.to_string(), entry("qwen3.6/moe", 84.0));
    models.insert(DENSE.to_string(), entry("qwen3.8/dense", 87.0));
    GateBaseline {
        schema: 2,
        hardware: BTreeMap::from([(
            TEST_HW.to_string(),
            HardwareBaseline {
                default: MODEL.to_string(),
                models,
            },
        )]),
    }
}

fn plant_variant(root: &Path, model: &str, secs: u64) -> std::path::PathBuf {
    let mut record = run_record(
        BTreeMap::from([("overall_accuracy".to_string(), 90.0)]),
        Verdict::pass("ok"),
    );
    record.target_model = model.to_string();
    record.recorded_at = secs;
    let mut gate = GateRecord::from_run(&record, hw(), SHA.to_string(), Vec::new(), None).unwrap();
    gate.benchmark_id = "ssm-state-poisoning-gate".to_string();
    write_record(root, &gate).unwrap()
}

#[test]
fn the_variant_slug_is_filename_safe_and_lossy_on_purpose() {
    assert_eq!(variant_slug(DENSE), "unsloth-qwen3.8-27b-nvfp4");
    assert_eq!(
        variant_slug("Qwen/Qwen3.6-35B-A3B-FP8"),
        "qwen-qwen3.6-35b-a3b-fp8"
    );
    assert_eq!(variant_slug("a//b"), "a-b", "no doubled separators");
    assert_eq!(variant_slug("/x/"), "x", "no leading or trailing separator");
}

/// 2026-09-26: One commit, one UTC day, both variants measured: the default
/// gets `YYYY-MM-DD-<sha>.json`, the other a slugged name, and neither
/// replaces the other's record.
#[test]
fn both_variants_records_coexist_for_one_commit_and_day() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    write_baseline(root, "ssm-state-poisoning-gate", &two_variant_baseline());

    let default_path = plant_variant(root, MODEL, 1_785_891_382);
    let dense_path = plant_variant(root, DENSE, 1_785_891_382 + 60);

    assert_eq!(
        default_path,
        root.join(".benchmarks/ssm-state-poisoning-gate/2026-08-05-b72dad1893.json")
    );
    assert_eq!(
        dense_path,
        root.join(".benchmarks/ssm-state-poisoning-gate/2026-08-05-b72dad1893-unsloth-qwen3.8-27b-nvfp4.json")
    );
    assert!(default_path.exists() && dense_path.exists());
    // 2026-09-26: Each record names its model, whatever the file is called.
    assert_eq!(read_record(&default_path).unwrap().target_model, MODEL);
    assert_eq!(read_record(&dense_path).unwrap().target_model, DENSE);
}

#[test]
fn lossy_variant_slugs_cannot_overwrite_each_other() {
    const ALIAS: &str = "unsloth/Qwen3.8/27B/NVFP4";
    assert_eq!(variant_slug(DENSE), variant_slug(ALIAS));
    let dir = tempdir::Dir::new();
    let root = dir.path();
    let mut baseline = two_variant_baseline();
    let dense = baseline.hardware[TEST_HW].models[DENSE].clone();
    baseline
        .hardware
        .get_mut(TEST_HW)
        .unwrap()
        .models
        .insert(ALIAS.into(), dense);
    write_baseline(root, "ssm-state-poisoning-gate", &baseline);

    let first = plant_variant(root, DENSE, 1_785_891_382);
    let second = plant_variant(root, ALIAS, 1_785_891_382 + 60);
    assert_ne!(first, second);
    assert_eq!(read_record(&first).unwrap().target_model, DENSE);
    assert_eq!(read_record(&second).unwrap().target_model, ALIAS);
}

/// 2026-09-26: A same-day re-run of one non-default variant replaces that
/// variant's record.
#[test]
fn a_variant_rerun_replaces_only_its_own_record() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    write_baseline(root, "ssm-state-poisoning-gate", &two_variant_baseline());
    let first = plant_variant(root, DENSE, 1_785_891_382);
    let second = plant_variant(root, DENSE, 1_785_891_382 + 3_600);
    assert_eq!(first, second, "same variant + sha + UTC day = same file");
    assert_eq!(read_record(&second).unwrap().recorded_at, 1_785_894_982);
}

/// 2026-09-26: A benchmark with no baseline gets the default filename
/// `YYYY-MM-DD-<sha>.json` for any model.
#[test]
fn no_baseline_means_the_legacy_filename() {
    let dir = tempdir::Dir::new();
    let path = plant_variant(dir.path(), DENSE, 1_785_891_382);
    assert_eq!(
        path,
        dir.path()
            .join(".benchmarks/ssm-state-poisoning-gate/2026-08-05-b72dad1893.json")
    );
}

/// 2026-09-26: With an unknown box class (`fetch_hardware` returns
/// `Hardware::unknown()` on error), a model that is no hardware's default
/// still gets a slugged name, so it cannot replace the default's record; a
/// model that is some hardware's default gets the default filename.
#[test]
fn an_unknown_hardware_variant_record_does_not_clobber_the_default() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    write_baseline(root, "ssm-state-poisoning-gate", &two_variant_baseline());
    let default_path = plant_variant(root, MODEL, 1_785_891_382);

    let mut record = run_record(
        BTreeMap::from([("overall_accuracy".to_string(), 90.0)]),
        Verdict::pass("ok"),
    );
    record.target_model = DENSE.to_string();
    record.recorded_at = 1_785_891_382 + 60;
    let mut gate = GateRecord::from_run(
        &record,
        crate::hardware::Hardware::unknown(),
        SHA.to_string(),
        Vec::new(),
        None,
    )
    .unwrap();
    gate.benchmark_id = "ssm-state-poisoning-gate".to_string();
    let dense_path = write_record(root, &gate).unwrap();

    assert_eq!(
        dense_path,
        root.join(".benchmarks/ssm-state-poisoning-gate/2026-08-05-b72dad1893-unsloth-qwen3.8-27b-nvfp4.json")
    );
    assert_eq!(read_record(&default_path).unwrap().target_model, MODEL);

    let mut record = run_record(
        BTreeMap::from([("overall_accuracy".to_string(), 90.0)]),
        Verdict::pass("ok"),
    );
    record.target_model = MODEL.to_string();
    record.recorded_at = 1_785_891_382 + 120;
    let mut gate = GateRecord::from_run(
        &record,
        crate::hardware::Hardware::unknown(),
        SHA.to_string(),
        Vec::new(),
        None,
    )
    .unwrap();
    gate.benchmark_id = "ssm-state-poisoning-gate".to_string();
    assert_eq!(write_record(root, &gate).unwrap(), default_path);
}

/// 2026-09-26: A required gate's subject is the default model. A newer,
/// passing record of another model does not discharge it, and alone it reads
/// as missing.
#[test]
fn a_non_default_variant_record_cannot_discharge_the_required_gate() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    for id in REQUIRED_GATES {
        write_baseline(root, id, &two_variant_baseline());
    }
    // 2026-09-26: The dense record is newer than the default's.
    plant_variant(root, MODEL, 1_785_891_382);
    plant_variant(root, DENSE, 1_785_891_382 + 60);

    let gates = check_gates(root, SHA);
    assert!(
        matches!(gates["ssm-state-poisoning-gate"], GateStatus::Pass),
        "the OLDER default record still discharges the gate: {:?}",
        gates["ssm-state-poisoning-gate"]
    );

    std::fs::remove_file(record_path(
        root,
        "ssm-state-poisoning-gate",
        1_785_891_382,
        SHA,
    ))
    .unwrap();
    let gates = check_gates(root, SHA);
    assert!(matches!(
        &gates["ssm-state-poisoning-gate"],
        GateStatus::Missing(reason)
            if reason == "latest record measured the unsloth/Qwen3.8-27B-NVFP4 variant; the required subject on gb10 is Qwen/Qwen3.6-35B-A3B-FP8, which has no covering record"
    ));
}

/// 2026-09-26: `check_record` scores a variant record against its own entry.
#[test]
fn a_variant_record_is_scored_against_its_own_entry() {
    let baseline = two_variant_baseline();
    let mut record = run_record(
        BTreeMap::from([("overall_accuracy".to_string(), 86.0)]),
        Verdict::pass("ok"),
    );
    record.target_model = DENSE.to_string();
    let gate = GateRecord::from_run(&record, hw(), SHA.to_string(), Vec::new(), None).unwrap();
    // 2026-09-26: 86.0 clears the default's floor (84.0) but not the dense
    // one (87.0), so only the dense entry fails it.
    assert_eq!(
        check_record(&gate, &baseline),
        Some(vec![
            "overall_accuracy 86.00 is below the floor 87.00 (noise 0.00)".into()
        ])
    );
}
