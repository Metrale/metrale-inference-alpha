// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `dirty_perf_paths`, and for a gate record that carries
//! dirty paths. The `git status` cases build a scratch git repository.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::coverage_tests::scratch_repo;
use super::tests::{tempdir, *};
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;

/// 2026-09-26: `.benchmarks/` is not a perf path, so an uncommitted record file is not
/// reported; an edited file under `crates/` is.
#[test]
fn a_dirty_record_file_is_silent_and_a_dirty_crate_is_not() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    scratch_repo::commit(root, "crates/a/src/lib.rs", "fn a() {}", "add a");

    assert!(
        dirty_perf_paths(root).unwrap().is_empty(),
        "a freshly committed tree has nothing uncommitted"
    );

    std::fs::create_dir_all(root.join(".benchmarks/bfcl-subset")).unwrap();
    std::fs::write(
        root.join(".benchmarks/bfcl-subset/2026-08-07-abc.json"),
        "{}",
    )
    .unwrap();
    assert!(
        dirty_perf_paths(root).unwrap().is_empty(),
        "an uncommitted gate record is the normal state of a campaign"
    );

    std::fs::write(root.join("crates/a/src/lib.rs"), "fn a() { todo!() }").unwrap();
    assert_eq!(dirty_perf_paths(root).unwrap(), ["crates/a/src/lib.rs"]);
}

#[test]
fn an_untracked_kernel_counts_but_an_ignored_file_does_not() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    std::fs::create_dir_all(root.join("kernels")).unwrap();
    std::fs::write(root.join("kernels/new.cu"), "__global__ void k() {}").unwrap();
    assert_eq!(dirty_perf_paths(root).unwrap(), ["kernels/new.cu"]);

    scratch_repo::commit(root, ".gitignore", "kernels/*.cu\n", "ignore built kernels");
    assert!(
        dirty_perf_paths(root).unwrap().is_empty(),
        "an ignored file is not evidence of an unrecorded source change"
    );
}

#[test]
fn a_non_checkout_errs_rather_than_reporting_a_clean_tree() {
    let dir = tempdir::Dir::new();
    assert!(
        dirty_perf_paths(dir.path()).is_err(),
        "no git metadata means the question is unanswered, not answered clean"
    );
}

/// 2026-09-26: A record with non-empty `dirty_paths` fails the gate, although its
/// metrics clear the baseline and its own verdict passes.
#[test]
fn a_record_measured_from_a_dirty_tree_fails_the_gate() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    std::fs::create_dir_all(gate_dir(root, "ssm-state-poisoning-gate")).unwrap();
    write_baseline(root, "ssm-state-poisoning-gate", &bfcl_baseline());
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 90.0);

    let mut gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        vec!["crates/model-layers/src/layers/gdn.rs".to_string()],
        None,
    )
    .unwrap();
    gate.benchmark_id = "ssm-state-poisoning-gate".to_string();
    gate.recorded_at = 1_785_891_382;
    write_record(root, &gate).unwrap();

    assert!(gate.verdict_passes());
    assert!(check_record(&gate, &bfcl_baseline()).is_none());

    match &check_gates(root, SHA)["ssm-state-poisoning-gate"] {
        GateStatus::Fail(reasons) => assert_eq!(
            reasons,
            &[format!(
                "measured from a dirty tree — 1 uncommitted invalidation-set file(s) \
                 when the run started (crates/model-layers/src/layers/gdn.rs), so the binary \
                 was not {SHA}"
            )]
        ),
        other => panic!("a record that names no commit is not a pass: {other:?}"),
    }
}

/// 2026-09-26: A clean record serialises without a `dirty_paths` key, and JSON without
/// the key parses to an empty list.
#[test]
fn the_field_is_absent_when_clean_and_optional_when_reading() {
    let gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap();
    let json = serde_json::to_string(&gate).unwrap();
    assert!(!json.contains("dirty_paths"), "{json}");

    let older: GateRecord = serde_json::from_str(&json).expect("an older record still parses");
    assert!(older.dirty_paths.is_empty());
}
