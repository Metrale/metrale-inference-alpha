// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for BENCH.toml loading and baseline assembly (`bench.rs`), against the
//! committed tree and against temporary fixtures.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::*;

pub(super) fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("metrale-bench-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

pub(super) fn fixture(name: &str, bench_toml: &str) -> std::path::PathBuf {
    let root = tmp(name);
    let m = root.join("kernels/gb10/modelA");
    std::fs::create_dir_all(m.join("nvfp4")).unwrap();
    std::fs::write(
        root.join("kernels/gb10/HARDWARE.toml"),
        "[hardware]\nvendor = \"nvidia\"\n",
    )
    .unwrap();
    std::fs::write(m.join("MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(m.join("BENCH.toml"), bench_toml).unwrap();
    root
}

/// 2026-09-26: Every required gate resolves a baseline from the committed tree, with a default
/// checkpoint on each hardware that is among that hardware's own models.
#[test]
fn every_required_gate_still_resolves_a_baseline() {
    let root = repo_root();
    for id in super::super::REQUIRED_GATES {
        let baseline = baseline_for(&root, id).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert!(
            !baseline.hardware.is_empty(),
            "{id}: no hardware entries after the move"
        );
        for (hw, entry) in &baseline.hardware {
            assert!(
                !entry.default.is_empty(),
                "{id}/{hw}: no default checkpoint"
            );
            assert!(
                entry.models.contains_key(&entry.default),
                "{id}/{hw}: default {:?} is not among its own models",
                entry.default
            );
        }
    }
}

/// 2026-09-26: `BENCH.toml` is not a closure input, so ratcheting a bar does not invalidate the
/// record that justified it; `MODEL.toml` is one.
#[test]
fn bench_toml_is_not_a_closure_input() {
    let root = repo_root();
    let target = taxon::Target {
        hardware: "gb10".into(),
        model: "qwen3.6-27b".into(),
        quant: "nvfp4".into(),
    };
    let configs = taxon::configs(&root, &target);
    assert!(
        !configs.iter().any(|p| p.ends_with("BENCH.toml")),
        "BENCH.toml must not be hashed: a threshold ratchet would invalidate \
         the very record that justified it. Found: {configs:?}"
    );
    assert!(
        configs.iter().any(|p| p.ends_with("MODEL.toml")),
        "MODEL.toml IS compiled in and must stay hashed: {configs:?}"
    );
}

/// 2026-09-26: An unmeasured entry carrying thresholds is rejected at load, so an unmeasured bar
/// can never report PASS.
#[test]
fn an_unmeasured_entry_carrying_thresholds_is_rejected() {
    let root = fixture(
        "guessed",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/M"
gate = "bfcl-subset"
status = "unmeasured"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    let err = load_all(&root).unwrap_err().to_string();
    assert_eq!(
        err,
        format!(
            "{}: bfcl-subset / org/M is unmeasured but carries thresholds. A guessed number a run can clear is worse than no number — it reports PASS for something nobody measured.",
            root.join("kernels/gb10/modelA/BENCH.toml").display()
        )
    );
}

/// 2026-09-26: A measured entry with no metrics table, or an empty one, is rejected at load.
#[test]
fn a_measured_entry_without_thresholds_is_rejected() {
    for (name, metrics) in [("absent", ""), ("empty", "[benchmarks.metrics]")] {
        let root = fixture(
            name,
            &format!(
                r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/M"
gate = "bfcl-subset"
status = "measured"
{metrics}
"#
            ),
        );
        let err = load_all(&root).unwrap_err().to_string();
        assert_eq!(
            err,
            format!(
                "{}: bfcl-subset / org/M claims to be measured but declares no metrics",
                root.join("kernels/gb10/modelA/BENCH.toml").display()
            ),
            "partition {name}"
        );
    }
}

#[test]
fn an_unknown_status_is_rejected_rather_than_treated_as_unmeasured() {
    let root = fixture(
        "bad-status",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/M"
gate = "bfcl-subset"
status = "probably-fine"
"#,
    );
    assert_eq!(
        load_all(&root).unwrap_err().to_string(),
        format!(
            "{}: status must be \"measured\" or \"unmeasured\", got \"probably-fine\"",
            root.join("kernels/gb10/modelA/BENCH.toml").display()
        )
    );
}

/// 2026-09-26: An unmeasured entry is dropped from the baseline, which assembles empty, and
/// resolving against it fails with "no baseline for hardware".
#[test]
fn an_unmeasured_entry_produces_no_baseline_at_all() {
    let root = fixture(
        "unmeasured",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/M"
gate = "bfcl-subset"
status = "unmeasured"
"#,
    );
    let baseline = baseline_for(&root, "bfcl-subset").unwrap();
    assert_eq!(baseline.schema, 2);
    assert!(baseline.hardware.is_empty(), "{baseline:?}");
    let err = baseline.resolve("gb10", None).unwrap_err().to_string();
    assert_eq!(
        err,
        "no baseline for hardware \"gb10\"; this benchmark has entries for []"
    );
}

/// 2026-09-26: Two checkpoints claiming `default = true` on one hardware is an error: the gate
/// would have no single subject.
#[test]
fn two_checkpoints_claiming_default_is_an_error() {
    let root = fixture(
        "two-defaults",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 85.0

[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/B"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 86.0
"#,
    );
    let err = baseline_for(&root, "bfcl-subset").unwrap_err().to_string();
    assert_eq!(
        err,
        "bfcl-subset: both org/A (in modelA) and org/B (in modelA) claim to be the default on gb10"
    );
}

/// 2026-09-26: A lone checkpoint must still set `default = true`, so adding a second one later
/// cannot silently change which one the gate scores.
#[test]
fn a_lone_checkpoint_must_still_declare_itself_default() {
    let root = fixture(
        "no-default",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    assert_eq!(
        baseline_for(&root, "bfcl-subset").unwrap_err().to_string(),
        "bfcl-subset: no checkpoint on gb10 sets `default = true`; one must, or the gate has no defined subject"
    );
}

#[test]
fn the_same_checkpoint_declared_twice_for_one_gate_is_an_error() {
    let root = fixture(
        "dupe",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 85.0

[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 90.0
"#,
    );
    assert_eq!(
        baseline_for(&root, "bfcl-subset").unwrap_err().to_string(),
        "bfcl-subset: org/A is declared twice on gb10"
    );
}

/// 2026-09-26: BENCH.toml is per model, so a model with two quant dirs still yields each entry
/// once.
#[test]
fn entries_are_not_duplicated_across_a_models_quant_dirs() {
    let root = fixture(
        "multi-quant",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    std::fs::create_dir_all(root.join("kernels/gb10/modelA/fp8")).unwrap();
    let all = load_all(&root).unwrap();
    assert_eq!(all.len(), 1, "one entry, not one per quant dir: {all:?}");
}
