// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for which committed record still covers a head (`record_covers`,
//! `check_gates`) and for threshold comparison, plus the scratch git repository
//! helpers other gate tests share.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::tests::{tempdir, *};
use super::*;
use crate::result::{RunStatus, Verdict};
use std::collections::BTreeMap;

/// 2026-09-26: A gate coverage with no exclusions, so these tests do not depend on any
/// real gate's exclusion list.
pub(super) fn any_gate() -> super::coverage::GateCoverage {
    super::coverage::GateCoverage {
        id: "test-strictest",
        excludes: &[],
    }
}

pub(super) mod scratch_repo {
    use std::path::Path;
    use std::process::Command;

    fn git(root: &Path, args: &[&str]) {
        let out = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .env("GIT_AUTHOR_NAME", "test")
            .env("GIT_AUTHOR_EMAIL", "t@t")
            .env("GIT_COMMITTER_NAME", "test")
            .env("GIT_COMMITTER_EMAIL", "t@t")
            .output()
            .expect("git runs");
        assert!(out.status.success(), "git {args:?}: {:?}", out);
    }

    pub fn init(root: &Path) {
        git(root, &["init", "-q"]);
        std::fs::write(root.join("README.md"), "first").unwrap();
        git(root, &["add", "."]);
        git(root, &["commit", "-q", "-m", "first"]);
    }

    pub fn head(root: &Path) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["rev-parse", "--short=10", "HEAD"])
            .output()
            .expect("git runs");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    pub fn branch(root: &Path, name: &str) {
        git(root, &["checkout", "-q", "-b", name]);
    }

    /// 2026-09-26: Switch to `name`, the branch `git init` created, which callers read
    /// with `current_branch` because `init.defaultBranch` is user configuration.
    pub fn checkout_default(root: &Path, name: &str) {
        git(root, &["checkout", "-q", name]);
    }

    pub fn current_branch(root: &Path) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .output()
            .expect("git runs");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// 2026-09-26: Whether `a` is an ancestor of `b`. Only a fixture assertion uses it;
    /// `record_covers` diffs content and never asks.
    pub fn is_ancestor(root: &Path, a: &str, b: &str) -> bool {
        Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["merge-base", "--is-ancestor", a, b])
            .output()
            .is_ok_and(|o| o.status.success())
    }

    pub fn commit(root: &Path, file: &str, contents: &str, message: &str) {
        let path = root.join(file);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, contents).unwrap();
        git(root, &["add", "."]);
        git(root, &["commit", "-q", "-m", message]);
    }
}

#[test]
fn an_ancestor_record_covers_head_until_a_perf_path_changes() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);

    // 2026-09-26: Commit the fixture scaffolding under `kernels/` before `sha_a`.
    // `scratch_repo::commit` runs `git add .`, so left uncommitted it would join the
    // next commit and put a perf-path change between the record and the head.
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(root, id)).unwrap();
        write_baseline(root, id, &bfcl_baseline());
    }
    scratch_repo::commit(root, "docs/seed.md", "seed", "baseline fixtures");
    let sha_a = scratch_repo::head(root);

    for id in REQUIRED_GATES {
        plant_required(root, id, &sha_a, 1_785_891_382, "PASS");
    }

    scratch_repo::commit(root, "docs/notes.md", "hello", "docs only");
    let sha_b = scratch_repo::head(root);
    assert!(
        record_covers(root, &sha_b, &sha_a, &any_gate()),
        "docs-only diff is inert"
    );
    let gates = check_gates(root, &sha_b);
    for id in REQUIRED_GATES {
        assert!(
            matches!(gates[id], GateStatus::Pass),
            "{id}: {:?}",
            gates[id]
        );
    }

    scratch_repo::commit(root, "crates/x.rs", "// code", "touch a crate");
    let sha_c = scratch_repo::head(root);
    assert!(
        !record_covers(root, &sha_c, &sha_a, &any_gate()),
        "crates/ diff invalidates"
    );
    let gates = check_gates(root, &sha_c);
    for id in REQUIRED_GATES {
        assert!(
            matches!(&gates[id], GateStatus::Missing(m) if m.contains(&sha_a)),
            "{id}: {:?}",
            gates[id]
        );
    }
}

/// 2026-09-26: Records are ordered by `recorded_at`, not by file name, which would
/// rank two records from one UTC day by their sha. The shas are real commits, so
/// the pass and fail roles are assigned from their observed order.
#[test]
fn the_newest_record_is_the_one_measured_last_not_the_higher_sha() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    let sha_a = scratch_repo::head(root);
    scratch_repo::commit(root, "docs/a.md", "a", "docs only");
    let sha_b = scratch_repo::head(root);
    scratch_repo::commit(root, "docs/b.md", "b", "docs only");
    let head = scratch_repo::head(root);

    std::fs::create_dir_all(gate_dir(root, "ssm-state-poisoning-gate")).unwrap();
    write_baseline(root, "ssm-state-poisoning-gate", &bfcl_baseline());

    // 2026-09-26: Both on one UTC day. The PASS is measured earlier and has the
    // lexically greater sha, the order a file-name sort gets backwards.
    let day = 1_785_891_382;
    let (earlier_pass, later_fail) = if sha_a > sha_b {
        (&sha_a, &sha_b)
    } else {
        (&sha_b, &sha_a)
    };
    plant(root, "ssm-state-poisoning-gate", earlier_pass, day, "PASS");
    plant(
        root,
        "ssm-state-poisoning-gate",
        later_fail,
        day + 3_600,
        "FAIL",
    );

    let ordered = records_newest_first(root, "ssm-state-poisoning-gate");
    assert!(
        ordered[0].to_string_lossy().contains(later_fail.as_str()),
        "the record measured last must come first, got {ordered:?}"
    );
    // 2026-09-26: Only docs changed, so both records cover head and the order alone
    // decides the verdict.
    assert!(record_covers(root, &head, earlier_pass, &any_gate()));
    assert!(record_covers(root, &head, later_fail, &any_gate()));
    match &check_gates(root, &head)["ssm-state-poisoning-gate"] {
        GateStatus::Fail(reasons) => assert_eq!(reasons, &["run verdict is not PASS: ok"]),
        other => panic!("a superseded PASS must not speak for the branch, got {other:?}"),
    }
}

/// 2026-09-26: A `PERF_PATHS` entry that names nothing in the tree matches no changed
/// path, so it would never invalidate a record.
#[test]
fn every_invalidating_path_exists_in_this_repo() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate");
    for path in PERF_PATHS {
        assert!(
            root.join(path).exists(),
            "{path} is in PERF_PATHS but not in the tree — the guard matches \
             nothing and invalidates nothing"
        );
    }
}

/// 2026-09-26: Neither file is under `crates/`. The server prefers
/// `jinja-templates/<model_type>.jinja` to the checkpoint's chat template unless
/// `--disable-template-overrides` is set (`tokenizer/chat_impl.rs`), and
/// `rust-toolchain.toml` pins the compiler.
#[test]
fn a_prompt_template_or_toolchain_change_invalidates_an_earlier_record() {
    for (file, contents) in [
        ("jinja-templates/qwen3_5_moe.jinja", "{{ messages }}"),
        ("rust-toolchain.toml", "[toolchain]\nchannel = \"1.94.0\"\n"),
    ] {
        let dir = tempdir::Dir::new();
        let root = dir.path();
        scratch_repo::init(root);
        let before = scratch_repo::head(root);
        scratch_repo::commit(root, "docs/n.md", "inert", "docs only");
        assert!(
            record_covers(root, &scratch_repo::head(root), &before, &any_gate()),
            "a docs commit must stay inert"
        );
        scratch_repo::commit(root, file, contents, "change what gets measured");
        assert!(
            !record_covers(root, &scratch_repo::head(root), &before, &any_gate()),
            "{file} changes what a run measures, so an earlier record cannot speak for head"
        );
    }
}

/// 2026-09-26: The metric loop does nothing over an empty map, so `check_record`
/// refuses an entry with no thresholds rather than passing it.
#[test]
fn a_baseline_entry_with_no_thresholds_is_not_a_pass() {
    let gate = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap();
    let problems = check_record(&gate, &baseline_for(MODEL, BTreeMap::new())).expect("refused");
    assert_eq!(
        problems,
        [format!(
            "the baseline entry for {MODEL} on {TEST_HW} declares no thresholds — \
             there is nothing here for this run to have passed"
        )]
    );
}

#[test]
fn a_failed_frame_fails_the_gate_even_with_passing_numbers() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    std::fs::create_dir_all(gate_dir(root, "ssm-state-poisoning-gate")).unwrap();
    write_baseline(root, "ssm-state-poisoning-gate", &bfcl_baseline());
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 90.0);
    let mut record = run_record(metrics.clone(), Verdict::fail("scoring crashed"));
    record.frame = frame(RunStatus::Failed, metrics, Verdict::fail("scoring crashed"));
    let mut gate = GateRecord::from_run(&record, hw(), SHA.into(), Vec::new(), None).unwrap();
    gate.benchmark_id = "ssm-state-poisoning-gate".to_string();
    gate.recorded_at = 1_785_891_382;
    write_record(root, &gate).unwrap();

    let gates = check_gates(root, SHA);
    match &gates["ssm-state-poisoning-gate"] {
        GateStatus::Fail(reasons) => {
            assert_eq!(reasons, &["the run itself failed: scoring crashed"])
        }
        other => panic!("wanted Fail, got {other:?}"),
    }
}

#[test]
fn the_summary_names_the_model_the_numbers_and_the_verdict() {
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
    assert_eq!(
        gate.summary,
        format!("{MODEL} · overall_accuracy=87.74 · Pass: ok")
    );
}

#[test]
fn required_gates_are_registered_benchmarks() {
    for id in REQUIRED_GATES {
        assert!(
            crate::registry::find(id).is_some(),
            "{id} is not registered"
        );
    }
}

#[test]
fn an_exact_pin_passes_only_on_the_pinned_value() {
    use crate::gate::{Bound, Comparison, compare};

    let pin = Bound {
        min: Some(1004.0),
        max: Some(1004.0),
        noise: None,
    };
    assert!(matches!(compare("samples", 1004.0, &pin), Comparison::Pass));

    let Comparison::Fail(msg) = compare("samples", 972.0, &pin) else {
        panic!("a draw of 972 against a pin of 1004 must FAIL, not pass or skip");
    };
    assert_eq!(
        msg,
        "samples is 972, but this gate is pinned to exactly 1004 — \
         the run measured something other than what the baseline describes"
    );
}

#[test]
fn a_two_sided_range_accepts_its_interior_and_rejects_outside() {
    use crate::gate::{Bound, Comparison, compare};

    let range = Bound {
        min: Some(10.0),
        max: Some(20.0),
        noise: None,
    };
    for v in [10.0, 15.0, 20.0] {
        assert!(
            matches!(compare("m", v, &range), Comparison::Pass),
            "{v} is inside [10, 20]"
        );
    }
    for v in [9.0, 21.0] {
        assert!(
            matches!(compare("m", v, &range), Comparison::Fail(_)),
            "{v} is outside [10, 20]"
        );
    }
}

#[test]
fn a_bound_with_no_side_at_all_is_still_reported() {
    use crate::gate::{Bound, Comparison, compare};

    let empty = Bound {
        min: None,
        max: None,
        noise: None,
    };
    let Comparison::Skip(reason) = compare("m", 1.0, &empty) else {
        panic!("a bound with neither side must be reported as uncheckable");
    };
    assert_eq!(reason, "m has no bound");
}

/// 2026-09-26: The committed baselines, assembled from every
/// `kernels/<hw>/<model>/BENCH.toml`, load for each required gate, resolve their
/// default, name a recipe, and give `compare` a verdict on every bound.
#[test]
fn every_committed_baseline_parses_resolves_and_is_checkable() {
    use crate::gate::{Comparison, compare, read_baseline};

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate")
        .to_path_buf();

    for id in REQUIRED_GATES {
        let baseline = read_baseline(&root, id)
            .unwrap_or_else(|e| panic!("{id}: committed baseline does not load: {e:#}"));
        assert_eq!(baseline.schema, 2, "{id}: unexpected schema version");
        assert!(!baseline.hardware.is_empty(), "{id}: no hardware entries");

        for (hw, entry) in &baseline.hardware {
            // 2026-09-26: `resolve(hw, None)` falls back to `default`, so it must name
            // an entry in `models`.
            assert!(
                entry.models.contains_key(&entry.default),
                "{id}/{hw}: default {:?} has no entry in models",
                entry.default
            );
            let (model, mb) = baseline
                .resolve(hw, None)
                .unwrap_or_else(|e| panic!("{id}/{hw}: default does not resolve: {e:#}"));
            assert_eq!(&model, &entry.default);

            for (model, mb) in entry.models.iter().chain(std::iter::once((&model, mb))) {
                let recipe = mb.recipe.as_deref().unwrap_or_default();
                assert!(
                    recipe.contains('/'),
                    "{id}/{hw}/{model}: recipe {recipe:?} is not <family>/<stem>"
                );
                assert!(!mb.metrics.is_empty(), "{id}/{hw}/{model}: no thresholds");

                for (name, bound) in &mb.metrics {
                    assert!(
                        bound.min.is_some() || bound.max.is_some(),
                        "{id}/{hw}/{model}/{name}: bound has neither min nor max"
                    );
                    for probe in [-1e9, 0.0, 1e9] {
                        assert!(
                            !matches!(compare(name, probe, bound), Comparison::Skip(_)),
                            "{id}/{hw}/{model}/{name}: compare abstains on {probe} — \
                             a bound the comparator cannot act on is not a threshold"
                        );
                    }
                }
            }
        }
    }
}
