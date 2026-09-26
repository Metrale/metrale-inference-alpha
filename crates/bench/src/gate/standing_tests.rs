// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `check::record_standing`, which both the gate verdict
//! (`record_still_stands`) and `agreement::standing_at` read. A record stands
//! at a head whose diff from the record's commit touches nothing its gate
//! reads, judged by content, not ancestry; a perf-path change invalidates it
//! and names the path; a commit git cannot diff is `Unknown`.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::check::{Standing, record_standing};
use super::coverage_tests::{any_gate, scratch_repo};
use super::tests::{hw, run_record, tempdir};
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;

fn record_at(sha: &str) -> GateRecord {
    GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        sha.to_string(),
        Vec::new(),
        None,
    )
    .unwrap()
}

#[test]
fn a_record_stands_across_harmless_commits_and_falls_to_a_perf_path_or_an_unknown_commit() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    scratch_repo::commit(
        root,
        "crates/x/src/lib.rs",
        "// measured",
        "the measured tree",
    );
    let measured = scratch_repo::head(root);
    let record = record_at(&measured);
    let gate = any_gate();

    assert_eq!(
        record_standing(root, &measured, &record, &gate),
        Standing::Stands
    );
    scratch_repo::commit(root, "docs/notes.md", "words", "docs only");
    let docs = scratch_repo::head(root);
    assert_eq!(
        record_standing(root, &docs, &record, &gate),
        Standing::Stands
    );

    // 2026-09-26: A record from a side branch that is not an ancestor of the
    // head stands when the diff touches no perf path; a sha the repository
    // does not have is `Unknown`.
    let main_branch = scratch_repo::current_branch(root);
    scratch_repo::branch(root, "side");
    scratch_repo::commit(root, "docs/other.md", "aside", "side branch");
    let side = scratch_repo::head(root);
    scratch_repo::checkout_default(root, &main_branch);
    assert_eq!(
        record_standing(root, &docs, &record_at(&side), &gate),
        Standing::Stands
    );
    assert_eq!(
        record_standing(root, &docs, &record_at("0000000000"), &gate),
        Standing::Unknown
    );

    // 2026-09-26: Negative control: a later commit edits a perf path that
    // `any_gate` (no excludes) reads, so the record is invalidated and the
    // path is named.
    scratch_repo::commit(root, "crates/x/src/lib.rs", "// changed", "perf path");
    let moved = scratch_repo::head(root);
    assert_eq!(
        record_standing(root, &moved, &record, &gate),
        Standing::Invalidated(vec!["crates/x/src/lib.rs".to_string()])
    );
}
