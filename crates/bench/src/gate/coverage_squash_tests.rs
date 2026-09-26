// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests that `record_covers` judges a record by content, not ancestry: it
//! still covers a squash-merged commit with the same perf-path content, and does not
//! cover a commit that differs or that git cannot diff.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::coverage_tests::{any_gate, scratch_repo};
use super::tests::{tempdir, *};
use super::*;

#[test]
fn a_record_survives_its_pr_being_squash_merged() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);

    // 2026-09-26: Commit the fixture scaffolding first, so the `git add .` in
    // `scratch_repo::commit` does not put it in the commit under test.
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(root, id)).unwrap();
        write_baseline(root, id, &bfcl_baseline());
    }
    scratch_repo::commit(root, "docs/seed.md", "seed", "baseline fixtures");
    let default_branch = scratch_repo::current_branch(root);

    scratch_repo::branch(root, "pr");
    scratch_repo::commit(root, "crates/feature.rs", "// the change", "the feature");
    let branch_tip = scratch_repo::head(root);
    for id in REQUIRED_GATES {
        plant_required(root, id, &branch_tip, 1_785_891_382, "PASS");
    }

    // 2026-09-26: The squash: the default branch gets the same file contents in a new
    // commit that does not descend from `branch_tip`.
    scratch_repo::checkout_default(root, &default_branch);
    scratch_repo::commit(
        root,
        "crates/feature.rs",
        "// the change",
        "the feature (#1)",
    );
    let squashed = scratch_repo::head(root);

    assert!(
        !scratch_repo::is_ancestor(root, &branch_tip, &squashed),
        "fixture must reproduce the real shape: the record's commit is NOT an \
         ancestor of the squash"
    );
    assert!(
        record_covers(root, &squashed, &branch_tip, &any_gate()),
        "the squash has byte-identical perf-path content — the record that \
         measured it still speaks for it"
    );
}

#[test]
fn an_unrelated_commit_that_differs_is_still_not_covered() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(root, id)).unwrap();
        write_baseline(root, id, &bfcl_baseline());
    }
    scratch_repo::commit(root, "docs/seed.md", "seed", "baseline fixtures");
    let default_branch = scratch_repo::current_branch(root);

    scratch_repo::branch(root, "pr");
    scratch_repo::commit(root, "crates/feature.rs", "// version A", "feature A");
    let branch_tip = scratch_repo::head(root);
    for id in REQUIRED_GATES {
        plant_required(root, id, &branch_tip, 1_785_891_382, "PASS");
    }

    scratch_repo::checkout_default(root, &default_branch);
    scratch_repo::commit(root, "crates/feature.rs", "// version B", "feature B");
    let other = scratch_repo::head(root);

    assert!(
        !record_covers(root, &other, &branch_tip, &any_gate()),
        "different perf-path content must still invalidate — content is the \
         test, and this content differs"
    );
}

#[test]
fn a_record_for_an_unknown_commit_is_not_covered() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    for id in REQUIRED_GATES {
        std::fs::create_dir_all(gate_dir(root, id)).unwrap();
        write_baseline(root, id, &bfcl_baseline());
    }
    scratch_repo::commit(root, "docs/seed.md", "seed", "baseline fixtures");
    let head = scratch_repo::head(root);

    assert!(
        !record_covers(root, &head, "deadbeefcafe", &any_gate()),
        "git cannot diff a commit that is not here; that must read as \
         not-covered, never as a pass"
    );
}
