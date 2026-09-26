// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of a PR with `paths_unknown`: it is counted at the widest
//! reach, its rows say they are assumed, it is not ranked, it collides, and
//! the flag survives JSON parsing.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::*;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

fn blind(number: u64) -> PrFacts {
    PrFacts {
        number,
        title: format!("pr {number}"),
        author: "someone".into(),
        draft: false,
        merged: false,
        changed_paths: Vec::new(),
        paths_unknown: true,
    }
}

fn known(number: u64, paths: &[&str]) -> PrFacts {
    PrFacts {
        number,
        title: format!("pr {number}"),
        author: "someone".into(),
        draft: false,
        merged: false,
        changed_paths: paths.iter().map(|s| s.to_string()).collect(),
        paths_unknown: false,
    }
}

const FLAGSHIP: &str = "kernels/gb10/qwen3.6-27b/nvfp4/w4a4_gemm.cu";

/// 2026-09-26: Unknown paths make the view whole-repo, with every target and
/// every promotion candidate; a known docs-only PR still owes nothing.
#[test]
fn an_unreadable_diff_re_opens_every_target_not_none() {
    let root = repo_root();
    let v = &views(&root, &[blind(1)])[0];
    assert!(v.whole_repo, "unknown paths must be treated as whole-repo");
    assert_eq!(
        v.targets,
        taxon::walk(&root).into_iter().collect::<BTreeSet<_>>(),
        "an unreadable diff must re-open every target, not zero"
    );
    assert_eq!(
        v.promotion_debt,
        coverage::PROMOTION_CANDIDATES
            .iter()
            .map(|g| g.id)
            .collect::<Vec<_>>(),
        "unknown paths owe every promotion candidate"
    );
    let clean = &views(&root, &[known(2, &["docs/adr/README.md"])])[0];
    assert_eq!(clean.promotion_debt, Vec::<&str>::new());
}

/// 2026-09-26: The warning, the PR row and the debt row of an unknown PR say
/// so; the readable PR's row does not.
#[test]
fn the_comment_labels_every_assumed_row_as_assumed() {
    let root = repo_root();
    let body = render(&root, &[blind(7), known(8, &[FLAGSHIP])]);
    assert!(
        body.contains("⚠ **Changed files unavailable for #7.**"),
        "the banner must name the affected PRs: {body}"
    );
    let row = body
        .lines()
        .find(|l| l.starts_with("| #7"))
        .expect("the blind PR still has a row");
    assert!(
        row.contains("ALL — **assumed**, changed files unreadable"),
        "the targets cell must not read as a measurement: {row}"
    );
    assert!(
        row.contains("| unknown |"),
        "no matched owner and no paths to match are different cells: {row}"
    );
    assert!(
        row.contains("unknown (changed files unreadable)"),
        "\"host / non-kernel\" is a claim about a diff nobody read: {row}"
    );
    let debt = body
        .lines()
        .find(|l| l.starts_with("| #7 |"))
        .expect("the blind PR appears in the debt table");
    assert!(
        debt.contains("**assumed** (paths unreadable)"),
        "assumed debt must not be quoted as owed: {debt}"
    );
    let good = body
        .lines()
        .find(|l| l.starts_with("| #8"))
        .expect("the readable PR has a row");
    assert!(!good.contains("assumed"), "{good}");
}

/// 2026-09-26: An unknown PR is left out of the merge order and named as
/// unrankable in the next steps.
#[test]
fn an_unreadable_diff_is_never_recommended_as_merge_next() {
    let root = repo_root();
    let vs = views(&root, &[blind(7), known(8, &[FLAGSHIP])]);
    assert_eq!(
        order::merge_order(&vs),
        vec![8],
        "an unrankable PR must not be ranked at all, least of all first"
    );
    let body = render(&root, &[blind(7), known(8, &[FLAGSHIP])]);
    assert!(body.contains("**Merge next: #8**"), "{body}");
    assert!(
        body.contains("**Cannot rank #7:**"),
        "exclusion from the order must be stated, not silent: {body}"
    );
}

/// 2026-09-26: An unknown PR appears in every collision.
#[test]
fn a_blind_pr_still_collides_with_everything() {
    let root = repo_root();
    let vs = views(&root, &[blind(7), known(8, &[FLAGSHIP])]);
    let c = collisions(&vs);
    assert!(
        !c.is_empty(),
        "a maximal-radius PR contends with every gated PR"
    );
    for (target, prs) in &c {
        assert!(prs.contains(&7), "#7 missing from `{target}`: {prs:?}");
    }
}

/// 2026-09-26: When the only open PR is unknown, the next steps say "No
/// recommendation", not "Nothing to merge".
#[test]
fn an_all_blind_run_does_not_claim_there_is_nothing_to_merge() {
    let root = repo_root();
    let body = render(&root, &[blind(7)]);
    assert!(
        !body.contains("Nothing to merge: no open, non-draft PRs."),
        "there IS an open PR; the run just cannot see it: {body}"
    );
    assert!(body.contains("**No recommendation.**"), "{body}");
    assert!(body.contains("came back unreadable"), "{body}");
}

/// 2026-09-26: `paths_unknown` parses from JSON, and is false when absent
/// (`#[serde(default)]`).
#[test]
fn the_marker_survives_the_json_the_workflow_actually_writes() {
    let facts: Vec<PrFacts> = serde_json::from_str(
        r#"[{"number":7,"title":"t","author":"a","draft":false,"merged":false,
             "changed_paths":[],"paths_unknown":true},
            {"number":8,"title":"t","author":"a","draft":false,"merged":false,
             "changed_paths":[]}]"#,
    )
    .expect("the workflow's shape parses");
    assert!(facts[0].paths_unknown, "the flag must round-trip");
    assert!(
        !facts[1].paths_unknown,
        "absent means known-and-empty, which is what pre-fix records mean"
    );
}
