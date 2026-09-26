// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `telemetry` against the real kernel tree and
//! CODEOWNERS: per-PR views, collisions, order, and the rendered body.
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

fn pr(number: u64, paths: &[&str]) -> PrFacts {
    PrFacts {
        number,
        title: format!("pr {number}"),
        author: "someone".into(),
        draft: false,
        merged: false,
        paths_unknown: false,
        changed_paths: paths.iter().map(|s| s.to_string()).collect(),
    }
}

const COMMON: &str = "kernels/gb10/common/paged_decode_attn_fp8.cu";
const FLAGSHIP: &str = "kernels/gb10/qwen3.6-27b/nvfp4/w4a4_gemm.cu";

/// 2026-09-26: A gb10 `common/` kernel re-opens every target except Metal's:
/// all of gb10 by path, and the b200, b300, hopper, strix and strix-hip
/// targets because the file is among their resolved inputs.
#[test]
fn a_common_kernel_change_reopens_every_target_that_compiles_it() {
    let root = repo_root();
    let v = &views(&root, &[pr(1, &[COMMON])])[0];
    let cuda: BTreeSet<_> = taxon::walk(&root)
        .into_iter()
        .filter(|t| t.hardware != "metal")
        .collect();
    assert_eq!(v.targets, cuda, "every target but Metal's");
    let hardware: BTreeSet<&str> = v.targets.iter().map(|t| t.hardware.as_str()).collect();
    assert_eq!(
        hardware,
        BTreeSet::from(["b200", "b300", "gb10", "hopper", "strix", "strix-hip"])
    );
    assert!(!v.whole_repo, "a kernels-only diff is not whole-repo");
}

/// 2026-09-26: The owner, its redirected consumer (qwen3.8-27b's
/// `kernel_source`), and both again on hopper, which inherits gb10.
#[test]
fn a_source_owner_change_reopens_the_owner_and_redirected_consumer() {
    let root = repo_root();
    let v = &views(&root, &[pr(2, &[FLAGSHIP])])[0];
    let target = |hardware: &str, model: &str| taxon::Target {
        hardware: hardware.into(),
        model: model.into(),
        quant: "nvfp4".into(),
    };
    assert_eq!(
        v.targets,
        BTreeSet::from([
            target("gb10", "qwen3.6-27b"),
            target("gb10", "qwen3.8-27b"),
            target("hopper", "qwen3.6-27b"),
            target("hopper", "qwen3.8-27b"),
        ])
    );
}

/// 2026-09-26: A diff that reaches outside `kernels/` is whole-repo: every
/// target, and "ALL" in the PR table, whatever kernel paths it also has.
#[test]
fn a_diff_reaching_outside_kernels_is_marked_whole_repo() {
    let root = repo_root();
    let v = &views(
        &root,
        &[pr(3, &[FLAGSHIP, "crates/model-layers/src/lib.rs"])],
    )[0];
    assert!(v.whole_repo);
    assert_eq!(
        v.targets,
        taxon::walk(&root).into_iter().collect(),
        "a whole-repository diff re-opens every target"
    );
    let body = render(
        &root,
        &[pr(3, &[FLAGSHIP, "crates/model-layers/src/lib.rs"])],
    );
    assert!(
        body.contains("ALL (diff reaches outside kernels/)"),
        "the table must say ALL, not 1: {body}"
    );
}

#[test]
fn codeowners_are_resolved_from_the_changed_paths() {
    let root = repo_root();
    let v = &views(&root, &[pr(4, &["crates/model-layers/src/lib.rs"])])[0];
    assert_eq!(
        v.owners,
        ["@SeedSource", "@TheTom", "@rsafier", "@tbraun96"]
    );
}

/// 2026-09-26: Two PRs on one source owner collide on it and on every
/// consumer: the redirected model, and both on hopper.
#[test]
fn two_prs_touching_one_target_collide() {
    let root = repo_root();
    let v = views(&root, &[pr(1, &[FLAGSHIP]), pr(2, &[FLAGSHIP])]);
    let c = collisions(&v);
    assert_eq!(
        c,
        BTreeMap::from([
            ("gb10/qwen3.6-27b/nvfp4".into(), vec![1, 2]),
            ("gb10/qwen3.8-27b/nvfp4".into(), vec![1, 2]),
            ("hopper/qwen3.6-27b/nvfp4".into(), vec![1, 2]),
            ("hopper/qwen3.8-27b/nvfp4".into(), vec![1, 2]),
        ])
    );
}

#[test]
fn prs_on_different_targets_do_not_collide() {
    let root = repo_root();
    let v = views(
        &root,
        &[
            pr(1, &[FLAGSHIP]),
            pr(2, &["kernels/gb10/qwen3.6-35b-a3b/nvfp4/x.cu"]),
        ],
    );
    assert!(collisions(&v).is_empty());
}

/// 2026-09-26: A shared-kernel PR collides with a model PR on every target
/// the model PR re-opens.
#[test]
fn a_shared_kernel_pr_collides_with_every_model_pr_beneath_it() {
    let root = repo_root();
    let v = views(&root, &[pr(1, &[COMMON]), pr(2, &[FLAGSHIP])]);
    let c = collisions(&v);
    assert_eq!(
        c,
        BTreeMap::from([
            ("gb10/qwen3.6-27b/nvfp4".into(), vec![1, 2]),
            ("gb10/qwen3.8-27b/nvfp4".into(), vec![1, 2]),
            ("hopper/qwen3.6-27b/nvfp4".into(), vec![1, 2]),
            ("hopper/qwen3.8-27b/nvfp4".into(), vec![1, 2]),
        ]),
        "the shared change meets both the source owner and its consumer"
    );
}

#[test]
fn a_whole_repo_pr_collides_with_a_kernel_pr() {
    let root = repo_root();
    let v = views(
        &root,
        &[
            pr(1, &["crates/model-layers/src/lib.rs"]),
            pr(2, &[FLAGSHIP]),
        ],
    );
    assert_eq!(
        collisions(&v),
        BTreeMap::from([
            ("gb10/qwen3.6-27b/nvfp4".into(), vec![1, 2]),
            ("gb10/qwen3.8-27b/nvfp4".into(), vec![1, 2]),
            ("hopper/qwen3.6-27b/nvfp4".into(), vec![1, 2]),
            ("hopper/qwen3.8-27b/nvfp4".into(), vec![1, 2]),
        ])
    );
}

#[test]
fn a_merged_pr_does_not_remain_in_the_open_collision_map() {
    let root = repo_root();
    let mut merged = pr(1, &[FLAGSHIP]);
    merged.merged = true;
    let v = views(&root, &[merged, pr(2, &[FLAGSHIP])]);
    assert_eq!(collisions(&v), BTreeMap::new());
}

#[test]
fn the_narrowest_pr_is_suggested_first() {
    let root = repo_root();
    let v = views(&root, &[pr(1, &[COMMON]), pr(2, &[FLAGSHIP])]);
    assert_eq!(
        merge_order(&v),
        vec![2, 1],
        "4 targets before every CUDA target"
    );
}

/// 2026-09-26: The order does not depend on input order; ties break by PR
/// number.
#[test]
fn the_order_is_deterministic_under_input_permutation() {
    let root = repo_root();
    let a = views(&root, &[pr(7, &[FLAGSHIP]), pr(3, &[FLAGSHIP])]);
    let b = views(&root, &[pr(3, &[FLAGSHIP]), pr(7, &[FLAGSHIP])]);
    assert_eq!(merge_order(&a), merge_order(&b));
    assert_eq!(merge_order(&a), vec![3, 7], "ties break by PR number");
}

/// 2026-09-26: The targets table lists every target, with "—" for those no
/// PR re-opens.
#[test]
fn every_target_appears_even_when_no_pr_touches_it() {
    let root = repo_root();
    let body = render(&root, &[pr(1, &[FLAGSHIP])]);
    for target in taxon::walk(&root) {
        let reopened = if matches!(
            target.to_string().as_str(),
            "gb10/qwen3.6-27b/nvfp4"
                | "gb10/qwen3.8-27b/nvfp4"
                | "hopper/qwen3.6-27b/nvfp4"
                | "hopper/qwen3.8-27b/nvfp4"
        ) {
            "#1"
        } else {
            "—"
        };
        assert!(body.contains(&format!("| `{target}` | {reopened} |\n")));
    }
}

/// 2026-09-26: The body opens with `MARKER_START` and closes with
/// `MARKER_END`, each exactly once.
#[test]
fn the_body_is_delimited_so_it_can_be_rewritten_in_place() {
    let root = repo_root();
    let body = render(&root, &[pr(1, &[FLAGSHIP])]);
    assert!(body.starts_with(MARKER_START));
    assert!(body.trim_end().ends_with(MARKER_END));
    assert_eq!(body.matches(MARKER_START).count(), 1);
    assert_eq!(body.matches(MARKER_END).count(), 1);
}

#[test]
fn an_empty_pr_list_still_renders_a_valid_body() {
    let root = repo_root();
    assert_eq!(
        render(&root, &[]),
        format!("{MARKER_START}\n## Open-PR telemetry\n\n_No open pull requests._\n{MARKER_END}\n")
    );
}

/// 2026-09-26: A `|` in a title is escaped and a newline becomes a space, so
/// a title cannot break the table or the row.
#[test]
fn pr_titles_cannot_break_the_table() {
    let root = repo_root();
    let hostile = PrFacts {
        number: 9,
        title: "evil | row\ninjection".into(),
        author: "x".into(),
        draft: false,
        merged: false,
        paths_unknown: false,
        changed_paths: vec![FLAGSHIP.to_string()],
    };
    let body = render(&root, &[hostile]);
    let row = body
        .lines()
        .find(|l| l.starts_with("| #9"))
        .expect("the row rendered");
    assert_eq!(
        row,
        "| #9 evil \\| row injection | gb10 | 4 | @SeedSource @TheTom @rsafier @tbraun96 |"
    );
}

#[test]
fn a_draft_is_marked_as_one() {
    let root = repo_root();
    let mut facts = pr(5, &[FLAGSHIP]);
    facts.draft = true;
    let row = render(&root, &[facts])
        .lines()
        .find(|line| line.starts_with("| #5"))
        .unwrap()
        .to_string();
    assert_eq!(
        row,
        "| #5 (draft) pr 5 | gb10 | 4 | @SeedSource @TheTom @rsafier @tbraun96 |"
    );
}

/// 2026-09-26: The debt section's policy text, table header and a
/// scheduler PR's row, verbatim.
#[test]
fn the_promotion_debt_section_is_always_rendered() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let prs = vec![super::PrFacts {
        number: 1,
        title: "a scheduler change".into(),
        author: "someone".into(),
        draft: false,
        merged: false,
        paths_unknown: false,
        changed_paths: vec!["crates/server/src/scheduler/mod.rs".into()],
    }];
    let body = super::render(&root, &prs);
    assert!(
        body.contains(
            "### Promotion-candidate debt\n\nThese gates are NOT required, so these PRs can merge without them. Each row is coverage this repository chose not to buy — recorded so the choice stays visible rather than becoming an assumption.\n\n| PR | merged? | title | gates that wanted to run |\n|---|---|---|---|\n| #1 | not yet | a scheduler change | cross-contamination, scheduler-equivalence |\n"
        ),
        "the unconditional debt section must retain its policy, schema, and row: {body}"
    );
}

/// 2026-09-26: Each PR's debt comes from its own paths: a docs PR owes
/// nothing, a scheduler PR owes `cross-contamination` and
/// `scheduler-equivalence`.
#[test]
fn debt_is_derived_from_the_prs_own_paths() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let prs = vec![
        super::PrFacts {
            number: 1,
            title: "docs".into(),
            author: "a".into(),
            draft: false,
            merged: false,
            paths_unknown: false,
            changed_paths: vec!["docs/adr/README.md".into()],
        },
        super::PrFacts {
            number: 2,
            title: "engine".into(),
            author: "b".into(),
            draft: false,
            merged: false,
            paths_unknown: false,
            changed_paths: vec!["crates/server/src/scheduler/mod.rs".into()],
        },
    ];
    let views = super::views(&root, &prs);
    assert_eq!(views[0].promotion_debt, Vec::<&str>::new());
    assert_eq!(
        views[1].promotion_debt,
        vec!["cross-contamination", "scheduler-equivalence"]
    );
}

/// 2026-09-26: The debt table's "merged?" column tells an open PR's debt
/// from a merged PR's.
#[test]
fn the_debt_table_distinguishes_merged_from_open() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let prs = vec![
        super::PrFacts {
            number: 1,
            title: "still open".into(),
            author: "a".into(),
            draft: false,
            merged: false,
            paths_unknown: false,
            changed_paths: vec!["crates/server/src/scheduler/mod.rs".into()],
        },
        super::PrFacts {
            number: 2,
            title: "already landed".into(),
            author: "b".into(),
            draft: false,
            merged: true,
            paths_unknown: false,
            changed_paths: vec!["crates/server/src/scheduler/mod.rs".into()],
        },
    ];
    let body = super::render(&root, &prs);
    assert!(
        body.contains(
            "| #1 | not yet | still open | cross-contamination, scheduler-equivalence |\n| #2 | **yes** | already landed | cross-contamination, scheduler-equivalence |\n"
        ),
        "open warning and accrued merged debt must remain distinct: {body}"
    );
}
