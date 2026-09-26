// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `telemetry::order`: the chart's position and bound,
//! the ordering rule, label escaping, the next steps, and `cap_prs`, rendered
//! against the real kernel tree.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::super::{PrFacts, render, views};
use super::*;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
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

const FLAGSHIP: &str = "kernels/gb10/qwen3.6-27b/nvfp4/w4a4_gemm.cu";
const MOE: &str = "kernels/gb10/qwen3.6-35b-a3b/nvfp4/x.cu";

/// 2026-09-26: Only the start marker and the title precede the chart.
#[test]
fn the_chart_is_the_first_thing_in_the_comment() {
    let root = repo_root();
    let body = render(&root, &[pr(1, &[FLAGSHIP]), pr(2, &[MOE])]);
    assert!(
        body.starts_with(
            "<!-- metrale-pr-telemetry:start -->\n## Open-PR telemetry\n\n```mermaid\ngitGraph\n  \
             commit id: \"main\"\n"
        ),
        "only the replacement marker and title may precede the chart: {body}"
    );
}

/// 2026-09-26: With 12 orderable PRs the chart draws `CHART_PR_BOUND`
/// branches.
#[test]
fn the_chart_is_bounded_to_chart_pr_bound() {
    let root = repo_root();
    let prs: Vec<PrFacts> = (101..=112).map(|n| pr(n, &[FLAGSHIP])).collect();
    let body = render(&root, &prs);
    assert_eq!(
        body.matches("  branch pr-").count(),
        CHART_PR_BOUND,
        "exactly the bound, not all 12"
    );
}

/// 2026-09-26: A truncated chart says "Showing N of M" and names the PRs it
/// left out.
#[test]
fn truncation_is_disclosed_with_showing_n_of_m() {
    let root = repo_root();
    let prs: Vec<PrFacts> = (101..=112).map(|n| pr(n, &[FLAGSHIP])).collect();
    let body = render(&root, &prs);
    assert!(
        body.contains("Showing 10 of 12 open PRs"),
        "the caption must say showing N of M: {body}"
    );
    assert!(
        body.contains("Not charted: #111, #112."),
        "the dropped PRs must be named: {body}"
    );
}

/// 2026-09-26: Under the bound, N equals M and no "Not charted" appears.
#[test]
fn a_small_input_is_not_truncated() {
    let root = repo_root();
    let body = render(&root, &[pr(1, &[FLAGSHIP]), pr(2, &[MOE])]);
    assert!(
        body.contains("Showing 2 of 2 open PRs, left to right in recommended merge order.\nOrder:"),
        "{body}"
    );
    assert!(!body.contains("Not charted"), "nothing was dropped");
}

/// 2026-09-26: With no orderable PR, a sentence replaces the chart.
#[test]
fn only_merged_prs_yield_a_note_not_a_degenerate_chart() {
    let root = repo_root();
    let mut merged = pr(1, &[FLAGSHIP]);
    merged.merged = true;
    let v = views(&root, &[merged]);
    assert_eq!(
        render_order_chart(&v),
        "_No open, non-draft PRs to order._\n"
    );
}

/// 2026-09-26: Fewest conflict partners first: #3 shares no target with the
/// others, so it leads.
#[test]
fn the_uncontended_pr_is_recommended_first() {
    let root = repo_root();
    let v = views(
        &root,
        &[pr(1, &[FLAGSHIP]), pr(2, &[FLAGSHIP]), pr(3, &[MOE])],
    );
    assert_eq!(merge_order(&v), vec![3, 1, 2]);
}

/// 2026-09-26: A whole-repo PR ranks after a narrow kernel PR it contends
/// with, on breadth.
#[test]
fn a_whole_repo_pr_orders_last() {
    let root = repo_root();
    let v = views(
        &root,
        &[
            pr(1, &["crates/model-layers/src/lib.rs"]),
            pr(2, &[FLAGSHIP]),
        ],
    );
    assert_eq!(
        merge_order(&v),
        vec![2, 1],
        "whole-repo after the narrow PR"
    );
}

/// 2026-09-26: Merged and draft PRs are neither ranked nor charted.
#[test]
fn merged_and_draft_prs_are_not_ranked_or_charted() {
    let root = repo_root();
    let mut merged = pr(1, &[FLAGSHIP]);
    merged.merged = true;
    let mut draft = pr(2, &[FLAGSHIP]);
    draft.draft = true;
    let open = pr(3, &[FLAGSHIP]);
    let v = views(&root, &[merged.clone(), draft.clone(), open.clone()]);
    assert_eq!(merge_order(&v), vec![3], "only the open, non-draft PR");
    let body = render(&root, &[merged, draft, open]);
    assert!(!body.contains("branch pr-1\n"), "merged not charted");
    assert!(!body.contains("branch pr-2\n"), "draft not charted");
    assert!(body.contains("branch pr-3\n"), "open PR charted");
}

/// 2026-09-26: Non-kernel diffs are whole-repo and contend with every gated
/// PR, each other included; with partners tied, breadth and then diff size
/// decide.
#[test]
fn whole_repo_prs_contend_and_tie_break_by_size() {
    let root = repo_root();
    let v = views(
        &root,
        &[
            pr(1, &["docs/adr/README.md", "docs/adr/0002.md"]),
            pr(2, &["docs/other.md"]),
            pr(3, &[MOE]),
        ],
    );
    // 2026-09-26: All three contend pairwise, so partners tie at 2; #3 leads
    // on breadth (whole-repo ranks last), then #2's one path precedes #1's
    // two.
    assert_eq!(merge_order(&v), vec![3, 2, 1]);
}

/// 2026-09-26: A quote in a title becomes an apostrophe and a newline a
/// space, so the title cannot close the quoted commit id.
#[test]
fn a_hostile_title_cannot_break_the_chart() {
    let root = repo_root();
    let mut hostile = pr(9, &[FLAGSHIP]);
    hostile.title = "evil \"quote\"\ninject".into();
    let v = views(&root, &[hostile]);
    assert_eq!(commit_label(&v[0]), "#9 evil 'quote' inject");
}

/// 2026-09-26: A title longer than `TITLE_CLIP` (24) characters is clipped
/// and ends with an ellipsis.
#[test]
fn a_long_title_is_clipped_in_the_chart() {
    let root = repo_root();
    let mut long = pr(9, &[FLAGSHIP]);
    long.title = "a".repeat(80);
    let v = views(&root, &[long]);
    assert_eq!(commit_label(&v[0]), format!("#9 {}…", "a".repeat(24)));
}

/// 2026-09-26: While both PRs are open, the head of the order is the
/// recommendation; once one merges, the other contends with it and is listed
/// for re-gating.
#[test]
fn next_steps_change_when_a_partner_merges() {
    let root = repo_root();
    let before = render(&root, &[pr(1, &[FLAGSHIP]), pr(2, &[FLAGSHIP])]);
    assert!(before.contains("**Merge next: #1**"), "{before}");
    assert!(
        !before.contains("Re-gate before merging"),
        "nothing merged yet, nothing to re-gate: {before}"
    );

    let mut merged = pr(2, &[FLAGSHIP]);
    merged.merged = true;
    let after = render(&root, &[pr(1, &[FLAGSHIP]), merged]);
    assert!(after.contains("**Merge next: #1**"), "{after}");
    assert!(
        after.contains("Re-gate before merging:** #1"),
        "the merged partner moved #1's baseline: {after}"
    );
}

/// 2026-09-26: A merged PR with promotion debt is named in the next steps.
#[test]
fn next_steps_surface_merged_promotion_debt() {
    let root = repo_root();
    let mut merged = pr(7, &["crates/server/src/scheduler/mod.rs"]);
    merged.merged = true;
    let body = render(&root, &[pr(1, &[FLAGSHIP]), merged]);
    assert!(
        body.contains("Discharge promotion debt:** merged #7"),
        "the merged debtor must be named: {body}"
    );
}

/// 2026-09-26: The section ends with the sentence that lists what its inputs
/// cannot see.
#[test]
fn next_steps_admit_what_they_cannot_know() {
    let root = repo_root();
    let v = views(&root, &[pr(1, &[FLAGSHIP])]);
    let body = render_next_steps(&v);
    assert!(
        body.ends_with(
            "\n_Derived only from changed paths and merge state. This section cannot see check \
             or required-context status, review state, true git mergeability, or whether one PR \
             unblocks another — those live on each PR._\n"
        ),
        "the complete limits must close the section: {body}"
    );
}

/// 2026-09-26: `cap_prs` shows `CHART_PR_BOUND` numbers and counts the rest;
/// under the bound it adds nothing.
#[test]
fn cap_prs_caps_at_the_bound_and_discloses_the_rest() {
    let numbers: Vec<u64> = (1..=12).collect();
    let capped = cap_prs(&numbers);
    assert_eq!(capped, "#1, #2, #3, #4, #5, #6, #7, #8, #9, #10 … +2 more");
    assert_eq!(cap_prs(&[1, 2]), "#1, #2", "no noise under the bound");
}

/// 2026-09-26: The targets table's cells go through `cap_prs`.
#[test]
fn target_table_cells_are_bounded() {
    let root = repo_root();
    let prs: Vec<PrFacts> = (101..=112).map(|n| pr(n, &[FLAGSHIP])).collect();
    let body = render(&root, &prs);
    let row = body
        .lines()
        .find(|l| l.starts_with("| `gb10/qwen3.6-27b/nvfp4` |"))
        .expect("the flagship target row rendered");
    assert!(row.contains("+2 more"), "cell capped and disclosed: {row}");
    assert_eq!(row.matches('#').count(), CHART_PR_BOUND, "{row}");
}
