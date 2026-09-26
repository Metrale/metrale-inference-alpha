// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The recommended merge order, the mermaid `gitGraph` that draws
//! it (`main` as a line, each PR a branch merged back in order), and the
//! "Recommended next steps" list.
//!
//! Owner: bench gate.
//! Invariants:
//! - The chart and every list of PR numbers built by [`cap_prs`] show at most
//!   [`CHART_PR_BOUND`] PRs, and say how many more there are.
//! - [`merge_order`] ranks only unmerged, non-draft PRs whose changed files
//!   were read.

use super::PrView;

/// 2026-09-26: The most PRs the chart draws and [`cap_prs`] lists.
pub const CHART_PR_BOUND: usize = 10;

/// 2026-09-26: True when the two PRs re-open a common target, or one is
/// whole-repo and the other is whole-repo or re-opens any target. Shared
/// changed files are not compared.
fn contend(a: &PrView, b: &PrView) -> bool {
    let a_gated = a.whole_repo || !a.targets.is_empty();
    let b_gated = b.whole_repo || !b.targets.is_empty();
    if (a.whole_repo && b_gated) || (b.whole_repo && a_gated) {
        return true;
    }
    !a.targets.is_disjoint(&b.targets)
}

/// 2026-09-26: The PRs a merge order can contain: unmerged, non-draft, and
/// with a changed-file list this run could read. [`render_next_steps`] lists
/// the unreadable ones.
fn orderable(views: &[PrView]) -> Vec<&PrView> {
    views
        .iter()
        .filter(|v| !v.facts.merged && !v.facts.draft && !v.facts.paths_unknown)
        .collect()
}

/// 2026-09-26: The recommended merge order over the orderable PRs, sorted by,
/// in turn:
/// 1. fewest conflict partners (the private `contend` helper);
/// 2. fewest targets re-opened, with a whole-repo PR last;
/// 3. fewest changed paths;
/// 4. lowest PR number, which makes the order total.
///
/// [`super::PrFacts`] carries no check, review, mergeability or dependency
/// state, so the order ignores them.
pub fn merge_order(views: &[PrView]) -> Vec<u64> {
    let open = orderable(views);
    let mut ranked: Vec<(usize, usize, usize, u64)> = open
        .iter()
        .map(|v| {
            let partners = open
                .iter()
                .filter(|o| o.facts.number != v.facts.number && contend(v, o))
                .count();
            let breadth = if v.whole_repo {
                usize::MAX
            } else {
                v.targets.len()
            };
            (
                partners,
                breadth,
                v.facts.changed_paths.len(),
                v.facts.number,
            )
        })
        .collect();
    ranked.sort_unstable();
    ranked.into_iter().map(|(_, _, _, n)| n).collect()
}

/// 2026-09-26: `#1, #2 … +k more`: at most [`CHART_PR_BOUND`] numbers, then
/// the count of the rest.
pub fn cap_prs(numbers: &[u64]) -> String {
    let shown = numbers.len().min(CHART_PR_BOUND);
    let mut s = numbers[..shown]
        .iter()
        .map(|n| format!("#{n}"))
        .collect::<Vec<_>>()
        .join(", ");
    if numbers.len() > shown {
        s.push_str(&format!(" … +{} more", numbers.len() - shown));
    }
    s
}

/// 2026-09-26: The commit label for one PR inside the chart. The label is
/// written inside double quotes, so a quote in the title becomes an
/// apostrophe and a newline a space; the title is clipped to `TITLE_CLIP`
/// characters with an ellipsis.
fn commit_label(v: &PrView) -> String {
    const TITLE_CLIP: usize = 24;
    let clean = v.facts.title.replace('"', "'").replace('\n', " ");
    let clipped: String = clean.chars().take(TITLE_CLIP).collect();
    let ellipsis = if clean.chars().count() > TITLE_CLIP {
        "…"
    } else {
        ""
    };
    format!("#{} {}{}", v.facts.number, clipped, ellipsis)
}

/// 2026-09-26: The chart: `main` as a line, each ordered PR as a branch off
/// it, merged back in order, for the first [`CHART_PR_BOUND`] PRs. The
/// caption says "Showing N of M" and lists the rest through [`cap_prs`]. With
/// nothing to order, a sentence replaces the chart.
pub fn render_order_chart(views: &[PrView]) -> String {
    let order = merge_order(views);
    if order.is_empty() {
        if views
            .iter()
            .any(|v| !v.facts.merged && v.facts.paths_unknown)
        {
            return "_Nothing could be ordered: the changed files of the open PR(s) came \
                    back unreadable on this run._\n"
                .to_string();
        }
        return "_No open, non-draft PRs to order._\n".to_string();
    }
    let shown = order.len().min(CHART_PR_BOUND);
    let mut out = String::from("```mermaid\ngitGraph\n  commit id: \"main\"\n");
    for number in &order[..shown] {
        let v = views
            .iter()
            .find(|v| v.facts.number == *number)
            .expect("order is drawn from these views");
        out.push_str(&format!(
            "  branch pr-{number}\n  commit id: \"{}\"\n  checkout main\n  merge pr-{number}\n",
            commit_label(v)
        ));
    }
    out.push_str("```\n\n");
    out.push_str(&format!(
        "Showing {shown} of {} open PRs, left to right in recommended merge order.",
        order.len()
    ));
    if order.len() > shown {
        out.push_str(&format!(" Not charted: {}.", cap_prs(&order[shown..])));
    }
    out.push_str(
        "\nOrder: fewest conflict partners first (a partner shares a gate target; \
         a whole-repo diff contends with every gated PR), then fewest targets \
         re-opened, then smallest diff, then lowest PR number. Drafts and merged \
         PRs are not ranked.\n",
    );
    out
}

/// 2026-09-26: "Recommended next steps": the head of [`merge_order`], the
/// unmerged PRs whose changed files were unreadable, unmerged PRs that
/// contend with a merged one, and merged PRs with promotion debt. Its inputs
/// are the changed paths and the draft and merged flags; the closing line
/// names what they cannot show.
pub fn render_next_steps(views: &[PrView]) -> String {
    let mut out = String::from("\n### Recommended next steps\n\n");
    let order = merge_order(views);
    match order.first() {
        None if views
            .iter()
            .any(|v| !v.facts.merged && v.facts.paths_unknown) =>
        {
            out.push_str(
                "- **No recommendation.** Every rankable PR's changed files came \
                       back unreadable on this run.\n",
            )
        }
        None => out.push_str("- Nothing to merge: no open, non-draft PRs.\n"),
        Some(head) => {
            let v = views.iter().find(|v| v.facts.number == *head).unwrap();
            out.push_str(&format!(
                "- **Merge next: #{head}** ({}) — least disruptive open PR under the \
                 order rule above.\n",
                super::escape(&v.facts.title)
            ));
        }
    }

    let blind: Vec<u64> = views
        .iter()
        .filter(|v| !v.facts.merged && v.facts.paths_unknown)
        .map(|v| v.facts.number)
        .collect();
    if !blind.is_empty() {
        out.push_str(&format!(
            "- **Cannot rank {}:** their changed files came back unreadable, so this \
             run has no input to order them by. They are counted as touching \
             everything elsewhere in this comment; re-run before trusting the order.\n",
            cap_prs(&blind)
        ));
    }

    let merged: Vec<&PrView> = views.iter().filter(|v| v.facts.merged).collect();
    let mut regate: Vec<u64> = Vec::new();
    for v in views.iter().filter(|v| !v.facts.merged) {
        if merged.iter().any(|m| contend(v, m)) {
            regate.push(v.facts.number);
        }
    }
    if !regate.is_empty() {
        out.push_str(&format!(
            "- **Re-gate before merging:** {} — a merged PR in this window moved a \
             baseline they were measured against.\n",
            cap_prs(&regate)
        ));
    }

    let mut owing: Vec<u64> = merged
        .iter()
        .filter(|v| !v.promotion_debt.is_empty())
        .map(|v| v.facts.number)
        .collect();
    owing.sort_unstable();
    if !owing.is_empty() {
        out.push_str(&format!(
            "- **Discharge promotion debt:** merged {} shipped without a \
             promotion-candidate gate (see the debt table below).\n",
            cap_prs(&owing)
        ));
    }

    out.push_str(
        "\n_Derived only from changed paths and merge state. This section cannot \
         see check or required-context status, review state, true git \
         mergeability, or whether one PR unblocks another — those live on each \
         PR._\n",
    );
    out
}

#[cfg(test)]
#[path = "telemetry_order_tests.rs"]
mod telemetry_order_tests;
