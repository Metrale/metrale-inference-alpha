// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The open-PR telemetry comment: which kernel targets each PR
//! re-opens, which targets more than one open PR re-opens, a recommended
//! merge order, and promotion-candidate debt.
//!
//! [`render`] reads only its [`PrFacts`] and the tree (the kernel taxonomy and
//! CODEOWNERS); the `pr-telemetry` workflow collects the facts from GitHub and
//! posts the body. The comment is advisory: [`render`] returns only text.
//!
//! Owner: bench gate.
//! Invariants:
//! - A PR with `paths_unknown` is counted at the widest reach, every target
//!   and every promotion candidate, and [`render`] lists such PRs (through
//!   `order::cap_prs`) in a warning before the tables.
//! - The body always starts with [`MARKER_START`] and ends with
//!   [`MARKER_END`] and a newline.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::{codeowners, coverage, taxon};

#[path = "telemetry_order.rs"]
pub mod order;
pub use order::{CHART_PR_BOUND, merge_order};

/// 2026-09-26: The markers around the body. The workflow's publish job finds
/// its previous comment by `MARKER_START` and replaces it
/// (`.github/workflows/pr-telemetry.yml`, `marker:`).
pub const MARKER_START: &str = "<!-- metrale-pr-telemetry:start -->";
pub const MARKER_END: &str = "<!-- metrale-pr-telemetry:end -->";

/// 2026-09-26: What the workflow collects about one open or merged PR.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PrFacts {
    pub number: u64,
    pub title: String,
    pub author: String,
    #[serde(default)]
    pub draft: bool,
    /// 2026-09-26: True when GitHub reports the PR merged (`merged_at` set),
    /// into whatever base it targeted. A merged PR stays in the debt table,
    /// the next steps and the targets table, and leaves the PR table, the
    /// collisions and the merge order. The workflow collects open and merged
    /// PRs only, so a closed, unmerged PR never appears.
    #[serde(default)]
    pub merged: bool,
    /// 2026-09-26: Repo-relative paths this PR changes. Meaningful only when
    /// [`Self::paths_unknown`] is false.
    #[serde(default)]
    pub changed_paths: Vec<String>,
    /// 2026-09-26: True when the collector could not read this PR's file list.
    /// The PR is kept, and [`views`] counts it as whole-repo with every
    /// promotion candidate as debt, so an unread diff never renders as an
    /// empty one.
    #[serde(default)]
    pub paths_unknown: bool,
}

/// 2026-09-26: One PR's derived position in the kernel taxonomy.
#[derive(Debug, Clone)]
pub struct PrView {
    pub facts: PrFacts,
    pub hardware: BTreeSet<String>,
    pub models: BTreeSet<(String, String)>,
    pub targets: BTreeSet<taxon::Target>,
    pub owners: Vec<String>,
    /// 2026-09-26: True when the paths are unknown or any changed path is
    /// outside `kernels/`; `targets` is then every target in the tree.
    pub whole_repo: bool,
    /// 2026-09-26: The [`coverage::PROMOTION_CANDIDATES`] this PR's paths
    /// would invalidate (`coverage::promotion_debt`): gates that are not
    /// required, so the PR can merge without them.
    pub promotion_debt: Vec<&'static str>,
}

/// 2026-09-26: Derive every PR's view from its facts, the kernel taxonomy and
/// CODEOWNERS.
pub fn views(root: &Path, prs: &[PrFacts]) -> Vec<PrView> {
    let rules = codeowners::load(root);
    let all_targets: BTreeSet<taxon::Target> = taxon::walk(root).into_iter().collect();
    prs.iter()
        .map(|facts| {
            let kernel_paths: Vec<String> = facts
                .changed_paths
                .iter()
                .filter(|p| taxon::hardware_of(p).is_some())
                .cloned()
                .collect();
            // 2026-09-26: Unknown paths take the whole-repo branch, which sets
            // every target here and makes the PR contend with every gated PR
            // in `order`.
            let whole_repo = facts.paths_unknown || facts.changed_paths.len() > kernel_paths.len();
            PrView {
                hardware: taxon::hardware_span(&kernel_paths),
                models: taxon::model_span(&kernel_paths),
                targets: if whole_repo {
                    all_targets.clone()
                } else {
                    taxon::affected(root, &kernel_paths)
                },
                owners: codeowners::owners_for_paths(&rules, &facts.changed_paths),
                whole_repo,
                // 2026-09-26: With unknown paths the debt is every candidate;
                // `render` labels the row as assumed.
                promotion_debt: if facts.paths_unknown {
                    coverage::PROMOTION_CANDIDATES
                        .iter()
                        .map(|gate| gate.id)
                        .collect()
                } else {
                    coverage::promotion_debt(facts.changed_paths.iter().map(String::as_str))
                },
                facts: facts.clone(),
            }
        })
        .collect()
}

/// 2026-09-26: Targets that more than one unmerged PR re-opens, with those
/// PRs' numbers.
pub fn collisions(views: &[PrView]) -> BTreeMap<String, Vec<u64>> {
    let mut by_target: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for view in views.iter().filter(|view| !view.facts.merged) {
        for target in &view.targets {
            by_target
                .entry(target.to_string())
                .or_default()
                .push(view.facts.number);
        }
    }
    by_target.retain(|_, prs| prs.len() > 1);
    by_target
}

fn escape(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

/// 2026-09-26: The comment body, markers included.
pub fn render(root: &Path, prs: &[PrFacts]) -> String {
    let views = views(root, prs);
    let all_targets = taxon::walk(root);
    let mut out = String::new();

    out.push_str(MARKER_START);
    out.push_str("\n## Open-PR telemetry\n\n");

    if views.is_empty() {
        out.push_str("_No open pull requests._\n");
        out.push_str(MARKER_END);
        out.push('\n');
        return out;
    }

    out.push_str(&order::render_order_chart(&views));
    out.push_str(
        "\nAdvisory. Nothing here blocks a merge — the blocking checks live on each PR.\n",
    );

    // 2026-09-26: Name the PRs whose rows below are assumed rather than
    // measured.
    let blind: Vec<u64> = views
        .iter()
        .filter(|v| v.facts.paths_unknown)
        .map(|v| v.facts.number)
        .collect();
    if !blind.is_empty() {
        out.push_str(&format!(
            "\n> ⚠ **Changed files unavailable for {}.** The API call for their diffs \
             failed on this run, so nothing below measures them. They are counted at \
             MAXIMUM blast radius — every target, every promotion candidate — because \
             an unknown diff must never render as an empty one. Re-run this workflow \
             before using any row they appear in.\n",
            order::cap_prs(&blind)
        ));
    }

    out.push_str(&order::render_next_steps(&views));

    // 2026-09-26: Unmerged PRs, grouped by the hardware they touch.
    let merged_count = views.iter().filter(|v| v.facts.merged).count();
    out.push_str("\n### Pull requests\n\n");
    if merged_count > 0 {
        out.push_str(&format!(
            "_{merged_count} merged PR(s) tracked for debt only — see the ledger below._\n\n"
        ));
    }
    out.push_str("| PR | category | targets re-opened | codeowners |\n");
    out.push_str("|---|---|---|---|\n");
    let mut grouped: BTreeMap<String, Vec<&PrView>> = BTreeMap::new();
    for view in views.iter().filter(|v| !v.facts.merged) {
        // 2026-09-26: "host / non-kernel" means no changed path is under
        // `kernels/`, which an unread diff cannot show.
        let key = if view.facts.paths_unknown {
            "unknown (changed files unreadable)".to_string()
        } else if view.hardware.is_empty() {
            "host / non-kernel".to_string()
        } else {
            view.hardware
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(" + ")
        };
        grouped.entry(key).or_default().push(view);
    }
    for (category, group) in &grouped {
        for view in group {
            let targets = if view.facts.paths_unknown {
                "ALL — **assumed**, changed files unreadable".to_string()
            } else if view.whole_repo {
                "ALL (diff reaches outside kernels/)".to_string()
            } else if view.targets.is_empty() {
                "none".to_string()
            } else {
                format!("{}", view.targets.len())
            };
            // 2026-09-26: "—" means CODEOWNERS matched nobody; an unread diff
            // is "unknown".
            let owners = if view.facts.paths_unknown {
                "unknown".to_string()
            } else if view.owners.is_empty() {
                "—".to_string()
            } else {
                view.owners.join(" ")
            };
            out.push_str(&format!(
                "| #{} {}{} | {} | {} | {} |\n",
                view.facts.number,
                if view.facts.draft { "(draft) " } else { "" },
                escape(&view.facts.title),
                escape(category),
                targets,
                escape(&owners),
            ));
        }
    }

    // 2026-09-26: The debt section is rendered even when no PR owes
    // anything: a placeholder row, or a sentence when no gate is a
    // candidate, says so.
    out.push_str("\n### Promotion-candidate debt\n\n");
    if coverage::PROMOTION_CANDIDATES.is_empty() {
        out.push_str(
            "No gates are on a promotion path, so nothing can be owed. When one \
             is registered (`coverage::PROMOTION_CANDIDATES`), every PR whose \
             paths it covers appears here until a record discharges it.\n",
        );
    } else {
        let owing: Vec<&PrView> = views
            .iter()
            .filter(|v| !v.promotion_debt.is_empty())
            .collect();
        out.push_str(
            "These gates are NOT required, so these PRs can merge without them. \
             Each row is coverage this repository chose not to buy — recorded so \
             the choice stays visible rather than becoming an assumption.\n\n",
        );
        out.push_str("| PR | merged? | title | gates that wanted to run |\n|---|---|---|---|\n");
        if owing.is_empty() {
            out.push_str("| — | — | _no tracked PR touches a promotion candidate's paths_ | — |\n");
        }
        for v in owing {
            out.push_str(&format!(
                "| #{} | {} | {} | {} |\n",
                v.facts.number,
                if v.facts.merged { "**yes**" } else { "not yet" },
                escape(&v.facts.title),
                if v.facts.paths_unknown {
                    format!(
                        "{} — **assumed** (paths unreadable)",
                        v.promotion_debt.join(", ")
                    )
                } else {
                    v.promotion_debt.join(", ")
                }
            ));
        }
    }

    let collisions = collisions(&views);
    out.push_str("\n### Collisions\n\n");
    if collisions.is_empty() {
        out.push_str("None: no target is re-opened by more than one open PR.\n");
    } else {
        out.push_str(
            "Each PR below is measured against a baseline another open PR will \
             move. Whichever lands second needs re-gating.\n\n\
             | target | PRs |\n|---|---|\n",
        );
        for (target, prs) in &collisions {
            out.push_str(&format!("| `{target}` | {} |\n", order::cap_prs(prs)));
        }
    }

    out.push_str(&format!(
        "\n### Targets ({} total)\n\nEvery target is listed, including the ones no open PR \
         touches. Showing only the affected ones would silently turn *ungated* into \
         *unaffected*.\n\n| target | re-opened by |\n|---|---|\n",
        all_targets.len()
    ));
    for target in &all_targets {
        let key = target.to_string();
        let touching: Vec<u64> = views
            .iter()
            .filter(|v| v.targets.contains(target))
            .map(|v| v.facts.number)
            .collect();
        out.push_str(&format!(
            "| `{key}` | {} |\n",
            if touching.is_empty() {
                "—".to_string()
            } else {
                order::cap_prs(&touching)
            }
        ));
    }

    out.push_str(MARKER_END);
    out.push('\n');
    out
}

#[cfg(test)]
#[path = "telemetry_tests.rs"]
mod telemetry_tests;

#[cfg(test)]
#[path = "telemetry_unknown_tests.rs"]
mod telemetry_unknown_tests;
