// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `met doctor`: is this box ready to run a benchmark? Checks the
//! metrale home resolves and is writable, whether the signing identity is
//! committed, and whether the recipe index is populated and readable.
//!
//! Owner: server CLI (`met doctor`).
//! Invariants:
//! - Every check has a branch that reports a problem.
//! - The exit code is 1 when any finding is a problem, else 0.

use anyhow::Result;
use metrale_bench::artifacts::{HomeFault, MetraleHome};
use metrale_bench::gate;

pub struct Finding {
    pub label: &'static str,
    pub problem: bool,
    pub detail: String,
    /// 2026-09-26: What to do about it. Empty when there is nothing to do.
    pub remedy: String,
}

impl Finding {
    fn ok(label: &'static str, detail: impl Into<String>) -> Self {
        Self {
            label,
            problem: false,
            detail: detail.into(),
            remedy: String::new(),
        }
    }
    fn bad(label: &'static str, detail: impl Into<String>, remedy: impl Into<String>) -> Self {
        Self {
            label,
            problem: true,
            detail: detail.into(),
            remedy: remedy.into(),
        }
    }
}

/// 2026-09-26: Where the home is, and where that came from (`MetraleHome::describe`).
pub fn check_home() -> Finding {
    match MetraleHome::resolve() {
        Err(e) => Finding::bad(
            "home",
            format!("cannot be resolved: {e:#}"),
            "set METRALE_HOME, or ensure HOME is set and non-empty.",
        ),
        Ok(h) => Finding::ok("home", h.describe()),
    }
}

/// 2026-09-26: Can this process write there? `MetraleHome::fault` probes by
/// writing, not by reading a mode bit.
pub fn check_writable() -> Finding {
    let Ok(h) = MetraleHome::resolve() else {
        return Finding::bad(
            "writable",
            "skipped — the home could not be resolved",
            "fix `home` first.",
        );
    };
    match h.fault() {
        None => Finding::ok("writable", format!("{} is writable", h.root.display())),
        Some(f @ HomeFault::NotWritable { .. }) => Finding::bad(
            "writable",
            format!("{} {f}", h.root.display()),
            "every gate writes its run frames and provisioned artifacts here; \
             an unwritable home fails the run minutes in, reported as an empty \
             recipe index.",
        ),
        Some(f) => Finding::bad(
            "writable",
            format!("{} {f}", h.root.display()),
            "point METRALE_HOME at a directory this user can create and write.",
        ),
    }
}

/// 2026-09-26: The signing identity, and whether its fingerprint is committed.
///
/// A `.pub` that `signing::register` wrote on disk proves only that this box
/// once signed something, so `committed_signers` asks `git ls-files`.
pub fn check_identity(repo_root: Option<&std::path::Path>) -> Finding {
    let Ok(h) = MetraleHome::resolve() else {
        return Finding::bad(
            "identity",
            "skipped — the home could not be resolved",
            "fix `home` first.",
        );
    };
    let key = h.root.join("identity").join("ed25519.pk8");
    if !key.exists() {
        return Finding::ok(
            "identity",
            "no signing key yet — one is minted on this box's first gate record",
        );
    }
    let Some(root) = repo_root else {
        return Finding::bad(
            "identity",
            format!("{} exists, but this is not a git repo", key.display()),
            "run from inside the metrale checkout so the committed signer list \
             can be read.",
        );
    };
    let Ok(identity) = gate::signing::load_or_create(&h.root) else {
        return Finding::bad(
            "identity",
            format!("{} is present but unusable", key.display()),
            "delete it and let the next gate run mint a fresh one.",
        );
    };
    let fp = identity.fingerprint().to_string();
    match gate::signing::committed_signers(root) {
        Err(e) => Finding::bad(
            "identity",
            format!("{fp}; could not read .github/record-signers/: {e:#}"),
            "run from inside the checkout.",
        ),
        Ok(list) if list.contains(&fp) => Finding::ok("identity", format!("{fp}, committed")),
        Ok(list) => Finding::bad(
            "identity",
            format!(
                "{fp} is NOT committed in .github/record-signers/ ({} signer(s) are)",
                list.len()
            ),
            "commit the one-line .pub beside the records this box produces, and \
             remember every record one PR adds for a SPEED-class gate must carry \
             the same fingerprint — a campaign split across boxes cannot be \
             certified.",
        ),
    }
}

/// 2026-09-26: Has the recipe index been populated, and can it be read?
pub fn check_recipes() -> Finding {
    let Ok(h) = MetraleHome::resolve() else {
        return Finding::bad(
            "recipes",
            "skipped — the home could not be resolved",
            "fix `home` first.",
        );
    };
    // 2026-09-26: Through `cache_dir`, the directory `recipe::fetch::cached`
    // reads.
    let index = crate::recipe::fetch::cache_dir(&h.root).join("index.json");
    match std::fs::read_to_string(&index) {
        // 2026-09-26: Parsed by `recipe::fetch::parse_cache`, the reader
        // `cached` uses, so doctor and the gate agree on the schema.
        Ok(text) => match crate::recipe::fetch::parse_cache(&text) {
            Ok(idx) => {
                let n = idx.recipes.len();
                if n == 0 {
                    Finding::bad(
                        "recipes",
                        format!("{} parses but lists none", index.display()),
                        "run `met sync-recipes`.",
                    )
                } else {
                    Finding::ok("recipes", format!("{n} cached in {}", index.display()))
                }
            }
            Err(e) => Finding::bad(
                "recipes",
                format!(
                    "{} does not parse as a recipe index: {e:#}",
                    index.display()
                ),
                "delete it and run `met sync-recipes`.",
            ),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Finding::bad(
            "recipes",
            format!("{} has never been written", index.display()),
            "run `met sync-recipes`.",
        ),
        // 2026-09-26: Unreadable is not absent, so it gets its own finding.
        Err(e) => Finding::bad(
            "recipes",
            format!("{} exists but cannot be read: {e}", index.display()),
            "this is a permission problem, not a missing sync — check `writable` \
             above; `met sync-recipes` would fail on the same path.",
        ),
    }
}

pub fn run(repo_root: Option<&std::path::Path>) -> (Vec<Finding>, bool) {
    let findings = vec![
        check_home(),
        check_writable(),
        check_identity(repo_root),
        check_recipes(),
    ];
    let bad = findings.iter().any(|f| f.problem);
    (findings, bad)
}

/// 2026-09-26: `met doctor`. Exit code 1 when any finding is a problem.
pub fn dispatch() -> Result<i32> {
    let repo_root = super::bench_run::repo_root().ok();
    let (findings, bad) = run(repo_root.as_deref());
    for f in &findings {
        let mark = if f.problem { "PROBLEM" } else { "ok" };
        println!("{:>8}  {:<9} {}", mark, f.label, f.detail);
        if f.problem && !f.remedy.is_empty() {
            println!("          {:<9} {}", "", f.remedy);
        }
    }
    println!();
    if bad {
        println!(
            "{} problem(s) found.",
            findings.iter().filter(|f| f.problem).count()
        );
    } else {
        println!("no problems found.");
    }
    Ok(i32::from(bad))
}

#[cfg(test)]
#[path = "doctor_tests.rs"]
mod doctor_tests;
