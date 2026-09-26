// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Print the benchmarks a classified PR intent implies
//! (`implied_benches --root . --path performance/decode`), by the same walk
//! the gate uses, [`metrale_bench::gate::pr_taxonomy::benches_for_matched`].
//! `ci.yml` calls it to render the PR summary table.
//!
//! | stream | meaning |
//! |---|---|
//! | stdout | one bench id per line, sorted; empty means "implies nothing" |
//! | stderr | an unknown-segment warning (exit 0), or the cause (exit != 0) |
//! | exit 0 | an answer was computed, possibly empty |
//! | exit 1 | no answer: unreadable or malformed taxonomy, or bad invocation |
//!
//! An unknown segment warns rather than fails: the walk stops at it and keeps
//! the benches of the segments before it, as the gate does.
//!
//! Owner: bench gate.
//! Invariants:
//! - A taxonomy that `pr_taxonomy::load` rejects exits non-zero; it is never
//!   printed as an empty answer.

use anyhow::{Context, Result, bail};
use metrale_bench::gate::{pr_taxonomy, required};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() -> Result<()> {
    let root = std::path::PathBuf::from(arg("--root").unwrap_or_else(|| ".".into()));

    // 2026-09-26: `--path` has no default. An empty value means "not
    // classified"; an absent flag is an error.
    let path = arg("--path").context(
        "--path is required (pass an empty string for \"not classified\"). \
         There is no default intent: guessing one would attribute benchmarks \
         to a PR nobody classified.",
    )?;

    // 2026-09-26: A load error propagates; defaulting it to an empty tree
    // would print "implies nothing" for a broken taxonomy.
    let roots = pr_taxonomy::load(&root)
        .with_context(|| format!("reading the taxonomy under {}", root.display()))?;

    let segments = required::parse_category(&path);
    let (benches, matched) = pr_taxonomy::benches_for_matched(&roots, &segments);

    if matched < segments.len() {
        // 2026-09-26: Without this warning, `performance/decodes` and
        // `performance` would print the same set with no sign of the typo.
        eprintln!(
            "warning: segment {:?} is not in the taxonomy; matched {} of {} segments",
            segments[matched],
            matched,
            segments.len()
        );
    }

    for bench in &benches {
        println!("{bench}");
    }
    if benches.is_empty() && segments.is_empty() && !path.trim().is_empty() {
        // 2026-09-26: A non-empty `--path` with no segments (e.g. "///") is a
        // caller error.
        bail!("--path {path:?} contains no usable segments");
    }
    Ok(())
}
