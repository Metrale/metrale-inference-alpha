// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Validate a harvested ledger artifact and append its new events
//! to the journey ledger.
//!
//! ```text
//! ledger_harvest --root . --pr 433 --from downloaded/pr-433.jsonl
//! ```
//!
//! The `pr-categorize` job in `ci.yml` holds `permissions: contents: read`, so its
//! `category` line leaves as a workflow artifact; `governance-harvest.yml`,
//! running default-branch code, downloads it and calls this binary. The
//! artifact's content is untrusted: it was produced on the PR author's branch.
//!
//! Checks, each fatal unless noted:
//! - `--pr` comes from the caller (the run's own API record), and any event
//!   naming another PR is rejected, so an artifact cannot write into another
//!   PR's journey.
//! - Every non-blank line must parse as an [`Event`].
//! - Only [`EventKind::Category`] is accepted; gate and measurement events are
//!   never taken from an artifact.
//! - A category whose first segment is not in the taxonomy is dropped with a
//!   warning (not fatal). `pr_taxonomy::load` validates that every `_benches`
//!   id is a required gate.
//!
//! An error stops the run after the lines before it were appended; the
//! harvest workflow restores its snapshot of the ledger when this exits
//! non-zero. Events whose [`Event::identity`] (which excludes `at`) is already
//! in the ledger are skipped, so re-harvesting an artifact adds nothing.
//!
//! Owner: bench (governance ledger).
//! Invariants:
//! - An event whose `pr` differs from `--pr` is never appended.
//! - Only `Category` events are ever appended.

use anyhow::{Context, Result, bail};
use metrale_bench::gate::{pr_taxonomy, required};
use metrale_governance::event::{Event, EventKind};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn require(name: &str) -> Result<String> {
    arg(name).with_context(|| format!("{name} is required"))
}

fn main() -> Result<()> {
    let root = std::path::PathBuf::from(arg("--root").unwrap_or_else(|| ".".into()));

    // 2026-09-26: No default PR number: defaulting to the artifact's own
    // claim would let it choose which journey it writes.
    let pr: u64 = require("--pr")?
        .parse()
        .context("--pr must be a number, taken from the RUN's API record")?;
    let from = std::path::PathBuf::from(require("--from")?);

    let text =
        std::fs::read_to_string(&from).with_context(|| format!("reading {}", from.display()))?;

    // 2026-09-26: A taxonomy load failure is fatal; an empty tree would drop
    // every category.
    let roots = pr_taxonomy::load(&root).context("loading the taxonomy to validate categories")?;

    let dest = metrale_governance::ledger::path_for(&root, pr);
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // 2026-09-26: Identities already in the ledger, read before appending.
    let seen: std::collections::BTreeSet<String> = if dest.exists() {
        metrale_governance::ledger::read_all(&dest)
            .with_context(|| format!("reading the existing ledger at {}", dest.display()))?
            .events
            .iter()
            .map(metrale_governance::event::Event::identity)
            .collect()
    } else {
        Default::default()
    };

    let (mut appended, mut skipped, mut dropped) = (0usize, 0usize, 0usize);
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let event: Event = serde_json::from_str(line)
            .with_context(|| format!("{}:{} is not a well-formed Event", from.display(), i + 1))?;

        if event.pr != pr {
            bail!(
                "{}:{} claims pr={} but this artifact belongs to pr={pr} (per the run record). \
                 Refusing: an artifact naming its own PR is not evidence.",
                from.display(),
                i + 1,
                event.pr
            );
        }

        let label = event.node_label();
        let EventKind::Category { value, status } = &event.kind else {
            bail!(
                "{}:{} is a `{label}` event. Only `category` is harvested — Gate and Measurement \
                 are written where those things happen (beside the .benchmarks/ record, on the \
                 box that measured them), and accepting them from an artifact would let a PR \
                 assert its own gate verdicts.",
                from.display(),
                i + 1,
            );
        };

        // 2026-09-26: Dropped with a warning, not fatal, so a taxonomy rename
        // does not stop the harvest of older artifacts.
        let segments = required::parse_category(value);
        let (_, matched) = pr_taxonomy::benches_for_matched(&roots, &segments);
        if !segments.is_empty() && matched == 0 {
            eprintln!(
                "warning: dropping {value:?} (status {status:?}) — no segment resolves in the \
                 taxonomy"
            );
            dropped += 1;
            continue;
        }

        if seen.contains(&event.identity()) {
            skipped += 1;
            continue;
        }
        metrale_governance::ledger::append(&dest, &event)?;
        appended += 1;
    }

    println!(
        "{}: appended {appended}, already present {skipped}, dropped {dropped}",
        dest.display()
    );
    Ok(())
}
