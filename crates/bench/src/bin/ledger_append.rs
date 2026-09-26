// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Append one `category` event to a PR's journey ledger,
//! `governance/pr-<n>.jsonl` under `--root`.
//!
//! ```text
//! ledger_append category --root . --pr 433 --head-sha abc123 \
//!               --run-id 42 --attempt 1 --at 1786280000 \
//!               --value performance/decode --status ok
//! ```
//!
//! It does not touch git. In CI the `pr-categorize` job runs it and the file leaves
//! as a workflow artifact, which `governance-harvest.yml` validates with
//! `ledger_harvest` and commits.
//!
//! Owner: bench (governance ledger).
//! Invariants: none beyond the types.

use anyhow::{Context, Result, bail};
use metrale_governance::event::{Event, EventKind};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn require(name: &str) -> Result<String> {
    // 2026-09-26: No default: a line that guessed its PR number or sha would
    // record the wrong thing.
    arg(name).with_context(|| format!("{name} is required"))
}

fn main() -> Result<()> {
    let root = std::path::PathBuf::from(arg("--root").unwrap_or_else(|| ".".into()));

    let pr: u64 = require("--pr")?.parse().context("--pr must be a number")?;
    let head_sha = require("--head-sha")?;
    let run_id = require("--run-id")?;
    let attempt: u32 = arg("--attempt")
        .unwrap_or_else(|| "1".into())
        .parse()
        .context("--attempt must be a number")?;

    // 2026-09-26: `at` is supplied by the caller, not read from the clock, so
    // a replay is reproducible. `Event::identity` excludes it, so a re-run's
    // duplicate line collapses when the journey is read.
    let at: u64 = require("--at")?
        .parse()
        .context("--at must be a unix time")?;

    let kind = match std::env::args().nth(1).as_deref() {
        Some("category") => EventKind::Category {
            value: require("--value")?,
            status: require("--status")?,
        },
        other => bail!(
            "unknown event kind {other:?}; this binary currently writes `category` only. \
             Gate and Measurement events are produced where those things happen, not here."
        ),
    };

    let event = Event {
        pr,
        head_sha,
        run_id,
        attempt,
        at,
        kind,
    };
    let path = metrale_governance::ledger::path_for(&root, pr);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    metrale_governance::ledger::append(&path, &event)?;
    println!("appended to {}", path.display());
    Ok(())
}
