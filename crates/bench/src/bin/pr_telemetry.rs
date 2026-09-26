// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Render the open-PR telemetry comment body.
//!
//! Reads a JSON array of `PrFacts` on stdin and writes markdown to stdout; the
//! optional first argument is the repository root (default `.`). It does not
//! talk to GitHub: `pr-telemetry.yml` fetches the facts and posts the body.
//!
//! ```text
//! pr_telemetry . < facts.json > body.md
//! ```
//!
//! Owner: bench gate.
//! Invariants:
//! - Input that is not a JSON array of `PrFacts` is an error; nothing is
//!   printed.

use std::io::Read;

use metrale_bench::gate::telemetry::{PrFacts, render};

fn main() -> anyhow::Result<()> {
    let root = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    // 2026-09-26: A parse error propagates. Defaulted to an empty list, it
    // would render "_No open pull requests._" and the workflow would publish
    // it; failing the render job instead skips the publish job, which
    // `needs: render`, so the previous comment stands.
    let prs: Vec<PrFacts> = serde_json::from_str(input.trim())
        .map_err(|e| anyhow::anyhow!("the PR feed on stdin is not a JSON array of PrFacts: {e}"))?;

    print!("{}", render(&root, &prs));
    Ok(())
}
