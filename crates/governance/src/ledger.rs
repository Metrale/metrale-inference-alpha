// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Reading, appending and materialising a journey file.
//!
//! Owner: metrale-governance.
//! Invariants:
//! - [`append`] only appends; it never truncates or rewrites a file.
//! - [`read_all`] returns every non-blank line or an error; it never drops a
//!   line it cannot parse.

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{Context, Result};
use lattice_core::{CollectionConfig, CollectionEngine, Distance, HnswConfig, Point, VectorConfig};

use crate::event::Event;

/// 2026-09-26: The events of one journey file, in file order.
#[derive(Debug, Clone, Default)]
pub struct Journey {
    pub events: Vec<Event>,
}

impl Journey {
    /// 2026-09-26: Deduplicate by [`Event::identity`], keeping the first
    /// occurrence and the order.
    pub fn deduplicated(mut self) -> Self {
        let mut seen = BTreeSet::new();
        self.events.retain(|e| seen.insert(e.identity()));
        self
    }

    /// 2026-09-26: The `Gate` events with this id, in `events` order.
    pub fn gate_history<'a>(&'a self, gate: &'a str) -> impl Iterator<Item = &'a Event> {
        self.events.iter().filter(
            move |e| matches!(&e.kind, crate::event::EventKind::Gate { id, .. } if id == gate),
        )
    }
}

/// 2026-09-26: Append one event as a JSON line, creating the file and its
/// parent directories if needed. Callers pass [`path_for`], one file per pull
/// request.
pub fn append(path: &Path, event: &Event) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let line = serde_json::to_string(event).context("encoding journey event")?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {} for append", path.display()))?;
    writeln!(file, "{line}").with_context(|| format!("appending to {}", path.display()))
}

/// 2026-09-26: Read a journey, skipping blank lines. Does not deduplicate.
///
/// # Errors
/// When the file cannot be opened or read, or any non-blank line does not
/// parse as an [`Event`]; the message names the line number.
pub fn read_all(path: &Path) -> Result<Journey> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut events = Vec::new();
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("reading {} line {}", path.display(), n + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        events.push(
            serde_json::from_str(&line)
                .with_context(|| format!("parsing {} line {}", path.display(), n + 1))?,
        );
    }
    Ok(Journey { events })
}

/// 2026-09-26: Vector width of the materialised collection. Every point gets
/// the constant vector `[1.0]`: only the graph is used, and the vectors are
/// not embeddings.
const PLACEHOLDER_DIM: usize = 1;

/// 2026-09-26: Build an in-memory graph from a journey: one node per distinct
/// head sha plus one per event, and an `observed` edge from each commit node to
/// each of its events. Nothing is written to disk.
pub fn materialize(journey: &Journey) -> Result<CollectionEngine> {
    let config = CollectionConfig::new(
        "journey",
        VectorConfig::new(PLACEHOLDER_DIM, Distance::Cosine),
        HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 100,
            ef_construction: 200,
        },
    )
    .with_relation("observed", 0)
    .with_relation("precedes", 1);

    let mut engine =
        CollectionEngine::new(config).map_err(|e| anyhow::anyhow!("creating collection: {e}"))?;

    // 2026-09-26: Commits take ids `0..n` in sorted-sha order; events take
    // `n..` in journey order. A new sha that sorts earlier shifts every id.
    let mut commits: Vec<&str> = journey.events.iter().map(|e| e.head_sha.as_str()).collect();
    commits.sort_unstable();
    commits.dedup();

    let mut points = Vec::new();
    for (i, sha) in commits.iter().enumerate() {
        points.push(
            Point::new_vector(i as u64, vec![1.0; PLACEHOLDER_DIM])
                .with_field("label", br#""commit""#.to_vec())
                .with_field("sha", serde_json::to_vec(sha).unwrap_or_default()),
        );
    }
    let commit_base = commits.len() as u64;
    for (i, event) in journey.events.iter().enumerate() {
        points.push(
            Point::new_vector(commit_base + i as u64, vec![1.0; PLACEHOLDER_DIM])
                .with_field(
                    "label",
                    serde_json::to_vec(event.node_label()).unwrap_or_default(),
                )
                .with_field("at", serde_json::to_vec(&event.at).unwrap_or_default())
                .with_field("kind", serde_json::to_vec(&event.kind).unwrap_or_default()),
        );
    }
    engine
        .upsert_points(points)
        .map_err(|e| anyhow::anyhow!("upserting journey points: {e}"))?;

    for (i, event) in journey.events.iter().enumerate() {
        let Ok(commit_idx) = commits.binary_search(&event.head_sha.as_str()) else {
            continue;
        };
        engine
            .add_edge(commit_idx as u64, commit_base + i as u64, "observed", 1.0)
            .map_err(|e| anyhow::anyhow!("adding observed edge: {e}"))?;
    }

    Ok(engine)
}

/// 2026-09-26: `<root>/governance/pr-<pr>.jsonl`.
pub fn path_for(root: &Path, pr: u64) -> std::path::PathBuf {
    root.join("governance").join(format!("pr-{pr}.jsonl"))
}
