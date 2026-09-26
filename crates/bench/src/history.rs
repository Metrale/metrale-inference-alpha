// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Run history: [`RunRecord`]s written by [`save`] and read by
//! [`load`], [`load_all`] and [`find`], under the artifact store's
//! `runs/<benchmark>` directory. A record built by [`RunRecord::new`] carries
//! every parameter, the target, the source and the engine version beside the
//! terminal frame. The headless driver and the TUI both save through
//! [`save`].
//!
//! Owner: bench.
//! Invariants:
//! - [`save`] never overwrites another run: it claims a fresh
//!   `run-<nanos>.json` with `create_new`, and the finished record replaces
//!   only its own claim, by rename.
//! - The readers skip files they cannot read or parse.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::artifacts::ArtifactStore;
use crate::benchmark::BenchmarkDescriptor;
use crate::params::{ParamSpec, ParamValues};
use crate::plugin::TargetEndpoint;
use crate::result::{BenchmarkResult, VerdictKind};

/// 2026-09-26: Current record schema. `0` marks a bare-frame file read
/// through `from_legacy`, or a record with no `schema` field.
pub const SCHEMA: u32 = 1;

/// 2026-09-26: Where a run was started from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunSource {
    Tui,
    Cli,
    /// 2026-09-26: A bare-frame file, which carries no provenance, or a
    /// record with no `source` field.
    #[default]
    Unknown,
}

/// 2026-09-26: One finished run, as stored. Only `benchmark_id`,
/// `recorded_at` and `frame` are required and every other field defaults, so
/// a bare [`BenchmarkResult`] (no `frame` key) never parses as a record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunRecord {
    #[serde(default)]
    pub schema: u32,
    /// 2026-09-26: The file stem [`save`] claimed; [`find`] looks a run up by
    /// it.
    #[serde(default)]
    pub run_id: String,
    pub benchmark_id: String,
    #[serde(default)]
    pub benchmark_name: String,
    /// 2026-09-26: Unix seconds when [`RunRecord::new`] assembled the record,
    /// after the run ended. A bare-frame file takes the number in its stem.
    pub recorded_at: u64,
    /// 2026-09-26: Stored flat rather than as a `TargetEndpoint`, whose
    /// constructor trims trailing slashes; [`RunRecord::target`] rebuilds it
    /// through that constructor.
    #[serde(default)]
    pub target_url: String,
    #[serde(default)]
    pub target_model: String,
    /// 2026-09-26: Every parameter, defaults included.
    #[serde(default)]
    pub params: BTreeMap<String, String>,
    /// 2026-09-26: The serve overrides the caller put on the target. The CLI
    /// sets them only under `--pull-request-gate`, to the served plan's merged
    /// set (the baseline's `[benchmarks.serve_overrides]` and
    /// `--serve-override`, after `--hermetic` expansion); the TUI sets none.
    /// Empty therefore does not mean the endpoint had no overrides.
    #[serde(default)]
    pub serve_overrides: BTreeMap<String, String>,
    #[serde(default)]
    pub source: RunSource,
    #[serde(default)]
    pub metrale_version: String,
    /// 2026-09-26: The run's terminal frame.
    pub frame: BenchmarkResult,
}

impl RunRecord {
    /// 2026-09-26: Assemble a record when the run ends. `run_id` stays empty
    /// until [`save`] sets it.
    pub fn new(
        descriptor: &BenchmarkDescriptor,
        values: &ParamValues,
        target: &TargetEndpoint,
        serve_overrides: BTreeMap<String, String>,
        source: RunSource,
        metrale_version: &str,
        frame: BenchmarkResult,
    ) -> Self {
        Self {
            schema: SCHEMA,
            run_id: String::new(),
            benchmark_id: descriptor.id.to_string(),
            benchmark_name: descriptor.name.to_string(),
            recorded_at: now_secs(),
            target_url: target.base_url.clone(),
            target_model: target.model.clone(),
            params: values.to_strings(),
            serve_overrides,
            source,
            metrale_version: metrale_version.to_string(),
            frame,
        }
    }

    /// 2026-09-26: The endpoint, rebuilt through `TargetEndpoint::new`.
    pub fn target(&self) -> TargetEndpoint {
        TargetEndpoint::new(&self.target_url, &self.target_model)
    }

    /// 2026-09-26: Rehydrate the stored parameters against a live schema
    /// through `ParamValues::from_overrides`, which errors on an unknown key or
    /// a value the spec's `ParamKind` rejects.
    pub fn values(&self, specs: &[ParamSpec]) -> Result<ParamValues> {
        let pairs = self
            .params
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect::<Vec<_>>();
        ParamValues::from_overrides(specs, pairs)
    }

    pub fn verdict_kind(&self) -> Option<VerdictKind> {
        self.frame.verdict.as_ref().map(|v| v.kind)
    }

    /// 2026-09-26: True when `schema` is 0.
    pub fn is_legacy(&self) -> bool {
        self.schema == 0
    }

    /// 2026-09-26: Compact age, e.g. `3m ago`.
    pub fn age_text(&self) -> String {
        let secs = now_secs().saturating_sub(self.recorded_at);
        match secs {
            0..=59 => format!("{secs}s ago"),
            60..=3599 => format!("{}m ago", secs / 60),
            3600..=86_399 => format!("{}h ago", secs / 3600),
            _ => format!("{}d ago", secs / 86_400),
        }
    }

    fn from_legacy(benchmark_id: &str, run_id: &str, frame: BenchmarkResult) -> Self {
        Self {
            schema: 0,
            recorded_at: run_id
                .trim_start_matches("run-")
                .parse()
                .unwrap_or_default(),
            run_id: run_id.to_string(),
            benchmark_id: benchmark_id.to_string(),
            benchmark_name: String::new(),
            target_url: String::new(),
            target_model: String::new(),
            params: BTreeMap::new(),
            serve_overrides: BTreeMap::new(),
            source: RunSource::Unknown,
            metrale_version: String::new(),
            frame,
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// 2026-09-26: Write `record`, setting its `run_id` to the file stem chosen,
/// and return the path.
///
/// The name is `run-<unix_nanos>`, zero-padded to 19 digits, so a filename
/// sort is a time sort while the nanos have 19 digits (until 2286). A taken
/// name moves the nanos up by one until a name is free.
pub fn save(store: &ArtifactStore, record: &mut RunRecord) -> Result<PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or_default();
    save_at(store, record, nanos, |_| {})
}

/// 2026-09-26: [`save`] with the clock and a post-claim hook supplied, so
/// tests can force a collision and inspect the claimed placeholder.
fn save_at<F>(
    store: &ArtifactStore,
    record: &mut RunRecord,
    mut nanos: u64,
    after_claim: F,
) -> Result<PathBuf>
where
    F: FnOnce(&Path),
{
    let dir = store.runs_dir(&record.benchmark_id)?;
    // 2026-09-26: `create_new` claims the name atomically, so two processes
    // saving into this directory cannot both take it.
    let (run_id, path) = loop {
        let id = format!("run-{nanos:019}");
        let path = dir.join(format!("{id}.json"));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(_) => break (id, path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                nanos = nanos.saturating_add(1);
            }
            Err(e) => {
                return Err(e).with_context(|| format!("claiming {}", path.display()));
            }
        }
    };
    record.run_id = run_id;
    after_claim(&path);
    let json = serde_json::to_string_pretty(&record).context("serializing the run record")?;
    // 2026-09-26: Write beside the claim and rename over it, so a reader sees
    // the empty placeholder or the whole record, never part of one.
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("publishing {}", path.display()))?;
    Ok(path)
}

/// 2026-09-26: Every run for one benchmark, newest first. A file that cannot
/// be read or parsed is skipped; a directory that cannot be created or read
/// gives an empty list.
pub fn load(store: &ArtifactStore, benchmark_id: &str) -> Vec<RunRecord> {
    let Ok(dir) = store.runs_dir(benchmark_id) else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<RunRecord> = entries
        .flatten()
        .filter_map(|e| read_one(benchmark_id, &e.path()))
        .collect();
    sort_newest_first(&mut out);
    out
}

/// 2026-09-26: Every run for every registered benchmark, newest first across
/// all of them.
pub fn load_all(store: &ArtifactStore) -> Vec<RunRecord> {
    let mut out: Vec<RunRecord> = crate::registry::all()
        .iter()
        .flat_map(|d| load(store, d.id))
        .collect();
    sort_newest_first(&mut out);
    out
}

/// 2026-09-26: One run by its `run_id`, across every registered benchmark.
pub fn find(store: &ArtifactStore, run_id: &str) -> Option<RunRecord> {
    load_all(store).into_iter().find(|r| r.run_id == run_id)
}

fn sort_newest_first(records: &mut [RunRecord]) {
    records.sort_by(|a, b| {
        b.recorded_at
            .cmp(&a.recorded_at)
            .then_with(|| b.run_id.cmp(&a.run_id))
    });
}

/// 2026-09-26: Parse one `run-*.json` file as a record, or else as a bare
/// frame. Other files in the directory are ignored.
fn read_one(benchmark_id: &str, path: &Path) -> Option<RunRecord> {
    let stem = path.file_stem()?.to_str()?;
    if !stem.starts_with("run-") || path.extension()? != "json" {
        return None; // 2026-09-26: baseline files and the agentic sandbox live here too
    }
    let text = std::fs::read_to_string(path).ok()?;
    if let Ok(mut record) = serde_json::from_str::<RunRecord>(&text) {
        if record.run_id.is_empty() {
            record.run_id = stem.to_string();
        }
        return Some(record);
    }
    let frame = serde_json::from_str::<BenchmarkResult>(&text).ok()?;
    Some(RunRecord::from_legacy(benchmark_id, stem, frame))
}

#[cfg(test)]
#[path = "history_tests.rs"]
mod tests;
