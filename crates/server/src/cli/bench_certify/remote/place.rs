// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A fetched record becomes a repository record only after it is checked.
//!
//! Owner: server CLI (`met benchmark certify`).
//! `place` checks that the record is for this unit and shard, at the anchor, on
//! the class being certified, not failed, from a clean tree, and signed under
//! a committed key. The remote runner reports any refusal as a non-retryable
//! harness failure.
//! Invariants: a record is never overwritten (`copy_new` uses `create_new`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use metrale_bench::gate::{self, signing};

use super::metralectl::FetchedFile;

/// 2026-09-26: The record and its sidecar, where the repository keeps them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    pub record: PathBuf,
    pub signature: PathBuf,
    /// 2026-09-26: The child's log, copied under the campaign's log dir, if it was
    /// fetched and the copy worked.
    pub log: Option<PathBuf>,
}

/// 2026-09-26: What a placed record must satisfy.
pub struct Expect<'a> {
    pub unit_id: &'a str,
    /// 2026-09-26: The slice the record must say it measured; `None` for a whole draw.
    pub shard: Option<(usize, usize)>,
    /// 2026-09-26: The stem of the log's name under the campaign dir (`Unit::file_stem`).
    pub log_stem: &'a str,
    pub anchor: &'a str,
    pub hardware: &'a str,
}

/// 2026-09-26: Which fetched files are the record, its signature and the log.
///
/// # Errors
/// When the set has no record, no signature, two records or two signatures, a
/// signature that is not the record's, or a file whose path does not belong to
/// this unit. With no record, the error quotes the final `Error:` block of the
/// fetched log, when there is one.
pub fn sort_files(
    files: &[FetchedFile],
    unit_id: &str,
) -> Result<(FetchedFile, FetchedFile, Option<FetchedFile>)> {
    let record_dir = format!(".benchmarks/{unit_id}/");
    let mut record = None;
    let mut sig = None;
    let mut log = None;
    for f in files {
        let rel = f.relative_path.as_str();
        if rel.starts_with(&record_dir) && rel.ends_with(".json") {
            if record.replace(f.clone()).is_some() {
                bail!("the node returned two records for {unit_id}");
            }
        } else if rel.starts_with(&record_dir) && rel.ends_with(".json.sig") {
            if sig.replace(f.clone()).is_some() {
                bail!("the node returned two signatures for {unit_id}");
            }
        } else if rel.starts_with(".certify/") && rel.ends_with(".log") {
            log = Some(f.clone());
        } else {
            bail!(
                "the node returned {rel:?}, which is not a record, signature or log of {unit_id}"
            );
        }
    }
    let Some(record) = record else {
        let cause = log
            .as_ref()
            .and_then(|l| {
                crate::cli::bench_cause::tail_of_file(&l.path, crate::cli::bench_cause::TAIL_BYTES)
                    .ok()
            })
            .and_then(|tail| crate::cli::bench_cause::final_error_block(&tail));
        match cause {
            Some(block) => bail!(
                "the node returned no record for {unit_id} — the child's log ends with:\n{block}"
            ),
            None => bail!("the node returned no record for {unit_id}"),
        }
    };
    let Some(sig) = sig else {
        bail!("the node returned no signature for {unit_id}");
    };
    if sig.relative_path != format!("{}.sig", record.relative_path) {
        bail!(
            "signature {} does not belong to record {}",
            sig.relative_path,
            record.relative_path
        );
    }
    Ok((record, sig, log))
}

fn copy_new(from: &Path, to: &Path) -> Result<()> {
    if let Some(p) = to.parent() {
        std::fs::create_dir_all(p).with_context(|| format!("creating {}", p.display()))?;
    }
    let bytes = std::fs::read(from).with_context(|| format!("reading {}", from.display()))?;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(to)
        .with_context(|| {
            format!(
                "{} already exists — a record is never overwritten",
                to.display()
            )
        })?;
    std::io::Write::write_all(&mut f, &bytes).with_context(|| format!("writing {}", to.display()))
}

/// 2026-09-26: Check the fetched record and copy it, with its signature, into the
/// repository. A refused check copies nothing. A failed signature copy removes
/// the copied record, and a failed signature check removes both.
///
/// # Errors
/// Each names what was wrong.
pub fn place(
    root: &Path,
    log_dir: &Path,
    files: &[FetchedFile],
    expect: &Expect,
) -> Result<Placed> {
    let (rec, sig, log) = sort_files(files, expect.unit_id)?;
    let parsed = gate::read_record(&rec.path)
        .with_context(|| format!("parsing the fetched record {}", rec.path.display()))?;
    if parsed.benchmark_id != expect.unit_id {
        bail!(
            "the fetched record is for {}, not {}",
            parsed.benchmark_id,
            expect.unit_id
        );
    }
    if parsed.shard() != expect.shard {
        bail!(
            "the fetched record measured shard {}, this unit is {}",
            spell_shard(parsed.shard()),
            spell_shard(expect.shard)
        );
    }
    if !(parsed.git_sha.starts_with(expect.anchor) || expect.anchor.starts_with(&parsed.git_sha)) {
        bail!(
            "the fetched record names commit {}, not the anchor {}",
            parsed.git_sha,
            expect.anchor
        );
    }
    let class = parsed.hardware.gate_key();
    if class != expect.hardware {
        bail!(
            "the fetched record was measured on class {class}, this campaign certifies {}",
            expect.hardware
        );
    }
    if parsed.frame_status_failed() {
        bail!("the fetched record says its own run failed (frame status Failed)");
    }
    if !parsed.dirty_paths.is_empty() {
        bail!(
            "the fetched record was measured on a dirty tree ({})",
            parsed.dirty_paths.join(", ")
        );
    }
    let record_to = root.join(&rec.relative_path);
    let sig_to = root.join(&sig.relative_path);
    copy_new(&rec.path, &record_to)?;
    if let Err(e) = copy_new(&sig.path, &sig_to) {
        let _ = std::fs::remove_file(&record_to);
        return Err(e);
    }
    if let Err(e) = signing::verify_record(root, &record_to, &parsed.git_sha, parsed.recorded_at) {
        let _ = std::fs::remove_file(&record_to);
        let _ = std::fs::remove_file(&sig_to);
        return Err(
            e.context("the fetched record's signature does not verify under a committed key")
        );
    }
    let log_to = match &log {
        Some(l) => {
            let name = Path::new(&l.relative_path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| format!("{}.log", expect.unit_id));
            let to = log_dir.join(format!("{}.remote.{name}", expect.log_stem));
            match std::fs::copy(&l.path, &to) {
                Ok(_) => Some(to),
                Err(_) => None,
            }
        }
        None => None,
    };
    Ok(Placed {
        record: record_to,
        signature: sig_to,
        log: log_to,
    })
}

fn spell_shard(s: Option<(usize, usize)>) -> String {
    s.map_or("the whole draw".to_string(), |(i, n)| format!("{i}/{n}"))
}

#[cfg(test)]
#[path = "place_tests.rs"]
mod place_tests;
