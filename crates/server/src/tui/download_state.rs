// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The dashboard's view of a model download and of one Hub
//! freshness check. The download runs on `model_download`'s thread and the
//! check on a `worker::spawn` thread; `pump` only `try_recv`s.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use std::collections::HashMap;
use std::time::Instant;

use crate::model_download::stale::Freshness;
use crate::model_download::{DownloadError, DownloadMsg, Handle};

/// 2026-09-26: What a running job looks like to the renderer.
pub struct Job {
    pub repo: String,
    handle: Handle,
    /// 2026-09-26: Set by the first `cancel`. The worker checks its cancel flag
    /// once per 1 MiB chunk, so it stops after the current chunk.
    pub cancelling: bool,
    pub file: Option<(usize, usize, String)>,
    pub done: u64,
    pub total: u64,
    pub started: Instant,
    /// 2026-09-26: (when, bytes) of the last rate sample.
    last: (Instant, u64),
    pub rate_bps: f64,
    /// 2026-09-26: Bytes already on disk when `DownloadMsg::Planned` arrived.
    /// `done_message` compares it with `total` to report an already-complete
    /// model instead of a transfer.
    pub already_have: u64,
}

impl Job {
    /// 2026-09-26: `done / total`, clamped to 1.0; `None` while `total` is 0.
    pub fn fraction(&self) -> Option<f64> {
        (self.total > 0).then(|| (self.done as f64 / self.total as f64).clamp(0.0, 1.0))
    }
}

#[derive(Default)]
pub struct DownloadState {
    pub job: Option<Job>,
    /// 2026-09-26: Freshness by model id, held in memory only.
    pub freshness: HashMap<String, Freshness>,
    /// 2026-09-26: The model whose freshness is being checked; the Library row
    /// draws a placeholder badge for it.
    pub checking: Option<String>,
    pending_check: Option<std::sync::mpsc::Receiver<(String, Freshness)>>,
    /// 2026-09-26: The next toast, `(text, is_error)`, set when a download
    /// settles or a freshness check answers.
    pub last_message: Option<(String, bool)>,
}

/// 2026-09-26: How a download job ended, as `pump` saw it. The event loop
/// marks the Library for a rescan after either.
pub enum Settled {
    /// 2026-09-26: `DownloadMsg::Done`.
    Finished(String),
    /// 2026-09-26: `Cancelled`, `Failed`, or a worker that disconnected
    /// without a terminal message.
    Stopped(String),
}

/// 2026-09-26: The toast for a finished job: "already complete" when
/// `0 < total <= already_have`, else the bytes moved and the elapsed time.
/// `settle_done_for_test` calls it too.
fn done_message(job: &Job) -> (String, bool) {
    if job.total > 0 && job.already_have >= job.total {
        return (
            format!(
                "{} is already complete — {} on disk, nothing to download (u checks for updates)",
                job.repo,
                crate::tui::format::bytes(job.total)
            ),
            false,
        );
    }
    let moved = job.total.saturating_sub(job.already_have);
    (
        format!(
            "{} downloaded — {} in {}",
            job.repo,
            crate::tui::format::bytes(moved),
            fmt_elapsed(job.started.elapsed().as_secs())
        ),
        false,
    )
}

/// 2026-09-26: `58s`, `12m 40s`, `1h 03m`: a transfer duration for a receipt.
fn fmt_elapsed(secs: u64) -> String {
    match secs {
        s if s < 60 => format!("{s}s"),
        s if s < 3_600 => format!("{}m {:02}s", s / 60, s % 60),
        s => format!("{}h {:02}m", s / 3_600, s / 60 % 60),
    }
}

impl DownloadState {
    /// 2026-09-26: `pump`'s handling of `DownloadMsg::Done` (set the message
    /// from `done_message`, clear the job), without a worker thread.
    #[cfg(test)]
    pub(crate) fn settle_done_for_test(&mut self) {
        if let Some(job) = self.job.as_ref() {
            self.last_message = Some(done_message(job));
        }
        self.job = None;
    }

    /// 2026-09-26: Is a download running for this model?
    pub fn is_downloading(&self, repo: &str) -> bool {
        self.job.as_ref().is_some_and(|j| j.repo == repo)
    }

    /// 2026-09-26: Begin a download and return the toast. While a job runs the
    /// request is refused with a message naming it; the `App` keeps a pending
    /// request of its own (`start_pending_download`).
    pub fn start(&mut self, repo: &str, cache_root: std::path::PathBuf) -> (String, bool) {
        if let Some(j) = &self.job {
            return (
                format!("already downloading {} — press x to stop it", j.repo),
                true,
            );
        }
        let handle = crate::model_download::start(repo, cache_root);
        self.job = Some(Job {
            repo: repo.to_string(),
            handle,
            cancelling: false,
            file: None,
            done: 0,
            total: 0,
            started: Instant::now(),
            last: (Instant::now(), 0),
            rate_bps: 0.0,
            already_have: 0,
        });
        (format!("resolving {repo}…"), false)
    }

    /// 2026-09-26: Ask the running job to stop; a second call stops tracking
    /// it. `None` when no job is running.
    pub fn cancel(&mut self) -> Option<(String, bool)> {
        let job = self.job.as_mut()?;
        if job.cancelling {
            // 2026-09-26: Second call: drop the job and its handle. The worker
            // already has the cancel flag and stops after its current chunk;
            // what it wrote stays on disk for a later resume.
            let repo = job.repo.clone();
            self.job = None;
            return Some((format!("abandoned the {repo} download"), false));
        }
        job.cancelling = true;
        job.handle.cancel();
        Some(("stopping — finishing the current chunk".into(), false))
    }

    /// 2026-09-26: Drain the download and freshness channels. The event loop
    /// calls it on every iteration.
    pub fn pump(&mut self) -> Option<Settled> {
        let mut settled = None;
        if let Some(job) = self.job.as_mut() {
            // 2026-09-26: One `try_recv` at a time, so `Disconnected` is seen
            // only after every message the worker sent.
            let mut msgs = Vec::new();
            let mut disconnected = false;
            loop {
                match job.handle.rx.try_recv() {
                    Ok(m) => msgs.push(m),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
            for msg in msgs {
                match msg {
                    DownloadMsg::Planned {
                        total_bytes,
                        already_have,
                        ..
                    } => {
                        job.total = total_bytes;
                        job.done = already_have;
                        job.already_have = already_have;
                    }
                    DownloadMsg::File { index, of, name } => job.file = Some((index, of, name)),
                    DownloadMsg::Progress { done, total } => {
                        job.done = done;
                        job.total = total;
                        // 2026-09-26: The rate is recomputed over the bytes
                        // since the last sample, at most every 0.5 s.
                        let dt = job.last.0.elapsed().as_secs_f64();
                        if dt >= 0.5 {
                            job.rate_bps = (done.saturating_sub(job.last.1)) as f64 / dt;
                            job.last = (Instant::now(), done);
                        }
                    }
                    DownloadMsg::Done { .. } => {
                        settled = Some(Settled::Finished(job.repo.clone()));
                        self.last_message = Some(done_message(job));
                    }
                    DownloadMsg::Cancelled { completed, of } => {
                        settled = Some(Settled::Stopped(job.repo.clone()));
                        self.last_message = Some((
                            format!(
                                "{} stopped after {completed}/{of} files — press d to resume",
                                job.repo
                            ),
                            false,
                        ));
                    }
                    DownloadMsg::Failed(e) => {
                        settled = Some(Settled::Stopped(job.repo.clone()));
                        self.last_message = Some((describe(&job.repo, &e), true));
                    }
                }
            }
            // 2026-09-26: A worker gone without a terminal message still
            // settles the job; `start`'s thread sends `Failed` for an `Err`, so
            // this is a panic.
            if settled.is_none() && disconnected {
                settled = Some(Settled::Stopped(job.repo.clone()));
                self.last_message = Some((
                    format!(
                        "{} download stopped unexpectedly — press d to resume",
                        job.repo
                    ),
                    true,
                ));
            }
        }
        if settled.is_some() {
            self.job = None;
        }
        if let Some(rx) = &self.pending_check {
            match rx.try_recv() {
                Ok((id, f)) => {
                    self.pending_check = None;
                    self.checking = None;
                    // 2026-09-26: Every answer gets a toast; the row badge
                    // shows only `Stale`.
                    self.last_message = Some(checked_message(&id, &f));
                    self.freshness.insert(id, f);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.pending_check = None;
                    self.checking = None;
                    // 2026-09-26: A worker gone without answering settles with
                    // an error toast.
                    self.last_message = Some((
                        "the freshness check ended unexpectedly — u retries".into(),
                        true,
                    ));
                }
            }
        }
        settled
    }

    /// 2026-09-26: Start a freshness check for one model, off the render
    /// thread. Refused while another check is running.
    pub fn check(&mut self, repo: &str, cache_root: std::path::PathBuf) -> (String, bool) {
        if self.pending_check.is_some() {
            return ("already checking a model".into(), true);
        }
        let owned = repo.to_string();
        let fallback = repo.to_string();
        let rx = crate::tui::worker::spawn(
            "metrale-freshness",
            move || {
                // 2026-09-26: Any `stale::check` error is `Unknown`.
                let f = crate::model_download::stale::check(&owned, &cache_root)
                    .unwrap_or(Freshness::Unknown);
                (owned, f)
            },
            |_| (fallback, Freshness::Unknown),
        );
        self.pending_check = Some(rx);
        self.checking = Some(repo.to_string());
        (format!("checking {repo}…"), false)
    }
}

/// 2026-09-26: A failed download's toast: the repo and `DownloadError::hint`.
fn describe(repo: &str, e: &DownloadError) -> String {
    format!("{repo}: {}", e.hint())
}

/// 2026-09-26: The toast for a freshness answer. Only `Unknown` uses the error
/// tone; `Stale` and `Missing` name the `d` key.
fn checked_message(id: &str, f: &Freshness) -> (String, bool) {
    match f {
        Freshness::Current => (format!("{id} is up to date with the Hub"), false),
        Freshness::Stale { local, remote } => (
            format!(
                "{id} has an update — local {} vs Hub {}; d downloads it",
                crate::model_download::stale::short(local),
                crate::model_download::stale::short(remote)
            ),
            false,
        ),
        Freshness::Missing => (
            format!("{id} has nothing on disk to compare — d downloads it"),
            false,
        ),
        Freshness::Unknown => (
            format!("could not reach the Hub — {id} freshness unknown"),
            true,
        ),
    }
}

#[cfg(test)]
#[path = "download_state_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "download_state_more_tests.rs"]
mod more_tests;
