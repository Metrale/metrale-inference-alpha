// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Download a model from the Hugging Face Hub into the local cache on a worker thread, reporting progress on a channel.
//!
//! The work runs on a `std::thread` named `metrale-download`; the caller only
//! receives from `Handle::rx`, so a render loop never waits on it
//! (`.github/workflows/tui-threading.yml`).
//!
//! Owner: server (model download).
//! Invariants:
//! - `refs/main` is written last, by `hf::publish`, after every planned file is
//!   on disk. The resolver refuses a cached model without `refs/main`
//!   (`model_resolver.rs`), so an interrupted first download does not load.
//! - A job sends at most one of `Done`, `Failed` and `Cancelled`.

pub mod hf;
pub mod plan;
pub mod stale;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender, channel};

pub(crate) const HOST: &str = "https://huggingface.co";
pub(crate) const AGENT: &str = crate::identity::USER_AGENT;

/// 2026-09-26: Progress from a running download. `Done`, `Failed` and
/// `Cancelled` end the job.
#[derive(Clone, Debug)]
pub enum DownloadMsg {
    /// 2026-09-26: The file list is known: file count, total bytes and bytes
    /// already on disk.
    Planned {
        revision: String,
        files: usize,
        total_bytes: u64,
        already_have: u64,
    },
    /// 2026-09-26: Starting a file; `index` is 1-based.
    File {
        index: usize,
        of: usize,
        name: String,
    },
    /// 2026-09-26: Job-wide bytes, including files already on disk.
    Progress {
        done: u64,
        total: u64,
    },
    Done {
        snapshot: PathBuf,
        revision: String,
    },
    Failed(DownloadError),
    Cancelled {
        completed: usize,
        of: usize,
    },
}

/// 2026-09-26: Why a download stopped. Each variant gets its own advice in
/// [`DownloadError::hint`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadError {
    Offline(String),
    /// 2026-09-26: The Hub answered 401 or 403.
    Gated {
        repo: String,
        had_token: bool,
    },
    NotFound {
        repo: String,
    },
    RateLimited,
    DiskFull,
    /// 2026-09-26: The remaining bytes exceed the free space; refused before
    /// any file is fetched.
    NotEnoughSpace {
        need: u64,
        free: u64,
    },
    /// 2026-09-26: The selected files include no `.safetensors` file.
    NoSafetensors {
        repo: String,
    },
    Http {
        repo: String,
        status: u16,
    },
    Io(String),
}

impl DownloadError {
    /// 2026-09-26: One line for the user saying what to do about the error.
    pub fn hint(&self) -> String {
        match self {
            Self::Offline(e) => format!("no route to huggingface.co ({e}) — check the network"),
            Self::Gated {
                repo,
                had_token: false,
            } => format!(
                "{repo} requires credentials — set HF_TOKEN, or run `hf auth login` \
                 (`huggingface-cli login` before huggingface_hub 1.0)"
            ),
            Self::Gated {
                repo,
                had_token: true,
            } => format!(
                "this account has not accepted the licence for {repo} — visit https://huggingface.co/{repo}"
            ),
            Self::NotFound { repo } => format!("no such model on the Hub: {repo}"),
            Self::RateLimited => "the Hub is rate-limiting this IP — wait a few minutes".into(),
            Self::DiskFull => "the disk filled up mid-download — completed files were kept".into(),
            Self::NotEnoughSpace { need, free } => format!(
                "needs {:.1} GB, {:.1} GB free",
                *need as f64 / 1e9,
                *free as f64 / 1e9
            ),
            Self::NoSafetensors { repo } => {
                format!("{repo} publishes no safetensors — Metrale Engine cannot load it")
            }
            Self::Http { repo, status } => format!("the Hub answered {status} for {repo}"),
            Self::Io(e) => format!("could not write to the cache: {e}"),
        }
    }
}

/// 2026-09-26: A running download, from the caller's side.
pub struct Handle {
    pub repo: String,
    pub rx: Receiver<DownloadMsg>,
    cancel: Arc<AtomicBool>,
}

impl Handle {
    /// 2026-09-26: Ask the worker to stop. It takes effect within one 1 MiB
    /// chunk (`hf::fetch_file`), and the partial file stays for a resume.
    pub fn cancel(&self) {
        self.cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// 2026-09-26: Start downloading `repo` into `cache_root`. Returns at once;
/// progress arrives on the handle's channel, and a thread that cannot be
/// spawned is reported there as `Failed`.
pub fn start(repo: &str, cache_root: PathBuf) -> Handle {
    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let owned = repo.to_string();
    let spawned = std::thread::Builder::new()
        .name("metrale-download".into())
        .spawn({
            let tx = tx.clone();
            let cancel = Arc::clone(&cancel);
            let repo = owned.clone();
            move || {
                if let Err(e) = run(&repo, &cache_root, &tx, &cancel) {
                    let _ = tx.send(DownloadMsg::Failed(e));
                }
            }
        });
    if let Err(e) = spawned {
        let _ = tx.send(DownloadMsg::Failed(DownloadError::Io(format!(
            "could not start the download thread: {e}"
        ))));
    }
    Handle {
        repo: repo.to_string(),
        rx,
        cancel,
    }
}

fn run(
    repo: &str,
    cache_root: &std::path::Path,
    tx: &Sender<DownloadMsg>,
    cancel: &AtomicBool,
) -> Result<(), DownloadError> {
    let tok = hf::token();
    let (revision, listing) = hf::repo_info(repo, tok.as_deref())?;

    // 2026-09-26: Sizes come from a second endpoint; without them the job still
    // runs, with unknown sizes counted as 0.
    let sizes = hf::sizes(repo, &revision, tok.as_deref());
    let with_sizes: Vec<plan::RemoteFile> = listing
        .into_iter()
        .map(|mut f| {
            f.size = sizes.iter().find(|(p, _)| *p == f.name).map(|(_, s)| *s);
            f
        })
        .collect();

    let files = plan::select(&with_sizes);
    if !plan::has_weights(&files) {
        return Err(DownloadError::NoSafetensors { repo: repo.into() });
    }
    let total = plan::total_bytes(&files);

    let snapshot = hf::repo_dir(cache_root, repo)
        .join("snapshots")
        .join(&revision);
    // 2026-09-26: Files completed by an earlier run count at once.
    let already: u64 = files
        .iter()
        .filter_map(|f| {
            std::fs::metadata(snapshot.join(&f.name))
                .ok()
                .map(|m| m.len())
        })
        .sum();

    // 2026-09-26: Refuse before the first byte when the rest cannot fit. The
    // check is skipped when free space cannot be measured.
    if let Some(free) = hf::free_bytes(cache_root) {
        fits(total, already, free)?;
    }

    let _ = tx.send(DownloadMsg::Planned {
        revision: revision.clone(),
        files: files.len(),
        total_bytes: total,
        already_have: already,
    });

    let mut done_bytes = 0u64;
    for (i, f) in files.iter().enumerate() {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = tx.send(DownloadMsg::Cancelled {
                completed: i,
                of: files.len(),
            });
            return Ok(());
        }
        let dest = snapshot.join(&f.name);
        if let Ok(m) = std::fs::metadata(&dest) {
            // 2026-09-26: A file at `dest` is complete: `fetch_file` renames
            // onto it only after the last byte.
            done_bytes += m.len();
            let _ = tx.send(DownloadMsg::Progress {
                done: done_bytes,
                total,
            });
            continue;
        }
        let _ = tx.send(DownloadMsg::File {
            index: i + 1,
            of: files.len(),
            name: f.name.clone(),
        });
        let base = done_bytes;
        let mut last_sent = 0u64;
        let completed = hf::fetch_file(
            repo,
            &revision,
            &f.name,
            &dest,
            tok.as_deref(),
            cancel,
            &mut |n| {
                // 2026-09-26: At most one progress message per 1 MiB moved.
                if n.saturating_sub(last_sent) >= 1024 * 1024 {
                    last_sent = n;
                    let _ = tx.send(DownloadMsg::Progress {
                        done: base + n,
                        total,
                    });
                }
            },
        )?;
        if !completed {
            let _ = tx.send(DownloadMsg::Cancelled {
                completed: i,
                of: files.len(),
            });
            return Ok(());
        }
        done_bytes = base + std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
        let _ = tx.send(DownloadMsg::Progress {
            done: done_bytes,
            total,
        });
    }

    // 2026-09-26: Published only after every file is on disk.
    hf::publish(cache_root, repo, &revision).map_err(|e| DownloadError::Io(format!("{e:#}")))?;
    let _ = tx.send(DownloadMsg::Done { snapshot, revision });
    Ok(())
}

/// 2026-09-26: Is there room for `total - already` bytes? Pure, so it is
/// tested without a full filesystem.
fn fits(total: u64, already: u64, free: u64) -> Result<(), DownloadError> {
    let need = total.saturating_sub(already);
    if need > free {
        return Err(DownloadError::NotEnoughSpace { need, free });
    }
    Ok(())
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "net_tests.rs"]
mod net_tests;
