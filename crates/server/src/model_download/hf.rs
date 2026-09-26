// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Every request this server makes to the Hub (`HOST`): repo listing, file sizes, resumable file download, plus the cache's `refs/main` and disk-space helpers.
//!
//! Built on `ureq` so a download reports progress per chunk, resumes from a
//! `.part` file, and can be cancelled inside a file.
//!
//! Owner: server (model download).
//! Invariants:
//! - A file reaches its final name only by the rename after its last byte was
//!   written; until then it is `<name>.part`.
//! - `refs/main` is written only after `config.json` or `params.json` and at
//!   least one `.safetensors` file exist in the snapshot.

use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use super::plan::RemoteFile;
use super::{DownloadError, HOST};

/// 2026-09-26: Read/write chunk. The cancel flag is checked once per chunk, so
/// a cancel takes effect within 1 MiB.
const CHUNK: usize = 1024 * 1024;

/// 2026-09-26: One agent for the process, so connections are reused across
/// files.
fn agent() -> &'static ureq::Agent {
    static POOL: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        ureq::Agent::config_builder()
            // 2026-09-26: Only a connect timeout: a shard can legitimately take
            // minutes, and a stall shows in the progress display.
            .timeout_connect(Some(std::time::Duration::from_secs(30)))
            .build()
            .into()
    })
}

/// 2026-09-26: The Hub token, if any: `HF_TOKEN`, then
/// `HUGGING_FACE_HUB_TOKEN`, then `$HOME/.cache/huggingface/token`, with blank
/// values skipped (`pick_token`). `run` calls it once per download job and
/// passes the result down.
pub fn token() -> Option<String> {
    let env = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"].map(|v| std::env::var(v).ok());
    let file = std::env::var_os("HOME")
        .and_then(|h| std::fs::read_to_string(Path::new(&h).join(".cache/huggingface/token")).ok());
    pick_token(&env, file.as_deref())
}

/// 2026-09-26: The precedence rule of [`token`] without the lookups, so tests
/// need not set process environment variables. A blank or whitespace-only value
/// counts as absent, so an exported empty `HF_TOKEN` sends no `Authorization`
/// header; values are trimmed.
pub(super) fn pick_token(env: &[Option<String>], file: Option<&str>) -> Option<String> {
    for v in env.iter().flatten() {
        if !v.trim().is_empty() {
            return Some(v.trim().to_string());
        }
    }
    file.map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

pub(super) fn classify(repo: &str, status: u16, had_token: bool) -> DownloadError {
    match status {
        401 | 403 => DownloadError::Gated {
            repo: repo.into(),
            had_token,
        },
        404 => DownloadError::NotFound { repo: repo.into() },
        429 => DownloadError::RateLimited,
        s => DownloadError::Http {
            repo: repo.into(),
            status: s,
        },
    }
}

fn get(
    url: &str,
    token: Option<&str>,
    range_from: Option<u64>,
) -> Result<ureq::http::Response<ureq::Body>, (u16, anyhow::Error)> {
    let mut req = agent().get(url).header("User-Agent", super::AGENT);
    if let Some(t) = token {
        req = req.header("Authorization", &format!("Bearer {t}"));
    }
    if let Some(from) = range_from {
        req = req.header("Range", &format!("bytes={from}-"));
    }
    match req.call() {
        Ok(r) => Ok(r),
        Err(ureq::Error::StatusCode(code)) => Err((code, anyhow::anyhow!("HTTP {code}"))),
        Err(e) => Err((0, anyhow::Error::new(e))),
    }
}

/// 2026-09-26: [`get`], retried once without the token when a request with one
/// is answered 401 or 403, so an invalid token does not block a public repo. If
/// the anonymous retry also fails, the first status and error are returned, so
/// `classify` still reports `had_token: true`.
fn get_or_anon(
    url: &str,
    token: Option<&str>,
    range_from: Option<u64>,
) -> Result<ureq::http::Response<ureq::Body>, (u16, anyhow::Error)> {
    match get(url, token, range_from) {
        Err((s @ (401 | 403), e)) if token.is_some() => match get(url, None, range_from) {
            Ok(r) => Ok(r),
            Err(_) => Err((s, e)),
        },
        other => other,
    }
}

/// 2026-09-26: The revision sha that `/api/models/{repo}` reports, and every
/// file it lists (`siblings`), with sizes unknown.
pub fn repo_info(
    repo: &str,
    tok: Option<&str>,
) -> Result<(String, Vec<RemoteFile>), DownloadError> {
    let had = tok.is_some();
    let url = format!("{HOST}/api/models/{repo}");
    let body = match get_or_anon(&url, tok, None) {
        Ok(r) => r
            .into_body()
            .read_to_string()
            .map_err(|e| DownloadError::Io(e.to_string()))?,
        Err((0, e)) => return Err(DownloadError::Offline(e.to_string())),
        Err((s, _)) => return Err(classify(repo, s, had)),
    };
    let doc: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| DownloadError::Io(e.to_string()))?;
    let sha = doc
        .get("sha")
        .and_then(|s| s.as_str())
        .ok_or_else(|| DownloadError::Io("model info has no sha".into()))?
        .to_string();
    let files = doc
        .get("siblings")
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| e.get("rfilename").and_then(|n| n.as_str()))
                .map(|name| RemoteFile {
                    name: name.to_string(),
                    size: None,
                })
                .collect()
        })
        .unwrap_or_default();
    Ok((sha, files))
}

/// 2026-09-26: File sizes from the tree endpoint for `revision`. Any failure
/// returns an empty list; `run` then proceeds with the sizes unknown.
pub fn sizes(repo: &str, revision: &str, tok: Option<&str>) -> Vec<(String, u64)> {
    let url = format!("{HOST}/api/models/{repo}/tree/{revision}?recursive=1");
    let Ok(r) = get_or_anon(&url, tok, None) else {
        return Vec::new();
    };
    let Ok(body) = r.into_body().read_to_string() else {
        return Vec::new();
    };
    let Ok(doc) = serde_json::from_str::<serde_json::Value>(&body) else {
        return Vec::new();
    };
    doc.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let path = e.get("path")?.as_str()?.to_string();
                    // 2026-09-26: `lfs.size` first: for an LFS file, the outer
                    // `size` is that of its pointer.
                    let size = e
                        .get("lfs")
                        .and_then(|l| l.get("size"))
                        .and_then(|s| s.as_u64())
                        .or_else(|| e.get("size").and_then(|s| s.as_u64()))?;
                    Some((path, size))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// 2026-09-26: Download one file into `dest`, resuming from `<dest>.part`.
///
/// `on_bytes` gets the running byte total for this file. `cancel` is checked
/// every [`CHUNK`]. Returns `Ok(false)` when cancelled, leaving the `.part` in
/// place, and `Ok(true)` once the file was renamed to `dest`.
pub fn fetch_file(
    repo: &str,
    revision: &str,
    name: &str,
    dest: &Path,
    tok: Option<&str>,
    cancel: &AtomicBool,
    on_bytes: &mut dyn FnMut(u64),
) -> Result<bool, DownloadError> {
    let had = tok.is_some();
    // 2026-09-26: The `.part` sibling is the only resume record.
    let part = part_path(dest);
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

    let url = format!("{HOST}/{repo}/resolve/{revision}/{name}");
    let resp = match get_or_anon(&url, tok, (have > 0).then_some(have)) {
        Ok(r) => r,
        Err((0, e)) => return Err(DownloadError::Offline(e.to_string())),
        Err((s, _)) => return Err(classify(repo, s, had)),
    };
    // 2026-09-26: 206 means the range was honoured; 200 means the whole file is
    // coming, so the `.part` is truncated rather than appended to.
    let resuming = resp.status().as_u16() == 206;
    let mut written = if resuming { have } else { 0 };
    on_bytes(written);

    if let Some(parent) = part.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DownloadError::Io(e.to_string()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resuming)
        .truncate(!resuming)
        .open(&part)
        .map_err(|e| DownloadError::Io(e.to_string()))?;

    let mut reader = resp.into_body().into_reader();
    let mut buf = vec![0u8; CHUNK];
    loop {
        if cancel.load(Ordering::Relaxed) {
            // 2026-09-26: The `.part` stays for the next resume.
            file.flush().ok();
            return Ok(false);
        }
        let n = reader
            .read(&mut buf)
            .map_err(|e| DownloadError::Io(e.to_string()))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(write_error)?;
        written += n as u64;
        on_bytes(written);
    }
    file.flush().map_err(|e| DownloadError::Io(e.to_string()))?;
    drop(file);
    // 2026-09-26: Renamed last, so a kill at any point leaves a `.part` or a
    // complete file, never a partial file under the final name.
    std::fs::rename(&part, dest).map_err(|e| DownloadError::Io(e.to_string()))?;
    Ok(true)
}

/// 2026-09-26: The resume record for `dest`: `.part` is appended to the file
/// name. `with_extension` would replace the extension, so `foo.safetensors` and
/// `foo.json` would share `foo.part`, and a resume would append one file's
/// body to the other's.
pub fn part_path(dest: &Path) -> PathBuf {
    dest.with_file_name(format!(
        "{}.part",
        dest.file_name().unwrap_or_default().to_string_lossy()
    ))
}

/// 2026-09-26: A write failure: `StorageFull` becomes `DownloadError::DiskFull`,
/// anything else `DownloadError::Io`. Separate from the write so it is testable
/// without filling a filesystem.
pub(super) fn write_error(e: std::io::Error) -> DownloadError {
    if e.kind() == std::io::ErrorKind::StorageFull {
        DownloadError::DiskFull
    } else {
        DownloadError::Io(e.to_string())
    }
}

/// 2026-09-26: Free bytes on the filesystem that holds `path`, measured at
/// `path` or its nearest existing ancestor, because the cache directory may
/// not exist yet on a first download. `None` when no ancestor can be measured,
/// and always on non-Unix targets; `run` then skips the space check.
pub fn free_bytes(path: &Path) -> Option<u64> {
    let mut cur = Some(path);
    while let Some(p) = cur {
        if let Some(n) = statvfs_avail(p) {
            return Some(n);
        }
        cur = p.parent();
    }
    None
}

/// 2026-09-26: Non-Unix targets have no `statvfs`; Windows builds of
/// `metrale-server` (`.github/workflows/release-build.yml`) get `None` here.
#[cfg(not(unix))]
fn statvfs_avail(_path: &Path) -> Option<u64> {
    None
}

/// 2026-09-26: Free and total bytes of the filesystem holding `path` (or its
/// nearest existing ancestor), for `disk_guard`. `None` when unmeasurable or
/// when the filesystem reports zero blocks.
pub fn disk_usage(path: &Path) -> Option<(u64, u64)> {
    let mut cur = Some(path);
    while let Some(p) = cur {
        if let Some(u) = statvfs_usage(p) {
            return Some(u);
        }
        cur = p.parent();
    }
    None
}

#[cfg(not(unix))]
fn statvfs_usage(_path: &Path) -> Option<(u64, u64)> {
    None
}

#[cfg(unix)]
fn statvfs_usage(path: &Path) -> Option<(u64, u64)> {
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    // 2026-09-26: SAFETY: `c` is a valid NUL-terminated path; `stat` is read
    // only after a successful call.
    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c.as_ptr(), &mut stat) != 0 {
            return None;
        }
        // 2026-09-26: `f_bavail` excludes the blocks reserved for root, which
        // an unprivileged download cannot use; `f_bfree` includes them.
        #[allow(clippy::unnecessary_cast)]
        let free = stat.f_bavail as u64 * stat.f_frsize as u64;
        #[allow(clippy::unnecessary_cast)]
        let total = stat.f_blocks as u64 * stat.f_frsize as u64;
        (total > 0).then_some((free, total))
    }
}

#[cfg(unix)]
fn statvfs_avail(path: &Path) -> Option<u64> {
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    // 2026-09-26: SAFETY: `c` is a valid NUL-terminated path; `stat` is read
    // only after a successful call.
    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c.as_ptr(), &mut stat) != 0 {
            return None;
        }
        // 2026-09-26: The casts are needed off Linux: on macOS `f_bavail` is
        // u32 and `f_frsize` is u64 (libc's apple `statvfs`), while on Linux
        // both are u64 and clippy calls them unnecessary.
        #[allow(clippy::unnecessary_cast)]
        Some(stat.f_bavail as u64 * stat.f_frsize as u64)
    }
}

/// 2026-09-26: `models--<org>--<name>` under the cache root.
pub fn repo_dir(cache_root: &Path, repo: &str) -> PathBuf {
    cache_root.join(format!("models--{}", repo.replace('/', "--")))
}

/// 2026-09-26: The trimmed contents of the repo's `refs/main`, or `None` when
/// absent or empty.
pub fn local_revision(cache_root: &Path, repo: &str) -> Option<String> {
    let p = repo_dir(cache_root, repo).join("refs/main");
    let s = std::fs::read_to_string(p).ok()?;
    let s = s.trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// 2026-09-26: Publish a snapshot by writing `refs/main` (via a temp file and a
/// rename). `run` calls it after every planned file is on disk. Refuses a
/// snapshot without `config.json`/`params.json` or without a `.safetensors`
/// file.
pub fn publish(cache_root: &Path, repo: &str, revision: &str) -> Result<()> {
    let dir = repo_dir(cache_root, repo);
    let snapshot = dir.join("snapshots").join(revision);
    if !snapshot.join("config.json").exists() && !snapshot.join("params.json").exists() {
        bail!("refusing to publish {repo}@{revision}: no config.json or params.json");
    }
    // 2026-09-26: A real `.safetensors` file, not `snapshot_has_weights`: that
    // also accepts `model.safetensors.index.json`, which is small and is
    // fetched before the shards (`plan::select`).
    let has_shard = std::fs::read_dir(&snapshot)
        .map(|rd| {
            rd.filter_map(|e| e.ok()).any(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| n.ends_with(".safetensors"))
            })
        })
        .unwrap_or(false);
    if !has_shard {
        bail!("refusing to publish {repo}@{revision}: no .safetensors shard on disk");
    }
    let refs = dir.join("refs");
    std::fs::create_dir_all(&refs).context("creating refs/")?;
    let tmp = refs.join("main.tmp");
    std::fs::write(&tmp, revision)?;
    std::fs::rename(&tmp, refs.join("main"))?;
    Ok(())
}
