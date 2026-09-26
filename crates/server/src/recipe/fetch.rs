// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The recipe index: read from the on-disk cache, refreshed from GitHub on a worker thread.
//!
//! Every read returns what is on disk, marked with its age and with why it is
//! not fresh; a network failure never empties the Library. The TUI's render
//! thread only receives from the channels returned here
//! (`.github/workflows/tui-threading.yml`); the requests themselves are
//! blocking `ureq` calls on plain threads (`fetch_github.rs`).
//!
//! A refresh makes one GitHub API call, which lists the repo's tree at `main`,
//! then fetches each `recipes/**/*.yaml` from `raw.githubusercontent.com` at
//! that tree's sha, up to eight at a time.
//!
//! Owner: server (recipe).
//! Invariants:
//! - The cached `index.json` is replaced only by a refresh that fetched every
//!   listed recipe file, and only through a temp file and a rename.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::Recipe;
use super::fetch_github::{self, try_refresh};

pub(super) const REPO: &str = "Metrale/metralectl";
pub(super) const CACHE: &str = "metrale-recipes";
pub(super) const INDEX: &str = "index.json";
pub(super) const AGENT: &str = crate::identity::USER_AGENT;
pub(super) const TIMEOUT: Duration = Duration::from_secs(20);

/// 2026-09-26: What the Library renders: the recipes, and how fresh they are.
#[derive(Clone, Debug, Default)]
pub struct Index {
    pub recipes: Vec<Recipe>,
    pub tree_sha: String,
    /// 2026-09-26: Unix seconds of the network fetch; 0 means never.
    pub fetched_at: u64,
    /// 2026-09-26: Why this came off disk instead of the network: the fetch
    /// failed, or the cache file could not be read or parsed.
    pub offline: Option<String>,
    /// 2026-09-26: Set when the fetch returned but the cache was not replaced:
    /// some listed files did not come back, or the write failed. Unlike
    /// [`Self::offline`], this can happen on a working network.
    pub incomplete: Option<String>,
}

impl Index {
    /// 2026-09-26: How old the data is, for the panel title: "never fetched"
    /// when `fetched_at` is 0, else minutes, hours or days. Always `Some`.
    pub fn age_text(&self) -> Option<String> {
        if self.fetched_at == 0 {
            return Some("never fetched".into());
        }
        let now = unix_now();
        let secs = now.saturating_sub(self.fetched_at);
        Some(match secs {
            0..=3599 => format!("{} m old", secs / 60),
            3600..=86399 => format!("{} h old", secs / 3600),
            _ => format!("{} d old", secs / 86400),
        })
    }

    /// 2026-09-26: The one line the Library puts in its title.
    pub fn status_text(&self) -> String {
        match (&self.offline, self.age_text()) {
            (Some(_), Some(age)) => format!("⚠ {age} — offline"),
            (None, Some(age)) => age,
            (Some(e), None) => format!("⚠ offline — {e}"),
            (None, None) => "up to date".into(),
        }
    }

    /// 2026-09-26: The `offline` reason plus a hint picked from its text: no
    /// route (DNS, unreachable), rate limiting (403), timeout, or none of these.
    pub fn offline_detail(&self) -> Option<String> {
        let raw = self.offline.as_ref()?;
        let lowered = raw.to_lowercase();
        let hint = if lowered.contains("dns")
            || lowered.contains("resolve")
            || lowered.contains("unreachable")
            || lowered.contains("no route")
        {
            "This machine has no route to github.com. Set HTTPS_PROXY to a host \
             that does — recipes are then fetched through it — or copy the \
             cached index (~/.metrale/metrale-recipes/index.json) from a machine \
             that can reach it."
        } else if lowered.contains("403") || lowered.contains("rate") {
            "GitHub is rate-limiting this IP. The listing costs one API call per \
             refresh; the cached recipes below are still usable."
        } else if lowered.contains("timed out") || lowered.contains("timeout") {
            "The request timed out. A slow or filtered link will do this; the \
             cached recipes below are still usable."
        } else {
            "The cached recipes below are still usable."
        };
        Some(format!("{raw}. {hint}"))
    }
}

pub(super) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 2026-09-26: The recipe cache directory, `<root>/metrale-recipes`.
/// `pub(crate)` so the `cli` messages name the path this code reads.
pub(crate) fn cache_dir(root: &Path) -> PathBuf {
    root.join(CACHE)
}

/// 2026-09-26: Read the cached index; never touches the network.
pub fn cached(root: &Path) -> Index {
    let path = cache_dir(root).join(INDEX);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        // 2026-09-26: No index yet (never synced) is an empty Library.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Index::default(),
        // 2026-09-26: Any other read error (e.g. permissions) is reported in
        // `offline`, not shown as an empty index.
        Err(e) => {
            return Index {
                offline: Some(format!("{} could not be read: {e}", path.display())),
                ..Index::default()
            };
        }
    };
    match parse_cache(&text) {
        Ok(index) => index,
        // 2026-09-26: A corrupt cache is reported the same way; the next
        // successful refresh overwrites it.
        Err(e) => Index {
            offline: Some(format!("cached index unreadable: {e}")),
            ..Index::default()
        },
    }
}

/// 2026-09-26: Parse `index.json`. `pub(crate)` so `cli::doctor`'s
/// `check_recipes` reads the index with this same parser.
pub(crate) fn parse_cache(text: &str) -> Result<Index> {
    let doc: serde_json::Value = serde_json::from_str(text)?;
    let files = doc
        .get("files")
        .and_then(|f| f.as_object())
        .context("no `files` object")?;
    let mut recipes = Vec::new();
    for (id, content) in files {
        let Some(body) = content.as_str() else {
            bail!("{id} is not text");
        };
        // 2026-09-26: A recipe that fails to parse is skipped with a warning.
        match Recipe::parse(id.clone(), body) {
            Ok(r) => recipes.push(r),
            Err(e) => tracing::warn!("skipping cached recipe {id}: {e:#}"),
        }
    }
    recipes.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(Index {
        recipes,
        tree_sha: doc
            .get("tree_sha")
            .and_then(|s| s.as_str())
            .unwrap_or_default()
            .to_string(),
        fetched_at: doc.get("fetched_at").and_then(|s| s.as_u64()).unwrap_or(0),
        offline: None,
        incomplete: None,
    })
}

/// 2026-09-26: Fetch from GitHub, falling back to the cache on any failure.
///
/// Blocking; each request may take up to `TIMEOUT`. The UI uses
/// [`refresh_in_background`].
pub fn refresh(root: &Path, cancel: &AtomicBool) -> Index {
    refresh_with(root, || try_refresh(root, cancel))
}

/// 2026-09-26: The fallback rule with the fetch injected, so it is testable
/// offline: a failed fetch serves the cache with `offline` set to the error.
fn refresh_with(root: &Path, fetch: impl FnOnce() -> Result<Index>) -> Index {
    match fetch() {
        Ok(index) => index,
        Err(e) => {
            let mut fallback = cached(root);
            fallback.offline = Some(one_line(&format!("{e:#}")));
            fallback
        }
    }
}

/// 2026-09-26: Run [`refresh`] on a named `std::thread`; the result arrives on
/// the returned channel. If the thread cannot be spawned, the channel still
/// delivers the cache (`tui::worker::spawn`).
///
/// The returned flag cancels the refresh. Each fetch worker checks it before
/// taking the next file, so a request already in flight still completes.
pub fn refresh_in_background(root: &Path) -> (std::sync::mpsc::Receiver<Index>, Arc<AtomicBool>) {
    let owned = root.to_path_buf();
    let cancel = Arc::new(AtomicBool::new(false));
    let rx = crate::tui::worker::spawn(
        "metrale-recipes",
        {
            let cancel = Arc::clone(&cancel);
            move || refresh(&owned, &cancel)
        },
        |e| {
            let mut index = cached(root);
            index.offline = Some(format!("fetcher thread unavailable: {e}"));
            index
        },
    );
    (rx, cancel)
}

/// 2026-09-26: One line of at most 90 characters (plus an ellipsis), for a
/// title bar.
fn one_line(s: &str) -> String {
    let flat = s.replace('\n', " ");
    if flat.chars().count() <= 90 {
        return flat;
    }
    flat.chars().take(90).collect::<String>() + "…"
}

#[cfg(test)]
#[path = "fetch_tests.rs"]
mod tests;

/// 2026-09-26: Ask GitHub, on a worker thread, for the date of the last commit
/// to one recipe file (`fetch_github::commit_date`). The TUI calls it for one
/// recipe at a time, and only for a recipe without `metadata.updated`
/// (`tui/lib_dates.rs`).
///
/// The recipe id comes back with the date, because the selection may have
/// moved by the time the answer arrives.
pub fn updated_in_background(id: &str) -> std::sync::mpsc::Receiver<(String, Option<String>)> {
    let owned = id.to_string();
    let fallback_id = id.to_string();
    crate::tui::worker::spawn(
        "metrale-recipe-date",
        move || {
            let date = fetch_github::commit_date(&owned)
                .map_err(|e| {
                    // 2026-09-26: A failed lookup is logged at debug and yields
                    // no date.
                    tracing::debug!("could not date recipe {owned}: {e:#}");
                })
                .ok();
            (owned, date)
        },
        |_| (fallback_id, None),
    )
}
