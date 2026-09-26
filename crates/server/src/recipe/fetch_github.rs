// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The GitHub side of the recipe refresh: list the tree, fetch the files, write the cache, and date one recipe.
//!
//! Every call here blocks; `fetch.rs` runs them on worker threads.
//!
//! Owner: server (recipe).
//! Invariants:
//! - `try_refresh` writes the cache only when every listed file was fetched.

use anyhow::{Context, Result, bail};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::Recipe;
use super::fetch::{AGENT, INDEX, Index, REPO, TIMEOUT, cache_dir, unix_now};

/// 2026-09-26: How many recipe files are fetched at once. The files are small,
/// so a refresh is bound by round trips, which these workers overlap.
const FETCH_WIDTH: usize = 8;

pub(super) fn try_refresh(root: &Path, cancel: &AtomicBool) -> Result<Index> {
    let (tree_sha, paths) = list_recipe_paths()?;
    if paths.is_empty() {
        bail!("{REPO}@{tree_sha} lists no recipes/**/*.yaml");
    }
    // 2026-09-26: `FETCH_WIDTH` scoped workers each take the next unfetched
    // path until none is left or `cancel` is set.
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<(String, String)>> = Mutex::new(Vec::with_capacity(paths.len()));
    let width = FETCH_WIDTH.min(paths.len());
    std::thread::scope(|scope| {
        for _ in 0..width {
            scope.spawn(|| {
                loop {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(path) = paths.get(i) else { return };
                    let url = format!("https://raw.githubusercontent.com/{REPO}/{tree_sha}/{path}");
                    match get(&url) {
                        Ok(body) => out.lock().push((recipe_id(path), body)),
                        // 2026-09-26: A failed file is left out of this refresh.
                        Err(e) => tracing::warn!("skipping recipe {path}: {e:#}"),
                    }
                }
            });
        }
    });
    if cancel.load(Ordering::Relaxed) {
        bail!("refresh cancelled");
    }

    if out.lock().is_empty() {
        // 2026-09-26: Nothing fetched: an error, so the caller serves the cache
        // unchanged.
        bail!(
            "{REPO}@{tree_sha} listed {} recipe file(s) but none could be fetched",
            paths.len()
        );
    }

    let mut files: BTreeMap<String, String> = BTreeMap::new();
    let mut recipes = Vec::new();
    for (id, body) in out.into_inner() {
        match Recipe::parse(id.clone(), &body) {
            Ok(r) => recipes.push(r),
            // 2026-09-26: A recipe that fails to parse is skipped here but still
            // written to the cache.
            Err(e) => tracing::warn!("skipping recipe {id}: {e:#}"),
        }
        files.insert(id, body);
    }
    // 2026-09-26: The fetch order is nondeterministic; the Library's row order
    // is by id.
    recipes.sort_by(|a, b| a.id.cmp(&b.id));

    let fetched_at = unix_now();

    // 2026-09-26: The cache is replaced only by a complete fetch. Writing a
    // partial set would drop the recipes that did not come back, so a partial
    // fetch is returned for this session and reported in `incomplete`.
    let missing = paths.len().saturating_sub(files.len());
    let incomplete = if missing > 0 {
        Some(format!(
            "{missing} of {} recipe file(s) could not be fetched, so the cache was left alone",
            paths.len()
        ))
    } else {
        // 2026-09-26: A write failure goes into `incomplete`, not a log line:
        // `sync-recipes` runs before any tracing subscriber is installed
        // (`main.rs`), so a log line would not be seen.
        write_cache(root, &tree_sha, fetched_at, &files)
            .err()
            .map(|e| format!("the index could not be cached: {e:#}"))
    };

    Ok(Index {
        recipes,
        tree_sha,
        fetched_at,
        offline: None,
        incomplete,
    })
}

/// 2026-09-26: `recipes/qwen3.6/foo.yaml` → `qwen3.6/foo`.
pub(super) fn recipe_id(path: &str) -> String {
    path.trim_start_matches("recipes/")
        .trim_end_matches(".yaml")
        .to_string()
}

/// 2026-09-26: One API call: the tree sha at `main` and every
/// `recipes/**/*.yaml` blob under it, sorted.
fn list_recipe_paths() -> Result<(String, Vec<String>)> {
    let body = get(&format!(
        "https://api.github.com/repos/{REPO}/git/trees/main?recursive=1"
    ))
    .context("listing the recipe tree")?;
    let doc: serde_json::Value =
        serde_json::from_str(&body).context("tree response is not JSON")?;
    let sha = doc
        .get("sha")
        .and_then(|s| s.as_str())
        .context("tree response has no sha")?
        .to_string();
    // 2026-09-26: A truncated listing is an error, not a smaller Library.
    if doc.get("truncated").and_then(|t| t.as_bool()) == Some(true) {
        bail!("GitHub truncated the tree listing for {REPO}");
    }
    let entries = doc
        .get("tree")
        .and_then(|t| t.as_array())
        .context("tree response has no tree")?;
    let mut paths: Vec<String> = entries
        .iter()
        .filter(|e| e.get("type").and_then(|t| t.as_str()) == Some("blob"))
        .filter_map(|e| e.get("path").and_then(|p| p.as_str()))
        .filter(|p| p.starts_with("recipes/") && p.ends_with(".yaml"))
        .map(str::to_string)
        .collect();
    paths.sort();
    Ok((sha, paths))
}

/// 2026-09-26: One agent for the process, so connections are reused across
/// requests; every request has the global `TIMEOUT` of 20 s.
fn agent() -> &'static ureq::Agent {
    static AGENT_POOL: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    AGENT_POOL.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_global(Some(TIMEOUT))
            .build()
            .into()
    })
}

fn get(url: &str) -> Result<String> {
    let response = agent()
        .get(url)
        .header("User-Agent", AGENT)
        .call()
        .with_context(|| format!("GET {url}"))?;
    Ok(response.into_body().read_to_string()?)
}

/// 2026-09-26: Write `index.json` through a temp file and a rename, so a reader
/// never sees a half-written index.
pub(super) fn write_cache(
    root: &Path,
    tree_sha: &str,
    fetched_at: u64,
    files: &BTreeMap<String, String>,
) -> Result<()> {
    let dir = cache_dir(root);
    std::fs::create_dir_all(&dir)?;
    let doc = serde_json::json!({
        "tree_sha": tree_sha,
        "fetched_at": fetched_at,
        "files": files,
    });
    let tmp = dir.join("index.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&doc)?)?;
    std::fs::rename(&tmp, dir.join(INDEX))?;
    Ok(())
}

/// 2026-09-26: The committer date, as `YYYY-MM-DD`, of the first commit the
/// commits API returns for `recipes/{id}.yaml` (`per_page=1`). One API call per
/// recipe; `fetch::updated_in_background` is its only caller.
pub(super) fn commit_date(id: &str) -> Result<String> {
    let body = get(&format!(
        "https://api.github.com/repos/{REPO}/commits?path=recipes/{id}.yaml&per_page=1"
    ))
    .with_context(|| format!("dating recipe {id}"))?;
    let doc: serde_json::Value =
        serde_json::from_str(&body).context("commits response is not JSON")?;
    let date = doc
        .get(0)
        .and_then(|c| c.get("commit"))
        .and_then(|c| c.get("committer"))
        .and_then(|c| c.get("date"))
        .and_then(|d| d.as_str())
        .with_context(|| format!("no commit found for recipe {id}"))?;
    // 2026-09-26: Keep the part before `T` of the ISO-8601 timestamp.
    Ok(date.split('T').next().unwrap_or(date).to_string())
}
