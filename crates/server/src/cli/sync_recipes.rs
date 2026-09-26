// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `met sync-recipes`: fetch the recipe index into the local cache, with no TUI.
//!
//! The benchmark serve plan resolves a recipe id against this cache.
//!
//! Owner: server CLI.
//! Invariants: `run` returns `Ok` only when the fetch reached the repository,
//! returned at least one recipe and wrote the cache.

use anyhow::{Context, Result, bail};
use std::sync::atomic::AtomicBool;

/// 2026-09-26: Decide whether a finished refresh counts as a sync. Pure, so the
/// rule is tested without a network.
///
/// # Errors
/// When the refresh did not reach the repository, returned no recipes, or did
/// not write the cache.
pub fn verdict(
    offline: Option<&str>,
    incomplete: Option<&str>,
    recipes: usize,
    before: usize,
) -> Result<(), String> {
    // 2026-09-26: When the fetch fails, `refresh` returns the cache with
    // `offline` set; that is not a sync.
    if let Some(why) = offline {
        return Err(format!(
            "could not reach the recipe repository: {why}\n\
             The cache is unchanged ({before} recipe(s)). This command reports \
             failure rather than success-with-stale-data, because a stale index \
             fails later, somewhere less obvious."
        ));
    }
    if recipes == 0 {
        return Err(
            "the repository returned no recipes; refusing to call that a synced index".to_owned(),
        );
    }
    // 2026-09-26: `incomplete` means the repository was reached but the cache
    // was not replaced (a file did not arrive, or the write failed), so the file
    // on disk is unchanged. It is an error, not a `warn!`: this command runs
    // before any tracing subscriber is installed, so a warning would be lost.
    if let Some(why) = incomplete {
        return Err(format!(
            "the recipe index was not written: {why}\n\
             The cache is unchanged ({before} recipe(s)) — nothing was replaced."
        ));
    }
    Ok(())
}

/// 2026-09-26: Fetch the recipe index into the cache and print where it was
/// written.
///
/// # Errors
/// If the artifact store cannot be located, or [`verdict`] refuses the fetch.
pub fn run() -> Result<()> {
    let store = metrale_bench::ArtifactStore::discover()
        .context("locating the artifact store that holds the recipe cache")?;
    let root = store.root();

    let before = crate::recipe::fetch::cached(root).recipes.len();
    // 2026-09-26: Never cancelled: there is no UI to cancel from, and every
    // request is bounded by `fetch::TIMEOUT`.
    let index = crate::recipe::fetch::refresh(root, &AtomicBool::new(false));

    if let Err(why) = verdict(
        index.offline.as_deref(),
        index.incomplete.as_deref(),
        index.recipes.len(),
        before,
    ) {
        bail!("{why}");
    }

    println!(
        "recipe index written to {}",
        crate::recipe::fetch::cache_dir(root)
            .join("index.json")
            .display()
    );
    println!(
        "  {} recipe(s), tree {}",
        index.recipes.len(),
        index.tree_sha
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::verdict;

    #[test]
    fn a_real_fetch_with_recipes_is_a_sync() {
        assert!(verdict(None, None, 30, 0).is_ok());
    }

    #[test]
    fn falling_back_to_a_warm_cache_is_not_a_sync() {
        // 2026-09-26: The fallback carries the cached recipes, so the count
        // alone looks healthy.
        let e = verdict(Some("dns failure"), None, 30, 30).expect_err("must refuse");
        assert!(e.contains("could not reach"), "{e}");
        assert!(
            e.contains("30 recipe(s)"),
            "must say what is actually there: {e}"
        );
        assert!(
            e.contains("unchanged"),
            "must not imply anything was written: {e}"
        );
    }

    #[test]
    fn falling_back_with_no_cache_at_all_is_also_refused() {
        assert!(verdict(Some("timed out"), None, 0, 0).is_err());
    }

    #[test]
    fn an_empty_index_is_refused_even_when_the_network_worked() {
        let e = verdict(None, None, 0, 30).expect_err("must refuse");
        assert!(e.contains("no recipes"), "{e}");
    }

    #[test]
    fn a_fetch_that_could_not_be_cached_is_not_a_sync() {
        let e = verdict(
            None,
            Some("the index could not be cached: permission denied"),
            30,
            12,
        )
        .expect_err("an uncached fetch is not a synced index");
        assert!(e.contains("permission denied"), "{e}");
        assert!(
            e.contains("unchanged (12 recipe(s))"),
            "the operator needs to know what is still on disk: {e}"
        );
    }

    #[test]
    fn a_partial_fetch_is_not_a_sync_even_though_recipes_arrived() {
        let e = verdict(
            None,
            Some("27 of 30 recipe file(s) could not be fetched"),
            3,
            30,
        )
        .expect_err("3 of 30 recipes is not a synced index");
        assert!(e.contains("27 of 30"), "{e}");
    }

    /// 2026-09-26: Offline is reported first: naming a write problem for a fetch
    /// that never happened would point the operator at the wrong cause.
    #[test]
    fn being_offline_is_reported_before_any_write_problem() {
        let e = verdict(Some("dns failure"), Some("could not be cached"), 30, 30)
            .expect_err("must refuse");
        assert!(e.contains("dns failure"), "{e}");
        assert!(!e.contains("could not be cached"), "{e}");
    }
}
