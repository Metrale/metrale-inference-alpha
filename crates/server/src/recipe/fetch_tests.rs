// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the recipe index cache and its offline fallback.
//!
//! Owner: server (recipe).
//! Invariants: none beyond the types. Only the `#[ignore]` network tests and the
//! cancellation test call `refresh`; the cancellation test lists the repository
//! before the flag is checked. Every other test injects the fetch or reads the
//! cache.

use super::super::fetch_github::{recipe_id, write_cache};
use super::*;
use std::collections::BTreeMap;

struct Dir(PathBuf);
impl Dir {
    fn new(tag: &str) -> Self {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let p = std::env::temp_dir().join(format!("metrale-recipes-{tag}-{n}"));
        std::fs::create_dir_all(&p).expect("scratch");
        Self(p)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn a_recipe() -> String {
    "recipe_version: \"2\"\nmodel: Qwen/Qwen3.6-27B\nruntime: metrale\ncontainer: c\n\
     metadata:\n  description: test\n  maintainer: metrale\ndefaults:\n  port: 8888\n"
        .to_string()
}

fn seed(dir: &Dir, fetched_at: u64) {
    let files = BTreeMap::from([("qwen3.6/test".to_string(), a_recipe())]);
    write_cache(&dir.0, "deadbeef", fetched_at, &files).expect("cache written");
}

#[test]
fn an_empty_store_reports_never_fetched_rather_than_failing() {
    // 2026-09-26: A missing cache is an empty index, not an error.
    let dir = Dir::new("empty");
    let index = cached(&dir.0);
    assert!(index.recipes.is_empty());
    assert_eq!(index.fetched_at, 0);
    assert_eq!(index.age_text().as_deref(), Some("never fetched"));
}

#[test]
fn a_cached_index_round_trips() {
    let dir = Dir::new("roundtrip");
    seed(&dir, unix_now());
    let index = cached(&dir.0);
    assert_eq!(index.recipes.len(), 1);
    assert_eq!(index.recipes[0].id, "qwen3.6/test");
    assert_eq!(index.recipes[0].model, "Qwen/Qwen3.6-27B");
    assert_eq!(index.tree_sha, "deadbeef");
    assert!(
        index.offline.is_none(),
        "reading a cache is not being offline"
    );
}

#[test]
fn a_corrupt_cache_degrades_instead_of_crashing() {
    let dir = Dir::new("corrupt");
    std::fs::create_dir_all(cache_dir(&dir.0)).expect("dir");
    std::fs::write(cache_dir(&dir.0).join(INDEX), "{ not json").expect("write");
    let index = cached(&dir.0);
    assert!(index.recipes.is_empty());
    assert!(index.offline.is_some(), "says why it is empty");
}

#[test]
fn one_unreadable_recipe_does_not_blank_the_others() {
    // 2026-09-26: A single malformed recipe costs one row, not the whole index.
    let dir = Dir::new("partial");
    let files = BTreeMap::from([
        ("good/one".to_string(), a_recipe()),
        (
            "bad/two".to_string(),
            "this: is\n\tnot: valid\n".to_string(),
        ),
    ]);
    write_cache(&dir.0, "sha", unix_now(), &files).expect("write");
    let index = cached(&dir.0);
    assert_eq!(index.recipes.len(), 1);
    assert_eq!(index.recipes[0].id, "good/one");
}

#[test]
fn age_is_reported_in_the_largest_useful_unit() {
    let mut index = Index {
        fetched_at: unix_now(),
        ..Index::default()
    };
    assert_eq!(index.age_text().as_deref(), Some("0 m old"));
    index.fetched_at = unix_now() - 7200;
    assert_eq!(index.age_text().as_deref(), Some("2 h old"));
    index.fetched_at = unix_now() - 3 * 86400;
    assert_eq!(index.age_text().as_deref(), Some("3 d old"));
}

#[test]
fn offline_is_visible_in_the_status_line() {
    let index = Index {
        fetched_at: unix_now() - 3 * 86400,
        offline: Some("dns failure".into()),
        ..Index::default()
    };
    let text = index.status_text();
    assert!(text.contains("3 d old"), "{text}");
    assert!(text.contains("offline"), "{text}");
}

#[test]
fn a_failed_fetch_serves_the_cache_and_says_it_is_stale() {
    // 2026-09-26: The fetch is injected, so the test asserts the same thing with or
    // without a network.
    let dir = Dir::new("fallback");
    seed(&dir, unix_now() - 86400);
    let index = refresh_with(&dir.0, || anyhow::bail!("dns failure"));
    assert_eq!(index.recipes.len(), 1, "the cache still answered");
    assert!(index.offline.is_some(), "and it is marked stale");
    let text = index.status_text();
    assert!(text.contains("offline"), "{text}");
    assert!(text.contains("1 d old"), "with its age: {text}");
}

#[test]
fn a_successful_fetch_is_not_marked_offline() {
    let dir = Dir::new("live");
    seed(&dir, 0);
    let fresh = Index {
        recipes: Vec::new(),
        tree_sha: "abc".into(),
        fetched_at: unix_now(),
        offline: None,
        incomplete: None,
    };
    let index = refresh_with(&dir.0, || Ok(fresh));
    assert!(index.offline.is_none());
    assert_eq!(index.tree_sha, "abc", "the fetch wins over the cache");
}

#[test]
fn a_recipe_id_is_its_path_without_prefix_or_extension() {
    assert_eq!(recipe_id("recipes/qwen3.6/foo.yaml"), "qwen3.6/foo");
}

#[test]
fn a_long_error_is_trimmed_to_fit_a_title_bar() {
    let long = "e".repeat(500);
    let out = one_line(&long);
    assert_eq!(out.chars().count(), 91);
    assert!(
        !one_line("a\nb").contains('\n'),
        "newlines would break layout"
    );
}

#[test]
fn the_cache_is_written_atomically() {
    // 2026-09-26: A half-written index would read as a corrupt one, so the write
    // goes through a temp file and a rename.
    let dir = Dir::new("atomic");
    seed(&dir, unix_now());
    assert!(cache_dir(&dir.0).join(INDEX).exists());
    assert!(
        !cache_dir(&dir.0).join("index.json.tmp").exists(),
        "the temp file is renamed, not left behind"
    );
}

/// 2026-09-26: A live fetch from the recipe repository, `#[ignore]` because it needs
/// the network. Run it by hand after changing the fetch:
/// `cargo test -p metrale-server --bins live_fetch -- --ignored --nocapture`
#[test]
#[ignore = "needs the network"]
fn live_fetch_against_github() {
    let dir = Dir::new("live-github");
    let index = refresh(&dir.0, &std::sync::atomic::AtomicBool::new(false));
    assert!(index.offline.is_none(), "fetch failed: {:?}", index.offline);
    assert_eq!(index.recipes.len(), 25, "the corpus is 25 recipes");
    assert_eq!(index.recipes.iter().filter(|r| r.is_metrale()).count(), 23);
    assert_eq!(index.tree_sha.len(), 40, "a full tree sha");
    // 2026-09-26: Every live `runtime: metrale` recipe must produce a valid serve
    // config; the vendored-fixture tests cannot see the live index.
    for r in index.recipes.iter().filter(|r| r.is_metrale()) {
        r.serve_args(&BTreeMap::new())
            .unwrap_or_else(|e| panic!("live recipe {} is not servable: {e:#}", r.id));
    }
    // 2026-09-26: A complete fetch writes the cache, which must read back the same.
    let reread = cached(&dir.0);
    assert_eq!(reread.recipes.len(), index.recipes.len());
    assert_eq!(reread.tree_sha, index.tree_sha);
    eprintln!("{} recipes @ {}", index.recipes.len(), index.tree_sha);
}

#[test]
fn a_no_route_failure_tells_the_user_what_to_do_about_it() {
    // 2026-09-26: A no-route failure names the fix (HTTPS_PROXY), not just
    // "offline".
    let index = Index {
        offline: Some("GET https://api.github.com/…: dns error: Network is unreachable".into()),
        fetched_at: unix_now() - 86400,
        ..Index::default()
    };
    let detail = index.offline_detail().expect("a reason");
    assert!(detail.contains("no route"), "{detail}");
    assert!(detail.contains("HTTPS_PROXY"), "names the fix: {detail}");
    assert!(
        detail.contains(
            " or copy the cached index (~/.metrale/metrale-recipes/index.json) from a machine \
             that can reach it."
        ),
        "names the one cache path, rendered whole: {detail}"
    );
    assert!(index.status_text().len() < 40, "{}", index.status_text());
}

#[test]
fn a_rate_limit_is_distinguished_from_a_dead_link() {
    let index = Index {
        offline: Some("GET https://api.github.com/…: status 403 rate limit exceeded".into()),
        ..Index::default()
    };
    let detail = index.offline_detail().expect("a reason");
    assert!(detail.contains("rate-limiting"), "{detail}");
    assert!(
        !detail.contains("HTTPS_PROXY"),
        "a proxy does not fix a rate limit: {detail}"
    );
}

#[test]
fn a_healthy_index_has_no_reason_to_show() {
    let index = Index {
        fetched_at: unix_now(),
        ..Index::default()
    };
    assert!(index.offline_detail().is_none());
}

#[test]
fn a_cancelled_refresh_serves_the_cache_rather_than_a_partial_index() {
    // 2026-09-26: A cancelled refresh is an error, so it serves the cache instead
    // of writing a subset over it.
    let dir = Dir::new("cancelled");
    let cancel = std::sync::atomic::AtomicBool::new(true);
    let index = refresh(&dir.0, &cancel);
    assert!(
        index.offline.is_some(),
        "a cancelled refresh is not a live index"
    );
}

/// 2026-09-26: Network test: the concurrent fetch returns the recipes sorted by id.
#[test]
#[ignore = "needs the network"]
fn a_concurrent_refresh_is_ordered_and_complete() {
    let dir = Dir::new("concurrent-order");
    let index = refresh(&dir.0, &std::sync::atomic::AtomicBool::new(false));
    assert!(index.offline.is_none(), "fetch failed: {:?}", index.offline);
    assert_eq!(index.recipes.len(), 25);
    // 2026-09-26: Fetch order is nondeterministic; row order is not.
    let mut sorted = index.recipes.clone();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    assert_eq!(
        index.recipes.iter().map(|r| &r.id).collect::<Vec<_>>(),
        sorted.iter().map(|r| &r.id).collect::<Vec<_>>(),
        "recipes must be sorted regardless of which worker finished first"
    );
}

/// 2026-09-26: Network test: prints how long a cold refresh takes. It asserts only
/// that the fetch succeeded, not a wall-clock bound.
#[test]
#[ignore = "needs the network"]
fn measure_refresh_wall_time() {
    let dir = Dir::new("measure-refresh");
    let t = std::time::Instant::now();
    let index = refresh(&dir.0, &std::sync::atomic::AtomicBool::new(false));
    let elapsed = t.elapsed();
    eprintln!(
        "refresh: {:?} for {} recipes (offline={:?})",
        elapsed,
        index.recipes.len(),
        index.offline
    );
    assert!(index.offline.is_none());
}

/// 2026-09-26: An index that exists and cannot be read is not an empty index:
/// `cached` reports the path it could not read in `offline`.
#[test]
fn an_unreadable_index_is_not_reported_as_an_absent_one() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = Dir::new("unreadable");
        seed(&dir, unix_now());
        let path = cache_dir(&dir.0).join(INDEX);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).expect("chmod");
        // 2026-09-26: root, or an ACL, can read the file despite mode 000. The test
        // returns early when the fault cannot be staged, rather than pass for a
        // reason unrelated to the code.
        if std::fs::read_to_string(&path).is_ok() {
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
            return;
        }

        let index = cached(&dir.0);
        let why = index.offline.as_deref().unwrap_or_default();
        assert!(
            !why.is_empty(),
            "an unreadable index must say so, not answer `no recipes`"
        );
        assert!(
            why.contains(&path.display().to_string()),
            "the operator needs the path that could not be read: {why}"
        );

        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
}

/// 2026-09-26: Control for the test above: `NotFound` is the one read error that
/// is an empty index, with no fault reported.
#[test]
fn an_index_that_was_never_written_is_still_an_ordinary_empty_store() {
    let dir = Dir::new("never-written");
    let index = cached(&dir.0);
    assert!(index.recipes.is_empty());
    assert!(
        index.offline.is_none(),
        "a box that has never synced has no fault to report, got: {:?}",
        index.offline
    );
}

/// 2026-09-26: `cache_dir` is `<root>/metrale-recipes` when that directory exists.
#[test]
fn the_cache_dir_is_the_current_name_when_it_exists() {
    let dir = Dir::new("cache-current");
    std::fs::create_dir_all(dir.0.join(CACHE)).expect("current");
    assert_eq!(cache_dir(&dir.0), dir.0.join(CACHE));
}

/// 2026-09-26: `cache_dir` is `<root>/metrale-recipes` on a fresh store too.
#[test]
fn a_fresh_store_uses_the_current_cache_dir() {
    let dir = Dir::new("cache-neither");
    assert_eq!(cache_dir(&dir.0), dir.0.join(CACHE));
}
