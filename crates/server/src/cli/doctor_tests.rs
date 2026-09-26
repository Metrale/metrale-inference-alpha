// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `met doctor`: the home, writable and recipes checks
//! are each asserted both passing and reporting a problem. `check_identity`
//! is not covered here.
//!
//! They call the `check_*` functions directly with `METRALE_HOME` set around
//! each, under one mutex (`env_lock`), since the environment is process-global.
//!
//! Owner: server CLI (`met doctor`).
//! Invariants: none beyond the types.

use super::{check_home, check_recipes, check_writable};
use std::sync::{Mutex, MutexGuard, OnceLock};

fn env_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

struct Dir(std::path::PathBuf);
impl Dir {
    fn new(tag: &str) -> Self {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let p = std::env::temp_dir().join(format!("metrale-doctor-{tag}-{n}"));
        std::fs::create_dir_all(&p).expect("scratch");
        Self(p)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn with_home<T>(root: &std::path::Path, f: impl FnOnce() -> T) -> T {
    let _g = env_lock();
    let prev = std::env::var_os("METRALE_HOME");
    // 2026-09-26: SAFETY: `env_lock` serialises this file's tests; it does not
    // exclude other test modules in this binary that touch the environment.
    unsafe { std::env::set_var("METRALE_HOME", root) };
    let out = f();
    match prev {
        Some(v) => unsafe { std::env::set_var("METRALE_HOME", v) },
        None => unsafe { std::env::remove_var("METRALE_HOME") },
    }
    out
}

#[test]
fn home_is_green_when_resolvable_and_names_its_provenance() {
    let d = Dir::new("home");
    let f = with_home(&d.0, check_home);
    assert!(!f.problem, "{}", f.detail);
    assert!(
        f.detail.contains("from METRALE_HOME"),
        "the provenance is the point: {}",
        f.detail
    );
}

/// 2026-09-26: Problem: an empty `METRALE_HOME`, as
/// `export METRALE_HOME=$SOMETHING_UNSET` produces.
#[test]
fn home_goes_red_when_metrale_home_is_empty() {
    let _g = env_lock();
    let prev = std::env::var_os("METRALE_HOME");
    unsafe { std::env::set_var("METRALE_HOME", "") };
    let f = check_home();
    match prev {
        Some(v) => unsafe { std::env::set_var("METRALE_HOME", v) },
        None => unsafe { std::env::remove_var("METRALE_HOME") },
    }
    assert!(f.problem, "an empty METRALE_HOME must be a problem");
    assert!(f.detail.contains("empty"), "{}", f.detail);
}

#[test]
fn writable_is_green_on_a_writable_home() {
    let d = Dir::new("w-ok");
    let f = with_home(&d.0, check_writable);
    assert!(!f.problem, "{}", f.detail);
}

/// 2026-09-26: Problem: a home that exists but this process cannot write.
#[test]
fn writable_goes_red_on_a_read_only_home() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let d = Dir::new("w-ro");
        std::fs::set_permissions(&d.0, std::fs::Permissions::from_mode(0o500)).expect("chmod");
        let f = with_home(&d.0, check_writable);
        std::fs::set_permissions(&d.0, std::fs::Permissions::from_mode(0o755)).expect("restore");
        assert!(
            f.problem,
            "a read-only home must be a problem: {}",
            f.detail
        );
        assert!(
            f.detail.contains("not writable"),
            "must say what is wrong: {}",
            f.detail
        );
    }
}

/// 2026-09-26: Problem: a home path that exists and is not a directory.
#[test]
fn writable_goes_red_when_the_home_is_a_file() {
    let d = Dir::new("w-file");
    let file = d.0.join("notadir");
    std::fs::write(&file, b"x").expect("write");
    let f = with_home(&file, check_writable);
    assert!(f.problem, "{}", f.detail);
    assert!(f.detail.contains("not a directory"), "{}", f.detail);
}

fn a_recipe_body(n: usize) -> String {
    format!("recipe_version: \"2\"\nmodel: test/model-{n}\ncontainer: c\ndefaults:\n  port: 8888\n")
}

/// 2026-09-26: Passing: an index in the `files: { <path>: <recipe body> }`
/// schema that `parse_cache` reads, listing three recipes.
#[test]
fn recipes_is_green_when_the_files_map_lists_some() {
    let d = Dir::new("r-ok");
    std::fs::create_dir_all(d.0.join("metrale-recipes")).expect("mkdir");
    let doc = serde_json::json!({
        "tree_sha": "deadbeef",
        "fetched_at": 0,
        "files": {
            "family/one": a_recipe_body(1),
            "family/two": a_recipe_body(2),
            "family/three": a_recipe_body(3),
        },
    });
    std::fs::write(d.0.join("metrale-recipes/index.json"), doc.to_string()).expect("write");
    let f = with_home(&d.0, check_recipes);
    assert!(!f.problem, "{}", f.detail);
    assert!(f.detail.contains('3'), "{}", f.detail);
}

/// 2026-09-26: Problem: no index file; the remedy is `sync-recipes`.
#[test]
fn recipes_goes_red_when_never_written() {
    let d = Dir::new("r-none");
    let f = with_home(&d.0, check_recipes);
    assert!(f.problem);
    assert!(f.detail.contains("never been written"), "{}", f.detail);
    assert!(f.remedy.contains("sync-recipes"), "{}", f.remedy);
}

/// 2026-09-26: An index that exists but cannot be read is reported as
/// unreadable, not missing, and the remedy does not say to run `sync-recipes`.
#[test]
fn an_unreadable_index_is_not_reported_as_a_missing_one() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let d = Dir::new("r-perm");
        std::fs::create_dir_all(d.0.join("metrale-recipes")).expect("mkdir");
        let idx = d.0.join("metrale-recipes/index.json");
        std::fs::write(&idx, br#"{"recipes":[1]}"#).expect("write");
        std::fs::set_permissions(&idx, std::fs::Permissions::from_mode(0o000)).expect("chmod");
        let f = with_home(&d.0, check_recipes);
        std::fs::set_permissions(&idx, std::fs::Permissions::from_mode(0o644)).expect("restore");
        assert!(f.problem);
        assert!(
            f.detail.contains("cannot be read"),
            "must distinguish unreadable from absent: {}",
            f.detail
        );
        assert!(
            !f.remedy.contains("run `met sync-recipes`"),
            "must NOT send the operator to the network for a permission fault: {}",
            f.remedy
        );
    }
}

/// 2026-09-26: Problem: an index that parses and lists no recipes.
#[test]
fn recipes_goes_red_when_the_index_is_empty() {
    let d = Dir::new("r-empty");
    std::fs::create_dir_all(d.0.join("metrale-recipes")).expect("mkdir");
    std::fs::write(d.0.join("metrale-recipes/index.json"), br#"{"files":{}}"#).expect("write");
    let f = with_home(&d.0, check_recipes);
    assert!(f.problem);
    assert!(f.detail.contains("lists none"), "{}", f.detail);
}
