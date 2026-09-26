// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Structure rules for the `kernels/` tree, checked through
//! `metrale_closure::layout`, the resolver build.rs compiles from.
//! `scripts/check_kernel_shadows.py` checks the same rules in Python.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! - No symlinks under `kernels/`: sharing is `[sources] use` in a KERNEL.toml
//!   or `[hardware] inherits` in a HARDWARE.toml.
//! - Every target `walk` lists resolves with `discover`, has a non-empty module
//!   set, and states a reason for every shadow.
//! - Rule 1: no shadow is byte-identical to the entry it shadows.
//! - Rule 2: no two regular kernel files under `kernels/` share a name and
//!   their bytes; keep one and `use` it from the others.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use metrale_closure::layout::{discover, symlinks, walk};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .to_path_buf()
}

fn is_kernel_file(name: &str) -> bool {
    [".cu", ".cuh", ".h", ".metal"]
        .iter()
        .any(|e| name.ends_with(e))
}

#[test]
fn no_symlink_under_kernels() {
    let links = symlinks(&workspace_root());
    assert!(
        links.is_empty(),
        "{} symlink(s) under kernels/ — declare the file in [sources] use or inherit the tree:\n  {}",
        links.len(),
        links
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

#[test]
fn every_target_resolves_a_non_empty_module_set() {
    let root = workspace_root();
    let targets = walk(&root).expect("the tree resolves");
    assert!(
        targets.len() > 50,
        "{} targets — wrong root?",
        targets.len()
    );
    for t in &targets {
        let l = discover(&root, t).unwrap_or_else(|e| panic!("{t}: {e}"));
        assert!(!l.modules().is_empty(), "{t}: no modules");
        for s in &l.shadows {
            assert!(
                !s.reason.trim().is_empty(),
                "{t}: {} shadows {} for no stated reason",
                s.winner.display(),
                s.loser.display()
            );
        }
    }
}

/// 2026-09-25: Rule 1, over every shadow of every target.
#[test]
fn no_shadow_is_byte_identical_to_what_it_shadows() {
    let root = workspace_root();
    let mut dead = Vec::new();
    for t in walk(&root).expect("the tree resolves") {
        let l = discover(&root, &t).unwrap_or_else(|e| panic!("{t}: {e}"));
        for s in &l.shadows {
            if std::fs::read(&s.winner).unwrap() == std::fs::read(&s.loser).unwrap() {
                dead.push(format!(
                    "{t}: {} == {}",
                    s.winner.display(),
                    s.loser.display()
                ));
            }
        }
    }
    dead.sort();
    dead.dedup();
    assert!(
        dead.is_empty(),
        "dead override(s) — byte-identical to what they shadow:\n  {}",
        dead.join("\n  ")
    );
}

/// 2026-09-25: Rule 2, over the whole tree, so an overlay's copy of a gb10
/// file counts as a duplicate too.
#[test]
fn no_two_regular_kernel_files_share_a_name_and_their_bytes() {
    let mut by_key: BTreeMap<(String, Vec<u8>), Vec<PathBuf>> = BTreeMap::new();
    fn walk_dir(dir: &Path, out: &mut BTreeMap<(String, Vec<u8>), Vec<PathBuf>>) {
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if p.is_dir() {
                walk_dir(&p, out);
            } else if is_kernel_file(&name) {
                out.entry((name, std::fs::read(&p).unwrap()))
                    .or_default()
                    .push(p);
            }
        }
    }
    walk_dir(&workspace_root().join("kernels"), &mut by_key);
    let dups: Vec<String> = by_key
        .into_iter()
        .filter(|(_, paths)| paths.len() > 1)
        .map(|((name, _), paths)| {
            format!(
                "{name}: {}",
                paths
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(" == ")
            )
        })
        .collect();
    assert!(
        dups.is_empty(),
        "duplicate regular copies — keep one, `use` it from the rest:\n  {}",
        dups.join("\n  ")
    );
}
