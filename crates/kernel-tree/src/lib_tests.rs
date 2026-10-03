// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The embedded tree is the repository's kernels/ text, byte for byte, and unpacks to
//! a root those files can be read from.
//!
//! Owner: kernel tree (build embedding).
//! Invariants: none beyond the types.

use std::path::PathBuf;

use super::{check_path, files, materialize};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// 2026-10-02: Path A: every embedded file equals the checkout's, and the files the memory model
/// reads first are present.
#[test]
fn the_embedded_files_are_the_checkouts() {
    let all = files().unwrap();
    for must in [
        "kernels/DEVICES.toml",
        "kernels/circuits/COPIES.toml",
        "kernels/gb10/HARDWARE.toml",
        "kernels/gb10/common/KERNEL_FAMILIES.toml",
    ] {
        assert!(all.iter().any(|(p, _)| p == must), "{must} is not embedded");
    }
    for (rel, bytes) in &all {
        assert_eq!(&std::fs::read(repo().join(rel)).unwrap(), bytes, "{rel}");
    }
    assert!(
        all.windows(2).all(|w| w[0].0 < w[1].0),
        "sorted, no duplicates"
    );
}

/// 2026-10-02: Path B: an unpacked tree reads back the same bytes, and a second call reuses it.
#[test]
fn materialize_unpacks_once() {
    let dir = std::env::temp_dir().join(format!("metrale-kernel-tree-test-{}", std::process::id()));
    let root = materialize(&dir).unwrap();
    let devices = std::fs::read(root.join("kernels/DEVICES.toml")).unwrap();
    assert_eq!(
        devices,
        std::fs::read(repo().join("kernels/DEVICES.toml")).unwrap()
    );
    let marker = root.join("kernels/.reused");
    std::fs::write(&marker, b"").unwrap();
    assert_eq!(materialize(&dir).unwrap(), root);
    assert!(marker.exists(), "the second call rebuilt the tree");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// 2026-10-02: A path that would escape the tree is refused.
#[test]
fn escaping_paths_are_refused() {
    for bad in ["", "/etc/passwd", "../x", "kernels/../../x", "./kernels"] {
        assert!(check_path(bad).is_err(), "`{bad}` accepted");
    }
    assert!(check_path("kernels/gb10/HARDWARE.toml").is_ok());
}
