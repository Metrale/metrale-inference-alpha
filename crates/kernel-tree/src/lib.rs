// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The repository's `kernels/` manifests and sources (`*.toml`, `*.cu`, `*.cuh`,
//! `*.h`) as they were when this binary was built, so code that reads the repository
//! (`met circuit memory`'s model, which `met serve` also runs) works without a checkout.
//!
//! Owner: kernel tree (build embedding).
//! Invariants:
//! - [`materialize`] writes the tree under `<dir>/<SHA256>/` and returns that path; a tree
//!   already there is reused, and a partial one never appears under the final name (it is built
//!   under a temporary name and renamed).
//! - No embedded path is absolute or has a `..` component: unpacking cannot write outside the
//!   tree.

use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

const PACKED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernel_tree.bin"));

/// 2026-10-02: SHA-256 of the unpacked records: the tree's identity.
pub const SHA256: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel_tree.sha256"));

/// 2026-10-02: Every embedded file, `(repository-relative path, bytes)`, sorted by path.
pub fn files() -> Result<Vec<(String, Vec<u8>)>> {
    let mut raw = Vec::new();
    std::io::Read::read_to_end(&mut flate2::read::DeflateDecoder::new(PACKED), &mut raw)
        .context("inflate the embedded kernel tree")?;
    let mut out = Vec::new();
    let mut at = 0usize;
    let mut take = |n: usize| -> Result<&[u8]> {
        let end = at.checked_add(n).filter(|&e| e <= raw.len());
        let end = end.context("the embedded kernel tree is truncated")?;
        let s = &raw[at..end];
        at = end;
        Ok(s)
    };
    loop {
        let Ok(len) = take(4) else { break };
        let len = u32::from_le_bytes(len.try_into().expect("4 bytes")) as usize;
        let path = String::from_utf8(take(len)?.to_vec()).context("embedded path")?;
        let n = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes")) as usize;
        let bytes = take(n)?.to_vec();
        check_path(&path)?;
        out.push((path, bytes));
    }
    Ok(out)
}

fn check_path(rel: &str) -> Result<()> {
    let p = Path::new(rel);
    ensure!(
        !rel.is_empty() && p.components().all(|c| matches!(c, Component::Normal(_))),
        "embedded path `{rel}` is not a plain relative path"
    );
    Ok(())
}

/// 2026-10-02: The embedded tree unpacked under `dir/<SHA256>/`, a repository root that has the
/// `kernels/` files and nothing else.
pub fn materialize(dir: &Path) -> Result<PathBuf> {
    let root = dir.join(SHA256);
    if root.join("kernels").is_dir() {
        return Ok(root);
    }
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let tmp = dir.join(format!("{SHA256}.partial-{}", std::process::id()));
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp).with_context(|| format!("remove {}", tmp.display()))?;
    }
    for (rel, bytes) in files()? {
        let path = tmp.join(&rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        std::fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))?;
    }
    match std::fs::rename(&tmp, &root) {
        Ok(()) => Ok(root),
        // 2026-10-02: Another process unpacked the same tree first; its copy is identical.
        Err(_) if root.join("kernels").is_dir() => {
            std::fs::remove_dir_all(&tmp).with_context(|| format!("remove {}", tmp.display()))?;
            Ok(root)
        }
        Err(e) => bail!("rename {} to {}: {e}", tmp.display(), root.display()),
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
