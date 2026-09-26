// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Resolving a file-tool path inside the agent's sandbox.
//!
//! Owner: bench, agentic.
//! Invariants: none beyond the types.

use std::path::{Component, Path, PathBuf};

use anyhow::{Result, bail};

/// 2026-09-26: Resolve `path` inside `sandbox`. The check is lexical, because
/// the target usually does not exist yet: a relative path, or an absolute one
/// already under `sandbox`, is accepted unless it has a `..` component or
/// resolves, through an existing symlink, outside the sandbox.
///
/// This is not a privilege boundary: `bash` runs any command, and a symlink
/// created after this check is not seen.
pub fn resolve(sandbox: &Path, path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    let path = match path.strip_prefix(sandbox) {
        Ok(inside) => inside,
        Err(_) if path.is_absolute() => bail!(
            "path must be inside the project directory {}: {}",
            sandbox.display(),
            path.display()
        ),
        Err(_) => path,
    };
    let mut out = sandbox.to_path_buf();
    for component in path.components() {
        match component {
            Component::Normal(c) => out.push(c),
            Component::CurDir => {}
            Component::ParentDir => bail!("path must not leave the project directory"),
            Component::RootDir | Component::Prefix(_) => bail!("absolute paths are not allowed"),
        }
    }
    if leaves_via_symlink(sandbox, &out) {
        bail!("path must not leave the project directory through a symlink");
    }
    Ok(out)
}

/// 2026-09-26: Does `out`, already lexically inside `sandbox`, resolve outside
/// it? False when `sandbox` itself cannot be canonicalised; true when
/// `deepest_existing_real` finds nothing within 40 dangling-symlink hops.
fn leaves_via_symlink(sandbox: &Path, out: &Path) -> bool {
    let Ok(root) = std::fs::canonicalize(sandbox) else {
        return false;
    };
    deepest_existing_real(out, 40).is_none_or(|real| !real.starts_with(&root))
}

/// 2026-09-26: Canonicalise `path`, or else the deepest ancestor that
/// canonicalises, following a dangling symlink's target for up to `hops` links.
fn deepest_existing_real(path: &Path, hops: usize) -> Option<PathBuf> {
    let mut probe = path;
    loop {
        if let Ok(real) = std::fs::canonicalize(probe) {
            return Some(real);
        }
        if std::fs::symlink_metadata(probe).is_ok_and(|meta| meta.file_type().is_symlink()) {
            if hops == 0 {
                return None;
            }
            let target = std::fs::read_link(probe).ok()?;
            let target = match target.is_absolute() {
                true => target,
                false => probe.parent()?.join(target),
            };
            return deepest_existing_real(&target, hops - 1);
        }
        probe = probe.parent()?;
    }
}
