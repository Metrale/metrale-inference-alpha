// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What one layer directory of a [`super::layout::Layout`]
//! contributes: the kernel files it holds and the ones its `KERNEL.toml`
//! `[sources] use` brings in.
//!
//! Owner: metrale-closure (kernel layout).
//! Invariants:
//! - A `[sources] use` entry that `layer_contents` accepts is a relative path
//!   whose lexically normalised form stays under `kernels/` (links are not
//!   resolved), names an existing kernel source or header, and whose final
//!   component is not a symlink.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::layout::{Entry, Layer, Role, Tier};
use crate::layout_manifest::{HEADER_EXTS, LayoutError, kernel_manifest};

pub(crate) fn has_ext(name: &str, ext: &str) -> bool {
    name.rsplit_once('.').is_some_and(|(_, e)| e == ext)
}

fn is_kernel_file(name: &str, source_ext: &str) -> bool {
    has_ext(name, source_ext) || HEADER_EXTS.iter().any(|h| has_ext(name, h))
}

pub(crate) fn layer(
    role: Role,
    tier: Tier,
    hardware: &str,
    dir: PathBuf,
) -> Result<Layer, LayoutError> {
    let manifest = if dir.is_dir() {
        kernel_manifest(&dir)?
    } else {
        None
    };
    Ok(Layer {
        role,
        tier,
        hardware: hardware.to_string(),
        dir,
        manifest,
    })
}

/// 2026-09-26: The files and subdirectories one layer contributes: the
/// kernel files its directory holds, then its `[sources] use`. A symlink in
/// the directory is an error.
#[allow(clippy::type_complexity)]
pub(crate) fn layer_contents(
    kernels: &Path,
    l: &Layer,
    idx: usize,
    source_ext: &str,
) -> Result<(BTreeMap<String, Entry>, BTreeMap<String, PathBuf>), LayoutError> {
    let mut entries = BTreeMap::new();
    let mut dirs = BTreeMap::new();
    if let Ok(read) = std::fs::read_dir(&l.dir) {
        for e in read.flatten() {
            let path = e.path();
            if path.is_symlink() {
                return Err(LayoutError::Symlink(path));
            }
            let name = e.file_name().to_string_lossy().to_string();
            if path.is_dir() {
                dirs.insert(name, path);
            } else if is_kernel_file(&name, source_ext) {
                entries.insert(
                    name.clone(),
                    Entry {
                        name,
                        source: path,
                        layer: idx,
                        used: false,
                    },
                );
            }
        }
    }
    let Some(m) = &l.manifest else {
        return Ok((entries, dirs));
    };
    for raw in &m.uses {
        let fail = |message: String| LayoutError::Use {
            manifest: m.path.clone(),
            entry: raw.clone(),
            message,
        };
        let rel = Path::new(raw);
        if rel.is_absolute() {
            return Err(fail("must be relative to kernels/<hardware>".into()));
        }
        let path = normalize(&kernels.join(&l.hardware).join(rel));
        if !path.starts_with(kernels) {
            return Err(fail("escapes kernels/".into()));
        }
        if path.is_symlink() {
            return Err(LayoutError::Symlink(path));
        }
        if !path.is_file() {
            return Err(fail(format!("no such file: {}", path.display())));
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .ok_or_else(|| fail("names no file".into()))?;
        if !is_kernel_file(&name, source_ext) {
            return Err(fail(format!("not a .{source_ext} source or a header")));
        }
        if let Some(existing) = entries.get(&name) {
            return Err(LayoutError::UseCollides {
                manifest: m.path.clone(),
                name,
                existing: existing.source.clone(),
            });
        }
        entries.insert(
            name.clone(),
            Entry {
                name,
                source: path,
                layer: idx,
                used: true,
            },
        );
    }
    Ok((entries, dirs))
}

/// 2026-09-26: Lexical `..`/`.` removal, with no symlink following: `..`
/// drops the previous component even when that component is a link.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}
