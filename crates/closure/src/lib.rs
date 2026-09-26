// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: One SHA-256 over everything a kernel target's device code is
//! compiled from: the transitive quoted-`#include` closure of its sources, the
//! manifests, the nvcc flags, the arch string and the compiler string.
//!
//! Owner: metrale-closure (shared by the kernels build script and the bench gate).
//! Invariants:
//! - A path under the canonicalised `root` enters the digest relative to it, so
//!   two checkouts of one commit agree; a path outside `root` enters as given.
//! - Every file in the closure is read, and any read error returns `Err`; there
//!   is no partial digest.
//! - A quoted include that names no file on disk is not an error: it is kept in
//!   [`Closure::unresolved`] and hashed by name, not by content.
//!
//! The closure, not the resolved file set, is hashed because a leaf file can
//! `#include` the common file it shadows (the qwen3.6-27b nvfp4
//! `attn_prefill_paged_indirect.cu` includes its `common/` namesake), and
//! because headers are not in `layout::Layout::sources`, which lists one
//! source per module.
//!
//! Not covered:
//! - angle-bracket includes (only the compiler string stands for them);
//! - includes found through a search path or in another staged layer: an
//!   include is resolved only against the including file's own directory;
//! - preprocessor conditionals, which are not evaluated, so an include in a
//!   dead branch is still walked;
//! - host code under `crates/`: the gate's `excuses` refuses any non-kernel
//!   path before it consults this hash;
//! - anything not passed in [`ClosureInputs`].

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub mod layout;
mod layout_manifest;
mod layout_scan;

use sha2::{Digest, Sha256};

/// 2026-09-26: Version of the hash definition. It is the first value fed into
/// the digest after the domain tag, so changing it changes every digest.
/// Change it whenever what is fed into the digest changes (an input, an order,
/// a separator), or old and new records compare equal while meaning different
/// things.
pub const CLOSURE_SCHEMA: u32 = 2;

#[derive(Debug)]
pub enum ClosureError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl std::fmt::Display for ClosureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "reading {}: {source}", path.display()),
        }
    }
}

/// 2026-09-26: A computed closure: the digest, plus the includes that named no
/// file.
#[derive(Debug, Clone)]
pub struct Closure {
    /// 2026-09-26: Lower-case hex SHA-256.
    pub digest: String,
    /// 2026-09-26: Quoted includes naming a file that is not on disk, as
    /// `including-file -> include`, the including file relative to `root`.
    pub unresolved: BTreeSet<String>,
}

impl std::error::Error for ClosureError {}

type Result<T> = std::result::Result<T, ClosureError>;

/// 2026-09-26: The inputs of one target's hash.
#[derive(Debug, Clone)]
pub struct ClosureInputs {
    /// 2026-09-26: Resolved sources, in any order: the closure is a sorted set.
    pub sources: Vec<PathBuf>,
    /// 2026-09-26: Files hashed by content but not walked for includes. The
    /// kernels build passes HARDWARE.toml, the layers' KERNEL.tomls and
    /// MODEL.toml.
    pub configs: Vec<PathBuf>,
    /// 2026-09-26: Compiler flags, hashed in the given order.
    pub flags: Vec<String>,
    /// 2026-09-26: `[hardware] arch` of HARDWARE.toml, e.g. `"sm_121f"`.
    pub arch: String,
    /// 2026-09-26: Compiler identification. The kernels build passes the last
    /// non-empty line of `nvcc --version`.
    pub compiler: String,
}

/// 2026-09-26: The digest of [`hash_with_report`], without the unresolved set.
pub fn hash(root: &Path, inputs: &ClosureInputs) -> Result<String> {
    hash_with_report(root, inputs).map(|c| c.digest)
}

/// 2026-09-26: Hash a target's closure and report the quoted includes that
/// named no file on disk.
///
/// An unresolvable include is recorded, not fatal. Its name still reaches the
/// digest twice: through the including file's bytes, and through the
/// unresolved set, hashed under its own label. What stays uncovered is the
/// content of a header the compiler finds some other way. The kernels build
/// prints each unresolved entry as a `cargo:warning`.
///
/// # Errors
/// [`ClosureError::Io`] when a source, an included file or a config cannot be
/// read.
pub fn hash_with_report(root: &Path, inputs: &ClosureInputs) -> Result<Closure> {
    // 2026-09-26: `expand` canonicalises every source, so the root is
    // canonicalised too; otherwise an alias such as macOS `/var` ->
    // `/private/var` makes `strip_prefix` fail and puts absolute paths into the
    // report and the digest.
    let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut closure: BTreeSet<PathBuf> = BTreeSet::new();
    let mut raw: BTreeSet<(PathBuf, String)> = BTreeSet::new();
    for src in &inputs.sources {
        expand(src, &mut closure, &mut raw)?;
    }
    let unresolved: BTreeSet<String> = raw
        .into_iter()
        .map(|(from, include)| {
            let rel = from.strip_prefix(&canonical_root).unwrap_or(&from);
            format!("{} -> {include}", rel.display())
        })
        .collect();
    for cfg in &inputs.configs {
        closure.insert(cfg.canonicalize().unwrap_or_else(|_| cfg.clone()));
    }

    let mut digest = Sha256::new();
    // 2026-09-26: Domain separation: a tag, the schema and a label before each
    // field, so the same bytes in a different field give a different digest.
    digest.update(b"metrale-closure\x00");
    digest.update(CLOSURE_SCHEMA.to_le_bytes());
    digest.update(b"\x00arch\x00");
    digest.update(inputs.arch.as_bytes());
    digest.update(b"\x00compiler\x00");
    digest.update(inputs.compiler.as_bytes());
    digest.update(b"\x00flags\x00");
    for flag in &inputs.flags {
        digest.update(flag.as_bytes());
        digest.update(b"\x1f");
    }

    digest.update(b"\x00files\x00");
    for path in &closure {
        let rel = path.strip_prefix(&canonical_root).unwrap_or(path);
        // 2026-09-26: The path is hashed as well as the bytes: the same content
        // under another stem shadows a different module.
        digest.update(rel.to_string_lossy().as_bytes());
        digest.update(b"\x1f");
        let bytes = std::fs::read(path).map_err(|source| ClosureError::Io {
            path: path.clone(),
            source,
        })?;
        digest.update(bytes.len().to_le_bytes());
        digest.update(&bytes);
        digest.update(b"\x1e");
    }

    // 2026-09-26: Under its own label, so an include that becomes resolvable
    // or unresolvable moves the digest even when no file's bytes change.
    digest.update(b"\x00unresolved\x00");
    for entry in &unresolved {
        digest.update(entry.as_bytes());
        digest.update(b"\x1f");
    }

    Ok(Closure {
        digest: format!("{:x}", digest.finalize()),
        unresolved,
    })
}

/// 2026-09-26: Add `file` and everything it quoted-includes, transitively,
/// resolving each include against the including file's directory.
///
/// `out` is also the cycle guard: a file already inserted is not walked again,
/// so mutually including headers terminate.
fn expand(
    file: &Path,
    out: &mut BTreeSet<PathBuf>,
    unresolved: &mut BTreeSet<(PathBuf, String)>,
) -> Result<()> {
    let canonical = file.canonicalize().unwrap_or_else(|_| file.to_path_buf());
    if !out.insert(canonical.clone()) {
        return Ok(());
    }
    let text = std::fs::read_to_string(&canonical).map_err(|source| ClosureError::Io {
        path: canonical.clone(),
        source,
    })?;
    let dir = canonical.parent().unwrap_or(Path::new("."));
    for include in quoted_includes(&text) {
        let target = dir.join(&include);
        if !target.exists() {
            // 2026-09-26: Keyed by the including file, so two files naming the
            // same missing header are two entries.
            unresolved.insert((canonical.clone(), include));
            continue;
        }
        expand(&target, out, unresolved)?;
    }
    Ok(())
}

/// 2026-09-26: Quoted include paths, in source order.
///
/// Angle-bracket includes are skipped. So is a line that starts with `//`,
/// because it is not compiled. Block comments are not tracked: an include
/// inside `/* … */` is still followed, which over-includes.
fn quoted_includes(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        let Some(rest) = trimmed.strip_prefix("#include") else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix('"') else {
            continue;
        };
        if let Some(end) = rest.find('"') {
            found.push(rest[..end].to_string());
        }
    }
    found
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod lib_tests;
