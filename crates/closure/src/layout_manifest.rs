// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The manifest files [`super::layout`] reads, and the errors it
//! reports.
//!
//! Three manifests steer resolution:
//!
//! * `kernels/<hw>/HARDWARE.toml`: `[hardware] vendor` (default `nvidia`)
//!   picks the source extension, `[hardware] inherits = "<hw>"` layers this
//!   tree over another.
//! * `kernels/<hw>/<model>/MODEL.toml`: `[model] kernel_source = "<model>"`
//!   redirects the per-quant kernel directories to another model's.
//! * `KERNEL.toml` in a leaf or `common/` directory: `[sources] use = [...]`
//!   adds files from elsewhere in `kernels/` to that directory's layer, and
//!   `[shadow] <stem> = "<reason>"` declares which of the directory's entries
//!   replace a same-name entry in a lower layer.
//!
//! Other keys in those files belong to the build script and are not
//! interpreted here, except that `[hardware] arch` is carried verbatim, a
//! `[kernels] overrides` table in `HARDWARE.toml` is refused, and so is a
//! `[sources]` key other than `use`.
//!
//! Owner: metrale-closure (kernel layout).
//! Invariants:
//! - A `Hardware` that [`hardware`] returns has a known source extension, and
//!   its parent, if any, exists, is not itself, inherits nothing and has the
//!   same extension.
//! - A `[shadow]` reason that [`kernel_manifest`] returns is not blank.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// 2026-09-26: `kernels/<hw>` has no `HARDWARE.toml`.
    UnknownHardware(String),
    /// 2026-09-26: A manifest did not parse, or holds a key of the wrong
    /// shape.
    Manifest { path: PathBuf, message: String },
    /// 2026-09-26: `[hardware] vendor` names no known source extension.
    UnknownVendor { hardware: String, vendor: String },
    /// 2026-09-26: `[hardware] inherits` names itself, a missing tree, a tree
    /// that itself inherits (chains are refused, as `kernel_source` chains
    /// are), or a tree with another source extension.
    Inherits { hardware: String, message: String },
    /// 2026-09-26: `[model] kernel_source` names no model directory, or one
    /// that redirects.
    Redirect { model_dir: PathBuf, message: String },
    /// 2026-09-26: The model directory the target names does not exist.
    UnknownModel(PathBuf),
    /// 2026-09-26: No layer directory exists at all for the target.
    NoKernelDirectory(String),
    /// 2026-09-26: A symlink inside a consulted directory, or a
    /// `[sources] use` path that is one. Files are shared with
    /// `[sources] use`, not symlinks.
    Symlink(PathBuf),
    /// 2026-09-26: A `[sources] use` entry that resolves to nothing usable.
    Use {
        manifest: PathBuf,
        entry: String,
        message: String,
    },
    /// 2026-09-26: Two entries in one layer share a name (a `use` naming a
    /// file the directory already holds, or two `use`s of one name).
    UseCollides {
        manifest: PathBuf,
        name: String,
        existing: PathBuf,
    },
    /// 2026-09-26: One name resolves in two layers and the winner's
    /// `KERNEL.toml` does not declare it in `[shadow]`.
    UndeclaredShadow {
        name: String,
        winner: PathBuf,
        loser: PathBuf,
        manifest: PathBuf,
    },
    /// 2026-09-26: A `[shadow]` key whose entry shadows nothing.
    DeadShadow { manifest: PathBuf, stem: String },
    /// 2026-09-26: A winner that is the very file it would shadow, reached
    /// twice.
    SelfShadow { name: String, source: PathBuf },
    /// 2026-09-26: `HARDWARE.toml [kernels] overrides`, which is refused: a
    /// replacement is declared in `[shadow]` of the directory's `KERNEL.toml`.
    RetiredKey { path: PathBuf, key: String },
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownHardware(hw) => write!(f, "kernels/{hw}: no HARDWARE.toml"),
            Self::Manifest { path, message } => write!(f, "{}: {message}", path.display()),
            Self::UnknownVendor { hardware, vendor } => write!(
                f,
                "kernels/{hardware}/HARDWARE.toml: vendor {vendor:?} has no kernel source extension"
            ),
            Self::Inherits { hardware, message } => {
                write!(
                    f,
                    "kernels/{hardware}/HARDWARE.toml: [hardware] inherits: {message}"
                )
            }
            Self::Redirect { model_dir, message } => {
                write!(
                    f,
                    "{}/MODEL.toml: [model] kernel_source: {message}",
                    model_dir.display()
                )
            }
            Self::UnknownModel(dir) => write!(f, "{}: no such model directory", dir.display()),
            Self::NoKernelDirectory(t) => write!(f, "{t}: no kernel directory in any layer"),
            Self::Symlink(p) => write!(
                f,
                "{}: is a symlink — symlinks are not a sharing mechanism under kernels/; \
                 declare the file in [sources] use of the KERNEL.toml beside it",
                p.display()
            ),
            Self::Use {
                manifest,
                entry,
                message,
            } => {
                write!(
                    f,
                    "{}: [sources] use {entry:?}: {message}",
                    manifest.display()
                )
            }
            Self::UseCollides {
                manifest,
                name,
                existing,
            } => write!(
                f,
                "{}: [sources] use brings {name} but this layer already holds {}",
                manifest.display(),
                existing.display()
            ),
            Self::UndeclaredShadow {
                name,
                winner,
                loser,
                manifest,
            } => write!(
                f,
                "{} shadows {} and {} does not declare it: add `[shadow] {} = \"<reason>\"`",
                winner.display(),
                loser.display(),
                manifest.display(),
                stem_of(name)
            ),
            Self::DeadShadow { manifest, stem } => write!(
                f,
                "{}: [shadow] {stem} is declared but this directory's {stem} shadows nothing",
                manifest.display()
            ),
            Self::SelfShadow { name, source } => write!(
                f,
                "{name} resolves to {} in two layers — a use of a file the target already reaches",
                source.display()
            ),
            Self::RetiredKey { path, key } => write!(
                f,
                "{}: `{key}` is retired; an overlay's own files need no inventory, and one that \
                 replaces a parent's file declares it in [shadow] of its KERNEL.toml",
                path.display()
            ),
        }
    }
}

impl std::error::Error for LayoutError {}

/// 2026-09-26: The stem of a kernel-source file name: `w4a16_gemm.cu` ->
/// `w4a16_gemm`.
pub fn stem_of(name: &str) -> &str {
    name.rsplit_once('.').map_or(name, |(stem, _)| stem)
}

/// 2026-09-26: Kernel-source extension for a `[hardware] vendor`, or `None`
/// for an unknown vendor, which [`hardware`] refuses.
pub fn source_ext(vendor: &str) -> Option<&'static str> {
    match vendor {
        "nvidia" | "cuda" | "amd" | "rocm" | "scale" | "hip" => Some("cu"),
        "apple" | "metal" => Some("metal"),
        _ => None,
    }
}

/// 2026-09-26: Header extensions that are entries of every layer beside the
/// sources, so a source's headers are staged with it.
pub const HEADER_EXTS: &[&str] = &["cuh", "h"];

/// 2026-09-26: `kernels/<hw>/HARDWARE.toml`, the part resolution reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hardware {
    pub name: String,
    pub vendor: String,
    /// 2026-09-26: `[hardware] arch`, verbatim, when present.
    pub arch: Option<String>,
    pub inherits: Option<String>,
    pub source_ext: &'static str,
}

fn read_toml(path: &Path) -> Result<toml::Value, LayoutError> {
    let text = std::fs::read_to_string(path).map_err(|e| LayoutError::Manifest {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    toml::from_str(&text).map_err(|e| LayoutError::Manifest {
        path: path.to_path_buf(),
        message: format!("bad TOML: {e}"),
    })
}

fn string_at(value: &toml::Value, path: &Path, key: &str) -> Result<Option<String>, LayoutError> {
    let (table, field) = key.split_once('.').unwrap_or(("", key));
    let holder = if table.is_empty() {
        Some(value)
    } else {
        value.get(table)
    };
    match holder.and_then(|t| t.get(field)) {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_string()))
            .ok_or_else(|| LayoutError::Manifest {
                path: path.to_path_buf(),
                message: format!("{key} must be a string"),
            }),
    }
}

/// 2026-09-26: Read one `HARDWARE.toml` without validating `inherits`;
/// [`hardware`] does.
fn hardware_raw(kernels: &Path, hw: &str) -> Result<Hardware, LayoutError> {
    let path = kernels.join(hw).join("HARDWARE.toml");
    if !path.is_file() {
        return Err(LayoutError::UnknownHardware(hw.to_string()));
    }
    let value = read_toml(&path)?;
    if value
        .get("kernels")
        .and_then(|k| k.get("overrides"))
        .is_some()
    {
        return Err(LayoutError::RetiredKey {
            path,
            key: "[kernels] overrides".into(),
        });
    }
    let vendor = string_at(&value, &path, "hardware.vendor")?.unwrap_or_else(|| "nvidia".into());
    let source_ext = source_ext(&vendor).ok_or_else(|| LayoutError::UnknownVendor {
        hardware: hw.to_string(),
        vendor: vendor.clone(),
    })?;
    Ok(Hardware {
        name: hw.to_string(),
        vendor,
        arch: string_at(&value, &path, "hardware.arch")?,
        inherits: string_at(&value, &path, "hardware.inherits")?,
        source_ext,
    })
}

/// 2026-09-26: `kernels/<hw>/HARDWARE.toml`, with `inherits` checked: the
/// parent must exist, must not be the tree itself, must not inherit in turn,
/// and must compile the same source extension.
pub fn hardware(kernels: &Path, hw: &str) -> Result<Hardware, LayoutError> {
    let own = hardware_raw(kernels, hw)?;
    if let Some(parent) = &own.inherits {
        let fail = |message: String| LayoutError::Inherits {
            hardware: hw.to_string(),
            message,
        };
        if parent == hw {
            return Err(fail("a tree cannot inherit itself".into()));
        }
        let parent_hw =
            hardware_raw(kernels, parent).map_err(|e| fail(format!("parent {parent:?}: {e}")))?;
        if let Some(grand) = parent_hw.inherits {
            return Err(fail(format!(
                "{parent:?} itself inherits {grand:?} — chains are refused; inherit the tree that owns the sources"
            )));
        }
        if parent_hw.source_ext != own.source_ext {
            return Err(fail(format!(
                "{parent:?} compiles .{} sources, this tree .{}",
                parent_hw.source_ext, own.source_ext
            )));
        }
    }
    Ok(own)
}

/// 2026-09-26: `[model] kernel_source` of `<model_dir>/MODEL.toml`,
/// validated: the referent is a sibling model directory holding a MODEL.toml
/// that does not redirect. Returns the source model directory (the model's
/// own when there is no redirect). A model directory with no MODEL.toml owns
/// its own sources.
pub fn kernel_source_dir(model_dir: &Path) -> Result<PathBuf, LayoutError> {
    let Some(src) = kernel_source(model_dir)? else {
        return Ok(model_dir.to_path_buf());
    };
    let fail = |message: String| LayoutError::Redirect {
        model_dir: model_dir.to_path_buf(),
        message,
    };
    let hw_dir = model_dir
        .parent()
        .ok_or_else(|| fail("model dir has no parent".into()))?;
    let src_dir = hw_dir.join(&src);
    if !src_dir.is_dir() || !src_dir.join("MODEL.toml").is_file() {
        return Err(fail(format!(
            "{src:?} does not name a kernel target directory under {}",
            hw_dir.display()
        )));
    }
    if kernel_source(&src_dir)?.is_some() {
        return Err(fail(format!(
            "{src:?} itself redirects — chains are not allowed; point at the target that owns the sources"
        )));
    }
    Ok(src_dir)
}

fn kernel_source(model_dir: &Path) -> Result<Option<String>, LayoutError> {
    let path = model_dir.join("MODEL.toml");
    if !path.is_file() {
        return Ok(None);
    }
    let value = read_toml(&path)?;
    let src = string_at(&value, &path, "model.kernel_source")?;
    if let Some(s) = &src
        && s.trim().is_empty()
    {
        return Err(LayoutError::Manifest {
            path,
            message: "[model] kernel_source must name a kernel target directory".into(),
        });
    }
    Ok(src)
}

/// 2026-09-26: The resolution-relevant part of one `KERNEL.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KernelManifest {
    pub path: PathBuf,
    /// 2026-09-26: `[sources] use`, verbatim, in file order.
    pub uses: Vec<String>,
    /// 2026-09-26: `[shadow] <stem> = "<reason>"`.
    pub shadow: BTreeMap<String, String>,
}

/// 2026-09-26: Parse `<dir>/KERNEL.toml` if it exists.
pub fn kernel_manifest(dir: &Path) -> Result<Option<KernelManifest>, LayoutError> {
    let path = dir.join("KERNEL.toml");
    if !path.is_file() {
        return Ok(None);
    }
    let value = read_toml(&path)?;
    let fail = |message: String| LayoutError::Manifest {
        path: path.clone(),
        message,
    };
    let mut uses = Vec::new();
    if let Some(sources) = value.get("sources") {
        let table = sources
            .as_table()
            .ok_or_else(|| fail("[sources] must be a table".into()))?;
        for key in table.keys() {
            if key != "use" {
                return Err(fail(format!("[sources] has no key `{key}`; only `use`")));
            }
        }
        if let Some(list) = table.get("use") {
            let arr = list
                .as_array()
                .ok_or_else(|| fail("[sources] use must be an array of paths".into()))?;
            for v in arr {
                let s = v
                    .as_str()
                    .ok_or_else(|| fail("[sources] use entries must be strings".into()))?;
                uses.push(s.to_string());
            }
        }
    }
    let mut shadow = BTreeMap::new();
    if let Some(decl) = value.get("shadow") {
        let table = decl
            .as_table()
            .ok_or_else(|| fail("[shadow] must be a table of `stem = \"reason\"`".into()))?;
        for (stem, reason) in table {
            let reason = reason
                .as_str()
                .ok_or_else(|| fail(format!("[shadow] {stem} must be a reason string")))?;
            if reason.trim().is_empty() {
                return Err(fail(format!("[shadow] {stem} needs a stated reason")));
            }
            shadow.insert(stem.clone(), reason.to_string());
        }
    }
    Ok(Some(KernelManifest { path, uses, shadow }))
}
