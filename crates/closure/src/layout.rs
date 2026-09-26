// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Which files a `(hardware, model, quant)` kernel target is
//! compiled from.
//!
//! The kernels build script (`crates/kernels/build_resolve.rs`) compiles what
//! it returns, the benchmark gate (`metrale-bench`'s `gate::taxon`) reads
//! its inputs, and `crates/kernels/tests/kernels_structure.rs` grades the
//! tree through it. `scripts/lib/kernel_layout.py` mirrors it for the Python
//! checkers, and `crates/kernels/tests/layout_mirror.rs` holds the two to the
//! same answer on the real tree.
//!
//! # Layers
//!
//! A target reads up to four directories, each a layer:
//!
//! | role   | tier   | directory                                  |
//! |--------|--------|--------------------------------------------|
//! | leaf   | own    | `kernels/<hw>/<source model>/<quant>/`     |
//! | leaf   | parent | `kernels/<parent>/<source model>/<quant>/` |
//! | common | own    | `kernels/<hw>/common/`                     |
//! | common | parent | `kernels/<parent>/common/`                 |
//!
//! `<parent>` is `[hardware] inherits` in the tree's `HARDWARE.toml`; a tree
//! that inherits nothing has two layers. `<source model>` is the model itself
//! unless its `MODEL.toml` redirects with `[model] kernel_source`. A layer's
//! entries are the kernel sources and headers its directory holds plus what
//! its `KERNEL.toml` `[sources] use` brings in.
//!
//! Within a role the own tier wins by file name; the leaf role wins over the
//! common role by source file stem. Every such win must be declared in
//! `[shadow]` of the winner's `KERNEL.toml`, with a reason: an undeclared one
//! is an error, and so is a declaration with nothing to shadow.
//!
//! Module identity is the file stem ([`Layout::modules`]), whichever layer
//! the file came from.
//!
//! # Materialisation
//!
//! Entries are compiled in their role directory, not where the file lives:
//! the compiler resolves a quoted `#include` against the including file's
//! directory, so a `use`d source must see the headers of the layer that uses
//! it. The build script stages each role into `OUT_DIR` before compiling
//! (`crates/kernels/build_stage.rs`); [`Layout::leaf`] and [`Layout::common`]
//! are those two directories' contents.
//!
//! Owner: metrale-closure (kernel layout).
//! Invariants:
//! - In a `Layout` that [`discover`] returns, every win of one layer over
//!   another is in `shadows` with its declared reason, and every `[shadow]`
//!   key names a win.
//! - No entry's file is a symlink.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub use crate::layout_manifest::{
    HEADER_EXTS, Hardware, KernelManifest, LayoutError, hardware, kernel_manifest,
    kernel_source_dir, source_ext, stem_of,
};
use crate::layout_scan::{has_ext, layer, layer_contents};

/// 2026-09-26: One compiled unit: `(hardware, model, quant)`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Target {
    pub hardware: String,
    pub model: String,
    pub quant: String,
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.hardware, self.model, self.quant)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    Leaf,
    Common,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Own,
    Parent,
}

/// 2026-09-26: One of a target's directories, whether or not it exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer {
    pub role: Role,
    pub tier: Tier,
    /// 2026-09-26: The `kernels/<hardware>` tree this directory is under,
    /// which its `[sources] use` paths are relative to.
    pub hardware: String,
    pub dir: PathBuf,
    pub manifest: Option<KernelManifest>,
}

/// 2026-09-26: One file a target compiles or stages beside what it compiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// 2026-09-26: File name in its role directory.
    pub name: String,
    /// 2026-09-26: The file it resolves to.
    pub source: PathBuf,
    /// 2026-09-26: Index into [`Layout::layers`].
    pub layer: usize,
    /// 2026-09-26: Brought in by `[sources] use` rather than held by the
    /// directory.
    pub used: bool,
}

/// 2026-09-26: A declared, verified shadow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shadow {
    pub role: Role,
    pub name: String,
    pub winner: PathBuf,
    pub loser: PathBuf,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct Layout {
    pub target: Target,
    pub hardware: Hardware,
    /// 2026-09-26: Own leaf, parent leaf, own common, parent common; the
    /// parent pair only when the tree inherits.
    pub layers: Vec<Layer>,
    /// 2026-09-26: Leaf-role files by name.
    pub leaf: BTreeMap<String, Entry>,
    /// 2026-09-26: Common-role files by name.
    pub common: BTreeMap<String, Entry>,
    /// 2026-09-26: Subdirectories of the role directories (e.g. vendored
    /// headers), by name.
    pub leaf_subdirs: BTreeMap<String, PathBuf>,
    pub common_subdirs: BTreeMap<String, PathBuf>,
    pub shadows: Vec<Shadow>,
    /// 2026-09-26: The model whose own-tier leaf directory this target
    /// reads, after redirect.
    pub source_model: String,
    /// 2026-09-26: `kernels/<hw>/<model>`, the target's own model directory.
    pub model_dir: PathBuf,
}

impl Layout {
    /// 2026-09-26: The `KERNEL.toml`s that exist, least specific first:
    /// parent common, own common, parent leaf, own leaf. The build script
    /// merges them in this order (`build_resolve.rs`).
    pub fn configs(&self) -> Vec<PathBuf> {
        let mut order: Vec<&Layer> = self.layers.iter().collect();
        order.sort_by_key(|l| (std::cmp::Reverse(l.role), std::cmp::Reverse(l.tier)));
        order
            .into_iter()
            .filter_map(|l| l.manifest.as_ref().map(|m| m.path.clone()))
            .collect()
    }

    /// 2026-09-26: `(stem, entry)` for every module source, sorted by stem:
    /// leaf over common.
    pub fn modules(&self) -> Vec<(String, &Entry)> {
        let ext = self.hardware.source_ext;
        let mut out: BTreeMap<String, &Entry> = BTreeMap::new();
        for map in [&self.common, &self.leaf] {
            for (name, entry) in map {
                if has_ext(name, ext) {
                    out.insert(stem_of(name).to_string(), entry);
                }
            }
        }
        out.into_iter().collect()
    }

    /// 2026-09-26: Module sources, sorted.
    pub fn sources(&self) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = self
            .modules()
            .into_iter()
            .map(|(_, e)| e.source.clone())
            .collect();
        out.sort();
        out
    }

    /// 2026-09-26: Every file the target's compile reads through this
    /// resolution: the role entries, every file under the role
    /// subdirectories, and the `KERNEL.toml`s. The build script watches them
    /// for rebuilds (`build_plan.rs`).
    pub fn inputs(&self) -> BTreeSet<PathBuf> {
        let mut out: BTreeSet<PathBuf> = self
            .leaf
            .values()
            .chain(self.common.values())
            .map(|e| e.source.clone())
            .collect();
        for dir in self
            .leaf_subdirs
            .values()
            .chain(self.common_subdirs.values())
        {
            files_under(dir, &mut out);
        }
        out.extend(self.configs());
        out
    }

    pub fn role(&self, role: Role) -> (&BTreeMap<String, Entry>, &BTreeMap<String, PathBuf>) {
        match role {
            Role::Leaf => (&self.leaf, &self.leaf_subdirs),
            Role::Common => (&self.common, &self.common_subdirs),
        }
    }
}

fn files_under(dir: &Path, out: &mut BTreeSet<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files_under(&path, out);
        } else {
            out.insert(path);
        }
    }
}

/// 2026-09-26: Every symlink under `kernels/`, sorted. The structure test
/// (`kernels_structure.rs`) refuses any.
pub fn symlinks(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn walk_dir(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_symlink() {
                out.push(path);
            } else if path.is_dir() {
                walk_dir(&path, out);
            }
        }
    }
    walk_dir(&root.join("kernels"), &mut out);
    out.sort();
    out
}

fn subdirs(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .collect();
    out.sort();
    out
}

/// 2026-09-26: The leaf model directory of `model` in the tree `hw_dir`,
/// after that tree's own redirect, or where it would be when the tree has no
/// such model.
fn leaf_model_dir(hw_dir: &Path, model: &str) -> Result<PathBuf, LayoutError> {
    let dir = hw_dir.join(model);
    if dir.join("MODEL.toml").is_file() {
        kernel_source_dir(&dir)
    } else {
        Ok(dir)
    }
}

/// 2026-09-26: Every target in the tree: each `kernels/<hw>` with a
/// `HARDWARE.toml`, each model directory under it with a `MODEL.toml`, and
/// every quant directory its source model owns in the tree or in the tree it
/// inherits.
pub fn walk(root: &Path) -> Result<Vec<Target>, LayoutError> {
    let kernels = root.join("kernels");
    let mut out = Vec::new();
    for hw in subdirs(&kernels) {
        let hw_dir = kernels.join(&hw);
        if !hw_dir.join("HARDWARE.toml").is_file() {
            continue;
        }
        let hardware = hardware(&kernels, &hw)?;
        for model in subdirs(&hw_dir) {
            let model_dir = hw_dir.join(&model);
            if !model_dir.join("MODEL.toml").is_file() {
                continue;
            }
            let own_src = kernel_source_dir(&model_dir)?;
            let mut quants: BTreeSet<String> = subdirs(&own_src).into_iter().collect();
            if let Some(parent) = &hardware.inherits {
                let src_model = own_src
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&model);
                quants.extend(subdirs(&leaf_model_dir(&kernels.join(parent), src_model)?));
            }
            for quant in quants {
                out.push(Target {
                    hardware: hw.clone(),
                    model: model.clone(),
                    quant,
                });
            }
        }
    }
    Ok(out)
}

/// 2026-09-26: Resolve one target.
pub fn discover(root: &Path, target: &Target) -> Result<Layout, LayoutError> {
    let kernels = root.join("kernels");
    let hardware = hardware(&kernels, &target.hardware)?;
    let hw_dir = kernels.join(&target.hardware);
    let model_dir = hw_dir.join(&target.model);
    if !model_dir.is_dir() {
        return Err(LayoutError::UnknownModel(model_dir));
    }
    let own_src = kernel_source_dir(&model_dir)?;
    let source_model = own_src
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&target.model)
        .to_string();

    let mut layers = vec![layer(
        Role::Leaf,
        Tier::Own,
        &target.hardware,
        own_src.join(&target.quant),
    )?];
    if let Some(parent) = &hardware.inherits {
        let parent_dir = kernels.join(parent);
        let parent_src = leaf_model_dir(&parent_dir, &source_model)?;
        layers.push(layer(
            Role::Leaf,
            Tier::Parent,
            parent,
            parent_src.join(&target.quant),
        )?);
    }
    layers.push(layer(
        Role::Common,
        Tier::Own,
        &target.hardware,
        hw_dir.join("common"),
    )?);
    if let Some(parent) = &hardware.inherits {
        layers.push(layer(
            Role::Common,
            Tier::Parent,
            parent,
            kernels.join(parent).join("common"),
        )?);
    }
    if !layers.iter().any(|l| l.dir.is_dir()) {
        return Err(LayoutError::NoKernelDirectory(target.to_string()));
    }

    let mut leaf = BTreeMap::new();
    let mut common = BTreeMap::new();
    let mut leaf_subdirs = BTreeMap::new();
    let mut common_subdirs = BTreeMap::new();
    let mut shadows = Vec::new();
    let mut declared_used: BTreeSet<(usize, String)> = BTreeSet::new();
    for (idx, l) in layers.iter().enumerate() {
        let (entries, dirs) = match l.role {
            Role::Leaf => (&mut leaf, &mut leaf_subdirs),
            Role::Common => (&mut common, &mut common_subdirs),
        };
        let (own_entries, own_dirs) = layer_contents(&kernels, l, idx, hardware.source_ext)?;
        for (name, entry) in own_entries {
            match entries.get(&name) {
                None => {
                    entries.insert(name, entry);
                }
                // 2026-09-26: A lower tier's namesake: the earlier (own)
                // entry wins, and that win is a shadow the winner must have
                // declared.
                Some(winner) => {
                    let winner = winner.clone();
                    let shadow = declare(&layers, &winner, &entry.source, &mut declared_used)?;
                    shadows.push(shadow);
                }
            }
        }
        for (name, dir) in own_dirs {
            dirs.entry(name).or_insert(dir);
        }
    }
    // 2026-09-26: Leaf over common, by stem, sources only.
    for (name, entry) in &leaf {
        if !has_ext(name, hardware.source_ext) {
            continue;
        }
        let stem = stem_of(name);
        let Some((_, loser)) = common
            .iter()
            .find(|(n, _)| has_ext(n, hardware.source_ext) && stem_of(n) == stem)
        else {
            continue;
        };
        shadows.push(declare(&layers, entry, &loser.source, &mut declared_used)?);
    }
    for (idx, l) in layers.iter().enumerate() {
        if let Some(m) = &l.manifest {
            for stem in m.shadow.keys() {
                if !declared_used.contains(&(idx, stem.clone())) {
                    return Err(LayoutError::DeadShadow {
                        manifest: m.path.clone(),
                        stem: stem.clone(),
                    });
                }
            }
        }
    }
    shadows.sort_by(|a, b| (a.role, &a.name, &a.loser).cmp(&(b.role, &b.name, &b.loser)));
    Ok(Layout {
        target: target.clone(),
        hardware,
        layers,
        leaf,
        common,
        leaf_subdirs,
        common_subdirs,
        shadows,
        source_model,
        model_dir,
    })
}

/// 2026-09-26: A winner over `loser` is a shadow the winner's manifest must
/// declare.
fn declare(
    layers: &[Layer],
    winner: &Entry,
    loser: &Path,
    used: &mut BTreeSet<(usize, String)>,
) -> Result<Shadow, LayoutError> {
    if winner.source == loser {
        return Err(LayoutError::SelfShadow {
            name: winner.name.clone(),
            source: winner.source.clone(),
        });
    }
    let l = &layers[winner.layer];
    let stem = stem_of(&winner.name).to_string();
    let manifest_path = l.dir.join("KERNEL.toml");
    let reason = l
        .manifest
        .as_ref()
        .and_then(|m| m.shadow.get(&stem))
        .ok_or_else(|| LayoutError::UndeclaredShadow {
            name: winner.name.clone(),
            winner: winner.source.clone(),
            loser: loser.to_path_buf(),
            manifest: manifest_path,
        })?;
    used.insert((winner.layer, stem));
    Ok(Shadow {
        role: l.role,
        name: winner.name.clone(),
        winner: winner.source.clone(),
        loser: loser.to_path_buf(),
        reason: reason.clone(),
    })
}

#[cfg(test)]
#[path = "layout_tests.rs"]
mod layout_tests;
