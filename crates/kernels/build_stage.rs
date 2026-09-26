// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Copy a target's resolved layers into OUT_DIR before compiling.
//!
//! Owner: metrale-kernels build.
//! Invariants:
//! - A file already staged in this run is never overwritten: every target
//!   that resolves a role directory gets the first copy of it.
//!
//! Included via `#[path = "build_stage.rs"] mod build_stage;`.
//!
//! The compiler resolves a quoted `#include` against the directory of the
//! file being compiled. A source that one directory reaches through
//! `[sources] use` must therefore compile as if it sat in that directory: the
//! headers beside it are that layer's, and `../../common/x.cu` from a model
//! directory is that target's common/, not the file's home. strix-hip's own
//! `common/prefill_paged_compute.cuh` defines `BR64` as 32 where gb10's
//! defines 64 on the non-SCALE path.
//!
//! So every build stages two directories per target, `common/` and
//! `<source model>/<quant>/`, the same depth apart as in `kernels/`, and
//! compiles from the copies. Targets that share a role directory share the
//! staged copy, so the (path, arch, flags) compile cache deduplicates across
//! them.
//!
//! build.rs wipes the stage before staging any target, so a copy of a header
//! that has since been removed cannot keep resolving.

use std::path::{Path, PathBuf};

use metrale_closure::layout::{Layout, Role};

/// 2026-09-25: Where one target's two roles were staged.
pub(crate) struct Staged {
    pub leaf: PathBuf,
    pub common: PathBuf,
}

impl Staged {
    /// 2026-09-25: The staged path of the module `stem`'s source. Panics if the layout has no such module.
    pub(crate) fn module_path(&self, layout: &Layout, stem: &str) -> PathBuf {
        let (_, entry) = layout
            .modules()
            .into_iter()
            .find(|(s, _)| s == stem)
            .unwrap_or_else(|| panic!("{}: no module {stem}", layout.target));
        let role = layout.layers[entry.layer].role;
        self.dir(role).join(&entry.name)
    }

    pub(crate) fn dir(&self, role: Role) -> &Path {
        match role {
            Role::Leaf => &self.leaf,
            Role::Common => &self.common,
        }
    }
}

/// 2026-09-25: Empty the stage. build.rs calls it once, before staging any target.
pub(crate) fn reset(stage_root: &Path) {
    if stage_root.exists() {
        std::fs::remove_dir_all(stage_root)
            .unwrap_or_else(|e| panic!("clear stage {}: {e}", stage_root.display()));
    }
}

/// 2026-09-25: Stage `layout`'s roles under `stage_root`: `common/` and
/// `<source model>/<quant>/`. A directory already staged in this run is
/// reused — every target that names it resolves the same files into it.
pub(crate) fn stage(stage_root: &Path, layout: &Layout) -> Staged {
    let common = stage_root.join("common");
    stage_role(&common, layout, Role::Common);
    let leaf = stage_root
        .join(&layout.source_model)
        .join(&layout.target.quant);
    stage_role(&leaf, layout, Role::Leaf);
    Staged { leaf, common }
}

fn stage_role(dir: &Path, layout: &Layout, role: Role) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let (entries, subdirs) = layout.role(role);
    for (name, entry) in entries {
        let dst = dir.join(name);
        if dst.exists() {
            continue;
        }
        std::fs::copy(&entry.source, &dst).unwrap_or_else(|e| {
            panic!("stage {} -> {}: {e}", entry.source.display(), dst.display())
        });
    }
    for (name, src) in subdirs {
        let dst = dir.join(name);
        if !dst.exists() {
            copy_tree(src, &dst);
        }
    }
}

fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap_or_else(|e| panic!("create {}: {e}", dst.display()));
    let entries = std::fs::read_dir(src).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
    for entry in entries.flatten() {
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_tree(&from, &to);
        } else {
            std::fs::copy(&from, &to)
                .unwrap_or_else(|e| panic!("stage {} -> {}: {e}", from.display(), to.display()));
        }
    }
}
