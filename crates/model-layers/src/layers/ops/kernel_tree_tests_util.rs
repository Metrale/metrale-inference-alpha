// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Every kernel source the build compiles, at each place it compiles it, for the source-contract tests.
//!
//! Owner: model-layers ops (test support).
//! Invariants: none beyond the types.
//!
//! A test that checks "every copy" of a kernel must see every place the build compiles it, not
//! only the regular files under `kernels/`. A gb10 source reaches hopper and b200 through
//! `inherits = "gb10"` in their HARDWARE.toml, and the strix trees through `[sources] use` in
//! their KERNEL.toml. The places are resolved by `metrale_closure::layout`, the resolver
//! crates/kernels/build.rs uses.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use metrale_closure::layout::{Role, Tier, discover, walk};

/// 2026-09-25: One kernel source at one place the build compiles it.
#[derive(Debug)]
pub(crate) struct KernelFile {
    /// 2026-09-25: `kernels/<hw>/common/<name>` or `kernels/<hw>/<model>/<quant>/<name>`: the
    /// target's own directory for the role the file serves.
    pub path: PathBuf,
    /// 2026-09-25: The regular file that holds the bytes.
    pub source: PathBuf,
    /// 2026-09-25: `path` is that regular file: the directory holds it itself rather than reaching
    /// it through `[sources] use` or `inherits`.
    pub held: bool,
}

/// 2026-09-25: Every `.cu` the build compiles, one entry per location (common and leaf directory
/// of every target that `walk` finds), sorted by `path`. The first entry for a path wins.
pub(crate) fn compiled_cu_files() -> Vec<KernelFile> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/model-layers is two levels below the workspace root");
    let mut out: BTreeMap<PathBuf, KernelFile> = BTreeMap::new();
    let targets = walk(root).unwrap_or_else(|e| panic!("kernels/ does not resolve: {e}"));
    for target in &targets {
        let layout = discover(root, target).unwrap_or_else(|e| panic!("{target}: {e}"));
        for role in [Role::Leaf, Role::Common] {
            let (entries, _) = layout.role(role);
            let dir = &layout
                .layers
                .iter()
                .find(|l| l.role == role && l.tier == Tier::Own)
                .expect("every layout has an own layer per role")
                .dir;
            for (name, entry) in entries {
                if !name.ends_with(".cu") {
                    continue;
                }
                let path = dir.join(name);
                let held = !entry.used && layout.layers[entry.layer].tier == Tier::Own;
                out.entry(path.clone()).or_insert(KernelFile {
                    path,
                    source: entry.source.clone(),
                    held,
                });
            }
        }
    }
    out.into_values().collect()
}
