// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The gate's view of the kernel taxonomy
//! `kernels/<hardware>/<model>/<quant>/`: target enumeration, sources and
//! configs, path-to-node helpers, and the set of targets a change affects.
//!
//! What a target compiles comes from `metrale_closure::layout`, the resolver
//! the kernels build script also uses; that crate has no CUDA dependency.
//!
//! Owner: bench gate.
//! Invariants:
//! - A target that does not resolve is never reported unaffected: [`sources`]
//!   returns `None` for it, and [`affected`] includes it for any path under
//!   `kernels/`.
//! - [`walk`] panics on a tree the resolver refuses; [`resolves`] is the
//!   non-panicking check.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub use metrale_closure::layout::Target;
use metrale_closure::layout::{discover, walk as walk_tree};

/// 2026-09-26: Every target in the tree, as the resolver enumerates it: a
/// hardware dir holds `HARDWARE.toml`, a model dir holds `MODEL.toml` (no
/// `common/` does), and the quants are the subdirectories of the model's
/// resolved kernel source directory, here or in the tree it inherits.
///
/// Panics when the resolver refuses the tree (for example a manifest that
/// does not parse, or a `kernel_source` redirect chain).
pub fn walk(root: &Path) -> Vec<Target> {
    walk_tree(root).unwrap_or_else(|e| panic!("kernels/ does not resolve: {e}"))
}

/// 2026-09-26: Whether the resolver can enumerate the tree, without the
/// panic of [`walk`].
pub fn resolves(root: &Path) -> bool {
    walk_tree(root).is_ok()
}

/// 2026-09-26: Kernel module sources for `target`, after shadowing, sorted.
/// `None` when the target does not resolve or resolves to no sources; callers
/// treat that as affected.
pub fn sources(root: &Path, target: &Target) -> Option<Vec<PathBuf>> {
    let layout = discover(root, target).ok()?;
    let sources = layout.sources();
    // 2026-09-26: No sources is reported like a broken target: the kernels
    // build script attests no closure hash for such a target either.
    if sources.is_empty() {
        return None;
    }
    Some(sources)
}

/// 2026-09-26: Config files that steer a target's compile without being
/// sources, in the order `closure_attestation` in the kernels build script
/// hashes them: `HARDWARE.toml`, every `KERNEL.toml` the resolution read
/// (least specific first), the target's own `MODEL.toml`. Files that do not
/// exist are left out. When the target does not resolve, the list is the
/// hardware's and the model's manifests only.
pub fn configs(root: &Path, target: &Target) -> Vec<PathBuf> {
    let hw_dir = root.join("kernels").join(&target.hardware);
    let mut out = vec![hw_dir.join("HARDWARE.toml")];
    match discover(root, target) {
        Ok(layout) => {
            out.extend(layout.configs());
            out.push(layout.model_dir.join("MODEL.toml"));
        }
        Err(_) => out.push(hw_dir.join(&target.model).join("MODEL.toml")),
    }
    out.into_iter().filter(|p| p.exists()).collect()
}

/// 2026-09-26: The hardware node a repo-relative path sits under: the first
/// component after `kernels/`.
pub fn hardware_of(path: &str) -> Option<&str> {
    path.strip_prefix("kernels/")?.split('/').next()
}

/// 2026-09-26: The `(hardware, model)` node a path sits under. `None` for a
/// path outside `kernels/`, a file directly in a hardware dir, and anything
/// under `common/`.
pub fn model_of(path: &str) -> Option<(&str, &str)> {
    let rest = path.strip_prefix("kernels/")?;
    let mut parts = rest.split('/');
    let hw = parts.next()?;
    let model = parts.next()?;
    if model == "common" || parts.next().is_none() {
        return None;
    }
    Some((hw, model))
}

/// 2026-09-26: Targets a changed path set can affect: the union of two rules.
///
/// * By path. A path under a model directory affects the targets on that
///   hardware for that model, those whose source model it is, and those that
///   do not resolve. Any other path under `kernels/<hw>/` (`common/`,
///   `HARDWARE.toml`) affects every target on that hardware.
/// * By resolution. Every target, on any hardware, whose resolved inputs
///   (`Layout::inputs`: role entries, vendored subdirectories, `KERNEL.toml`s)
///   include the path, and every target that does not resolve. This reaches
///   targets on a hardware that `inherits` the tree, or that name the file in
///   `[sources] use`.
///
/// A path outside `kernels/` affects nothing here. `closure::excuses` refuses
/// to excuse a path set that holds one.
pub fn affected(root: &Path, changed: &[String]) -> BTreeSet<Target> {
    let all = walk(root);
    let mut out = BTreeSet::new();
    let mut resolved: Vec<(Target, BTreeSet<PathBuf>)> = Vec::new();
    for t in &all {
        match discover(root, t) {
            Ok(l) => resolved.push((t.clone(), l.inputs())),
            // 2026-09-26: Empty inputs: affected by any path under `kernels/`.
            Err(_) => resolved.push((t.clone(), BTreeSet::new())),
        }
    }
    for path in changed {
        let Some(hw) = hardware_of(path) else {
            continue;
        };
        match model_of(path) {
            Some((_, model)) => out.extend(
                all.iter()
                    .filter(|t| {
                        t.hardware == hw
                            && (t.model == model
                                || discover(root, t)
                                    .map(|l| l.source_model == model)
                                    .unwrap_or(true))
                    })
                    .cloned(),
            ),
            None => out.extend(all.iter().filter(|t| t.hardware == hw).cloned()),
        }
        let abs = root.join(path);
        for (t, inputs) in &resolved {
            if inputs.is_empty() || inputs.contains(&abs) {
                out.insert(t.clone());
            }
        }
    }
    out
}

/// 2026-09-26: Hardware nodes the changed paths sit under.
pub fn hardware_span(changed: &[String]) -> BTreeSet<String> {
    changed
        .iter()
        .filter_map(|p| hardware_of(p))
        .map(str::to_string)
        .collect()
}

/// 2026-09-26: `(hardware, model)` nodes the changed paths sit under.
pub fn model_span(changed: &[String]) -> BTreeSet<(String, String)> {
    changed
        .iter()
        .filter_map(|p| model_of(p))
        .map(|(h, m)| (h.to_string(), m.to_string()))
        .collect()
}

#[cfg(test)]
#[path = "taxon_tests.rs"]
mod taxon_tests;
