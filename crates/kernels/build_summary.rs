// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The one summary line `build.rs` prints per kernel target, as a
//! `cargo:warning=`.
//!
//! Owner: kernels build script.
//! Invariants:
//! - This file has no `super::` dependencies, so `tests/build_summary.rs`
//!   compiles the same code with `#[path]`; cargo does not run a build
//!   script's own `#[cfg(test)]` modules.

/// 2026-09-25: `metrale-kernels: <N> kernels (<hw>, <model>, <quant>), <M> model-dir kernels, <K> overlay-owned`
///
/// * `n_kernels`: sources compiled for the target.
/// * `n_model_dir`: how many of them came from a leaf-role
///   `<model>/<quant>/` directory (this hardware's or its parent's) rather
///   than from a `common/` directory.
/// * `n_overrides`: [`overlay_owned`] for the target's layout.
///
/// Every field prints, `0` included, so the line always has the same shape.
pub(crate) fn summary(
    n_kernels: usize,
    hw: &str,
    model: &str,
    quant: &str,
    n_model_dir: usize,
    n_overrides: usize,
) -> String {
    format!(
        "metrale-kernels: {n_kernels} kernels ({hw}, {model}, {quant}), \
         {n_model_dir} model-dir kernels, {n_overrides} overlay-owned"
    )
}

/// 2026-09-25: How many resolved `common/` entries come from this hardware's
/// own `common/` directory rather than from the parent it inherits: the files
/// an overlay replaces or adds. 0 for a hardware tree with no `inherits`.
pub(crate) fn overlay_owned(layout: &metrale_closure::layout::Layout) -> usize {
    use metrale_closure::layout::{Role, Tier};
    if layout.hardware.inherits.is_none() {
        return 0;
    }
    layout
        .common
        .values()
        .filter(|e| {
            let l = &layout.layers[e.layer];
            l.role == Role::Common && l.tier == Tier::Own
        })
        .count()
}
