// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Checks that every warp-level block-scaled FP4 instruction in the
//! nvfp4 sources compiled for `hopper`, `b200` and `b300` sits inside
//! `#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA`, the define those three
//! `HARDWARE.toml`s pass in `extra_nvcc_flags`.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! The two tokens matched are the ones ptxas names when it rejects the
//! instructions; `scripts/hopper_ptx_gate.sh` compiles a hardware set with nvcc:
//!
//! ```text
//! sm_90a : error : Instruction 'cvt with .e2m1x2' not supported on .target 'sm_90a'
//! sm_100a: error : Instruction 'mma with block scale' not supported on .target 'sm_100a'
//! ```
//!
//! This scan needs no nvcc and cannot prove the kernels correct. A token is
//! matched anywhere in a line, comments included: prose about the path belongs
//! inside the guard with the code, and telling an inline-asm string from a
//! comment would take a C preprocessor.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// 2026-09-25: The define that compiles the warp-level block-scale path out.
const GUARD: &str = "METRALE_NO_WARP_BLOCKSCALE_MMA";

/// 2026-09-25: The two tokens the ptxas errors in the module doc name.
const BLOCKSCALE_TOKENS: &[&str] = &["e2m1x2", "mxf4nvf4"];

/// 2026-09-25: The hardware sets whose `HARDWARE.toml` passes `-D<GUARD>`, so
/// every such site they compile must be behind it.
const GUARDED_HW: &[&str] = &["hopper", "b200", "b300"];

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .join("kernels")
}

/// 2026-09-25: The `.cu`/`.cuh` files of the `<hw>/<model>/nvfp4` target as
/// `metrale_closure::layout::discover` returns them, keyed by stem with the
/// leaf role over the common role, so a shadowed common file is not scanned.
fn sources(hw: &str, model: &str) -> BTreeMap<String, PathBuf> {
    let t = metrale_closure::layout::Target {
        hardware: hw.into(),
        model: model.into(),
        quant: "nvfp4".into(),
    };
    let root = kernels_root().parent().unwrap().to_path_buf();
    let l = metrale_closure::layout::discover(&root, &t).unwrap_or_else(|e| panic!("{t}: {e}"));
    let mut by_stem = BTreeMap::new();
    for map in [&l.common, &l.leaf] {
        for (name, e) in map {
            if name.ends_with(".cu") || name.ends_with(".cuh") {
                by_stem.insert(
                    name.rsplit_once('.').unwrap().0.to_string(),
                    e.source.clone(),
                );
            }
        }
    }
    assert!(!by_stem.is_empty(), "kernels/{hw}/{model}: no sources");
    by_stem
}

#[test]
fn hopper_dense_38_scans_the_same_sources_as_its_36_alias_target() {
    assert_eq!(
        sources("hopper", "qwen3.8-27b"),
        sources("hopper", "qwen3.6-27b")
    );
}

/// 2026-09-25: Model directories under `kernels/<hw>`: every subdirectory that
/// holds a MODEL.toml.
fn models(hw: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(kernels_root().join(hw))
        .unwrap_or_else(|e| panic!("kernels/{hw}: {e}"))
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            e.path().join("MODEL.toml").exists().then_some(name)
        })
        .collect();
    names.sort();
    names
}

/// 2026-09-25: Line numbers (1-based) in `text` that name a block-scaled
/// instruction and are not inside an `#ifndef <GUARD>` region.
///
/// Tracks a stack of preprocessor conditionals, so an `#ifdef`/`#if` nested
/// inside the guard cannot close it. `#else` and `#elif` mark the innermost
/// entry unguarded: the else-branch of `#ifndef GUARD` compiles when the
/// define is set. Panics if a conditional is still open at the end of `text`.
fn unguarded_sites(text: &str) -> Vec<usize> {
    let mut stack: Vec<bool> = Vec::new();
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let t = line.trim_start();
        let directive = t.strip_prefix('#').map(str::trim_start);
        match directive {
            Some(d) if d.starts_with("ifndef") => {
                let opens_guard = d
                    .strip_prefix("ifndef")
                    .map(|rest| rest.split_whitespace().next() == Some(GUARD))
                    .unwrap_or(false);
                stack.push(opens_guard);
                continue;
            }
            Some(d) if d.starts_with("ifdef") || d.starts_with("if") => {
                stack.push(false);
                continue;
            }
            Some(d) if d.starts_with("else") || d.starts_with("elif") => {
                if let Some(top) = stack.last_mut() {
                    *top = false;
                }
                continue;
            }
            Some(d) if d.starts_with("endif") => {
                stack.pop();
                continue;
            }
            _ => {}
        }
        if stack.iter().any(|g| *g) {
            continue;
        }
        if BLOCKSCALE_TOKENS.iter().any(|tok| line.contains(tok)) {
            out.push(i + 1);
        }
    }
    assert!(
        stack.is_empty(),
        "unbalanced preprocessor conditionals: {} still open at EOF",
        stack.len()
    );
    out
}

/// 2026-09-25: Negative control. Without it, the real-tree test below would
/// also pass against a scanner that returns nothing.
#[test]
fn the_scanner_reports_a_site_outside_the_guard() {
    let src = "\
asm(\"mma.sync.aligned.kind::mxf4nvf4.block_scale... \");
#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA
asm(\"cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\");
#endif
";
    assert_eq!(unguarded_sites(src), vec![1]);
}

/// 2026-09-25: A site in the guard's else-branch is reported: that branch
/// compiles when the define is set.
#[test]
fn the_scanner_reports_a_site_in_the_guards_else_branch() {
    let src = "\
#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA
asm(\"cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\");
#else
asm(\"mma.sync.aligned.kind::mxf4nvf4.block_scale... \");
#endif
";
    assert_eq!(unguarded_sites(src), vec![4]);
}

/// 2026-09-25: An unrelated conditional nested inside the guard does not close it.
#[test]
fn a_nested_conditional_does_not_close_the_guard() {
    let src = "\
#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA
#ifdef SOMETHING_ELSE
asm(\"cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\");
#endif
asm(\"mma.sync.aligned.kind::mxf4nvf4.block_scale... \");
#endif
";
    assert!(unguarded_sites(src).is_empty());
}

/// 2026-09-25: A different `#ifndef` is not the guard, however similar its name.
#[test]
fn another_ifndef_is_not_the_guard() {
    let src = "\
#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA_V2
asm(\"cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\");
#endif
";
    assert_eq!(unguarded_sites(src), vec![2]);
}

/// 2026-09-25: Every nvfp4 source compiled for a `GUARDED_HW` set has its
/// block-scaled sites behind the guard. Faults are reported as file and line.
#[test]
fn no_source_compiled_for_guarded_hardware_leaves_a_blockscale_site_unguarded() {
    let mut faults: Vec<String> = Vec::new();
    for hw in GUARDED_HW {
        for model in models(hw) {
            for (stem, path) in sources(hw, &model) {
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                for line in unguarded_sites(&text) {
                    faults.push(format!("{hw}/{model}: {stem}.cu:{line}"));
                }
            }
        }
    }
    faults.sort();
    faults.dedup();
    assert!(
        faults.is_empty(),
        "block-scaled FP4 instructions outside `#ifndef {GUARD}`. Neither \
         sm_90a nor sm_100a assembles these (see this file's oracle); the \
         region must be compiled out on those targets:\n  {}",
        faults.join("\n  ")
    );
}

/// 2026-09-25: The guard is not vacuous: the gb10 qwen3.6-35b-a3b MoE grouped
/// GEMM still holds at least 10 block-scaled lines, all inside the guard. A
/// file that lost them would pass the test above with the GB10 W4A4 path gone.
#[test]
fn the_qwen36_moe_gemm_still_carries_its_blockscale_path_inside_the_guard() {
    let path = kernels_root().join("gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        text.contains(&format!("#ifndef {GUARD}")),
        "{}: no guard at all",
        path.display()
    );
    let total = text
        .lines()
        .filter(|l| BLOCKSCALE_TOKENS.iter().any(|tok| l.contains(tok)))
        .count();
    assert!(
        total >= 10,
        "{}: only {total} block-scaled lines left — the GB10 W4A4 path looks \
         deleted rather than guarded",
        path.display()
    );
    assert!(
        unguarded_sites(&text).is_empty(),
        "{}: all {total} of them must be inside the guard",
        path.display()
    );
}
