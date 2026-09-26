// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests the one summary line the kernels build script prints per
//! target, as text.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! An integration test, because cargo does not run a build script's own unit
//! tests. It compiles `../build_summary.rs` itself, so it tests the formatter
//! the build uses.

#[path = "../build_summary.rs"]
mod build_summary;

use build_summary::{overlay_owned, summary};
use metrale_closure::layout::{Target, discover};

/// 2026-09-25: The workspace root, two levels above this crate's manifest.
fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/kernels is two levels below the workspace root")
        .to_path_buf()
}

/// 2026-09-25: The line is pinned whole rather than field by field, so a
/// reordering of its fields fails the test.
#[test]
fn the_summary_line_is_pinned_whole() {
    assert_eq!(
        summary(199, "hopper", "qwen3.8-27b", "nvfp4", 14, 15),
        "metrale-kernels: 199 kernels (hopper, qwen3.8-27b, nvfp4), \
         14 model-dir kernels, 15 overlay-owned"
    );
}

/// 2026-09-25: The model-dir count and the overlay-owned count are separate
/// fields, and the line shows them disagreeing. A formatter that collapsed them
/// into one field would fail here.
#[test]
fn the_model_dir_count_and_the_overrides_list_are_separate_fields() {
    let r14 = summary(196, "hopper", "qwen3.8-27b", "nvfp4", 14, 11);
    let r15 = summary(199, "hopper", "qwen3.8-27b", "nvfp4", 14, 15);
    assert!(
        r14.contains("14 model-dir kernels, 11 overlay-owned"),
        "{r14}"
    );
    assert!(
        r15.contains("14 model-dir kernels, 15 overlay-owned"),
        "{r15}"
    );
    assert_ne!(
        r14, r15,
        "the overrides list moved; the line must move with it"
    );
}

/// 2026-09-25: A zero count prints as `0` rather than dropping its clause, so
/// "none" and "not reported" read differently in a log.
#[test]
fn zero_counts_still_print() {
    let line = summary(158, "gb10", "qwen3.6-27b", "nvfp4", 0, 0);
    assert_eq!(
        line,
        "metrale-kernels: 158 kernels (gb10, qwen3.6-27b, nvfp4), \
         0 model-dir kernels, 0 overlay-owned"
    );
}

/// 2026-09-25: Each line names its target by `(hw, model, quant)`, so the lines
/// of a multi-target build differ by content.
#[test]
fn each_target_is_identified_by_its_own_triple() {
    let lines: Vec<String> = [
        (199usize, "hopper", "qwen3.8-27b", "nvfp4", 14usize, 15usize),
        (196, "b200", "qwen3.8-27b", "nvfp4", 14, 0),
        (158, "gb10", "qwen3.6-27b", "nvfp4", 2, 0),
    ]
    .into_iter()
    .map(|(n, hw, model, quant, md, ov)| summary(n, hw, model, quant, md, ov))
    .collect();
    let mut unique = lines.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(
        unique.len(),
        lines.len(),
        "two targets must not print the same line:\n{}",
        lines.join("\n")
    );
    for line in &lines {
        assert!(line.starts_with("metrale-kernels: "), "{line}");
    }
}

/// 2026-09-25: The kernel count is the first number in the line.
#[test]
fn the_kernel_count_leads_the_line() {
    let line = summary(199, "hopper", "qwen3.8-27b", "nvfp4", 14, 15);
    let first_number: String = line
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    assert_eq!(first_number, "199");
    assert!(line.contains("199 kernels ("), "{line}");
}

/// 2026-09-25: On the real tree, the overlay-owned count equals the `.cu`,
/// `.cuh` and `.h` files in the overlay's own `common/` (hopper), and is 0 for
/// a tree that inherits nothing (gb10, and b300 although its `common/` holds
/// four sources) and for b200, which has no `common/`.
#[test]
fn the_overlay_owned_count_is_the_overlays_own_common_files() {
    let root = workspace_root();
    let count = |hw: &str, model: &str, quant: &str| {
        let t = Target {
            hardware: hw.into(),
            model: model.into(),
            quant: quant.into(),
        };
        overlay_owned(&discover(&root, &t).unwrap_or_else(|e| panic!("{t}: {e}")))
    };
    let own = |hw: &str| -> usize {
        std::fs::read_dir(root.join("kernels").join(hw).join("common"))
            .unwrap()
            .flatten()
            .filter(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.ends_with(".cu") || n.ends_with(".cuh") || n.ends_with(".h")
            })
            .count()
    };
    assert_eq!(count("hopper", "qwen3.8-27b", "nvfp4"), own("hopper"));
    assert_eq!(count("hopper", "qwen3.8-27b", "nvfp4"), 16);
    assert_eq!(count("b300", "kimi-k3", "bf16"), 0, "b300 inherits nothing");
    assert_eq!(count("b200", "kimi-k3", "bf16"), 0);
    assert_eq!(
        count("gb10", "qwen3.6-27b", "nvfp4"),
        0,
        "gb10 inherits nothing"
    );
}
