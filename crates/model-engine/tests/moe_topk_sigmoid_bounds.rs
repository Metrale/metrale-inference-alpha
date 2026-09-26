// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Pins `moe_topk_sigmoid.cu`'s shared-memory bounds (`MAX_TOP_K`,
//! `MAX_EXPERTS`) to their Rust mirrors, and keeps that kernel in `common/`.
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! The kernel sizes its top-K staging arrays from those two defines, and the
//! load-time checks in `moe/init.rs` and `nemotron_moe.rs` compare a config
//! against the mirrors.

use std::path::{Path, PathBuf};

use metrale_model_layers::layers::ops::{MOE_TOPK_SIGMOID_MAX_EXPERTS, MOE_TOPK_SIGMOID_MAX_TOP_K};

const KERNEL: &str = "gb10/common/moe_topk_sigmoid.cu";

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/model-engine is two levels below the workspace root")
        .join("kernels")
}

/// 2026-09-25: The integer of `#define <name> <integer>`, ignoring any trailing
/// comment; panics when the define is missing or not an integer.
fn define(text: &str, name: &str) -> usize {
    let needle = format!("#define {name} ");
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with(&needle))
        .unwrap_or_else(|| panic!("{KERNEL} no longer defines {name}"));
    line.trim_start()[needle.len()..]
        .split_whitespace()
        .next()
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| panic!("{name} in {KERNEL} is not a plain integer: {line}"))
}

/// 2026-09-25: `MOE_TOPK_SIGMOID_MAX_TOP_K` and `MOE_TOPK_SIGMOID_MAX_EXPERTS`
/// mirror the kernel's defines. If only the kernel's arrays shrank, the load-time
/// checks would admit a config the kernel cannot hold.
#[test]
fn rust_bounds_mirror_the_kernel_defines() {
    let text = std::fs::read_to_string(kernels_root().join(KERNEL)).unwrap();
    assert_eq!(
        define(&text, "MAX_TOP_K"),
        MOE_TOPK_SIGMOID_MAX_TOP_K,
        "MAX_TOP_K in {KERNEL} and MOE_TOPK_SIGMOID_MAX_TOP_K must move together"
    );
    assert_eq!(
        define(&text, "MAX_EXPERTS"),
        MOE_TOPK_SIGMOID_MAX_EXPERTS,
        "MAX_EXPERTS in {KERNEL} and MOE_TOPK_SIGMOID_MAX_EXPERTS must move together"
    );
}

/// 2026-09-25: A copy of this kernel in a model directory would carry its own
/// bounds and tie-break, which nothing compares with `common/`. The kernel has
/// no model-specific content, so any copy outside a `common/` directory fails.
#[test]
fn no_model_directory_shadows_the_sigmoid_routing_kernel() {
    let root = kernels_root();
    let mut shadows = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().is_some_and(|n| n == "moe_topk_sigmoid.cu")
                && path.parent().is_some_and(|p| !p.ends_with("common"))
            {
                shadows.push(
                    path.strip_prefix(&root)
                        .unwrap_or(path.as_path())
                        .display()
                        .to_string(),
                );
            }
        }
    }
    shadows.sort();
    assert!(
        shadows.is_empty(),
        "moe_topk_sigmoid.cu belongs only in a `common/` directory; these copies \
         will drift in their MAX_TOP_K and their tie-break exactly as the two \
         Nemotron ones did: {shadows:?}"
    );
}

/// 2026-09-25: The mirrors must hold the largest routing the configs declare: 22
/// experts per token (Nemotron-Super-120B) and 512 experts. These are const
/// asserts, so a lower bound fails the build.
const _: () = assert!(
    MOE_TOPK_SIGMOID_MAX_TOP_K >= 22,
    "Nemotron-Super-120B-A12B routes num_experts_per_tok=22"
);
const _: () = assert!(
    MOE_TOPK_SIGMOID_MAX_EXPERTS >= 512,
    "the Nemotron and Step-3.7 families declare n_routed_experts=512"
);
