// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The hardware sets that inherit `kernels/gb10`, what each
//! declares, and path helpers for the tests that check them.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! Included with `#[path]` by `inherited_targets.rs`,
//! `inherited_targets_w4a4.rs` and `hopper_27b.rs`, so they share one list.
//! It sits below `tests/`, so cargo does not build it as a test target of its
//! own. Each includer uses part of it, hence `allow(dead_code)`.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// 2026-09-25: One hardware set that inherits gb10's kernels.
pub struct Inherited {
    /// 2026-09-25: `kernels/<hw>` directory name.
    pub hw: &'static str,
    /// 2026-09-25: `[hardware].arch`, verbatim.
    pub arch: &'static str,
    /// 2026-09-25: `[hardware].compute_capability`.
    pub cc: &'static str,
    /// 2026-09-25: Text each of this set's MODEL.toml files carries in its
    /// first six lines, naming where the kernels came from.
    pub provenance: &'static str,
    /// 2026-09-25: The model directories declared for this hardware.
    pub models: &'static [&'static str],
    /// 2026-09-25: The ptxas rejection this hardware's
    /// `METRALE_NO_WARP_BLOCKSCALE_MMA` answers, which its MODEL.toml
    /// `[expected_absent]` reasons must cite. It differs per hardware.
    pub blockscale_rejection: &'static str,
}

/// 2026-09-25: The models hopper and b200 share. deepseek-v4.1-flash owns no
/// quant directory: its MODEL.toml sets `kernel_source = "deepseek-v4-flash"`,
/// and gb10's deepseek-v4-flash leaf holds the V4.1 kernels (`attn_v41`,
/// `engram_v41`, `hc_v41`, `kquant_moe`, `moe_v41`).
pub const P0_MODELS: &[&str] = &[
    "deepseek-v4-flash",
    "deepseek-v4.1-flash",
    "nemotron-3-nano-30b-a3b",
    "nemotron-super-120b-a12b",
    "qwen3-next-80b-a3b",
    "qwen3.6-35b-a3b",
];

/// 2026-09-25: `P0_MODELS` plus qwen3.8-27b and qwen3.6-27b, the source its
/// `kernel_source` redirect names.
pub const HOPPER_MODELS: &[&str] = &[
    "deepseek-v4-flash",
    "deepseek-v4.1-flash",
    "nemotron-3-nano-30b-a3b",
    "nemotron-super-120b-a12b",
    "qwen3-next-80b-a3b",
    "qwen3.6-27b",
    "qwen3.6-35b-a3b",
    "qwen3.8-27b",
];

/// 2026-09-25: Every hardware set with `[hardware] inherits = "gb10"`. b300 is
/// not one: it lists the gb10 files it compiles in `[sources] use`, and
/// `b300_target.rs` covers it.
pub const INHERITED: &[Inherited] = &[
    Inherited {
        hw: "hopper",
        arch: "sm_90a",
        cc: "9.0",
        provenance: "Hopper target: kernel set inherited from gb10 (HARDWARE.toml `inherits`)",
        models: HOPPER_MODELS,
        blockscale_rejection: "cvt with .e2m1x2",
    },
    Inherited {
        hw: "b200",
        arch: "sm_100a",
        cc: "10.0",
        provenance: "B200 target: kernel set inherited from gb10 (HARDWARE.toml `inherits`)",
        models: P0_MODELS,
        blockscale_rejection: "mma with block scale",
    },
];

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .to_path_buf()
}

pub fn kernels_root() -> PathBuf {
    workspace_root().join("kernels")
}

pub fn hw_dir(hw: &str) -> PathBuf {
    kernels_root().join(hw)
}

pub fn gb10_dir() -> PathBuf {
    kernels_root().join("gb10")
}

pub fn hardware_toml(hw: &str) -> toml::Value {
    let path = hw_dir(hw).join("HARDWARE.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    toml::from_str(&text).unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()))
}
