// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Build script for metrale-model-layers: declares the AMD target cfgs
//! and sets them from `METRALE_TARGET_HW`.
//!
//! Owner: model-layers (build).
//! Invariants:
//! - `metrale_hip` is set only together with `metrale_scale`.

fn main() {
    println!("cargo:rerun-if-env-changed=METRALE_TARGET_HW");
    // 2026-09-25: `metrale_scale` is set for every `METRALE_TARGET_HW` that starts
    // with `strix` (the SCALE `strix` and native-HIP `strix-hip` targets), the same
    // rule gpu-runtime's build.rs uses.
    println!("cargo:rustc-check-cfg=cfg(metrale_scale)");
    // 2026-09-25: `metrale_hip` is set only for `strix-hip`, the native-HIP target.
    println!("cargo:rustc-check-cfg=cfg(metrale_hip)");
    let hw = std::env::var("METRALE_TARGET_HW").unwrap_or_default();
    if hw.starts_with("strix") {
        println!("cargo:rustc-cfg=metrale_scale");
    }
    if hw == "strix-hip" {
        println!("cargo:rustc-cfg=metrale_hip");
    }
}
