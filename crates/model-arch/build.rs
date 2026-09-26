// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Build script for metrale-model-arch: declares the AMD target cfgs
//! and sets them from `METRALE_TARGET_HW`.
//!
//! Owner: model-arch (build).
//! Invariants:
//! - `metrale_hip` is set only together with `metrale_scale`.

fn main() {
    println!("cargo:rerun-if-env-changed=METRALE_TARGET_HW");
    // 2026-09-25: `metrale_scale` is set for every `METRALE_TARGET_HW` that starts
    // with `strix` (`strix` and `strix-hip`), the same rule gpu-runtime's build.rs
    // uses. In this crate only the `attn_prefill_microtest` example reads it.
    println!("cargo:rustc-check-cfg=cfg(metrale_scale)");
    // 2026-09-25: `metrale_hip` is set only for `strix-hip`, the native-HIP target.
    // The Qwen3.5 loader reads it to skip `predequant_for_prefill` and the FP8
    // prefill weights (`weight_loader/qwen35/load_layers/linear_attn_arms/nvfp4.rs`).
    println!("cargo:rustc-check-cfg=cfg(metrale_hip)");
    let hw = std::env::var("METRALE_TARGET_HW").unwrap_or_default();
    if hw.starts_with("strix") {
        println!("cargo:rustc-cfg=metrale_scale");
    }
    if hw == "strix-hip" {
        println!("cargo:rustc-cfg=metrale_hip");
    }
}
