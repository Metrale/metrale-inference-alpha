// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Build script: sets the `metrale_scale` cfg when
//! `METRALE_TARGET_HW` starts with `strix`, and `metrale_hip` when it is
//! `strix-hip`.
//!
//! Owner: metrale-model-engine.
//! Invariants: none beyond the types.

fn main() {
    println!("cargo:rerun-if-env-changed=METRALE_TARGET_HW");
    // 2026-09-25: `metrale_scale` marks the AMD targets `strix` and
    // `strix-hip`, as gpu-runtime's build.rs does. In this crate it lets the
    // SSM mid-chunk capture plan run (`prefill_b/midchunk_capture.rs`).
    println!("cargo:rustc-check-cfg=cfg(metrale_scale)");
    // 2026-09-25: `metrale_hip` marks only the native-HIP target `strix-hip`.
    // No code in this crate reads it.
    println!("cargo:rustc-check-cfg=cfg(metrale_hip)");
    let hw = std::env::var("METRALE_TARGET_HW").unwrap_or_default();
    if hw.starts_with("strix") {
        println!("cargo:rustc-cfg=metrale_scale");
    }
    if hw == "strix-hip" {
        println!("cargo:rustc-cfg=metrale_hip");
    }
}
