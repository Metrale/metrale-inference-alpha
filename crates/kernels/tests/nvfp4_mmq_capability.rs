// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Checks that every `metrale_nvfp4_*` export of gb10
//! qwen3.6-27b's `nvfp4_mmq.cu` is inside `#if defined(BLACKWELL_MMA_AVAILABLE)`,
//! so a target without that capability has no such symbol to resolve, and
//! that hopper's qwen 27B MODEL.tomls list every export in
//! `[expected_absent.nvfp4_mmq]` while gb10's list none.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! Without the guard, the vendored `quantize_impl.cuh` compiles
//! `quantize_mmq_nvfp4_worker` to `NO_DEVICE_CODE` on such a target.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels")
}

fn source() -> String {
    std::fs::read_to_string(root().join("gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu")).unwrap()
}

fn exports(text: &str) -> Vec<String> {
    text.lines()
        .filter(|line| line.starts_with("extern \"C\" __global__ void "))
        .map(|line| {
            let name = line.split("metrale_nvfp4_").nth(1).unwrap();
            format!("metrale_nvfp4_{}", name.split('(').next().unwrap())
        })
        .collect()
}

#[test]
fn every_mmq_export_is_inside_the_vendor_capability_guard() {
    let text = source();
    let marker = "#if defined(BLACKWELL_MMA_AVAILABLE) // Metrale Engine optional module";
    let (before, inside) = text.split_once(marker).expect(
        "Hopper resolves trap-only MMQ symbols: guard exports before handle-based selection",
    );
    assert!(exports(before).is_empty());
    assert!(
        inside
            .trim_end()
            .ends_with("#endif // Metrale Engine optional module")
    );
    assert_eq!(exports(inside).len(), 13);
    // 2026-09-25: quantize_mmq_nvfp4_worker and the vendored MMA path use the
    // same macro. Its range, 1200 up to 1300, leaves out B200's sm_100.
    let vendor =
        std::fs::read_to_string(root().join("gb10/qwen3.6-27b/nvfp4/q4k_vendor/common.cuh"))
            .unwrap();
    assert!(vendor.contains("#define GGML_CUDA_CC_BLACKWELL       1200"));
    assert!(vendor.contains("#define GGML_CUDA_CC_RUBIN           1300"));
    assert!(
        vendor.contains(
            "__CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN"
        )
    );
}

#[test]
fn hopper_audit_explains_every_removed_export_and_gb10_keeps_them() {
    let functions = exports(&source());
    assert_eq!(functions.len(), 13);
    for hw in ["hopper", "gb10"] {
        for model in ["qwen3.6-27b", "qwen3.8-27b"] {
            let path = root().join(hw).join(model).join("MODEL.toml");
            let doc: toml::Value =
                toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let absent = doc.get("expected_absent").and_then(|t| t.get("nvfp4_mmq"));
            if hw == "gb10" {
                assert!(absent.is_none(), "GB10 must retain the supported MMQ path");
                continue;
            }
            let absent = absent
                .and_then(toml::Value::as_table)
                .expect("all excluded MMQ entry points need reasons in the real Hopper audit");
            assert_eq!(absent.len(), functions.len());
            for name in &functions {
                let reason = absent[name].as_str().unwrap();
                assert!(reason.contains("BLACKWELL_MMA_AVAILABLE"));
                assert!(reason.contains("W4A16"));
            }
        }
    }
}
