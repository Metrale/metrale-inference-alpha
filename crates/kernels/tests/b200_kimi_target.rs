// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Checks on the CPU which sources the b200 Kimi K3 targets
//! compile: `kda_decode`, `mla_decode` and the gb10 DeepSeek E8M0 grouped GEMM
//! for every quant, plus b200's own `dense_f32io.cu` for bf16 and mxfp4 only.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! The sources come from `metrale_closure::layout::discover`, not a directory
//! listing, because b200 inherits from gb10 (`kernels/b200/HARDWARE.toml`).

use std::path::{Path, PathBuf};

use metrale_closure::layout::{Target, discover};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn b200_kimi_compiles_all_quant_dependencies() {
    let kernels = root().join("kernels");
    let model = kernels.join("b200/kimi-k3");
    let manifest: toml::Value =
        toml::from_str(&std::fs::read_to_string(model.join("MODEL.toml")).unwrap()).unwrap();
    assert_eq!(manifest["model"]["name"].as_str(), Some("kimi-k3"));
    for quant in ["bf16", "mxfp4", "nvfp4"] {
        let t = Target {
            hardware: "b200".into(),
            model: "kimi-k3".into(),
            quant: quant.into(),
        };
        let l = discover(&root(), &t).unwrap_or_else(|e| panic!("{t}: {e}"));
        let modules: std::collections::BTreeMap<String, PathBuf> = l
            .modules()
            .into_iter()
            .map(|(s, e)| (s, e.source.clone()))
            .collect();
        for stem in ["kda_decode", "mla_decode", "moe_w4a16_grouped_gemm"] {
            assert!(modules.contains_key(stem), "{quant}/{stem}");
        }
        assert_eq!(
            modules["moe_w4a16_grouped_gemm"],
            kernels.join("gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"),
            "{quant}: the E8M0 grouped GEMM is DeepSeek's one source"
        );
        let source = std::fs::read_to_string(&modules["moe_w4a16_grouped_gemm"]).unwrap();
        assert!(source.contains("moe_w4a16_grouped_gemm_ptrtable_e8m0"));
        // 2026-09-25: bf16 holds dense_f32io.cu and mxfp4 `use`s it; nvfp4
        // does not compile it.
        if quant == "nvfp4" {
            assert!(
                !modules.contains_key("dense_f32io"),
                "nvfp4 gained dense_f32io"
            );
        } else {
            assert_eq!(
                modules["dense_f32io"],
                kernels.join("b200/kimi-k3/bf16/dense_f32io.cu")
            );
        }
        let mut module_of = std::collections::BTreeMap::new();
        for config in l.configs() {
            let kernel: toml::Value =
                toml::from_str(&std::fs::read_to_string(&config).unwrap()).unwrap();
            if let Some(m) = kernel.get("modules").and_then(|m| m.as_table()) {
                for (k, v) in m {
                    module_of.insert(k.clone(), v.as_str().unwrap().to_string());
                }
            }
        }
        assert_eq!(
            module_of.get("moe_w4a16_grouped_gemm").map(String::as_str),
            Some("moe_w4a16")
        );
    }
}
