// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Checks the b300 build inputs on the CPU (arch fields, the
//! divergent `common/` sources, the Kimi K3 sources each quant resolves), not
//! CUDA execution.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.

#[path = "../build_arch.rs"]
mod build_arch;

use std::path::{Path, PathBuf};

use metrale_closure::layout::{Target, discover};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels")
}

fn toml_at(path: &Path) -> toml::Value {
    toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn b300_build_identity_preserves_arch_specific_preflight() {
    let hw = toml_at(&root().join("b300/HARDWARE.toml"));
    let arch = hw["hardware"]["arch"].as_str().unwrap();
    assert_eq!(arch, "sm_103a");
    assert_eq!(
        build_arch::target_arch_fields(arch),
        ("sm_103".into(), "sm_103a")
    );
    assert_eq!(hw["hardware"]["compute_capability"].as_str(), Some("10.3"));
    assert_eq!(hw["hardware"]["sm_count"].as_integer(), Some(148));
    assert_eq!(hw["hardware"]["memory_gb"].as_integer(), Some(288));
    assert_eq!(metrale_core::arch::target_hint((10, 3)), Some("b300"));
    assert!(metrale_core::arch::ptx_arch_runs_on_device(arch, (10, 3)).is_ok());
    for other in [(9, 0), (10, 0), (12, 1)] {
        assert!(metrale_core::arch::ptx_arch_runs_on_device(arch, other).is_err());
    }
    assert!(
        hw["build"]["extra_nvcc_flags"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_str() == Some("-DMETRALE_NO_WARP_BLOCKSCALE_MMA"))
    );
}

/// 2026-09-25: b300 compiles a subset of gb10/common, so it inherits nothing
/// and names each gb10 file it compiles in `b300/common/KERNEL.toml`
/// `[sources] use`. Its own common/ holds only the four sources that differ
/// from gb10's. With `inherits = "gb10"` every gb10/common kernel would join
/// b300's module set, including the ones listed below that b300 leaves out.
#[test]
fn b300_compiles_a_listed_subset_of_gb10_and_holds_only_its_divergences() {
    let ws = root().parent().unwrap().to_path_buf();
    let common = root().join("b300/common");
    let gb10_common = root().join("gb10/common");
    let mut own: Vec<String> = std::fs::read_dir(&common)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".cu") || n.ends_with(".cuh"))
        .collect();
    own.sort();
    assert_eq!(
        own,
        [
            "dsa_indexer.cu",
            "moe_shared_expert_fused.cu",
            "moe_w4a16_grouped_gemm.cu",
            "w8a16_gemv_batch4.cu",
        ]
    );
    let t = Target {
        hardware: "b300".into(),
        model: "kimi-k3".into(),
        quant: "bf16".into(),
    };
    let l = discover(&ws, &t).unwrap();
    assert_eq!(l.hardware.inherits, None);
    for name in &own {
        let e = &l.common[name];
        assert_eq!(e.source, common.join(name));
        assert_ne!(
            std::fs::read(&e.source).unwrap(),
            std::fs::read(gb10_common.join(name)).unwrap(),
            "b300/common/{name} is byte-identical to gb10's: a copy, not a divergence"
        );
    }
    let used: Vec<&str> = l
        .common
        .values()
        .filter(|e| !own.contains(&e.name))
        .map(|e| {
            assert!(e.used, "{}: common/ holds only the divergences", e.name);
            assert_eq!(e.source, gb10_common.join(&e.name));
            e.name.as_str()
        })
        .collect();
    assert!(used.len() > 150, "{} gb10/common files used", used.len());
    let modules: std::collections::BTreeSet<String> =
        l.modules().into_iter().map(|(stem, _)| stem).collect();
    for predated in [
        "dense_gemv_bf16_tc",
        "paged_decode_attn_bf16_gqa",
        "paged_decode_attn_fp8_gqa",
        "reshape_and_cache_fused_k_fp8",
        "w4a16_gemv_tc",
        "w4a4_gemv_mx",
        "w8a16_gemm_pipelined_m32",
    ] {
        assert!(
            gb10_common.join(format!("{predated}.cu")).is_file(),
            "{predated}: gb10/common no longer holds it; this check is vacuous"
        );
        assert!(
            !modules.contains(predated),
            "b300 gained {predated}, a gb10/common kernel its snapshot predates"
        );
    }
    assert!(!l.common.contains_key("w4a4_gemv_mx_ps.cuh"));
}

#[test]
fn every_kimi_quant_resolves_the_real_e8m0_kernel_dependency() {
    let ws = root().parent().unwrap().to_path_buf();
    for quant in ["bf16", "mxfp4", "nvfp4"] {
        let t = Target {
            hardware: "b300".into(),
            model: "kimi-k3".into(),
            quant: quant.into(),
        };
        let l = discover(&ws, &t).unwrap_or_else(|e| panic!("{t}: {e}"));
        let modules: std::collections::BTreeMap<String, PathBuf> = l
            .modules()
            .into_iter()
            .map(|(s, e)| (s, e.source.clone()))
            .collect();
        // 2026-09-25: build.rs compiles the sources the resolver returns.
        assert_eq!(
            modules["moe_w4a16_grouped_gemm"],
            root().join("gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"),
            "{quant}: the leaf's E8M0 GEMM wins over b300's own common/ copy"
        );
        let source = std::fs::read_to_string(&modules["moe_w4a16_grouped_gemm"]).unwrap();
        assert!(source.contains("moe_w4a16_grouped_gemm_ptrtable_e8m0"));
        for stem in ["kda_decode", "mla_decode"] {
            assert_eq!(
                modules[stem].parent(),
                Some(root().join("gb10/kimi-k3/bf16").as_path()),
                "{quant}/{stem}: one source, gb10's"
            );
        }
        let mut alias = None;
        for config in l.configs() {
            let m = toml_at(&config);
            if let Some(v) = m
                .get("modules")
                .and_then(|m| m.get("moe_w4a16_grouped_gemm"))
            {
                alias = v.as_str().map(str::to_string);
            }
        }
        assert_eq!(alias.as_deref(), Some("moe_w4a16"));
    }
    let models: Vec<_> = std::fs::read_dir(root().join("b300"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("MODEL.toml").is_file())
        .collect();
    assert_eq!(
        models.len(),
        1,
        "initial target must not claim other models"
    );
    assert_eq!(models[0].file_name().unwrap(), "kimi-k3");
    let target = toml_at(&models[0].join("MODEL.toml"));
    assert!(
        target["model_types"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["model_type"].as_str() == Some("kimi_k3")
                && row["hidden_size"].as_integer() == Some(7168))
    );
}
