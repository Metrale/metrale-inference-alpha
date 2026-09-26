// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Checks which sources `metrale_closure::layout::discover` gives
//! the gb10 Kimi K3 targets: every quant reaches the DeepSeek E8M0 grouped GEMM
//! through `[sources] use`, mxfp4 and nvfp4 `use` bf16's `kda_decode.cu`, and
//! no KERNEL.toml carries a `[build] extra_cu` key.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
use std::path::Path;

#[test]
fn each_gb10_k3_quant_exposes_the_e8m0_expert_source() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .canonicalize()
        .unwrap();
    let expected = root.join("kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu");
    for quant in ["bf16", "mxfp4", "nvfp4"] {
        let t = metrale_closure::layout::Target {
            hardware: "gb10".into(),
            model: "kimi-k3".into(),
            quant: quant.into(),
        };
        let l = metrale_closure::layout::discover(&root, &t).unwrap();
        let (_, e) = l
            .modules()
            .into_iter()
            .find(|(s, _)| s == "moe_w4a16_grouped_gemm")
            .unwrap_or_else(|| panic!("{quant}: E8M0 source absent from build discovery"));
        assert_eq!(e.source, expected);
        assert!(e.used, "{quant}: reached through [sources] use, not a copy");
        assert!(
            std::fs::read_to_string(&e.source)
                .unwrap()
                .contains("extern \"C\" __global__ void moe_w4a16_grouped_gemm_ptrtable_e8m0(")
        );
        let mut alias = None;
        for config in l.configs() {
            let manifest: toml::Value =
                toml::from_str(&std::fs::read_to_string(&config).unwrap()).unwrap();
            assert!(
                manifest
                    .get("build")
                    .and_then(|b| b.get("extra_cu"))
                    .is_none(),
                "unused external-source declaration"
            );
            if let Some(v) = manifest
                .get("modules")
                .and_then(|m| m.get("moe_w4a16_grouped_gemm"))
            {
                alias = v.as_str().map(str::to_string);
            }
        }
        assert_eq!(alias.as_deref(), Some("moe_w4a16"));
    }
}

#[test]
fn k3_kda_aliases_keep_one_source_copy() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../");
    let root = root.canonicalize().unwrap();
    let bf16 = root.join("kernels/gb10/kimi-k3/bf16/kda_decode.cu");
    assert!(bf16.is_file());
    for quant in ["bf16", "mxfp4", "nvfp4"] {
        let t = metrale_closure::layout::Target {
            hardware: "gb10".into(),
            model: "kimi-k3".into(),
            quant: quant.into(),
        };
        let l = metrale_closure::layout::discover(&root, &t).unwrap();
        let (_, e) = l
            .modules()
            .into_iter()
            .find(|(s, _)| s == "kda_decode")
            .unwrap();
        assert_eq!(
            e.source, bf16,
            "{quant}: one source copy, `use`d by the other quants"
        );
        assert_eq!(e.used, quant != "bf16");
    }
}
