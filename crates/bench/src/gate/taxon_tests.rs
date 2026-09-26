// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `taxon`: the real tree's target list and vendors, and
//! shadowing, redirects, fail-closed resolution, path-to-node mapping,
//! affected sets and spans on a temp-dir fixture tree.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::*;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("metrale-taxon-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// 2026-09-26: A minimal kernel tree in a temp dir: gb10 with `common/`,
/// `modelA` (which shadows `common/shared.cu`) and `modelB`.
fn fixture(name: &str) -> PathBuf {
    let root = tmp(name);
    let hw = root.join("kernels/gb10");
    std::fs::create_dir_all(hw.join("common")).unwrap();
    std::fs::create_dir_all(hw.join("modelA/nvfp4")).unwrap();
    std::fs::create_dir_all(hw.join("modelB/nvfp4")).unwrap();
    std::fs::write(
        hw.join("HARDWARE.toml"),
        "[hardware]\nvendor = \"nvidia\"\n",
    )
    .unwrap();
    std::fs::write(hw.join("modelA/MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(hw.join("modelB/MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(hw.join("common/shared.cu"), "__global__ void s() {}\n").unwrap();
    std::fs::write(hw.join("common/other.cu"), "__global__ void o() {}\n").unwrap();
    std::fs::write(hw.join("common/helper.cuh"), "#define H 1\n").unwrap();
    std::fs::write(
        hw.join("modelA/nvfp4/shared.cu"),
        "__global__ void s2() {}\n",
    )
    .unwrap();
    // 2026-09-26: A model file over its common namesake must be declared in
    // `[shadow]`, or the resolver refuses the target.
    std::fs::write(
        hw.join("modelA/nvfp4/KERNEL.toml"),
        "[shadow]\nshared = \"fixture fork\"\n",
    )
    .unwrap();
    root
}

/// 2026-09-26: The real tree's targets, listed by value so that a new target
/// fails here until it is added, and each resolves a non-empty source set.
#[test]
fn every_real_target_resolves_a_nonempty_source_set() {
    let root = repo_root();
    let targets = walk(&root);
    assert_eq!(
        targets.iter().map(ToString::to_string).collect::<Vec<_>>(),
        [
            "b200/deepseek-v4-flash/nvfp4",
            "b200/deepseek-v4.1-flash/nvfp4",
            "b200/kimi-k3/bf16",
            "b200/kimi-k3/mxfp4",
            "b200/kimi-k3/nvfp4",
            "b200/nemotron-3-nano-30b-a3b/nvfp4",
            "b200/nemotron-super-120b-a12b/nvfp4",
            "b200/qwen3-next-80b-a3b/nvfp4",
            "b200/qwen3.6-35b-a3b/nvfp4",
            "b300/kimi-k3/bf16",
            "b300/kimi-k3/mxfp4",
            "b300/kimi-k3/nvfp4",
            "gb10/deepseek-v4-flash/nvfp4",
            "gb10/deepseek-v4.1-flash/nvfp4",
            "gb10/gemma-4-26b-a4b/nvfp4",
            "gb10/gemma-4-31b/nvfp4",
            "gb10/glm-5.3-flash/nvfp4",
            "gb10/holo-3.1-0.8b/nvfp4",
            "gb10/holo-3.1-35b-a3b/nvfp4",
            "gb10/holo-3.1-4b/nvfp4",
            "gb10/kimi-k3/bf16",
            "gb10/kimi-k3/mxfp4",
            "gb10/kimi-k3/nvfp4",
            "gb10/laguna-s-2.1/nvfp4",
            // 2026-09-26: Compiles laguna-s-2.1's kernels through
            // `kernel_source`.
            "gb10/laguna-xs-2.1/nvfp4",
            "gb10/longcat-flash-lite/nvfp4",
            "gb10/minimax-m2-229b/nvfp4",
            "gb10/mistral-small-4/nvfp4",
            "gb10/nemotron-3-nano-30b-a3b/nvfp4",
            "gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4",
            "gb10/nemotron-super-120b-a12b/nvfp4",
            "gb10/nllb-200-3.3b/bf16",
            "gb10/ornith-1.0-9b/nvfp4",
            "gb10/qwen3-next-80b-a3b/nvfp4",
            "gb10/qwen3-vl-30b-a3b/nvfp4",
            "gb10/qwen3.5-122b-a10b/nvfp4",
            "gb10/qwen3.5-27b/nvfp4",
            "gb10/qwen3.5-35b-a3b/nvfp4",
            "gb10/qwen3.5-397b-a17b/nvfp4",
            "gb10/qwen3.6-27b/nvfp4",
            "gb10/qwen3.6-35b-a3b/nvfp4",
            "gb10/qwen3.8-27b/nvfp4",
            "gb10/qwen3.8-flash-next/nvfp4",
            "gb10/step3p7-flash/nvfp4",
            "hopper/deepseek-v4-flash/nvfp4",
            "hopper/deepseek-v4.1-flash/nvfp4",
            "hopper/nemotron-3-nano-30b-a3b/nvfp4",
            "hopper/nemotron-super-120b-a12b/nvfp4",
            "hopper/qwen3-next-80b-a3b/nvfp4",
            "hopper/qwen3.6-27b/nvfp4",
            "hopper/qwen3.6-35b-a3b/nvfp4",
            "hopper/qwen3.8-27b/nvfp4",
            "metal/nllb-200-3.3b/bf16",
            "metal/qwen3-5-4b-vlm-mlx-int8/mlx_int8",
            "strix/qwen3.6-27b/nvfp4",
            "strix/qwen3.6-35b-a3b/nvfp4",
            "strix-hip/qwen3.6-27b/nvfp4",
            "strix-hip/qwen3.6-35b-a3b/nvfp4",
        ]
    );
    for t in &targets {
        let srcs = sources(&root, t)
            .unwrap_or_else(|| panic!("{t}: sources() returned None — vendor table stale?"));
        assert!(!srcs.is_empty(), "{t}: empty source set");
    }
}

/// 2026-09-26: Every hardware dir that has targets maps its vendor to a
/// source extension.
#[test]
fn every_hardware_vendor_is_known_to_the_source_extension_table() {
    let root = repo_root();
    for t in walk(&root) {
        let hw = metrale_closure::layout::hardware(&root.join("kernels"), &t.hardware)
            .unwrap_or_else(|e| panic!("{}: {e}", t.hardware));
        assert!(
            !hw.source_ext.is_empty(),
            "hardware {} declares vendor {:?}, which the resolver does not know — \
             teach it the extension or its targets can never be skipped",
            t.hardware,
            hw.vendor
        );
    }
}

#[test]
fn a_model_file_shadows_the_common_file_with_the_same_stem() {
    let root = fixture("shadow");
    let t = Target {
        hardware: "gb10".into(),
        model: "modelA".into(),
        quant: "nvfp4".into(),
    };
    let srcs = sources(&root, &t).unwrap();
    assert_eq!(
        srcs,
        [
            root.join("kernels/gb10/common/other.cu"),
            root.join("kernels/gb10/modelA/nvfp4/shared.cu"),
        ]
    );
}

/// 2026-09-26: `sources` holds only files with the vendor's source
/// extension, so the fixture's `.cuh` header is not in it.
#[test]
fn headers_are_not_in_the_source_set() {
    let root = fixture("headers");
    let t = Target {
        hardware: "gb10".into(),
        model: "modelB".into(),
        quant: "nvfp4".into(),
    };
    assert_eq!(
        sources(&root, &t).unwrap(),
        [
            root.join("kernels/gb10/common/other.cu"),
            root.join("kernels/gb10/common/shared.cu"),
        ]
    );
}

#[test]
fn a_redirect_uses_the_source_owners_kernels_but_the_targets_model_config() {
    let root = fixture("redirect");
    let hw = root.join("kernels/gb10");
    std::fs::create_dir_all(hw.join("modelC")).unwrap();
    std::fs::write(
        hw.join("modelC/MODEL.toml"),
        "[model]\nkernel_source = \"modelA\"\n",
    )
    .unwrap();
    std::fs::write(hw.join("common/KERNEL.toml"), "[build]\n").unwrap();

    let redirected = Target {
        hardware: "gb10".into(),
        model: "modelC".into(),
        quant: "nvfp4".into(),
    };
    assert!(walk(&root).contains(&redirected));
    assert_eq!(
        sources(&root, &redirected).unwrap(),
        [
            hw.join("common/other.cu"),
            hw.join("modelA/nvfp4/shared.cu"),
        ]
    );
    // 2026-09-26: The hardware's manifest, every `KERNEL.toml` the resolution
    // read (least specific first), then the target's own `MODEL.toml`, not
    // the source owner's.
    assert_eq!(
        configs(&root, &redirected),
        [
            hw.join("HARDWARE.toml"),
            hw.join("common/KERNEL.toml"),
            hw.join("modelA/nvfp4/KERNEL.toml"),
            hw.join("modelC/MODEL.toml"),
        ]
    );
}

#[test]
fn a_source_owner_change_affects_its_redirected_consumers() {
    let root = fixture("redirect-affected");
    let hw = root.join("kernels/gb10");
    std::fs::create_dir_all(hw.join("modelC")).unwrap();
    std::fs::write(
        hw.join("modelC/MODEL.toml"),
        "[model]\nkernel_source = \"modelA\"\n",
    )
    .unwrap();

    assert_eq!(
        affected(&root, &["kernels/gb10/modelA/nvfp4/shared.cu".to_string()]),
        [
            Target {
                hardware: "gb10".into(),
                model: "modelA".into(),
                quant: "nvfp4".into(),
            },
            Target {
                hardware: "gb10".into(),
                model: "modelC".into(),
                quant: "nvfp4".into(),
            },
        ]
        .into_iter()
        .collect()
    );
}

/// 2026-09-26: No resolvable sources is `None`, never `Some(vec![])`.
#[test]
fn an_unresolvable_target_is_none_not_empty() {
    let root = fixture("empty");
    let t = Target {
        hardware: "gb10".into(),
        model: "modelA".into(),
        quant: "does-not-exist".into(),
    };
    std::fs::remove_dir_all(root.join("kernels/gb10/common")).unwrap();
    assert!(
        sources(&root, &t).is_none(),
        "an empty resolution must be None — every empty set hashes alike"
    );
}

#[test]
fn an_unknown_vendor_resolves_to_none() {
    let root = fixture("vendor");
    std::fs::write(
        root.join("kernels/gb10/HARDWARE.toml"),
        "[hardware]\nvendor = \"quantum-abacus\"\n",
    )
    .unwrap();
    let t = Target {
        hardware: "gb10".into(),
        model: "modelA".into(),
        quant: "nvfp4".into(),
    };
    assert!(sources(&root, &t).is_none());
}

#[test]
fn paths_map_to_the_right_nodes() {
    assert_eq!(hardware_of("kernels/gb10/common/x.cu"), Some("gb10"));
    assert_eq!(hardware_of("kernels/strix/qwen/nvfp4/x.cu"), Some("strix"));
    assert_eq!(hardware_of("crates/bench/src/lib.rs"), None);

    assert_eq!(
        model_of("kernels/gb10/qwen3.6-27b/nvfp4/x.cu"),
        Some(("gb10", "qwen3.6-27b"))
    );
    assert_eq!(
        model_of("kernels/gb10/common/x.cu"),
        None,
        "a shared kernel belongs to no single model"
    );
    assert_eq!(
        model_of("kernels/gb10/HARDWARE.toml"),
        None,
        "a hardware-level file is not under a model"
    );
    assert_eq!(
        model_of("kernels/gb10/qwen3.6-27b/MODEL.toml"),
        Some(("gb10", "qwen3.6-27b"))
    );
}

/// 2026-09-26: Only a path that starts with `kernels/` is in the kernel tree.
#[test]
fn lookalike_paths_are_not_kernel_paths() {
    assert_eq!(hardware_of("kernels-old/gb10/common/x.cu"), None);
    assert_eq!(hardware_of("docs/kernels/gb10/x.cu"), None);
}

#[test]
fn a_common_change_affects_every_target_on_that_hardware() {
    let root = fixture("affected-common");
    let hit = affected(&root, &["kernels/gb10/common/shared.cu".to_string()]);
    assert_eq!(
        hit,
        [
            Target {
                hardware: "gb10".into(),
                model: "modelA".into(),
                quant: "nvfp4".into(),
            },
            Target {
                hardware: "gb10".into(),
                model: "modelB".into(),
                quant: "nvfp4".into(),
            },
        ]
        .into_iter()
        .collect()
    );
}

#[test]
fn a_model_change_affects_only_that_model() {
    let root = fixture("affected-model");
    let hit = affected(&root, &["kernels/gb10/modelA/nvfp4/shared.cu".to_string()]);
    assert_eq!(
        hit,
        [Target {
            hardware: "gb10".into(),
            model: "modelA".into(),
            quant: "nvfp4".into(),
        }]
        .into_iter()
        .collect()
    );
}

#[test]
fn a_non_kernel_change_affects_no_target_here() {
    let root = fixture("affected-host");
    assert!(
        affected(&root, &["crates/bench/src/gate/check.rs".to_string()]).is_empty(),
        "host code is handled by the path boundary, not the taxonomy"
    );
}

#[test]
fn spans_are_reported_per_node_level() {
    let changed = vec![
        "kernels/gb10/qwen3.6-27b/nvfp4/a.cu".to_string(),
        "kernels/strix/qwen3.6-27b/nvfp4/a.cu".to_string(),
    ];
    assert_eq!(
        hardware_span(&changed),
        ["gb10", "strix"].map(str::to_string).into()
    );
    assert_eq!(
        model_span(&changed),
        [
            ("gb10".to_string(), "qwen3.6-27b".to_string()),
            ("strix".to_string(), "qwen3.6-27b".to_string()),
        ]
        .into()
    );

    let one_hw = vec![
        "kernels/gb10/modelA/nvfp4/a.cu".to_string(),
        "kernels/gb10/modelB/nvfp4/a.cu".to_string(),
    ];
    assert_eq!(hardware_span(&one_hw), ["gb10".to_string()].into());
    assert_eq!(
        model_span(&one_hw),
        [
            ("gb10".to_string(), "modelA".to_string()),
            ("gb10".to_string(), "modelB".to_string()),
        ]
        .into()
    );
}
