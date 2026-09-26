// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests the entry-point scanner in `build_shadow.rs` and uses it
//! on the real tree: every common source declares an entry point, and no
//! leaf-role file drops a kernel its common-role namesake declares unless
//! `[shadow_exempt]` allows it.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! build.rs computes the dropped kernels per target and bakes them into the
//! binary (`shadowed_dropped`) for the startup kernel audit. It compiles the
//! same `build_shadow.rs`, but CI runs `cargo test` with
//! `METRALE_SKIP_BUILD=1`, where build.rs returns before that step.

#[path = "../build_shadow.rs"]
mod build_shadow;

use build_shadow::{entry_points, shadowed_missing_symbols};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// 2026-09-25: The hardware sets this file checks, the same list as
/// `HW_TREES` in `scripts/check_kernel_shadows.py`. The extension column is not
/// read: the tests take each set's extension from the resolver
/// (`layout.hardware.source_ext`).
const HW_SOURCE_EXT: &[(&str, &str)] = &[
    ("b200", "cu"),
    ("b300", "cu"),
    ("gb10", "cu"),
    ("hopper", "cu"),
    ("metal", "metal"),
    ("strix", "cu"),
    ("strix-hip", "cu"),
];

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .join("kernels")
}

/// 2026-09-25: A file that names its kernel with `#define KERNEL_NAME` and
/// includes the body from a header has no literal `__global__`. Both entry
/// points are found: `KERNEL_NAME` and, through a token-pasting macro,
/// `KERNEL_NAME##_64`.
#[test]
fn macro_declared_kernels_resolve_through_define_and_include() {
    let file = kernels_root().join("gb10/common/attn_prefill_paged_fp8.cu");
    assert!(file.is_file(), "fixture moved: {}", file.display());
    let text = std::fs::read_to_string(&file).unwrap();
    assert!(
        !text.contains("__global__"),
        "{} now contains a literal __global__, so it no longer exercises the \
         macro path this test exists to pin — point the test at another \
         `#define KERNEL_NAME` file",
        file.display()
    );

    let found = entry_points(&file);
    let want: BTreeSet<String> = ["attn_prefill_paged_fp8", "attn_prefill_paged_fp8_64"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        found, want,
        "entry-point resolution regressed for a macro-declared kernel file"
    );
}

/// 2026-09-25: A shadow that keeps the primary kernel but drops the `_64`
/// variant is reported, although the common file has no literal `__global__`.
#[test]
fn a_shadow_dropping_a_macro_declared_kernel_is_reported() {
    let dir = std::env::temp_dir().join(format!(
        "metrale-shadow-detector-{}-{}",
        std::process::id(),
        line!()
    ));
    let common = dir.join("common");
    let model = dir.join("model");
    std::fs::create_dir_all(&common).unwrap();
    std::fs::create_dir_all(&model).unwrap();

    std::fs::write(
        common.join("compute.cuh"),
        r#"
#define _CONCAT(a, b) a##b
#define CONCAT(a, b) _CONCAT(a, b)
extern "C" __global__ void KERNEL_NAME(const float* x) {}
extern "C" __global__ __launch_bounds__(128, 2) void CONCAT(KERNEL_NAME, _64)(const float* x) {}
"#,
    )
    .unwrap();
    std::fs::write(
        common.join("attn.cu"),
        "#define KERNEL_NAME metrale_attn\n#include \"compute.cuh\"\n",
    )
    .unwrap();
    // 2026-09-25: The shadow declares only the primary kernel.
    std::fs::write(
        model.join("attn.cu"),
        "extern \"C\" __global__ void metrale_attn(const float* x) {}\n",
    )
    .unwrap();

    let dropped = shadowed_missing_symbols(&common.join("attn.cu"), &model.join("attn.cu"));
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        dropped,
        vec!["metrale_attn_64".to_string()],
        "the detector did not see the dropped macro-declared kernel — this is \
         exactly the shape a text scan for `__global__` reports as clean"
    );
}

/// 2026-09-25: An instantiation macro declares the kernels its invocations
/// name, not its own parameter `NAME`.
#[test]
fn an_instantiation_macro_declares_its_invocations_not_its_parameter() {
    let dir = std::env::temp_dir().join(format!(
        "metrale-shadow-detector-{}-{}",
        std::process::id(),
        line!()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("entry.cu");
    std::fs::write(
        &file,
        "#define ENTRY(NAME, T) \\\n    extern \"C\" __global__ void NAME(T* x) { \\\n    }\n\
         ENTRY(k_a, float)\nENTRY(k_b, int)\n",
    )
    .unwrap();
    let found = entry_points(&file);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        found,
        BTreeSet::from(["k_a".to_string(), "k_b".to_string()]),
        "a macro parameter was reported as an entry point"
    );
}

/// 2026-09-25: No common-role source resolves to zero entry points: a file
/// that declares nothing cannot be told apart from one the scanner failed to
/// read. A kernel-free source added to `common/` needs an explicit exception
/// here.
#[test]
fn every_common_source_declares_at_least_one_entry_point() {
    let root = kernels_root();
    let mut checked = 0usize;
    let mut empty = Vec::new();
    // 2026-09-25: The common role of the first target of each hardware set,
    // as the resolver builds it, so an overlay's inherited files are scanned
    // under it too.
    let ws = root.parent().unwrap().to_path_buf();
    let mut seen_hw = BTreeSet::new();
    for target in metrale_closure::layout::walk(&ws).expect("the tree resolves") {
        if !seen_hw.insert(target.hardware.clone()) {
            continue;
        }
        let layout = metrale_closure::layout::discover(&ws, &target)
            .unwrap_or_else(|e| panic!("{target}: {e}"));
        let ext = layout.hardware.source_ext;
        let mut hardware_checked = 0usize;
        for (name, e) in &layout.common {
            if !name.ends_with(&format!(".{ext}")) {
                continue;
            }
            checked += 1;
            hardware_checked += 1;
            if entry_points(&e.source).is_empty() {
                empty.push(format!(
                    "{}: {}",
                    target.hardware,
                    e.source.strip_prefix(&root).unwrap().display()
                ));
            }
        }
        assert!(
            hardware_checked > 0,
            "no {ext} common sources found for {}: wrong extension or root?",
            target.hardware
        );
    }
    assert_eq!(
        seen_hw.iter().map(String::as_str).collect::<Vec<_>>(),
        HW_SOURCE_EXT.iter().map(|(hw, _)| *hw).collect::<Vec<_>>(),
        "every guarded hardware set has at least one target"
    );
    empty.sort();
    empty.dedup();
    assert!(
        checked > 100,
        "only {checked} common sources found — wrong root?"
    );
    assert!(
        empty.is_empty(),
        "{} common source(s) resolve to no kernel entry point:\n  {}",
        empty.len(),
        empty.join("\n  ")
    );
}

/// 2026-09-25: The parsed `KERNEL.toml` in `dir`, or an empty table if it is
/// missing or does not parse.
fn kernel_toml(dir: &Path) -> toml::Value {
    std::fs::read_to_string(dir.join("KERNEL.toml"))
        .ok()
        .and_then(|t| toml::from_str(&t).ok())
        .unwrap_or_else(|| toml::Value::Table(toml::map::Map::new()))
}

/// 2026-09-25: Adds `[modules]` file-stem -> module-name overrides to `out`.
fn modules(v: &toml::Value, out: &mut BTreeMap<String, String>) {
    let Some(t) = v.get("modules").and_then(|m| m.as_table()) else {
        return;
    };
    for (stem, name) in t {
        if let Some(name) = name.as_str() {
            out.insert(stem.clone(), name.to_string());
        }
    }
}

/// 2026-09-25: Adds `[shadow_exempt]` (module, kernel) pairs a shadow may omit
/// to `out`.
fn shadow_exempt(v: &toml::Value, out: &mut BTreeSet<(String, String)>) {
    let Some(t) = v.get("shadow_exempt").and_then(|m| m.as_table()) else {
        return;
    };
    for (module, kernels) in t {
        for k in kernels.as_array().into_iter().flatten() {
            if let Some(k) = k.as_str() {
                out.insert((module.clone(), k.to_string()));
            }
        }
    }
}

/// 2026-09-25: On every target the resolver walks, no leaf-role module drops a
/// kernel its common-role namesake declares, unless `[shadow_exempt]` lists the
/// (module, kernel) pair. `[modules]` and `[shadow_exempt]` are read from the
/// target's KERNEL.tomls in `layout.configs()` order, as build.rs reads them.
#[test]
fn no_model_shadow_drops_a_common_kernel() {
    let root = kernels_root().parent().unwrap().to_path_buf();
    let mut pairs = 0usize;
    let mut drift = Vec::new();
    for target in metrale_closure::layout::walk(&root).expect("the tree resolves") {
        let layout = metrale_closure::layout::discover(&root, &target)
            .unwrap_or_else(|e| panic!("{target}: {e}"));
        let ext = layout.hardware.source_ext;
        let mut module_of = BTreeMap::new();
        let mut exempt = BTreeSet::new();
        for config in layout.configs() {
            let dir = config.parent().unwrap();
            let toml = kernel_toml(dir);
            modules(&toml, &mut module_of);
            shadow_exempt(&toml, &mut exempt);
        }
        for (stem, entry) in layout.modules() {
            if layout.layers[entry.layer].role != metrale_closure::layout::Role::Leaf {
                continue;
            }
            let Some((_, namesake)) = layout.common.iter().find(|(n, _)| {
                n.rsplit_once('.')
                    .is_some_and(|(s, e)| s == stem && e == ext)
            }) else {
                continue;
            };
            pairs += 1;
            let module = module_of.get(&stem).cloned().unwrap_or(stem);
            for kernel in shadowed_missing_symbols(&namesake.source, &entry.source) {
                if exempt.contains(&(module.clone(), kernel.clone())) {
                    continue;
                }
                drift.push(format!(
                    "{target}: {} drops {module}::{kernel}",
                    entry.source.strip_prefix(&root).unwrap().display()
                ));
            }
        }
    }
    assert!(pairs > 0, "no shadow/common pairs found — wrong root?");
    drift.sort();
    drift.dedup();
    assert!(
        drift.is_empty(),
        "{} shadowing file(s) drop a kernel their common/ namesake declares:\n  {}",
        drift.len(),
        drift.join("\n  ")
    );
}

/// 2026-09-25: Every `kernels/<hw>` with a HARDWARE.toml is in
/// `HW_SOURCE_EXT`. build.rs's `resolve_targets` builds any such directory.
#[test]
fn every_hardware_set_in_the_tree_is_guarded() {
    let root = kernels_root();
    let mut unguarded: Vec<String> = std::fs::read_dir(&root)
        .expect("kernels/")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("HARDWARE.toml").is_file())
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .filter(|hw| !HW_SOURCE_EXT.iter().any(|(known, _)| known == hw))
        .collect();
    unguarded.sort();
    assert!(
        unguarded.is_empty(),
        "hardware set(s) not in HW_SOURCE_EXT, so no shadow or entry-point check \
         ever looks at them: {unguarded:?} — add them here AND in \
         scripts/check_kernel_shadows.py"
    );
}
