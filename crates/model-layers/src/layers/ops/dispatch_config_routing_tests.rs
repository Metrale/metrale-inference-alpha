// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Source-scan tests that every cuBLASLt dispatch site reads the
//! `CublasScope` field of its own projection family.
//!
//! A scan covers every model crate, including sites that no runtime test
//! drives: a new file that reads `cublas.ffn` outside the FFN and MoE code
//! fails here.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// 2026-09-25: The crates under `crates/` that hold model code; the scan
/// covers all of them.
const MODEL_CRATES: &[&str] = &[
    "model-weights",
    "model-layers",
    "model-arch",
    "model-engine",
];

/// 2026-09-25: The path prefixes (`<crate>/<path under its src>`) each family
/// may be read from.
const FAMILY_HOMES: &[(&str, &[&str])] = &[
    (
        "ffn",
        &[
            "model-layers/layers/dense_ffn.rs",
            "model-layers/layers/dense_ffn_w8a8_prefill.rs",
            "model-layers/layers/moe/",
        ],
    ),
    ("attn", &["model-layers/layers/qwen3_attention/"]),
    ("ssm", &["model-layers/layers/qwen3_ssm/"]),
    // 2026-09-25: `head` has no consumer (see `CublasScope::head`); an empty
    // list means it must not be read anywhere.
    ("head", &[]),
];

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("this crate sits under crates/")
        .to_path_buf()
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable source dir") {
        let path = entry.expect("readable dir entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// 2026-09-25: `(<crate>/<path under src>, contents)` for every `.rs` file
/// under each [`MODEL_CRATES`] `src/` whose name does not contain `test`. A
/// crate with no sources panics, so a renamed crate cannot pass as empty.
fn sources() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for krate in MODEL_CRATES {
        let root = crates_dir().join(krate).join("src");
        let mut paths = Vec::new();
        rust_sources(&root, &mut paths);
        paths.sort();
        let before = out.len();
        out.extend(
            paths
                .into_iter()
                .filter(|p| {
                    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
                    !name.contains("test")
                })
                .map(|p| {
                    let rel = p
                        .strip_prefix(&root)
                        .expect("path under src")
                        .to_string_lossy()
                        .replace('\\', "/");
                    (
                        format!("{krate}/{rel}"),
                        std::fs::read_to_string(&p).expect("readable source"),
                    )
                }),
        );
        assert!(out.len() > before, "no Rust sources found under {root:?}");
    }
    out
}

/// 2026-09-25: No scanned source contains `dispatch.cublas_gemm`, a single
/// switch for every family.
#[test]
fn the_global_cublas_gemm_boolean_has_no_readers_left() {
    let offenders: Vec<_> = sources()
        .into_iter()
        .filter(|(_, body)| body.contains("dispatch.cublas_gemm"))
        .map(|(rel, _)| rel)
        .collect();
    assert!(
        offenders.is_empty(),
        "these files still read the un-scoped `dispatch.cublas_gemm`: {offenders:?}"
    );
}

#[test]
fn each_cublas_family_is_read_only_from_its_own_projection_area() {
    let sources = sources();
    let mut seen: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for (family, homes) in FAMILY_HOMES {
        let needle = format!("dispatch.cublas.{family}");
        for (rel, body) in &sources {
            if !body.contains(&needle) {
                continue;
            }
            assert!(
                homes.iter().any(|home| rel.starts_with(home)),
                "{rel} reads `{needle}` but is not a {family} projection site \
                 (allowed: {homes:?}) — a family must not be armed from another's file"
            );
            seen.entry(family).or_default().push(rel.clone());
        }
    }
    // 2026-09-25: Every family with a home must have a consumer, or the lever
    // accepts a spelling that does nothing.
    for (family, homes) in FAMILY_HOMES {
        if homes.is_empty() {
            assert!(
                !seen.contains_key(family),
                "`{family}` is documented as having no consumer, but {:?} reads it — \
                 wire it into FAMILY_HOMES and CublasScope's doc comment",
                seen.get(family)
            );
        } else {
            assert!(
                seen.contains_key(family),
                "no dispatch site reads `dispatch.cublas.{family}`"
            );
        }
    }
}
