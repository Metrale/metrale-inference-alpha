// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Checks the `[[model_types]]` claims in the MODEL.toml files
//! that `metrale_kernels::resolve::ptx_for_config` routes checkpoints by: an
//! exact `(model_type, hidden_size)` claim beats a claim without
//! `hidden_size`, and a pair nobody claims resolves to `None`.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! The tests read the MODEL.toml files, not the compiled registry, because
//! under `METRALE_SKIP_BUILD=1` (CI) the registry is an empty stub.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .join("kernels")
}

/// 2026-09-25: `(model_type, hidden_size)` -> the `{hw}/{model}` targets
/// claiming it.
type Claims = BTreeMap<(String, Option<u64>), Vec<String>>;

/// 2026-09-25: Every `[[model_types]]` claim under `kernels/*/*/MODEL.toml`.
/// Panics if there are none.
fn claims() -> Claims {
    let mut out: Claims = BTreeMap::new();
    for (target, manifest) in manifests() {
        let text = std::fs::read_to_string(&manifest).expect("manifest is readable");
        let parsed: toml::Value = toml::from_str(&text)
            .unwrap_or_else(|e| panic!("{} is not valid TOML: {e}", manifest.display()));
        let Some(entries) = parsed.get("model_types").and_then(|v| v.as_array()) else {
            continue;
        };
        for entry in entries {
            let model_type = entry
                .get("model_type")
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| {
                    panic!(
                        "{} has a [[model_types]] entry with no model_type",
                        manifest.display()
                    )
                })
                .to_string();
            let hidden = entry.get("hidden_size").and_then(|v| v.as_integer());
            out.entry((model_type, hidden.map(|h| h as u64)))
                .or_default()
                .push(target.clone());
        }
    }
    assert!(
        !out.is_empty(),
        "no [[model_types]] claims found under {} — the walk is broken, not the tree",
        kernels_root().display()
    );
    out
}

/// 2026-09-25: `({hw}/{model}, path to its MODEL.toml)` for every directory
/// `kernels/*/*` that holds a MODEL.toml, sorted.
fn manifests() -> Vec<(String, PathBuf)> {
    let root = kernels_root();
    let mut out = Vec::new();
    let hw_dirs = std::fs::read_dir(&root).expect("kernels/ is readable");
    for hw in hw_dirs.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
        let Ok(models) = std::fs::read_dir(&hw) else {
            continue;
        };
        for model in models.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
            let manifest = model.join("MODEL.toml");
            if !manifest.is_file() {
                continue;
            }
            out.push((
                format!(
                    "{}/{}",
                    hw.file_name().unwrap().to_string_lossy(),
                    model.file_name().unwrap().to_string_lossy()
                ),
                manifest,
            ));
        }
    }
    out.sort();
    out
}

/// 2026-09-25: Each Laguna hidden size is claimed by its own target and no
/// other. A second claimant of the same pair would leave routing to the
/// `match_names` tie-break (`resolve::resolve_target`).
#[test]
fn both_laguna_hidden_sizes_route_to_their_own_target() {
    let claims = claims();
    let lookup = |hidden: u64| -> Vec<String> {
        claims
            .get(&("laguna".to_string(), Some(hidden)))
            .cloned()
            .unwrap_or_default()
    };

    assert_eq!(
        lookup(2048),
        vec!["gb10/laguna-xs-2.1".to_string()],
        "Laguna-XS-2.1 (hidden_size 2048) must be claimed by gb10/laguna-xs-2.1 \
         and only by it; unclaimed means ptx_for_config returns None and XS \
         cannot boot at all"
    );
    assert_eq!(
        lookup(3072),
        vec!["gb10/laguna-s-2.1".to_string()],
        "Laguna-S-2.1 (hidden_size 3072) must stay on gb10/laguna-s-2.1"
    );
    // 2026-09-25: A wildcard `laguna` claim would route every other hidden
    // size to the target that declared it.
    assert!(
        !claims.contains_key(&("laguna".to_string(), None)),
        "a wildcard `laguna` claim would swallow every future hidden size; \
         each variant gets an explicit claim"
    );
}

/// 2026-09-25: Where a MODEL.toml has both, its `[[model_types]]`
/// `hidden_size` claims include its `[model].hidden_dim`. Other rows are
/// allowed: `kernels/{b200,b300,gb10}/kimi-k3/MODEL.toml` claim 7168 and 1024
/// under `hidden_dim = 7168`.
#[test]
fn claimed_hidden_size_matches_documented_hidden_dim() {
    let mut checked = 0usize;
    let mut violations: Vec<String> = Vec::new();
    for (_, manifest) in manifests() {
        let text = std::fs::read_to_string(&manifest).expect("manifest is readable");
        let parsed: toml::Value = toml::from_str(&text).expect("MODEL.toml parses");
        let Some(hidden_dim) = parsed
            .get("model")
            .and_then(|m| m.get("hidden_dim"))
            .and_then(|v| v.as_integer())
        else {
            continue;
        };
        let Some(entries) = parsed.get("model_types").and_then(|v| v.as_array()) else {
            continue;
        };
        let claims: Vec<i64> = entries
            .iter()
            .filter_map(|e| e.get("hidden_size").and_then(|v| v.as_integer()))
            .collect();
        if claims.is_empty() {
            continue;
        }
        checked += claims.len();
        if !claims.contains(&hidden_dim) {
            violations.push(format!(
                "{}: no [[model_types]] hidden_size claim ({}) matches [model] hidden_dim {hidden_dim}",
                manifest.display(),
                claims
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
    assert!(
        checked >= 2,
        "expected at least the two Laguna targets to carry an explicit \
         hidden_size claim, checked {checked}"
    );
}
