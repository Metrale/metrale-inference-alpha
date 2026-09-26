// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Checks that `scripts/lib/kernel_layout.py dump`, the Python
//! mirror of `metrale_closure::layout` used by the Python checkers, resolves
//! the real tree exactly as the resolver does: every target's source model,
//! layers, entries, subdirectories, configs, modules and shadows, with
//! repo-relative paths.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! Unix-only (`#![cfg(unix)]`) because it runs `python3`.

#![cfg(unix)]

use std::path::{Path, PathBuf};

use metrale_closure::layout::{discover, walk};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .to_path_buf()
}

fn rust_dump(root: &Path) -> serde_json::Value {
    let rel = |p: &Path| {
        serde_json::Value::String(p.strip_prefix(root).unwrap().to_string_lossy().to_string())
    };
    let mut out = serde_json::Map::new();
    for t in walk(root).expect("the tree resolves") {
        let l = discover(root, &t).unwrap_or_else(|e| panic!("{t}: {e}"));
        let entries = |m: &std::collections::BTreeMap<String, metrale_closure::layout::Entry>| {
            m.iter()
                .map(|(n, e)| {
                    (
                        n.clone(),
                        serde_json::json!([rel(&e.source), e.layer, e.used]),
                    )
                })
                .collect::<serde_json::Map<_, _>>()
        };
        let dirs = |m: &std::collections::BTreeMap<String, PathBuf>| {
            m.iter()
                .map(|(n, d)| (n.clone(), rel(d)))
                .collect::<serde_json::Map<_, _>>()
        };
        let role = |r: metrale_closure::layout::Role| match r {
            metrale_closure::layout::Role::Leaf => "leaf",
            metrale_closure::layout::Role::Common => "common",
        };
        let tier = |r: metrale_closure::layout::Tier| match r {
            metrale_closure::layout::Tier::Own => "own",
            metrale_closure::layout::Tier::Parent => "parent",
        };
        out.insert(
            t.to_string(),
            serde_json::json!({
                "source_model": l.source_model,
                "layers": l.layers.iter().map(|x| serde_json::json!([role(x.role), tier(x.tier), rel(&x.dir)])).collect::<Vec<_>>(),
                "leaf": entries(&l.leaf),
                "common": entries(&l.common),
                "leaf_subdirs": dirs(&l.leaf_subdirs),
                "common_subdirs": dirs(&l.common_subdirs),
                "configs": l.configs().iter().map(|p| rel(p)).collect::<Vec<_>>(),
                "modules": l.modules().iter().map(|(s, e)| (s.clone(), rel(&e.source))).collect::<serde_json::Map<_, _>>(),
                "shadows": l.shadows.iter().map(|s| serde_json::json!([role(s.role), s.name, rel(&s.winner), rel(&s.loser), s.reason])).collect::<Vec<_>>(),
            }),
        );
    }
    serde_json::Value::Object(out)
}

#[test]
fn the_python_mirror_resolves_the_real_tree_exactly_as_the_resolver_does() {
    let root = workspace_root();
    let out = std::process::Command::new("python3")
        .arg(root.join("scripts/lib/kernel_layout.py"))
        .arg("dump")
        .arg(&root)
        .output()
        .expect("python3 runs scripts/lib/kernel_layout.py");
    assert!(
        out.status.success(),
        "kernel_layout.py dump failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let python: serde_json::Value = serde_json::from_slice(&out.stdout).expect("dump is JSON");
    let rust = rust_dump(&root);
    let (po, ro) = (python.as_object().unwrap(), rust.as_object().unwrap());
    assert_eq!(
        po.keys().collect::<std::collections::BTreeSet<_>>(),
        ro.keys().collect::<std::collections::BTreeSet<_>>(),
        "the two walks enumerate different targets"
    );
    for (k, r) in ro {
        assert_eq!(
            &po[k], r,
            "{k}: the Python mirror disagrees with the resolver"
        );
    }
}
