// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The kernel build (`closure_attestation` in
//! `crates/kernels/build_emit.rs`) bakes a closure hash per target into the
//! binary, and the gate recomputes it from the tree with
//! `metrale_bench::gate::taxon`. The two resolve each target's layout and
//! config list separately. If they disagreed, every baked hash would differ
//! from the tree's and look like a kernel change; these tests compare them.
//!
//! Owner: server tests.
//! Invariants: none beyond the types.

use std::path::{Path, PathBuf};

use metrale_bench::gate::closure::{Attestation, TargetClosure};
use metrale_bench::gate::taxon;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

/// 2026-09-26: For a binary built with real kernels, every baked hash must
/// reproduce from the tree.
///
/// Returns early when the binary carries no attestation: a
/// `METRALE_SKIP_BUILD=1` build, or one whose compiler was not identified,
/// bakes `{}`.
#[test]
fn the_baked_attestation_reproduces_from_the_tree() {
    let baked: Attestation = serde_json::from_str(metrale_kernels::TARGET_CLOSURES)
        .expect("TARGET_CLOSURES must parse into the type the gate reads");
    if baked.is_empty() {
        eprintln!(
            "no baked attestation (skip build) — the build-vs-gate agreement is \
             unverified in this run; it is checked on a real kernel build"
        );
        return;
    }

    let root = repo_root();

    // 2026-09-26: Every gb10 nvfp4 target the tree resolves must be attested,
    // not only every attested entry verifiable: `closure_attestation` leaves a
    // failed target out of the map, and nothing else would notice.
    let gb10: Vec<_> = taxon::walk(&root)
        .into_iter()
        .filter(|t| t.hardware == "gb10" && t.quant == "nvfp4")
        .collect();
    let missing: Vec<String> = gb10
        .iter()
        .map(|t| t.to_string())
        .filter(|k| !baked.contains_key(k))
        .collect();
    assert!(
        missing.is_empty(),
        "the build resolved these targets but attested none of them, so their \
         records can never be excused: {missing:?}. Check build.rs's cargo:warning \
         output for the reason."
    );

    let mut checked = 0;
    for (key, recorded) in &baked {
        let parts: Vec<&str> = key.split('/').collect();
        assert_eq!(
            parts.len(),
            3,
            "attestation key must be hw/model/quant: {key}"
        );
        let target = taxon::Target {
            hardware: parts[0].into(),
            model: parts[1].into(),
            quant: parts[2].into(),
        };

        let sources = taxon::sources(&root, &target).unwrap_or_else(|| {
            panic!(
                "{key}: build.rs resolved sources for this target but taxon::sources \
                 did not. The two source resolvers have drifted, and the gate can \
                 never excuse this target again."
            )
        });
        let current = metrale_closure::hash(
            &root,
            &metrale_closure::ClosureInputs {
                sources,
                configs: taxon::configs(&root, &target),
                flags: recorded.flags.clone(),
                arch: recorded.arch.clone(),
                compiler: recorded.compiler.clone(),
            },
        )
        .unwrap_or_else(|e| panic!("{key}: recomputation failed: {e}"));

        assert_eq!(
            current, recorded.hash,
            "{key}: the tree-side hash disagrees with the one baked at build \
             time. Either the source sets differ (collect_cu_files vs \
             taxon::sources) or the config lists do — not a kernel change, a \
             bug in one of the two resolvers."
        );
        checked += 1;
    }
    assert!(checked > 0, "an attestation with no usable entries");
    // 2026-09-26: Printed because a pass looks the same whether it checked
    // one target or many.
    eprintln!("closure attestation: {checked} target(s) reproduced from the tree");
}

/// 2026-09-26: The JSON `closure_attestation` builds field by field must
/// deserialize into `TargetClosure`, the struct the gate reads.
#[test]
fn the_baked_json_shape_matches_what_the_gate_deserializes() {
    let sample = r#"{"gb10/qwen3.6-27b/nvfp4":{
        "hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "arch":"sm_121a",
        "compiler":"nvcc release 13.0, V13.0.2",
        "flags":["-lineinfo"]}}"#;
    let parsed: Attestation = serde_json::from_str(sample).expect("shape must match");
    let entry: &TargetClosure = &parsed["gb10/qwen3.6-27b/nvfp4"];
    assert_eq!(entry.arch, "sm_121a");
    assert_eq!(entry.flags, vec!["-lineinfo"]);
    assert!(entry.hash.starts_with("0123"));
}

/// 2026-09-26: An absent `flags` parses as empty (`#[serde(default)]`), but
/// an absent `hash` must fail rather than default: a defaulted hash would
/// compare equal to another defaulted hash.
#[test]
fn a_malformed_entry_refuses_to_deserialize_rather_than_defaulting() {
    let no_hash = r#"{"gb10/m/q":{"arch":"sm_121a","compiler":"nvcc"}}"#;
    assert!(
        serde_json::from_str::<Attestation>(no_hash).is_err(),
        "an entry without a hash must not parse — a defaulted hash would match \
         every other defaulted hash"
    );
    let no_flags = r#"{"gb10/m/q":{"hash":"ab","arch":"sm_121a","compiler":"nvcc"}}"#;
    assert!(
        serde_json::from_str::<Attestation>(no_flags).is_ok(),
        "absent flags is the common case and must parse as empty"
    );
}
