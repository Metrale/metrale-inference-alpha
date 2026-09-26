// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the crate root, attached by `lib.rs` with `#[path]`.
//! Most need a real kernel build and are `#[ignore]`d: under
//! `METRALE_SKIP_BUILD=1` the registry is an empty stub.
//!
//! Owner: kernels crate.
//! Invariants: none beyond the types.

use super::*;

#[test]
#[ignore = "requires nvcc and METRALE_SKIP_BUILD unset"]
fn all_ptx_modules_non_empty() {
    for (name, blob) in ptx_modules() {
        assert!(
            !blob.is_empty(),
            "PTX module '{name}' is empty — nvcc compilation may have failed"
        );
        // 2026-09-25: For an NVIDIA build the blob is PTX text and must carry a
        // `.version` directive; a blob that is not UTF-8 decodes to "" and
        // fails this assert.
        let ptx = std::str::from_utf8(blob).unwrap_or("");
        assert!(
            ptx.contains(".version"),
            "PTX module '{name}' doesn't contain .version directive"
        );
    }
}

#[test]
#[ignore = "requires nvcc and METRALE_SKIP_BUILD unset"]
fn available_targets_non_empty() {
    let targets = available_targets();
    assert!(!targets.is_empty(), "No kernel targets available");
    assert!(
        targets.iter().any(|t| t.target.quant == "nvfp4"),
        "Expected at least one NVFP4 target"
    );
}

#[test]
#[ignore = "requires nvcc and METRALE_SKIP_BUILD unset"]
fn all_targets_have_modules() {
    for t in available_targets() {
        assert!(
            t.modules.len() >= 31,
            "Target {} has only {} modules (expected >= 31)",
            t.target,
            t.modules.len()
        );
    }
}

/// 2026-09-25: The exact-verify `_snap` sources exist only in
/// `kernels/gb10/qwen3.6-27b/nvfp4`, but `qwen3_ssm::init` looks up all three
/// pairs on every GDN model, and the boot gate refuses an unresolved lookup
/// not in `[expected_absent]`. So every GDN target must compile or declare
/// each pair. A GDN target is one that compiles or declares
/// `gated_delta_rule_wy17`, which the same init also looks up.
#[test]
#[ignore = "requires nvcc and METRALE_SKIP_BUILD unset"]
fn exact_verify_snap_lookups_resolve_or_are_declared_on_every_gdn_target() {
    const PAIRS: [(&str, &str); 3] = [
        (
            "gated_delta_rule_snap",
            "gated_delta_rule_decode_f32_norm_snap",
        ),
        (
            "gated_delta_rule_snap",
            "gated_delta_rule_decode_f32_strided_norm_snap",
        ),
        (
            "gdn_verify_fused_conv_kn_f32",
            "gdn_verify_fused_conv_kn_f32",
        ),
    ];
    let ships = |t: &TargetPtxSet, m: &str| t.modules.iter().any(|(name, _)| *name == m);
    let declares = |t: &TargetPtxSet, m: &str, f: &str| {
        t.expected_absent
            .iter()
            .any(|(em, ef)| *em == m && *ef == f)
    };

    let mut gdn_targets = 0usize;
    let mut by_presence = 0usize;
    let mut by_declaration = 0usize;
    let mut violations: Vec<String> = Vec::new();
    for t in available_targets() {
        let issues_gdn = ships(&t, "gated_delta_rule_wy17")
            || declares(&t, "gated_delta_rule_wy17", "gated_delta_rule_wy17");
        if !issues_gdn {
            continue;
        }
        gdn_targets += 1;
        for (m, f) in PAIRS {
            if ships(&t, m) {
                by_presence += 1;
            } else if declares(&t, m, f) {
                by_declaration += 1;
            } else {
                violations.push(format!(
                    "{} misses {m}::{f} UNDECLARED — its boot gate will refuse to serve",
                    t.target
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "GDN targets with undeclared unresolvable snap lookups:\n{}",
        violations.join("\n")
    );
    // 2026-09-25: Both routes, compiled and declared, must have been seen, or
    // an empty or mis-staged build would pass.
    assert!(
        gdn_targets >= 2,
        "expected at least the 27B and 35B GDN targets, saw {gdn_targets}"
    );
    assert!(
        by_presence >= 3,
        "qwen3.6-27b must still SHIP all three snap modules — fixing the 35B \
         by unshipping the 27B is not a fix (pairs resolved by presence: {by_presence})"
    );
    assert!(
        by_declaration >= 3,
        "at least the 35B must cover all three pairs by declaration \
         (pairs covered: {by_declaration})"
    );
}

#[test]
#[ignore = "requires nvcc and METRALE_SKIP_BUILD unset"]
fn ptx_for_model_lookup() {
    let found = ptx_for_model("qwen3-next-80b").expect("compiled qwen3-next target");
    assert_eq!(
        found.target.model, "qwen3-next-80b-a3b",
        "lookup returned a different compiled target"
    );
}

/// 2026-09-25: Resolution against the compiled registry of a multi-target
/// build: the dense-27B checkpoints land on their own targets, a reference
/// with no identity is ambiguous, and a pin resolves it.
/// `tests/target_resolution.rs` checks the same routes against the MODEL.toml
/// files without a kernel build.
#[test]
#[ignore = "requires nvcc and METRALE_SKIP_BUILD unset (multi-target build)"]
fn ptx_for_config_breaks_the_dense_27b_tie_in_the_compiled_registry() {
    let name = |r: Result<Option<TargetPtxSet>, TargetResolveError>| {
        r.expect("resolves").expect("some target").target.model
    };
    assert_eq!(
        name(ptx_for_config(
            "qwen3_5",
            5120,
            &["unsloth/Qwen3.8-27B-NVFP4"],
            None
        )),
        "qwen3.8-27b"
    );
    assert_eq!(
        name(ptx_for_config(
            "qwen3_5",
            5120,
            &["unsloth/Qwen3.6-27B-NVFP4"],
            None
        )),
        "qwen3.6-27b"
    );
    assert_eq!(
        name(ptx_for_config(
            "qwen3_5",
            5120,
            &["Kbenkhaled/Qwen3.5-27B-NVFP4"],
            None
        )),
        "qwen3.6-27b"
    );
    assert!(matches!(
        ptx_for_config("qwen3_5", 5120, &["/model"], None),
        Err(TargetResolveError::Ambiguous { .. })
    ));
    assert_eq!(
        name(ptx_for_config(
            "qwen3_5",
            5120,
            &["/model"],
            Some("qwen3.8-27b")
        )),
        "qwen3.8-27b"
    );
    // 2026-09-25: qwen3.8-27b's MODEL.toml sets `kernel_source = "qwen3.6-27b"`,
    // so both targets must embed the same module list.
    let q36 = ptx_for_exact_target("qwen3.6-27b", "nvfp4").expect("compiled");
    let q38 = ptx_for_exact_target("qwen3.8-27b", "nvfp4").expect("compiled");
    assert_eq!(q36.modules.len(), q38.modules.len());
    let names36: Vec<&str> = q36.modules.iter().map(|(n, _)| *n).collect();
    let names38: Vec<&str> = q38.modules.iter().map(|(n, _)| *n).collect();
    assert_eq!(names36, names38, "kernel_source must mirror the module set");
}

#[test]
fn behavior_default_prose_budget_matches_shared_constant() {
    // 2026-09-25: The default must be the constant the build-time parse also
    // uses, and at least 2048 tokens.
    let b = ModelBehavior::default();
    assert_eq!(b.max_inter_tool_prose, DEFAULT_MAX_INTER_TOOL_PROSE);
    assert!(
        b.max_inter_tool_prose >= 2048,
        "inter-tool prose budget default must fit a plan/analysis turn"
    );
}
