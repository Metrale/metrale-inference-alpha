// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of the target-resolution rules on synthetic candidates;
//! no compiled kernels are needed. The fixture copies the gb10 tree's dense-27B
//! declarations: `qwen3.6-27b` and `qwen3.8-27b` both declare
//! `(qwen3_5, 5120)`. `tests/target_resolution.rs` runs the same rules on the
//! real MODEL.toml files.
//!
//! Owner: kernels crate.
//! Invariants: none beyond the types.

use super::*;

const fn m(model_type: &'static str, hidden_size: Option<usize>) -> ModelTypeMatch {
    ModelTypeMatch {
        model_type,
        hidden_size,
    }
}

// 2026-09-25: Consts, so the borrowed fixtures are `'static`.
const Q35_MATCHES: &[ModelTypeMatch] = &[m("qwen3_5", None), m("qwen3_6_moe", Some(5120))];
const Q36_MATCHES: &[ModelTypeMatch] = &[m("qwen3_5", Some(5120)), m("qwen3_6_moe", Some(5120))];
const Q38_MATCHES: &[ModelTypeMatch] = &[m("qwen3_5", Some(5120))];

/// 2026-09-25: Two targets colliding on the exact `(qwen3_5, 5120)` plus
/// qwen3.5-27b's wildcard, in name order.
fn dense_27b_fixture() -> Vec<ResolveCandidate<'static>> {
    vec![
        ResolveCandidate {
            name: "qwen3.5-27b",
            type_matches: Q35_MATCHES,
            match_names: &["qwen3.5-27b"],
        },
        ResolveCandidate {
            name: "qwen3.6-27b",
            type_matches: Q36_MATCHES,
            match_names: &["qwen3.6-27b", "qwen3.5-27b"],
        },
        ResolveCandidate {
            name: "qwen3.8-27b",
            type_matches: Q38_MATCHES,
            match_names: &["qwen3.8-27b"],
        },
    ]
}

fn resolve_name(
    cands: &[ResolveCandidate<'_>],
    model_type: &str,
    hidden: usize,
    refs: &[&str],
) -> Result<Option<&'static str>, TargetResolveError> {
    resolve_target(cands, model_type, hidden, refs)
        .map(|o| o.map(|i| ["qwen3.5-27b", "qwen3.6-27b", "qwen3.8-27b"][i]))
}

#[test]
fn exact_collision_broken_by_checkpoint_reference() {
    let c = dense_27b_fixture();
    assert_eq!(
        resolve_name(&c, "qwen3_5", 5120, &["unsloth/Qwen3.6-27B-NVFP4"]),
        Ok(Some("qwen3.6-27b"))
    );
    assert_eq!(
        resolve_name(&c, "qwen3_5", 5120, &["unsloth/Qwen3.8-27B-NVFP4"]),
        Ok(Some("qwen3.8-27b"))
    );
    // 2026-09-25: Resolution ignores quant; serve checks it afterwards.
    assert_eq!(
        resolve_name(&c, "qwen3_5", 5120, &["centml/Qwen3.6-27B-W4A4-mlpinf"]),
        Ok(Some("qwen3.6-27b"))
    );
}

/// 2026-09-25: A Qwen3.5-27B checkpoint routes to qwen3.6-27b, whose needles
/// include "qwen3.5-27b"; qwen3.8-27b's do not.
#[test]
fn qwen35_checkpoint_still_routes_to_qwen36_target() {
    let c = dense_27b_fixture();
    assert_eq!(
        resolve_name(&c, "qwen3_5", 5120, &["Kbenkhaled/Qwen3.5-27B-NVFP4"]),
        Ok(Some("qwen3.6-27b"))
    );
}

#[test]
fn reference_matching_is_case_insensitive_and_scans_all_refs() {
    let c = dense_27b_fixture();
    // 2026-09-25: The identity is in the second reference, upper-cased.
    assert_eq!(
        resolve_name(&c, "qwen3_5", 5120, &["/model", "QWEN3.8-27B"]),
        Ok(Some("qwen3.8-27b"))
    );
    assert_eq!(
        resolve_name(
            &c,
            "qwen3_5",
            5120,
            &["/root/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/abc"]
        ),
        Ok(Some("qwen3.8-27b"))
    );
}

#[test]
fn exact_collision_with_no_matching_reference_is_a_hard_error() {
    let c = dense_27b_fixture();
    let err = resolve_name(&c, "qwen3_5", 5120, &["/model"]).unwrap_err();
    match &err {
        TargetResolveError::Ambiguous {
            tier,
            candidates,
            matched,
            ..
        } => {
            assert_eq!(*tier, "exact");
            assert!(
                matched.is_empty(),
                "nothing should have matched: {matched:?}"
            );
            let names: Vec<&str> = candidates.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(names, ["qwen3.6-27b", "qwen3.8-27b"]);
        }
        other => panic!("expected Ambiguous, got {other:?}"),
    }
    // 2026-09-25: The error text must name both remedies.
    let msg = err.to_string();
    assert!(msg.contains("--kernel-target"), "no pin remedy in: {msg}");
    assert!(
        msg.contains("METRALE_TARGET_MODEL"),
        "no build remedy in: {msg}"
    );
}

#[test]
fn reference_matching_multiple_candidates_is_a_hard_error() {
    let c = dense_27b_fixture();
    let err = resolve_name(
        &c,
        "qwen3_5",
        5120,
        &["myorg/Qwen3.6-27B-to-Qwen3.8-27B-distill"],
    )
    .unwrap_err();
    match err {
        TargetResolveError::Ambiguous { matched, .. } => {
            assert_eq!(matched, ["qwen3.6-27b", "qwen3.8-27b"]);
        }
        other => panic!("expected Ambiguous, got {other:?}"),
    }
}

/// 2026-09-25: An unresolved exact collision is an error; it does not fall
/// through to qwen3.5-27b's `(qwen3_5, None)` wildcard.
#[test]
fn ambiguous_exact_tier_never_falls_through_to_wildcard() {
    let c = dense_27b_fixture();
    assert!(matches!(
        resolve_name(&c, "qwen3_5", 5120, &["/model"]),
        Err(TargetResolveError::Ambiguous { tier: "exact", .. })
    ));
}

#[test]
fn single_exact_match_needs_no_reference() {
    // 2026-09-25: A single exact match wins without consulting references.
    const MATCHES: &[ModelTypeMatch] = &[m("qwen3_6_moe", Some(2048))];
    let c = vec![ResolveCandidate {
        name: "qwen3.6-35b-a3b",
        type_matches: MATCHES,
        match_names: &[],
    }];
    assert_eq!(resolve_target(&c, "qwen3_6_moe", 2048, &[]), Ok(Some(0)));
}

#[test]
fn wildcard_fallback_when_no_exact_match() {
    let c = dense_27b_fixture();
    assert_eq!(
        resolve_name(&c, "qwen3_5", 9999, &[]),
        Ok(Some("qwen3.5-27b"))
    );
}

#[test]
fn no_declaration_resolves_to_none() {
    let c = dense_27b_fixture();
    assert_eq!(resolve_name(&c, "deepseek_v4", 5120, &[]), Ok(None));
}

/// 2026-09-25: Candidates with the same name (quant variants of one target)
/// are not a collision; resolution returns the first.
#[test]
fn multi_quant_variants_of_one_target_are_not_ambiguous() {
    let c = vec![
        ResolveCandidate {
            name: "qwen3.6-27b",
            type_matches: Q38_MATCHES,
            match_names: &[],
        },
        ResolveCandidate {
            name: "qwen3.6-27b",
            type_matches: Q38_MATCHES,
            match_names: &[],
        },
    ];
    assert_eq!(resolve_target(&c, "qwen3_5", 5120, &[]), Ok(Some(0)));
}

#[test]
fn wildcard_tier_collisions_error_too() {
    const WILD: &[ModelTypeMatch] = &[m("qwen3_5", None)];
    let c = vec![
        ResolveCandidate {
            name: "a",
            type_matches: WILD,
            match_names: &["a"],
        },
        ResolveCandidate {
            name: "b",
            type_matches: WILD,
            match_names: &["b"],
        },
    ];
    assert!(matches!(
        resolve_target(&c, "qwen3_5", 1234, &["/model"]),
        Err(TargetResolveError::Ambiguous {
            tier: "wildcard",
            ..
        })
    ));
    assert_eq!(
        resolve_target(&c, "qwen3_5", 1234, &["org/b-7b"]),
        Ok(Some(1))
    );
}

/// 2026-09-25: An empty `match_names` list matches nothing, not everything.
#[test]
fn empty_match_names_never_matches() {
    const T1: &[ModelTypeMatch] = &[m("t", Some(1))];
    let c = vec![
        ResolveCandidate {
            name: "a",
            type_matches: T1,
            match_names: &[],
        },
        ResolveCandidate {
            name: "b",
            type_matches: T1,
            match_names: &["b"],
        },
    ];
    // 2026-09-25: "a" appears in the reference but declares no needles, so
    // only "b" can win, through its own needle.
    assert_eq!(resolve_target(&c, "t", 1, &["org/a-and-b"]), Ok(Some(1)));
    assert!(matches!(
        resolve_target(&c, "t", 1, &["org/a-only"]),
        Err(TargetResolveError::Ambiguous { .. })
    ));
}

#[test]
fn pin_overrides_the_tie_break() {
    let c = dense_27b_fixture();
    assert_eq!(resolve_pinned(&c, "qwen3.8-27b", "qwen3_5", 5120), Ok(2));
    assert_eq!(resolve_pinned(&c, "QWEN3.8-27B", "qwen3_5", 5120), Ok(2));
    assert_eq!(resolve_pinned(&c, "qwen3.6-27b", "qwen3_5", 5120), Ok(1));
    // 2026-09-25: A wildcard declaration satisfies a pin.
    assert_eq!(resolve_pinned(&c, "qwen3.5-27b", "qwen3_5", 7777), Ok(0));
}

#[test]
fn pin_to_unknown_target_errors_with_the_available_list() {
    let c = dense_27b_fixture();
    match resolve_pinned(&c, "qwen3.9-27b", "qwen3_5", 5120) {
        Err(TargetResolveError::PinNotFound { available, .. }) => {
            assert_eq!(available, ["qwen3.5-27b", "qwen3.6-27b", "qwen3.8-27b"]);
        }
        other => panic!("expected PinNotFound, got {other:?}"),
    }
}

#[test]
fn pin_to_incompatible_target_errors() {
    let c = dense_27b_fixture();
    // 2026-09-25: In the fixture qwen3.8-27b declares only (qwen3_5, 5120).
    assert!(matches!(
        resolve_pinned(&c, "qwen3.8-27b", "qwen3_6_moe", 2048),
        Err(TargetResolveError::PinIncompatible { .. })
    ));
}
