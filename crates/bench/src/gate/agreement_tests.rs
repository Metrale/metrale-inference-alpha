// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `agreement::check`: the standing rule, and the signer
//! rule for both classes, each with cases that must pass and cases that must
//! be refused.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::agreement::{AddedRecord, Disagreement, check, required_by_class, sensitivity_of};
use super::check::Standing;
use crate::hardware::equivalence::HardwareFingerprint;
use crate::hardware::policy::Sensitivity;

/// 2026-09-26: A standing gb10 record with no hardware capture.
fn rec(gate: &str, sha: &str, signer: &str) -> AddedRecord {
    AddedRecord {
        path: format!(".benchmarks/{gate}/2026-09-06-{sha}.json"),
        benchmark_id: gate.into(),
        git_sha: sha.into(),
        signer: signer.into(),
        hardware: None,
        hardware_class: "gb10".into(),
        standing: Standing::Stands,
    }
}

/// 2026-09-26: The repository root, so the equivalence policy is read from
/// the committed `kernels/<hw>/HARDWARE.toml`.
fn root() -> &'static std::path::Path {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
}

/// 2026-09-26: A GB10 capture with no thermal alert and a valid postcheck,
/// at `chassis` °C.
fn gb10(chassis: f64) -> HardwareFingerprint {
    HardwareFingerprint {
        gpu: "NVIDIA GB10".into(),
        driver_major: Some(580),
        sm_clock_max_mhz: Some(3_003.0),
        mem_total_kb: Some(127_601_452),
        thermal_alert: Some(false),
        hottest_chassis_c: Some(chassis),
        postcheck_valid: Some(true),
    }
}

fn rec_on(gate: &str, signer: &str, fp: HardwareFingerprint) -> AddedRecord {
    AddedRecord {
        hardware: Some(fp),
        ..rec(gate, "abc123", signer)
    }
}

/// 2026-09-26: Pins the class of each named required gate, since a flipped
/// class changes which signer rule applies to it.
#[test]
fn the_required_gates_split_the_way_the_rule_assumes() {
    let (speed, correctness) = required_by_class();
    for g in [
        "decode-floor",
        "ttft-warm-gate",
        "ttft-cold-gate",
        "agentic-webserver",
    ] {
        assert!(speed.contains(&g), "{g} must be speed-class, got {speed:?}");
    }
    for g in ["bfcl-subset", "bfcl-subset-echolp", "vision-fidelity"] {
        assert!(
            correctness.contains(&g),
            "{g} must be correctness-class, got {correctness:?}"
        );
    }
    assert!(
        !speed.is_empty() && !correctness.is_empty(),
        "a split with an empty side would make the rule vacuous"
    );
}

#[test]
fn an_empty_set_agrees_with_itself() {
    assert!(check(root(), &[]).is_empty());
}

/// 2026-09-26: One commit, one signer, mixed classes.
#[test]
fn one_commit_one_signer_is_fine() {
    let v = check(
        root(),
        &[
            rec("decode-floor", "abc123", "k1"),
            rec("bfcl-subset", "abc123", "k1"),
        ],
    );
    assert!(v.is_empty(), "{v:?}");
}

/// 2026-09-26: Two commits pass when every record stands at the head.
/// Negative controls: an `Unknown` record and an `Invalidated` one are each a
/// straggler, named with its commit and the reason.
#[test]
fn records_may_span_commits_when_each_stands_at_head() {
    let v = check(
        root(),
        &[
            rec("bfcl-subset", "abc123", "k1"),
            rec("vision-fidelity", "def456", "k1"),
            rec("decode-floor", "def456", "k1"),
        ],
    );
    assert!(v.is_empty(), "{v:?}");

    let mut off = rec("vision-fidelity", "0ff000", "k1");
    off.standing = Standing::Unknown;
    let v = check(root(), &[rec("bfcl-subset", "abc123", "k1"), off]);
    match &v[..] {
        [Disagreement::Straggler { path, git_sha, why }] => {
            assert!(path.contains("vision-fidelity"), "{path}");
            assert_eq!(git_sha, "0ff000");
            assert!(why.contains("cannot be diffed"), "{why}");
        }
        other => panic!("{other:?}"),
    }

    let mut stale = rec("decode-floor", "abc123", "k1");
    stale.standing = Standing::Invalidated(vec!["crates/model-layers/src/x.rs".into()]);
    let v = check(root(), &[rec("bfcl-subset", "abc123", "k1"), stale]);
    match &v[..] {
        [Disagreement::Straggler { why, .. }] => {
            assert!(why.contains("crates/model-layers/src/x.rs"), "{why}");
        }
        other => panic!("{other:?}"),
    }
    let text = v[0].to_string();
    assert!(text.contains("Re-measure it at the head"), "{text}");
}

/// 2026-09-26: Correctness-class records may come from several signers.
#[test]
fn correctness_gates_may_span_two_signers() {
    let v = check(
        root(),
        &[
            rec("bfcl-subset", "abc123", "dgx1key"),
            rec("bfcl-subset-echolp", "abc123", "dgx2key"),
            rec("vision-fidelity", "abc123", "dgx3key"),
        ],
    );
    assert!(v.is_empty(), "correctness may span boxes, got {v:?}");
}

/// 2026-09-26: Speed-class records from two signers with no hardware capture
/// are refused.
#[test]
fn speed_gates_may_not_span_signers() {
    let v = check(
        root(),
        &[
            rec("decode-floor", "abc123", "dgx1key"),
            rec("ttft-cold-gate", "abc123", "dgx2key"),
        ],
    );
    match &v[..] {
        [
            Disagreement::SpeedSigners {
                gates,
                signers,
                mismatches,
            },
        ] => {
            assert_eq!(signers.len(), 2, "{signers:?}");
            assert!(gates.contains(&"decode-floor".to_string()), "{gates:?}");
            assert!(
                mismatches[0].contains("no hardware capture"),
                "{mismatches:?}"
            );
        }
        other => panic!("expected a speed-signer disagreement, got {other:?}"),
    }
}

/// 2026-09-26: Two signers on two GB10s whose captures agree are one box.
/// Negative controls: a 24 °C chassis gap (the gb10 limit is 15 °C), an
/// invalid postcheck, and a missing capture on one side.
#[test]
fn speed_gates_may_span_signers_only_when_the_records_prove_equivalence() {
    let ok = check(
        root(),
        &[
            rec_on("decode-floor", "dgx2key", gb10(65.0)),
            rec_on("ttft-cold-gate", "dgx3key", gb10(70.0)),
            rec_on("ttft-warm-gate", "dgx2key", gb10(66.0)),
        ],
    );
    assert!(ok.is_empty(), "{ok:?}");
    let v = check(
        root(),
        &[
            rec_on("decode-floor", "dgx2key", gb10(65.0)),
            rec_on("ttft-cold-gate", "dgx3key", gb10(89.0)),
        ],
    );
    match &v[..] {
        [Disagreement::SpeedSigners { mismatches, .. }] => {
            assert_eq!(mismatches.len(), 1, "{mismatches:?}");
            assert!(mismatches[0].contains("chassis 65 vs 89"), "{mismatches:?}");
            let msg = v[0].to_string();
            assert!(msg.contains("chassis 65 vs 89"), "{msg}");
        }
        other => panic!("{other:?}"),
    }
    let mut bad = gb10(66.0);
    bad.postcheck_valid = Some(false);
    let v = check(
        root(),
        &[
            rec_on("decode-floor", "dgx2key", gb10(65.0)),
            rec_on("ttft-cold-gate", "dgx3key", bad),
        ],
    );
    assert!(
        matches!(&v[..], [Disagreement::SpeedSigners { .. }]),
        "{v:?}"
    );
    let v = check(
        root(),
        &[
            rec_on("decode-floor", "dgx2key", gb10(65.0)),
            rec("ttft-cold-gate", "abc123", "dgx3key"),
        ],
    );
    assert!(
        matches!(&v[..], [Disagreement::SpeedSigners { .. }]),
        "{v:?}"
    );
    // 2026-09-26: One signer is never compared with itself, however far apart
    // its captures are.
    let v = check(
        root(),
        &[
            rec_on("decode-floor", "dgx2key", gb10(65.0)),
            rec_on("ttft-cold-gate", "dgx2key", gb10(89.0)),
        ],
    );
    assert!(v.is_empty(), "{v:?}");
}

/// 2026-09-26: A set where only the Correctness records span signers passes.
#[test]
fn a_mixed_set_is_judged_per_class_not_as_a_whole() {
    let v = check(
        root(),
        &[
            rec("decode-floor", "abc123", "dgx1key"),
            rec("ttft-warm-gate", "abc123", "dgx1key"),
            rec("bfcl-subset", "abc123", "dgx2key"),
            rec("vision-fidelity", "abc123", "dgx3key"),
        ],
    );
    assert!(v.is_empty(), "{v:?}");
}

/// 2026-09-26: An id the registry does not know is refused rather than
/// treated as Correctness.
#[test]
fn an_unknown_benchmark_is_refused_rather_than_assumed_correctness() {
    let v = check(root(), &[rec("not-a-real-gate", "abc123", "k1")]);
    assert!(
        matches!(&v[..], [Disagreement::UnknownBenchmark(id)] if id == "not-a-real-gate"),
        "{v:?}"
    );
}

#[test]
fn the_message_names_the_gates_that_must_be_redone() {
    let v = check(
        root(),
        &[
            rec("decode-floor", "abc123", "k1"),
            rec("ttft-cold-gate", "abc123", "k2"),
        ],
    );
    let msg = v[0].to_string();
    assert!(msg.contains("decode-floor"), "{msg}");
    assert!(msg.contains("ttft-cold-gate"), "{msg}");
    assert!(msg.contains("ONE box"), "{msg}");
}

#[test]
fn sensitivity_comes_from_the_registry_not_the_record() {
    assert_eq!(sensitivity_of("decode-floor"), Some(Sensitivity::Speed));
    assert_eq!(
        sensitivity_of("bfcl-subset"),
        Some(Sensitivity::Correctness)
    );
    assert_eq!(sensitivity_of("nope"), None);
}

/// 2026-09-26: A class whose HARDWARE.toml has no `[benchmarks.limits]`
/// (kernels/hopper) makes every cross-signer Speed pair a mismatch; gb10's
/// limits are not borrowed.
#[test]
fn a_class_without_an_envelope_never_lets_two_signers_agree() {
    let mut a = rec("decode-floor", "abc", "sig-a");
    let mut b = rec("ttft-warm-gate", "abc", "sig-b");
    a.hardware = Some(gb10(65.0));
    b.hardware = Some(gb10(66.0));
    assert!(
        check(root(), &[a.clone(), b.clone()]).is_empty(),
        "gb10 declares one"
    );
    a.hardware_class = "hopper".into();
    b.hardware_class = "hopper".into();
    let v = check(root(), &[a, b]);
    assert_eq!(v.len(), 1, "{v:?}");
    assert!(
        v[0].to_string()
            .contains("declares no [benchmarks.limits.thermal]"),
        "{}",
        v[0]
    );
}
