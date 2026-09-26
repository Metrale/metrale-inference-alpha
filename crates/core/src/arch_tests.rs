// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the compatibility rules of [`super::ptx_arch_runs_on_device`] and for [`super::target_hint`].
//!
//! The rule numbers in the test docs are the three rules listed on
//! `ptx_arch_runs_on_device`.
//!
//! Owner: core.
//! Invariants: none beyond the types.

use super::{ArchMismatch, SmSuffix, parse_sm_arch, ptx_arch_runs_on_device, target_hint};

/// 2026-09-25: Rule 3, on the pair `kernels/gb10/HARDWARE.toml` declares:
/// `arch = "sm_121f"`, `compute_capability = "12.1"`.
#[test]
fn family_ptx_runs_on_the_device_it_was_built_for() {
    assert!(ptx_arch_runs_on_device("sm_121f", (12, 1)).is_ok());
}

#[test]
fn family_ptx_does_not_run_on_a_hopper_device() {
    let err = ptx_arch_runs_on_device("sm_121f", (9, 0)).expect_err("sm_121f cannot run on CC 9.0");
    assert_eq!(err.compiled_arch, "sm_121f");
    assert_eq!(err.device_cc, (9, 0));
}

#[test]
fn family_ptx_does_not_run_below_its_own_compute_capability() {
    assert!(ptx_arch_runs_on_device("sm_121f", (12, 0)).is_err());
}

#[test]
fn family_ptx_does_not_run_on_a_different_family() {
    assert!(ptx_arch_runs_on_device("sm_121f", (10, 0)).is_err());
}

/// 2026-09-25: Rule 2, on the pair `kernels/hopper/HARDWARE.toml` declares:
/// `sm_90a` on CC 9.0.
#[test]
fn arch_specific_ptx_runs_on_its_exact_compute_capability() {
    assert!(ptx_arch_runs_on_device("sm_90a", (9, 0)).is_ok());
}

#[test]
fn arch_specific_ptx_does_not_run_on_a_newer_architecture() {
    assert!(ptx_arch_runs_on_device("sm_90a", (10, 0)).is_err());
    assert!(ptx_arch_runs_on_device("sm_90a", (12, 1)).is_err());
}

#[test]
fn plain_ptx_runs_on_its_own_and_on_newer_devices() {
    assert!(ptx_arch_runs_on_device("sm_90", (9, 0)).is_ok());
    assert!(ptx_arch_runs_on_device("sm_90", (12, 1)).is_ok());
    assert!(ptx_arch_runs_on_device("sm_80", (9, 0)).is_ok());
}

#[test]
fn plain_ptx_does_not_run_on_an_older_device() {
    assert!(ptx_arch_runs_on_device("sm_121", (9, 0)).is_err());
}

#[test]
fn a_non_nvidia_arch_is_not_judged_by_compute_capability() {
    for arch in ["gfx1151", "metal3.1", "", "sm_", "sm_9", "sm_12x"] {
        assert!(parse_sm_arch(arch).is_none(), "{arch} must not parse");
        assert!(
            ptx_arch_runs_on_device(arch, (9, 0)).is_ok(),
            "{arch} must not fail a CUDA compute-capability check"
        );
    }
}

#[test]
fn the_parser_splits_the_last_digit_as_the_minor_version() {
    let cases = [
        ("sm_80", 8, 0, SmSuffix::None),
        ("sm_90", 9, 0, SmSuffix::None),
        ("sm_90a", 9, 0, SmSuffix::Arch),
        ("sm_100", 10, 0, SmSuffix::None),
        ("sm_121f", 12, 1, SmSuffix::Family),
    ];
    for (text, major, minor, suffix) in cases {
        let got = parse_sm_arch(text).unwrap_or_else(|| panic!("{text} must parse"));
        assert_eq!((got.major, got.minor, got.suffix), (major, minor, suffix));
    }
}

#[test]
fn the_mismatch_message_names_both_sides_and_the_fix() {
    let err: ArchMismatch =
        ptx_arch_runs_on_device("sm_121f", (9, 0)).expect_err("mismatch expected");
    let msg = err.to_string();
    assert!(msg.contains("sm_121f"), "names the compiled arch: {msg}");
    assert!(msg.contains("9.0"), "names the device CC: {msg}");
    assert!(
        msg.contains("METRALE_TARGET_HW=hopper"),
        "hints the target that fits CC 9.0: {msg}"
    );
    assert!(
        msg.contains("HARDWARE.toml"),
        "points at the file to change: {msg}"
    );
    assert!(
        msg.contains("use the image built for this GPU"),
        "offers the no-rebuild way out: {msg}"
    );
}

#[test]
fn the_mismatch_message_hints_gb10_for_a_twelve_one_device() {
    let msg = ptx_arch_runs_on_device("sm_90a", (12, 1))
        .expect_err("mismatch expected")
        .to_string();
    assert!(msg.contains("METRALE_TARGET_HW=gb10"), "{msg}");
}

/// 2026-09-25: A CC 10.0 device is pointed at the b200 target
/// (`kernels/b200/HARDWARE.toml`: `compute_capability = "10.0"`), and neither
/// the gb10 nor the hopper arch runs on it.
#[test]
fn a_blackwell_datacentre_device_is_pointed_at_the_b200_target() {
    assert_eq!(target_hint((10, 0)), Some("b200"));
    for compiled in ["sm_121f", "sm_90a"] {
        let msg = ptx_arch_runs_on_device(compiled, (10, 0))
            .expect_err("neither shipped arch runs on CC 10.0")
            .to_string();
        assert!(msg.contains("METRALE_TARGET_HW=b200"), "{msg}");
    }
}

/// 2026-09-25: A CC 10.3 device is pointed at the b300 target, never at
/// b200, and the other three targets' arches are refused on it.
#[test]
fn blackwell_ultra_is_pointed_at_its_own_target() {
    assert_eq!(target_hint((10, 3)), Some("b300"));
    for compiled in ["sm_100a", "sm_121f", "sm_90a"] {
        let msg = ptx_arch_runs_on_device(compiled, (10, 3))
            .expect_err("another target must not pass the B300 preflight")
            .to_string();
        assert!(msg.contains("METRALE_TARGET_HW=b300"), "{msg}");
        assert!(!msg.contains("METRALE_TARGET_HW=b200"), "{msg}");
    }
}

#[test]
fn b300_ptx_requires_exactly_compute_capability_10_3() {
    assert!(ptx_arch_runs_on_device("sm_103a", (10, 3)).is_ok());
    for device in [(9, 0), (10, 0), (10, 7), (12, 1)] {
        let err = ptx_arch_runs_on_device("sm_103a", device).unwrap_err();
        assert_eq!(err.compiled_arch, "sm_103a");
        assert_eq!(err.device_cc, device);
    }
}

/// 2026-09-25: A capability with no target (7.5) gets `None` and a message
/// that names no `METRALE_TARGET_HW`. The NVIDIA targets under `kernels/` are
/// hopper (9.0), b200 (10.0), b300 (10.3) and gb10 (12.1).
#[test]
fn a_compute_capability_with_no_shipped_target_says_so() {
    assert_eq!(target_hint((9, 0)), Some("hopper"));
    assert_eq!(target_hint((12, 1)), Some("gb10"));
    assert_eq!(target_hint((10, 0)), Some("b200"));
    assert_eq!(target_hint((7, 5)), None);
    let msg = ptx_arch_runs_on_device("sm_121f", (7, 5))
        .expect_err("mismatch expected")
        .to_string();
    assert!(msg.contains("no shipped target"), "{msg}");
    assert!(!msg.contains("METRALE_TARGET_HW="), "{msg}");
}
