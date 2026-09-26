// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests `build_arch.rs`: the HARDWARE.toml `arch` string becomes
//! `KernelTarget.arch` with an `a` or `f` suffix stripped from `sm_<digits>`,
//! and any other string passes through unchanged.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! The constants in `crates/core/src/target.rs` spell GB10's arch `sm_121`.
//! This is an integration test because cargo does not run a build script's own
//! unit tests.

#[path = "../build_arch.rs"]
mod build_arch;

use build_arch::kernel_target_arch;

#[test]
fn family_specific_suffix_is_stripped() {
    // 2026-09-25: kernels/gb10/HARDWARE.toml declares `arch = "sm_121f"`.
    assert_eq!(kernel_target_arch("sm_121f"), "sm_121");
}

#[test]
fn arch_specific_suffix_is_stripped() {
    // 2026-09-25: hopper declares `sm_90a` and b200 `sm_100a`.
    assert_eq!(kernel_target_arch("sm_90a"), "sm_90");
    assert_eq!(kernel_target_arch("sm_100a"), "sm_100");
    assert_eq!(kernel_target_arch("sm_121a"), "sm_121");
}

#[test]
fn a_plain_sm_number_is_unchanged() {
    assert_eq!(kernel_target_arch("sm_121"), "sm_121");
    assert_eq!(kernel_target_arch("sm_90"), "sm_90");
}

#[test]
fn non_nvidia_arch_strings_pass_through() {
    // 2026-09-25: SCALE selects its toolchain directory by this exact string
    // and Metal passes it to `-std=`.
    assert_eq!(kernel_target_arch("gfx1151"), "gfx1151");
    assert_eq!(kernel_target_arch("gfx90a"), "gfx90a");
    assert_eq!(kernel_target_arch("metal3.1"), "metal3.1");
}

use build_arch::target_arch_fields;

/// 2026-09-25: `target_arch_fields` returns `(KernelTarget.arch, ptx_arch)`:
/// the base SM, and the arch string verbatim as nvcc receives it. Codegen
/// takes both from this one call.
#[test]
fn a_target_records_both_the_base_sm_and_the_arch_nvcc_was_handed() {
    assert_eq!(
        target_arch_fields("sm_90a"),
        ("sm_90".to_string(), "sm_90a")
    );
    assert_eq!(
        target_arch_fields("sm_121f"),
        ("sm_121".to_string(), "sm_121f")
    );
    assert_eq!(
        target_arch_fields("sm_100a"),
        ("sm_100".to_string(), "sm_100a")
    );
}

/// 2026-09-25: With no suffix to strip, both fields hold the same string, for
/// plain SM, SCALE and Metal arch strings alike.
#[test]
fn an_arch_with_no_feature_suffix_records_the_same_string_twice() {
    assert_eq!(target_arch_fields("sm_90"), ("sm_90".to_string(), "sm_90"));
    assert_eq!(
        target_arch_fields("gfx1151"),
        ("gfx1151".to_string(), "gfx1151")
    );
    assert_eq!(
        target_arch_fields("metal3.1"),
        ("metal3.1".to_string(), "metal3.1")
    );
}
