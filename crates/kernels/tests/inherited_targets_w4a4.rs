// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Checks that the hardware sets inheriting gb10 compile out the
//! warp-level block-scale W4A4 path with `METRALE_NO_WARP_BLOCKSCALE_MMA`, and
//! declare the entry points it removes in `[expected_absent]`, each with a
//! reason naming that hardware's ptxas rejection. gb10 does neither.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! The hardware list is `INHERITED` from `support/inherited.rs`, the one
//! `inherited_targets.rs` uses.

#[path = "support/inherited.rs"]
mod inherited;

use inherited::{INHERITED, gb10_dir, hardware_toml, hw_dir};

/// 2026-09-25: The define that compiles the warp-level block-scale W4A4 path
/// out.
const GUARD_FLAG: &str = "-DMETRALE_NO_WARP_BLOCKSCALE_MMA";

/// 2026-09-25: The model whose kernel set contains that path, its module, and
/// the two entry points the define removes. A test below derives the pair from
/// the gb10 source's guarded region.
const GUARDED_MODEL: &str = "qwen3.6-35b-a3b";
const GUARDED_MODULE: &str = "moe_w4a16";
const GUARDED_KERNELS: &[&str] = &[
    "moe_w4a16_fused_gate_up_t_k64_fp4",
    "moe_w4a16_down_t_k64_fp4",
];

/// 2026-09-25: Each inheriting set's HARDWARE.toml passes the define in
/// `[build] extra_nvcc_flags`; gb10's has no `[build]` table.
#[test]
fn both_inherited_targets_compile_out_the_warp_block_scale_path() {
    for t in INHERITED {
        let flags = hardware_toml(t.hw)
            .get("build")
            .and_then(|b| b.get("extra_nvcc_flags"))
            .and_then(|f| f.as_array())
            .map(|a| {
                a.iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        assert!(
            flags.iter().any(|f| f == GUARD_FLAG),
            "kernels/{}/HARDWARE.toml must define {GUARD_FLAG}: ptxas rejects \
             the W4A4 region here with {:?}",
            t.hw,
            t.blockscale_rejection
        );
    }
    let gb10: toml::Value =
        toml::from_str(&std::fs::read_to_string(gb10_dir().join("HARDWARE.toml")).unwrap())
            .unwrap();
    assert!(
        gb10.get("build").is_none(),
        "kernels/gb10/HARDWARE.toml must add no flags — the W4A4 path is \
         compiled IN on GB10 and its PTX may not move"
    );
}

/// 2026-09-25: `GUARDED_KERNELS` are exactly the `extern "C" __global__` entry
/// points after the guard's `#ifndef` in the gb10 source. On the inheriting
/// sets, a looked-up kernel that is missing and not listed in
/// `[expected_absent]` is counted as required-unresolved by the boot kernel
/// gate.
#[test]
fn the_declared_absences_are_the_entry_points_the_define_removes() {
    let src = gb10_dir()
        .join(GUARDED_MODEL)
        .join("nvfp4/moe_w4a16_grouped_gemm.cu");
    let text = std::fs::read_to_string(&src).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
    let (_, guarded) = text
        .split_once("#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA")
        .unwrap_or_else(|| panic!("{}: no guard", src.display()));
    let mut inside: Vec<&str> = guarded
        .lines()
        .filter_map(|l| l.strip_prefix("extern \"C\" __global__ void "))
        .filter_map(|l| l.split('(').next())
        .collect();
    inside.sort_unstable();
    let mut expected: Vec<&str> = GUARDED_KERNELS.to_vec();
    expected.sort_unstable();
    assert_eq!(
        inside,
        expected,
        "{}: the entry points inside the guard are not the ones both \
         MODEL.tomls declare expected-absent",
        src.display()
    );
}

/// 2026-09-25: Each inheriting set declares both kernels in
/// `[expected_absent.moe_w4a16]`, with a reason that names its own ptxas
/// rejection and `METRALE_NO_WARP_BLOCKSCALE_MMA`.
#[test]
fn both_inherited_targets_declare_the_w4a4_kernels_expected_absent() {
    for t in INHERITED {
        let path = hw_dir(t.hw).join(GUARDED_MODEL).join("MODEL.toml");
        let toml: toml::Value = toml::from_str(&std::fs::read_to_string(&path).unwrap())
            .unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()));
        let table = toml
            .get("expected_absent")
            .and_then(|e| e.get(GUARDED_MODULE))
            .and_then(|m| m.as_table())
            .unwrap_or_else(|| panic!("{}: no [expected_absent.{GUARDED_MODULE}]", path.display()));
        for kernel in GUARDED_KERNELS {
            let reason = table
                .get(*kernel)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{}: {kernel} is not declared", path.display()));
            assert!(
                reason.contains(t.blockscale_rejection),
                "{}: {kernel}'s reason does not name this architecture's ptxas \
                 rejection {:?}:\n{reason}",
                path.display(),
                t.blockscale_rejection
            );
            assert!(
                reason.contains("METRALE_NO_WARP_BLOCKSCALE_MMA"),
                "{}: {kernel}'s reason does not say what compiles it out",
                path.display()
            );
        }
    }
}

/// 2026-09-25: gb10 declares neither kernel absent, because it compiles both.
#[test]
fn gb10_declares_neither_w4a4_kernel_absent() {
    let path = gb10_dir().join(GUARDED_MODEL).join("MODEL.toml");
    let toml: toml::Value = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let table = toml
        .get("expected_absent")
        .and_then(|e| e.get(GUARDED_MODULE))
        .and_then(|m| m.as_table());
    for kernel in GUARDED_KERNELS {
        assert!(
            table.map(|t| t.get(*kernel).is_none()).unwrap_or(true),
            "{}: {kernel} is compiled on GB10 and must not be declared absent",
            path.display()
        );
    }
}
