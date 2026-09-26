// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: HARDWARE.toml `arch` → `KernelTarget.arch` mapping for build.rs.
//!
//! Owner: metrale-kernels build.
//! Invariants: none beyond the types.
//!
//! Included via `#[path = "build_arch.rs"] mod build_arch;` and reached from
//! the codegen module as `super::build_arch::target_arch_fields`. Its own
//! file, with no `super::` dependencies, so `tests/kernel_target_arch.rs` and
//! `tests/b300_target.rs` can compile the same code: cargo never runs a build
//! script's `#[cfg(test)]` modules.

/// 2026-09-25: The base SM string a compiled target is recorded under, given
/// the `arch` its HARDWARE.toml declares.
///
/// nvcc's `-arch=` takes a feature architecture: a base SM number plus an
/// optional suffix selecting an extended instruction set, `a` for
/// arch-specific (`sm_90a`, `sm_100a`) and `f` for family-specific
/// (`sm_121f`). `KernelTarget.arch` records the base SM (`sm_121` for GB10),
/// so the suffix is stripped.
///
/// Only `sm_<digits>` strings are touched. SCALE/HIP `gfx*` names are kept
/// verbatim (`gfx90a` is a whole architecture, not `gfx90` plus a suffix), and
/// so is Metal's `metal3.1`, which the Metal target passes to `-std=`.
pub(crate) fn kernel_target_arch(arch: &str) -> String {
    let Some(digits) = arch.strip_prefix("sm_") else {
        return arch.to_string();
    };
    let trimmed = digits.trim_end_matches(['a', 'f']);
    if trimmed.is_empty() || !trimmed.bytes().all(|b| b.is_ascii_digit()) {
        return arch.to_string();
    }
    format!("sm_{trimmed}")
}
/// 2026-09-25: The two arch strings one HARDWARE.toml `arch` declaration
/// produces, as `(KernelTarget.arch, ptx_arch)`.
///
/// `TargetPtxSet` records both because they answer different questions:
///
/// * `KernelTarget.arch` is the base SM (`sm_90`, `sm_121`), the identity the
///   constants in `crates/core/src/target.rs` are spelled with.
/// * `ptx_arch` is the string nvcc was handed, verbatim (`sm_90a`,
///   `sm_121f`). Only it can answer a compatibility question, because the
///   feature suffix decides where the PTX may run; with the suffix stripped,
///   every arch-specific build looks portable. `kernel_gate::gate_device_arch`
///   judges the device through `arch_preflight::preflight_arch`.
///
/// Returned as a pair, from one function, so the two readings of one
/// declaration cannot drift apart in codegen.
pub(crate) fn target_arch_fields(arch: &str) -> (String, &str) {
    (kernel_target_arch(arch), arch)
}
