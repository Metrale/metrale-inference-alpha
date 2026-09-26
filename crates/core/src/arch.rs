// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Decide whether PTX built for one SM architecture runs on a device, and name the target to rebuild when it does not.
//!
//! A kernel build compiles for the single arch in `kernels/<hw>/HARDWARE.toml`
//! `[hardware].arch` (one `-arch=` flag per nvcc call), so a binary and a GPU
//! can disagree. The CUDA preflight (`arch_preflight::check_arch`) calls
//! [`ptx_arch_runs_on_device`], and `--check-kernels` reads the resulting
//! [`ArchMismatch`] (`serve_phases/kernel_gate.rs`).
//!
//! Owner: core.
//! Invariants: every function here is pure; the module uses only `std`, so it
//! needs no GPU and no driver.

/// 2026-09-25: The suffix on an `sm_XY…` architecture string. It selects the
/// compatibility rule in [`ptx_arch_runs_on_device`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmSuffix {
    /// 2026-09-25: No suffix: accepted on any device with a compute capability
    /// at or above the compiled one.
    None,
    /// 2026-09-25: `a`, architecture-specific: accepted only on exactly the
    /// compiled compute capability.
    Arch,
    /// 2026-09-25: `f`, family-specific: accepted on devices of the same major
    /// version at or above the compiled compute capability.
    Family,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmArch {
    /// 2026-09-25: Compute-capability major (the `12` of `sm_121f`).
    pub major: u32,
    /// 2026-09-25: Compute-capability minor (the `1` of `sm_121f`).
    pub minor: u32,
    pub suffix: SmSuffix,
}

/// 2026-09-25: Parse an NVIDIA `sm_XY[a|f]` architecture string.
///
/// Returns `None` for anything that is not an NVIDIA SM arch: `gfx1151`,
/// `metal3.1`, fewer than two digits, or a non-digit. `None` is the "not an
/// NVIDIA target" marker.
///
/// The last digit is the minor version and the rest is the major
/// (`sm_90` = 9.0, `sm_100` = 10.0, `sm_121` = 12.1).
pub fn parse_sm_arch(arch: &str) -> Option<SmArch> {
    let rest = arch.strip_prefix("sm_")?;
    let (digits, suffix) = match rest.as_bytes().last()? {
        b'a' => (&rest[..rest.len() - 1], SmSuffix::Arch),
        b'f' => (&rest[..rest.len() - 1], SmSuffix::Family),
        _ => (rest, SmSuffix::None),
    };
    // 2026-09-25: Two digits minimum, one for the major and one for the minor;
    // `sm_9` is refused rather than guessed at.
    if digits.len() < 2 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (major, minor) = digits.split_at(digits.len() - 1);
    Some(SmArch {
        major: major.parse().ok()?,
        minor: minor.parse().ok()?,
        suffix,
    })
}

/// 2026-09-25: Which `kernels/<hw>/` target ships for a device compute
/// capability, so the mismatch message can say what to rebuild. `None` means
/// no target ships for that capability.
///
/// The table is hand-written although each NVIDIA `kernels/<hw>/HARDWARE.toml`
/// declares its `compute_capability`: metrale-core has no TOML parser
/// dependency. `crates/kernels/tests/target_hints.rs` keeps the two in step: it
/// asserts that every `vendor = "nvidia"` HARDWARE.toml's capability maps back
/// to its own directory, and that every name returned here is a directory.
pub fn target_hint(device_cc: (u32, u32)) -> Option<&'static str> {
    match device_cc {
        (9, 0) => Some("hopper"),
        // 2026-09-25: b200 (`sm_100a`) and b300 (`sm_103a`) are separate targets:
        // an `a` arch runs only on its own capability.
        (10, 0) => Some("b200"),
        (10, 3) => Some("b300"),
        (12, 1) => Some("gb10"),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchMismatch {
    /// 2026-09-25: The arch string the kernels were compiled for, verbatim.
    pub compiled_arch: String,
    /// 2026-09-25: The parsed form of `compiled_arch`; its suffix picks the
    /// reason printed by `Display`.
    pub compiled: SmArch,
    /// 2026-09-25: `(major, minor)` compute capability of the device.
    pub device_cc: (u32, u32),
}

impl std::fmt::Display for ArchMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (major, minor) = self.device_cc;
        let arch = &self.compiled_arch;
        let reason = match self.compiled.suffix {
            SmSuffix::None => format!(
                "portable PTX for {arch} needs compute capability {}.{} or later",
                self.compiled.major, self.compiled.minor
            ),
            SmSuffix::Arch => format!(
                "architecture-specific PTX ({arch}) runs only on compute capability {}.{}",
                self.compiled.major, self.compiled.minor
            ),
            SmSuffix::Family => format!(
                "family-specific PTX ({arch}) runs only on compute capability {}.{} or later \
                 within the {}.x family",
                self.compiled.major, self.compiled.minor, self.compiled.major
            ),
        };
        let fix = match target_hint(self.device_cc) {
            Some(hw) => format!(
                "rebuild with METRALE_TARGET_HW={hw} (kernels/{hw}/HARDWARE.toml arch must match \
                 this GPU) or use the image built for this GPU"
            ),
            None => format!(
                "no shipped target matches compute capability {major}.{minor} \
                 (kernels/<hw>/HARDWARE.toml arch must match this GPU) — \
                 use the image built for this GPU"
            ),
        };
        write!(
            f,
            "kernels compiled for {arch} cannot run on this GPU \
             (compute capability {major}.{minor}): {reason}; fix: {fix}"
        )
    }
}

impl std::error::Error for ArchMismatch {}

/// 2026-09-25: Can PTX compiled for `compiled_arch` run on a device of
/// `device_cc`? Three rules:
///
/// 1. plain `sm_XY` runs on any device with CC >= X.Y;
/// 2. `sm_XYa` runs only on CC == X.Y;
/// 3. `sm_XYf` runs on the same major version with CC >= X.Y.
///
/// A `compiled_arch` that is not an NVIDIA SM arch (`gfx1151`, `metal3.1`)
/// returns `Ok(())`, so AMD and Apple builds are never refused here. Callers
/// that need to tell the cases apart use [`parse_sm_arch`], whose `None` is the
/// marker.
pub fn ptx_arch_runs_on_device(
    compiled_arch: &str,
    device_cc: (u32, u32),
) -> Result<(), ArchMismatch> {
    let Some(compiled) = parse_sm_arch(compiled_arch) else {
        return Ok(());
    };
    let compiled_cc = (compiled.major, compiled.minor);
    let runs = match compiled.suffix {
        SmSuffix::None => device_cc >= compiled_cc,
        SmSuffix::Arch => device_cc == compiled_cc,
        SmSuffix::Family => device_cc.0 == compiled.major && device_cc >= compiled_cc,
    };
    if runs {
        return Ok(());
    }
    Err(ArchMismatch {
        compiled_arch: compiled_arch.to_string(),
        compiled,
        device_cc,
    })
}

#[cfg(test)]
#[path = "arch_tests.rs"]
mod tests;
