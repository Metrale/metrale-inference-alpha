// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A hardware class's benchmark limits: the numbers certification and
//! the hardware policies judge a box by, declared in `kernels/<hw>/HARDWARE.toml`
//! under `[benchmarks.limits]`.
//!
//! Owner: bench hardware.
//! Invariants:
//! - [`limits`] returns `Some` only with all four sub-tables present, no unknown
//!   key, every float finite, resume below park, positive tolerances, a free
//!   fraction inside `(0, 1)` and non-zero allowances.
//!
//! The numbers are facts about a box class, so they live beside its other facts
//! rather than in Rust constants. A class without `[benchmarks.limits]` has no
//! limits, and callers say so rather than borrowing another class's.
//! `HARDWARE.toml` is one of the configs hashed into the class's target closures
//! (`gate::taxon::configs`), so changing a limit re-opens the class's gates.

use std::path::Path;

use anyhow::{Context, Result, bail};

/// 2026-09-26: `[benchmarks.limits]` of one hardware class.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub thermal: ThermalEnvelope,
    pub memory: MemoryLimits,
    pub timing: TimingLimits,
    pub equivalence: EquivalenceLimits,
}

/// 2026-09-26: `[benchmarks.limits.thermal]`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalEnvelope {
    /// 2026-09-26: The hottest-chassis-zone reading at which certification parks a
    /// node, °C; also the precheck's chassis ceiling ([`super::policy::TempCeilings::of`]).
    pub chassis_park_c: f64,
    /// 2026-09-26: The reading a parked node must fall to before it resumes, °C.
    pub chassis_resume_c: f64,
    /// 2026-09-26: The largest hottest-zone difference at which two boxes of this
    /// class are still one box for Speed numbers, °C.
    pub chassis_equivalence_delta_c: f64,
    /// 2026-09-26: The precheck's GPU die ceiling, °C.
    pub gpu_ceiling_c: f64,
}

/// 2026-09-26: `[benchmarks.limits.memory]`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryLimits {
    /// 2026-09-26: `MemAvailable / MemTotal` a gate needs before its server may start.
    pub min_free_fraction: f64,
}

/// 2026-09-26: `[benchmarks.limits.timing]`, seconds.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimingLimits {
    /// 2026-09-26: Wall time a self-served gate may spend before its first sample
    /// (server start plus checkpoint load) before its deadline counts.
    pub serve_allowance_s: u64,
    /// 2026-09-26: How long a started server may take to name its model.
    pub boot_timeout_s: u64,
    /// 2026-09-26: Added to a shard's share of the draw when planning it.
    pub shard_overhead_s: u64,
    /// 2026-09-26: The least a shard is planned at, however thin the slice.
    pub shard_floor_s: u64,
    /// 2026-09-26: Time a remote node may spend building the anchor before its
    /// unit's deadline counts.
    pub build_allowance_s: u64,
}

/// 2026-09-26: `[benchmarks.limits.equivalence]`: the non-thermal half of "one box".
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EquivalenceLimits {
    /// 2026-09-26: Largest `|a - b| / max(a, b)` on the clock ceiling.
    pub clock_spread: f64,
    /// 2026-09-26: Largest `|a - b| / max(a, b)` on memory.
    pub mem_spread: f64,
}

#[derive(serde::Deserialize)]
struct HardwareToml {
    benchmarks: Option<Benchmarks>,
}

#[derive(serde::Deserialize)]
struct Benchmarks {
    limits: Option<Limits>,
}

/// 2026-09-26: The limits `hardware` declares, `None` when its `HARDWARE.toml` has
/// no `[benchmarks.limits]`.
///
/// # Errors
/// A missing or unparseable `HARDWARE.toml`, a `[benchmarks.limits]` missing one of
/// its tables or carrying a stray key, a non-finite value, or values that
/// contradict themselves (resume at or above park, a non-positive tolerance, a
/// fraction outside `(0, 1)`, a zero serve, boot or build allowance).
pub fn limits(root: &Path, hardware: &str) -> Result<Option<Limits>> {
    let path = root.join("kernels").join(hardware).join("HARDWARE.toml");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let parsed: HardwareToml =
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let Some(l) = parsed.benchmarks.and_then(|b| b.limits) else {
        return Ok(None);
    };
    let at = |what: &str| format!("{}: [benchmarks.limits] {what}", path.display());
    let t = l.thermal;
    for (name, v) in [
        ("thermal.chassis_park_c", t.chassis_park_c),
        ("thermal.chassis_resume_c", t.chassis_resume_c),
        (
            "thermal.chassis_equivalence_delta_c",
            t.chassis_equivalence_delta_c,
        ),
        ("thermal.gpu_ceiling_c", t.gpu_ceiling_c),
        ("memory.min_free_fraction", l.memory.min_free_fraction),
        ("equivalence.clock_spread", l.equivalence.clock_spread),
        ("equivalence.mem_spread", l.equivalence.mem_spread),
    ] {
        if !v.is_finite() {
            bail!("{}", at(&format!("{name} must be finite")));
        }
    }
    if t.chassis_resume_c >= t.chassis_park_c {
        bail!(
            "{}",
            at(&format!(
                "thermal.chassis_resume_c ({}) must be below chassis_park_c ({}) — without \
                 hysteresis a box flaps on the line",
                t.chassis_resume_c, t.chassis_park_c
            ))
        );
    }
    if t.chassis_equivalence_delta_c <= 0.0 {
        bail!(
            "{}",
            at("thermal.chassis_equivalence_delta_c must be positive")
        );
    }
    let f = l.memory.min_free_fraction;
    if !(f > 0.0 && f < 1.0) {
        bail!("{}", at("memory.min_free_fraction must be inside (0, 1)"));
    }
    if l.equivalence.clock_spread <= 0.0 || l.equivalence.mem_spread <= 0.0 {
        bail!("{}", at("equivalence spreads must be positive"));
    }
    let ti = l.timing;
    if ti.serve_allowance_s == 0 || ti.boot_timeout_s == 0 || ti.build_allowance_s == 0 {
        bail!("{}", at("timing allowances must be positive"));
    }
    Ok(Some(l))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB10: &str = "[benchmarks.limits.thermal]\nchassis_park_c = 80\nchassis_resume_c = 70\n\
        chassis_equivalence_delta_c = 15\ngpu_ceiling_c = 75\n[benchmarks.limits.memory]\n\
        min_free_fraction = 0.85\n[benchmarks.limits.timing]\nserve_allowance_s = 600\n\
        boot_timeout_s = 900\nshard_overhead_s = 420\nshard_floor_s = 300\nbuild_allowance_s = 1800\n\
        [benchmarks.limits.equivalence]\nclock_spread = 0.01\nmem_spread = 0.05\n";

    fn root_with(hw: &str, body: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("limits-{}-{}", std::process::id(), hw));
        std::fs::create_dir_all(d.join("kernels").join(hw)).unwrap();
        std::fs::write(d.join("kernels").join(hw).join("HARDWARE.toml"), body).unwrap();
        d
    }

    #[test]
    fn declared_limits_are_read_and_absent_ones_are_none() {
        let r = root_with("x1", &format!("[hardware]\nname = \"x1\"\n{GB10}"));
        let l = limits(&r, "x1").unwrap().unwrap();
        assert_eq!(
            (l.thermal.chassis_park_c, l.thermal.chassis_resume_c),
            (80.0, 70.0)
        );
        assert_eq!(l.memory.min_free_fraction, 0.85);
        assert_eq!(l.timing.serve_allowance_s, 600);
        assert_eq!(l.equivalence.clock_spread, 0.01);
        let r = root_with("x2", "[hardware]\nname = \"x2\"\n");
        assert_eq!(limits(&r, "x2").unwrap(), None);
        assert!(limits(&r, "nope").is_err(), "no file is an error, not None");
    }

    /// 2026-09-26: Negative controls: a missing sub-table, a stray key and each
    /// contradiction are refused.
    #[test]
    fn a_wrong_table_is_refused_by_name() {
        let r = root_with(
            "y1",
            &GB10.replace("[benchmarks.limits.memory]\nmin_free_fraction = 0.85\n", ""),
        );
        assert!(format!("{:#}", limits(&r, "y1").unwrap_err()).contains("memory"));
        let r = root_with("y2", &format!("{GB10}chassis_typo_c = 1\n"));
        assert!(limits(&r, "y2").is_err());
        let r = root_with(
            "y3",
            &GB10.replace("chassis_resume_c = 70", "chassis_resume_c = 85"),
        );
        assert!(
            limits(&r, "y3")
                .unwrap_err()
                .to_string()
                .contains("hysteresis")
        );
        let r = root_with(
            "y4",
            &GB10.replace("min_free_fraction = 0.85", "min_free_fraction = 1.5"),
        );
        assert!(
            limits(&r, "y4")
                .unwrap_err()
                .to_string()
                .contains("min_free_fraction")
        );
        let r = root_with(
            "y5",
            &GB10.replace(
                "chassis_equivalence_delta_c = 15",
                "chassis_equivalence_delta_c = 0",
            ),
        );
        assert!(limits(&r, "y5").is_err());
        let r = root_with(
            "y6",
            &GB10.replace("serve_allowance_s = 600", "serve_allowance_s = 0"),
        );
        assert!(
            limits(&r, "y6")
                .unwrap_err()
                .to_string()
                .contains("allowances")
        );
    }

    /// 2026-09-26: The committed GB10 table reads as declared; the other classes
    /// in the tree declare none.
    #[test]
    fn gb10_declares_the_measured_limits_and_others_declare_none() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let l = limits(root, "gb10")
            .unwrap()
            .expect("gb10 declares [benchmarks.limits]");
        assert_eq!(
            (
                l.thermal.chassis_park_c,
                l.thermal.chassis_resume_c,
                l.thermal.chassis_equivalence_delta_c,
                l.thermal.gpu_ceiling_c
            ),
            (80.0, 70.0, 15.0, 75.0)
        );
        assert_eq!(l.memory.min_free_fraction, 0.85);
        assert_eq!(
            (
                l.timing.serve_allowance_s,
                l.timing.boot_timeout_s,
                l.timing.shard_overhead_s,
                l.timing.shard_floor_s,
                l.timing.build_allowance_s
            ),
            (600, 900, 420, 300, 1800)
        );
        assert_eq!(
            (l.equivalence.clock_spread, l.equivalence.mem_spread),
            (0.01, 0.05)
        );
        for hw in ["hopper", "b200", "strix", "strix-hip", "metal"] {
            assert_eq!(limits(root, hw).unwrap(), None, "{hw}");
        }
    }
}
