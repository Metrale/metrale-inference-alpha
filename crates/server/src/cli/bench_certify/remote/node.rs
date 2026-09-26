// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Which nodes may take part, and what each is for the scheduler.
//!
//! Owner: server CLI (`met benchmark certify`).
//! `admit` refuses a node, before any unit is sent and with every reason, when:
//! - it sent no report, or bench is off there;
//! - its box class is absent or not the one being certified;
//! - its signer is absent or not committed in `.github/record-signers/`;
//! - it is busy or has jobs queued;
//! - its free-memory reading is absent or below the floor;
//! - its disk reading is below its own `min_free_disk_bytes` (an absent disk
//!   reading is not a refusal);
//! - it reports no GPU name.
//!
//! Invariants: none beyond the types.

use metrale_bench::hardware::equivalence::{HardwareFingerprint, driver_major};
use metrale_bench::hardware::{Hardware, HardwareState};

use super::metralectl::{NodeInfo, NodeRow};

/// 2026-09-26: A node the campaign may run on.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// 2026-09-26: The address as given to `--with-nodes`, or `local`.
    pub addr: String,
    /// 2026-09-26: The node's display name (the hostname for this box).
    pub name: String,
    /// 2026-09-26: The metralectl node id (empty for local).
    pub node_id: String,
    /// 2026-09-26: The signing key its records will carry.
    pub signer: String,
    pub hardware: HardwareFingerprint,
    /// 2026-09-26: `MemAvailable / MemTotal` at admission; the first key for the box
    /// a bundled Speed set goes to (`schedule::bundle_home`).
    pub free_fraction: Option<f64>,
    /// 2026-09-26: Whether the anchor is already built there (no build allowance needed;
    /// always true for this box).
    pub built: bool,
    pub local: bool,
}

/// 2026-09-26: Why a node was not admitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejection {
    pub addr: String,
    pub why: String,
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.addr, self.why)
    }
}

/// 2026-09-26: The fingerprint a node's report implies. An absent fact is `None` (an
/// empty string for `gpu`), which `equivalent` treats as undecidable.
pub fn fingerprint_of(info: &NodeInfo) -> HardwareFingerprint {
    let t = info.thermal.as_ref();
    HardwareFingerprint {
        gpu: info
            .gpu
            .as_ref()
            .map(|g| g.name.clone())
            .unwrap_or_default(),
        driver_major: info
            .gpu
            .as_ref()
            .and_then(|g| driver_major(&g.driver_version)),
        sm_clock_max_mhz: t.and_then(|t| t.sm_clock_max_mhz),
        mem_total_kb: t.and_then(|t| t.mem_total_kb),
        thermal_alert: t.and_then(|t| t.throttle_thermal),
        hottest_chassis_c: t.and_then(|t| {
            t.chassis_temps_c
                .iter()
                .copied()
                .fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x))))
        }),
        postcheck_valid: None,
    }
}

/// 2026-09-26: What admission needs to know about this side.
pub struct Wanted<'a> {
    /// 2026-09-26: The box class being certified (`--hardware`, or probed).
    pub hardware: &'a str,
    /// 2026-09-26: Fingerprints committed in `.github/record-signers/`.
    pub committed_signers: &'a [String],
    /// 2026-09-26: The full anchor sha, to tell whether a node has it built.
    pub anchor: &'a str,
    /// 2026-09-26: The free-memory floor a gate needs to start.
    pub min_free_fraction: f64,
}

/// 2026-09-26: Judge one row. Every reason is collected, not just the first.
pub fn admit(row: &NodeRow, wanted: &Wanted) -> Result<Node, Rejection> {
    let reject = |why: String| Rejection {
        addr: row.node.clone(),
        why,
    };
    let Some(info) = &row.info else {
        return Err(reject(match &row.error {
            Some(e) => e.to_string(),
            None => "no report and no error — metralectl said nothing about it".into(),
        }));
    };
    let mut why = Vec::new();
    if !info.bench_enabled {
        why.push(format!(
            "bench is off there ({})",
            info.disabled_reason.as_deref().unwrap_or("no reason given")
        ));
    }
    match info.hardware_class.as_deref() {
        Some(c) if c == wanted.hardware => {}
        Some(c) => why.push(format!(
            "its box class is {c}, this campaign certifies {}",
            wanted.hardware
        )),
        None => why.push("it reports no box class".into()),
    }
    match &info.signer_fp {
        Some(fp) if wanted.committed_signers.iter().any(|s| s == fp) => {}
        Some(fp) => why.push(format!(
            "its signer {fp} is not committed in .github/record-signers/ (commit {fp}.pub first)"
        )),
        None => why.push("it reports no signing identity (no METRALE_HOME identity there)".into()),
    }
    if info.busy {
        why.push(format!(
            "it is busy ({})",
            info.busy_reason.as_deref().unwrap_or("no reason given")
        ));
    }
    if info.queued > 0 {
        why.push(format!("it already has {} job(s) queued", info.queued));
    }
    let free = info.host_free_fraction.value();
    match free {
        Some(f) if f < wanted.min_free_fraction => why.push(format!(
            "only {:.0} % of host memory is free; a gate needs {:.0} %",
            f * 100.0,
            wanted.min_free_fraction * 100.0
        )),
        Some(_) => {}
        None => why.push("it cannot report free memory".into()),
    }
    if let Some(d) = info.disk_free_bytes.value()
        && (d as u64) < info.min_free_disk_bytes
    {
        why.push(format!(
            "only {} MiB free on its bench cache; it requires {} MiB",
            (d as u64) >> 20,
            info.min_free_disk_bytes >> 20
        ));
    }
    let hw = fingerprint_of(info);
    if hw.gpu.is_empty() {
        why.push("it reports no GPU".into());
    }
    if !why.is_empty() {
        return Err(reject(why.join("; ")));
    }
    Ok(Node {
        addr: row.node.clone(),
        name: info.name.clone(),
        node_id: info.node.clone(),
        signer: info.signer_fp.clone().unwrap_or_default(),
        hardware: hw,
        free_fraction: free,
        built: info.built_shas.iter().any(|b| b.sha == wanted.anchor),
        local: false,
    })
}

/// 2026-09-26: This machine as a node. It is not judged by `admit`.
pub fn local(signer: &str, hardware: &Hardware, state: &HardwareState) -> Node {
    Node {
        addr: "local".into(),
        name: state
            .machine
            .hostname
            .clone()
            .unwrap_or_else(|| "local".into()),
        node_id: String::new(),
        signer: signer.to_owned(),
        hardware: HardwareFingerprint::from_live(hardware, state),
        free_fraction: match (state.mem_available_kb, state.mem_total_kb) {
            (Some(a), Some(t)) if t > 0 => Some(a as f64 / t as f64),
            _ => None,
        },
        built: true,
        local: true,
    }
}

#[cfg(test)]
#[path = "node_tests.rs"]
mod node_tests;
