// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The I/O half of the hardware-state collector: run `nvidia-smi`,
//! read procfs and sysfs, and hand the text to [`super::parse`], which does
//! the interpreting.
//!
//! A tool that is missing, fails, exits non-zero or prints nothing, and a file
//! that cannot be read, leave the fields they would have filled at their
//! defaults, with one exception: a compute-apps query that gives no output is
//! recorded as an empty list when `nvidia-smi --list-gpus` answers.
//! [`super::policy`] decides what a missing reading means.
//!
//! [`collect`] blocks: it runs subprocesses, and `std::process` has no
//! timeout. The bench executor calls it through `spawn_blocking`
//! (`executor.rs`).
//!
//! Owner: bench hardware.
//! Invariants:
//! - `collect` returns a `HardwareState` on every path; it has no error type.
//! - `sources` names only sources that answered.

use std::time::{SystemTime, UNIX_EPOCH};

use super::parse;
use super::state::{HardwareState, MachineIdentity, ThermalZone};

/// 2026-09-26: Run a tool and return its stdout when it exits 0 with
/// non-blank output; otherwise `None`.
fn run(tool: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(tool)
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).to_string())
        .filter(|s| !s.trim().is_empty())
}

fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 2026-09-26: Every readable `/sys/class/thermal/thermal_zone*`, in numeric
/// zone order; a zone whose `temp` cannot be read or parsed is left out.
///
/// `None` when the directory itself cannot be read, which is a different fact
/// from a box reporting zero zones.
fn thermal_zones() -> Option<Vec<ThermalZone>> {
    let mut entries: Vec<_> = std::fs::read_dir("/sys/class/thermal")
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("thermal_zone"))
        })
        .collect();
    // 2026-09-26: Numeric, not lexical, order: zone10 sorts after zone2.
    entries.sort_by_key(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.trim_start_matches("thermal_zone").parse::<u32>().ok())
            .unwrap_or(u32::MAX)
    });
    Some(
        entries
            .iter()
            .filter_map(|p| {
                let temp = std::fs::read_to_string(p.join("temp")).ok()?;
                let kind = std::fs::read_to_string(p.join("type")).ok();
                parse::thermal_zone(kind.as_deref(), &temp)
            })
            .collect(),
    )
}

/// 2026-09-26: The scaling governor of cpu0 only; other cores are not read.
fn cpu_governor() -> Option<String> {
    read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").map(|s| s.trim().to_string())
}

fn machine() -> MachineIdentity {
    MachineIdentity {
        // 2026-09-26: Read from procfs, which costs no subprocess.
        hostname: read("/proc/sys/kernel/hostname")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
        machine_id: read("/etc/machine-id")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
        gpu: None,
        driver: None,
    }
}

/// 2026-09-26: Capture the local box's state. Each source fills only its own
/// fields, and one that does not answer leaves them at their defaults, except
/// the compute-apps list (see the module doc).
pub fn collect() -> HardwareState {
    let mut sources = Vec::new();
    let mut state = HardwareState {
        captured_at: now_secs(),
        machine: machine(),
        ..HardwareState::default()
    };

    if let Some(text) = run(
        "nvidia-smi",
        &[
            "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,persistence_mode",
            "--format=csv,noheader,nounits",
        ],
    ) {
        let q = parse::gpu_query(&text);
        state.machine.gpu = q.name;
        state.machine.driver = q.driver;
        state.sm_clock_mhz = q.sm_clock_mhz;
        state.sm_clock_max_mhz = q.sm_clock_max_mhz;
        state.gpu_temp_c = q.gpu_temp_c;
        state.persistence_mode = q.persistence_mode;
        sources.push("nvidia-smi".to_string());
    }

    if let Some(text) = run("nvidia-smi", &["-q", "-d", "PERFORMANCE"]) {
        let (counters, active) = parse::performance(&text);
        state.throttle_counters = counters;
        state.throttle_active = active;
        if !sources.iter().any(|s| s == "nvidia-smi") {
            sources.push("nvidia-smi".to_string());
        }
    }

    // 2026-09-26: Per-process `used_memory` from the compute-apps query, not
    // `--query-gpu=memory.used`, which reads `[N/A]` on GB10 (checked
    // 2026-09-26).
    if let Some(text) = run(
        "nvidia-smi",
        &[
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    ) {
        state.gpu_compute_apps = Some(parse::compute_apps(&text));
    } else if run("nvidia-smi", &["--list-gpus"]).is_some() {
        // 2026-09-26: The compute-apps query failed or printed nothing, but
        // `--list-gpus` answers, so this is recorded as an empty list (an idle
        // GPU). Left `None`, the policy would warn that processes could not be
        // listed (`policy.rs`).
        state.gpu_compute_apps = Some(Vec::new());
    }

    if let Some(text) = read("/proc/meminfo") {
        let m = parse::meminfo(&text);
        state.mem_total_kb = m.total_kb;
        state.mem_available_kb = m.available_kb;
        state.page_cache_kb = m.cached_kb;
        sources.push("procfs".to_string());
    }

    state.chassis_temps_c = thermal_zones();
    if state.chassis_temps_c.is_some() {
        sources.push("sysfs".to_string());
    }
    state.cpu_governor = cpu_governor();

    state.sources = sources;
    state
}
