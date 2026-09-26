// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The telemetry section of `/metrics` in the Prometheus text
//! format, which the server appends to its registry output.
//!
//! Names carry their unit (`_seconds`, `_joules`, `_watts`, `_bytes`,
//! `_celsius`); the NVML-sourced energy counters, clocks and PCIe throughput
//! keep NVML's units (`_millijoules`, `_mhz`, `_kbps`). TTFT is not exported
//! here: the server's `metrale_time_to_first_token_seconds` carries it,
//! labelled by model.
//!
//! Owner: telemetry.
//! Invariants: the section is empty at level `Off`.

use std::fmt::Write;

use crate::device::CLOCK_EVENT_REASONS;
use crate::hub::Telemetry;
use crate::level::Level;
use crate::snapshot::TelemetrySnapshot;

use super::prometheus_layers::{render_kernel, render_sched, render_spec_and_requests};
use super::prometheus_text::{head, histogram, one, opt};

/// 2026-09-26: Render the telemetry section. `kernel_name` names a kernel
/// handle; a handle it cannot name is exported by address.
pub fn render(t: &Telemetry, kernel_name: &dyn Fn(u64) -> Option<String>) -> String {
    let mut out = String::new();
    if t.level() == Level::Off {
        return out;
    }
    let snap = t.snapshot();
    render_level_and_device(&mut out, &snap);
    render_energy(&mut out, t, &snap);
    render_sched(&mut out, t, &snap);
    render_spec_and_requests(&mut out, t, &snap);
    if t.level() == Level::Kernel {
        render_kernel(&mut out, t, kernel_name);
    }
    out
}
fn render_level_and_device(out: &mut String, snap: &TelemetrySnapshot) {
    head(
        out,
        "metrale_telemetry_level",
        "gauge",
        "Telemetry level in force (1 on the active label)",
    );
    for l in Level::NAMES {
        let _ = writeln!(
            out,
            "metrale_telemetry_level{{level=\"{l}\"}} {}",
            u8::from(l == snap.level)
        );
    }
    one(
        out,
        "metrale_gpu_device_available",
        "gauge",
        "1 while the NVML device layer is reading",
        u8::from(snap.device_state == "available"),
    );
    let Some(d) = &snap.device else { return };
    let r = &d.reading;
    one(
        out,
        "metrale_gpu_samples_total",
        "counter",
        "Device samples taken",
        d.samples,
    );
    one(
        out,
        "metrale_gpu_sample_age_seconds",
        "gauge",
        "Age of the latest device sample",
        d.age_ns as f64 / 1e9,
    );
    one(
        out,
        "metrale_gpu_read_errors_total",
        "counter",
        "NVML field reads that failed",
        d.read_errors,
    );
    opt(
        out,
        "metrale_gpu_power_watts",
        "GPU-rail power draw (nvmlDeviceGetPowerUsage)",
        r.power_mw.map(|v| v as f64 / 1e3),
    );
    opt(
        out,
        "metrale_gpu_temperature_celsius",
        "GPU temperature",
        r.temperature_c.map(|v| v as f64),
    );
    head(
        out,
        "metrale_gpu_clock_mhz",
        "gauge",
        "Current clocks by domain",
    );
    for (dom, v) in [
        ("graphics", r.graphics_clock_mhz),
        ("sm", r.sm_clock_mhz),
        ("mem", r.mem_clock_mhz),
    ] {
        if let Some(v) = v {
            let _ = writeln!(out, "metrale_gpu_clock_mhz{{clock=\"{dom}\"}} {v}");
        }
    }
    if let Some(mask) = r.clocks_event_reasons {
        one(
            out,
            "metrale_gpu_clocks_event_reasons",
            "gauge",
            "NVML clocks-event-reasons bitmask",
            mask,
        );
        head(
            out,
            "metrale_gpu_clocks_event_reason_active",
            "gauge",
            "1 while the named clock-event reason is asserted",
        );
        for (name, bit) in CLOCK_EVENT_REASONS {
            let _ = writeln!(
                out,
                "metrale_gpu_clocks_event_reason_active{{reason=\"{name}\"}} {}",
                u8::from(mask & bit != 0)
            );
        }
    }
    opt(
        out,
        "metrale_gpu_memory_used_bytes",
        "Device memory in use (NVML)",
        r.mem_used_bytes.map(|v| v as f64),
    );
    opt(
        out,
        "metrale_gpu_memory_total_bytes",
        "Device memory total (NVML)",
        r.mem_total_bytes.map(|v| v as f64),
    );
    if r.pcie_tx_kbps.is_some() || r.pcie_rx_kbps.is_some() {
        head(
            out,
            "metrale_gpu_pcie_throughput_kbps",
            "gauge",
            "PCIe throughput by direction",
        );
        for (dir, v) in [("tx", r.pcie_tx_kbps), ("rx", r.pcie_rx_kbps)] {
            if let Some(v) = v {
                let _ = writeln!(
                    out,
                    "metrale_gpu_pcie_throughput_kbps{{direction=\"{dir}\"}} {v}"
                );
            }
        }
    }
    opt(
        out,
        "metrale_gpu_nvlink_active_links",
        "Active NVLink links",
        r.nvlink_active_links.map(|v| v as f64),
    );
}
fn render_energy(out: &mut String, t: &Telemetry, snap: &TelemetrySnapshot) {
    let e = &snap.energy;
    one(
        out,
        "metrale_gpu_energy_counter_millijoules",
        "gauge",
        "Raw NVML total-energy counter at the latest sample (GPU rail, mJ since driver load)",
        e.gpu_counter_mj,
    );
    one(
        out,
        "metrale_gpu_energy_millijoules_total",
        "counter",
        "GPU-rail energy since the first sample, reset- and wrap-corrected",
        e.gpu_mj_total,
    );
    one(
        out,
        "metrale_gpu_energy_counter_resets_total",
        "counter",
        "Times the NVML energy counter restarted",
        e.counter_resets,
    );
    one(
        out,
        "metrale_gpu_energy_counter_wraps_total",
        "counter",
        "Times the NVML energy counter wrapped",
        e.counter_wraps,
    );
    one(
        out,
        "metrale_energy_tokens_total",
        "counter",
        "Tokens emitted since the first energy sample (the J/token denominator)",
        e.tokens_total,
    );
    opt(
        out,
        "metrale_energy_joules_per_token",
        "GPU-rail J/token over the last second of samples",
        e.joules_per_token_live,
    );
    opt(
        out,
        "metrale_energy_joules_per_token_cumulative",
        "GPU-rail J/token since the first energy sample",
        e.joules_per_token_cumulative,
    );
    histogram(
        out,
        "metrale_request_energy_joules",
        "GPU-rail joules attributed to each finished request",
        &t.requests.energy.snapshot(),
        1e3,
    );
    one(
        out,
        "metrale_request_energy_unattributed_total",
        "counter",
        "Finished requests with no energy window to attribute from",
        snap.requests.energy_unattributed,
    );
}
#[cfg(test)]
#[path = "prometheus_tests.rs"]
mod tests;
