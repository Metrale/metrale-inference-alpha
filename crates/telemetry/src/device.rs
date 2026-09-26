// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: L0: what the GPU reports through NVML: power, the cumulative
//! energy counter, clocks, temperature, clock-event reasons, memory, PCIe and
//! NVLink.
//!
//! [`DeviceSource`] is the I/O seam: [`NvmlSource`] reads the driver, and a
//! test supplies its own source or, for the NVML path itself, its own NVML
//! function table (`nvml_source_tests`). Energy is the driver's counter
//! (`nvmlDeviceGetTotalEnergyConsumption`), not an integral of sampled power.
//!
//! Owner: telemetry.
//! Invariants: a field [`NvmlSource`] cannot read is `None`, never 0.

use metrale_gpu_sys::nvml::{
    NVML_CLOCK_GRAPHICS, NVML_CLOCK_MEM, NVML_CLOCK_SM, NVML_PCIE_UTIL_RX_BYTES,
    NVML_PCIE_UTIL_TX_BYTES, Nvml, NvmlFns, NvmlUnavailable,
};

/// 2026-09-26: One reading. `None`: the device or driver does not report the
/// field, or reading it failed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct DeviceReading {
    pub power_mw: Option<u64>,
    /// 2026-09-26: The raw cumulative counter, millijoules since driver load.
    pub energy_counter_mj: Option<u64>,
    pub graphics_clock_mhz: Option<u64>,
    pub sm_clock_mhz: Option<u64>,
    pub mem_clock_mhz: Option<u64>,
    pub temperature_c: Option<u64>,
    /// 2026-09-26: NVML clocks-event-reasons bitmask.
    pub clocks_event_reasons: Option<u64>,
    pub mem_used_bytes: Option<u64>,
    pub mem_total_bytes: Option<u64>,
    pub pcie_tx_kbps: Option<u64>,
    pub pcie_rx_kbps: Option<u64>,
    pub nvlink_active_links: Option<u64>,
}

/// 2026-09-26: The number of [`DeviceReading`] fields; `fields` lists them in
/// declaration order.
const FIELDS: usize = 12;
/// 2026-09-26: Words in the encoded form: a presence mask, the sample time,
/// the fields.
pub const SAMPLE_WORDS: usize = FIELDS + 2;

impl DeviceReading {
    fn fields(&self) -> [Option<u64>; FIELDS] {
        [
            self.power_mw,
            self.energy_counter_mj,
            self.graphics_clock_mhz,
            self.sm_clock_mhz,
            self.mem_clock_mhz,
            self.temperature_c,
            self.clocks_event_reasons,
            self.mem_used_bytes,
            self.mem_total_bytes,
            self.pcie_tx_kbps,
            self.pcie_rx_kbps,
            self.nvlink_active_links,
        ]
    }

    fn from_fields(f: [Option<u64>; FIELDS]) -> Self {
        Self {
            power_mw: f[0],
            energy_counter_mj: f[1],
            graphics_clock_mhz: f[2],
            sm_clock_mhz: f[3],
            mem_clock_mhz: f[4],
            temperature_c: f[5],
            clocks_event_reasons: f[6],
            mem_used_bytes: f[7],
            mem_total_bytes: f[8],
            pcie_tx_kbps: f[9],
            pcie_rx_kbps: f[10],
            nvlink_active_links: f[11],
        }
    }

    /// 2026-09-26: The word form a [`crate::seqcell::SeqCell`] carries.
    pub fn encode(&self, at_ns: u64) -> [u64; SAMPLE_WORDS] {
        let mut w = [0u64; SAMPLE_WORDS];
        w[1] = at_ns;
        for (i, f) in self.fields().into_iter().enumerate() {
            if let Some(v) = f {
                w[0] |= 1 << i;
                w[i + 2] = v;
            }
        }
        w
    }

    /// 2026-09-26: Inverse of [`DeviceReading::encode`]: the reading and its time.
    pub fn decode(w: &[u64; SAMPLE_WORDS]) -> (Self, u64) {
        let mut f = [None; FIELDS];
        for (i, slot) in f.iter_mut().enumerate() {
            if w[0] & (1 << i) != 0 {
                *slot = Some(w[i + 2]);
            }
        }
        (Self::from_fields(f), w[1])
    }
}

/// 2026-09-26: Named bits of the clocks-event-reasons mask, as `nvml.h`
/// defines them.
pub const CLOCK_EVENT_REASONS: [(&str, u64); 6] = [
    ("gpu_idle", 0x1),
    ("sw_power_cap", 0x4),
    ("hw_slowdown", 0x8),
    ("sw_thermal_slowdown", 0x20),
    ("hw_thermal_slowdown", 0x40),
    ("hw_power_brake_slowdown", 0x80),
];

/// 2026-09-26: Where device readings come from.
pub trait DeviceSource: Send {
    /// 2026-09-26: One reading, now. Never fails as a whole: an unreadable
    /// field is `None`.
    fn read(&mut self) -> DeviceReading;
    /// 2026-09-26: Failed reads so far. A "not supported" or "no permission"
    /// answer is not a failure. `NvmlSource` does not count the NVLink state
    /// queries: a link that does not answer is skipped.
    fn read_errors(&self) -> u64;
}

/// 2026-09-26: NVML-backed source for one device ordinal.
pub struct NvmlSource {
    nvml: Nvml,
    index: u32,
    errors: u64,
}

impl NvmlSource {
    /// 2026-09-26: Over the system library. `Err` when NVML is not loadable or
    /// has no such device.
    pub fn open(index: u32) -> Result<Self, NvmlUnavailable> {
        Self::checked(Nvml::open()?, index)
    }

    /// 2026-09-26: Over an explicit NVML function table.
    pub fn with_fns(fns: NvmlFns, index: u32) -> Result<Self, NvmlUnavailable> {
        Self::checked(Nvml::with_fns(fns)?, index)
    }

    fn checked(nvml: Nvml, index: u32) -> Result<Self, NvmlUnavailable> {
        nvml.device(index)?;
        Ok(Self {
            nvml,
            index,
            errors: 0,
        })
    }
}

impl DeviceSource for NvmlSource {
    fn read(&mut self) -> DeviceReading {
        let Ok(dev) = self.nvml.device(self.index) else {
            self.errors += 1;
            return DeviceReading::default();
        };
        let mut errors = 0u64;
        let mut ok = |r: Result<Option<u64>, i32>| match r {
            Ok(v) => v,
            Err(_) => {
                errors += 1;
                None
            }
        };
        let widen = |r: Result<Option<u32>, i32>| r.map(|v| v.map(u64::from));
        let memory = dev.memory();
        let reading = DeviceReading {
            power_mw: ok(widen(dev.power_mw())),
            energy_counter_mj: ok(dev.total_energy_mj()),
            graphics_clock_mhz: ok(widen(dev.clock_mhz(NVML_CLOCK_GRAPHICS))),
            sm_clock_mhz: ok(widen(dev.clock_mhz(NVML_CLOCK_SM))),
            mem_clock_mhz: ok(widen(dev.clock_mhz(NVML_CLOCK_MEM))),
            temperature_c: ok(widen(dev.temperature_c())),
            clocks_event_reasons: ok(dev.clocks_event_reasons()),
            mem_used_bytes: ok(memory.map(|m| m.map(|m| m.used))),
            mem_total_bytes: memory.ok().flatten().map(|m| m.total),
            pcie_tx_kbps: ok(widen(dev.pcie_kbps(NVML_PCIE_UTIL_TX_BYTES))),
            pcie_rx_kbps: ok(widen(dev.pcie_kbps(NVML_PCIE_UTIL_RX_BYTES))),
            nvlink_active_links: dev.nvlink_active_links().map(u64::from),
        };
        self.errors += errors;
        reading
    }

    fn read_errors(&self) -> u64 {
        self.errors
    }
}

#[cfg(test)]
#[path = "nvml_source_tests.rs"]
mod tests;
