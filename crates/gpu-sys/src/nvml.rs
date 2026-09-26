// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: NVML (the NVIDIA management library), resolved with `dlopen`.
//!
//! There is no link-time dependency: on a host without the driver,
//! `NvmlLibrary::open` returns `NvmlUnavailable` instead of the process
//! failing to load. Only `nvmlInit_v2`, `nvmlShutdown` and
//! `nvmlDeviceGetHandleByIndex_v2` are required; a missing query symbol leaves
//! its `NvmlFns` slot `None`.
//!
//! `NvmlFns` is a table of function pointers. `NvmlLibrary` fills it from the
//! shared object; tests, here and in the telemetry crate, fill it with
//! functions of their own through `Nvml::with_fns`. `Nvml` and `NvmlDevice`
//! run the same code over either.
//!
//! Values are in NVML's units, which the accessor names carry: `power_mw`
//! (milliwatts), `total_energy_mj` (millijoules), `clock_mhz`, `temperature_c`,
//! `pcie_kbps` (KB/s); `memory` is in bytes.
//!
//! Owner: metrale-gpu-sys.
//! Invariants:
//! - An `Nvml` exists only after its table's `init` succeeded, and its drop
//!   calls the table's `shutdown` once.

#![allow(non_camel_case_types)]

use std::ffi::{c_uint, c_ulonglong, c_void};

use libloading::Library;

/// 2026-09-26: `nvmlReturn_t`.
pub type NvmlReturn = std::ffi::c_int;
/// 2026-09-26: `nvmlDevice_t`, an opaque handle owned by the library.
pub type NvmlDeviceHandle = *mut c_void;

pub const NVML_SUCCESS: NvmlReturn = 0;
pub const NVML_ERROR_NOT_SUPPORTED: NvmlReturn = 3;
pub const NVML_ERROR_NO_PERMISSION: NvmlReturn = 4;

/// 2026-09-26: `nvmlClockType_t` values.
pub const NVML_CLOCK_GRAPHICS: c_uint = 0;
pub const NVML_CLOCK_SM: c_uint = 1;
pub const NVML_CLOCK_MEM: c_uint = 2;
/// 2026-09-26: `nvmlTemperatureSensors_t::NVML_TEMPERATURE_GPU`.
pub const NVML_TEMPERATURE_GPU: c_uint = 0;
/// 2026-09-26: `nvmlPcieUtilCounter_t` values.
pub const NVML_PCIE_UTIL_TX_BYTES: c_uint = 0;
pub const NVML_PCIE_UTIL_RX_BYTES: c_uint = 1;
/// 2026-09-26: `NVML_NVLINK_MAX_LINKS` in the CUDA toolkit's nvml.h.
pub const NVML_NVLINK_MAX_LINKS: c_uint = 18;

/// 2026-09-26: `nvmlMemory_t` (v1), in bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NvmlMemory {
    pub total: c_ulonglong,
    pub free: c_ulonglong,
    pub used: c_ulonglong,
}

pub type FnVoid = unsafe extern "C" fn() -> NvmlReturn;
pub type FnHandleByIndex = unsafe extern "C" fn(c_uint, *mut NvmlDeviceHandle) -> NvmlReturn;
pub type FnDevU32 = unsafe extern "C" fn(NvmlDeviceHandle, *mut c_uint) -> NvmlReturn;
pub type FnDevU64 = unsafe extern "C" fn(NvmlDeviceHandle, *mut c_ulonglong) -> NvmlReturn;
pub type FnDevArgU32 = unsafe extern "C" fn(NvmlDeviceHandle, c_uint, *mut c_uint) -> NvmlReturn;
pub type FnDevMemory = unsafe extern "C" fn(NvmlDeviceHandle, *mut NvmlMemory) -> NvmlReturn;

/// 2026-09-26: The entry points this crate binds. The first three are required;
/// each query is `None` when the library does not export it.
#[derive(Clone, Copy, Debug)]
pub struct NvmlFns {
    pub init: FnVoid,
    pub shutdown: FnVoid,
    pub device_by_index: FnHandleByIndex,
    /// 2026-09-26: `nvmlDeviceGetPowerUsage`, milliwatts.
    pub power_usage: Option<FnDevU32>,
    /// 2026-09-26: `nvmlDeviceGetTotalEnergyConsumption`, millijoules since the
    /// driver was last reloaded.
    pub total_energy: Option<FnDevU64>,
    /// 2026-09-26: `nvmlDeviceGetClockInfo(type)`, MHz.
    pub clock_info: Option<FnDevArgU32>,
    /// 2026-09-26: `nvmlDeviceGetTemperature(sensor)`, degrees C.
    pub temperature: Option<FnDevArgU32>,
    /// 2026-09-26: `nvmlDeviceGetCurrentClocksEventReasons`, or
    /// `nvmlDeviceGetCurrentClocksThrottleReasons` when the library lacks it.
    pub clocks_event_reasons: Option<FnDevU64>,
    /// 2026-09-26: `nvmlDeviceGetMemoryInfo`.
    pub memory_info: Option<FnDevMemory>,
    /// 2026-09-26: `nvmlDeviceGetPcieThroughput(counter)`, KB/s.
    pub pcie_throughput: Option<FnDevArgU32>,
    /// 2026-09-26: `nvmlDeviceGetNvLinkState(link)`, 1 = active.
    pub nvlink_state: Option<FnDevArgU32>,
}

/// 2026-09-26: Why NVML cannot be used. The telemetry sampler records it as
/// the device being unavailable and serving continues.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NvmlUnavailable {
    /// 2026-09-26: No candidate shared object could be opened.
    NotInstalled(String),
    /// 2026-09-26: The object opened but lacks a required entry point.
    MissingSymbol(&'static str),
    /// 2026-09-26: `nvmlInit_v2` returned this code.
    InitFailed(NvmlReturn),
    /// 2026-09-26: `nvmlDeviceGetHandleByIndex_v2(index)` returned this code.
    NoDevice { index: u32, code: NvmlReturn },
}

impl std::fmt::Display for NvmlUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInstalled(why) => write!(f, "NVML library not loadable: {why}"),
            Self::MissingSymbol(s) => write!(f, "NVML library lacks required symbol {s}"),
            Self::InitFailed(c) => write!(f, "nvmlInit_v2 failed with code {c}"),
            Self::NoDevice { index, code } => {
                write!(f, "NVML has no device {index} (code {code})")
            }
        }
    }
}

impl std::error::Error for NvmlUnavailable {}

/// 2026-09-26: The shared object names tried, in order.
pub const LIBRARY_NAMES: [&str; 2] = ["libnvidia-ml.so.1", "libnvidia-ml.so"];

/// 2026-09-26: An opened `libnvidia-ml` and the table resolved from it. The
/// library stays loaded as long as this value lives.
pub struct NvmlLibrary {
    fns: NvmlFns,
    _lib: Library,
}

impl NvmlLibrary {
    /// 2026-09-26: `open_from(&LIBRARY_NAMES)`.
    pub fn open() -> Result<Self, NvmlUnavailable> {
        Self::open_from(&LIBRARY_NAMES)
    }

    /// 2026-09-26: Open the first of `candidates` that loads. When that one
    /// lacks a required symbol, the error is returned and later candidates are
    /// not tried.
    pub fn open_from(candidates: &[&str]) -> Result<Self, NvmlUnavailable> {
        let mut last = String::from("no candidate names");
        for name in candidates {
            // 2026-09-26: SAFETY: loading runs the object's initialisers; NVML
            // itself is initialised only by the `nvmlInit_v2` call in
            // `Nvml::init`.
            match unsafe { Library::new(name) } {
                Ok(lib) => {
                    let fns = resolve(&lib)?;
                    return Ok(Self { fns, _lib: lib });
                }
                Err(e) => last = format!("{name}: {e}"),
            }
        }
        Err(NvmlUnavailable::NotInstalled(last))
    }

    pub fn fns(&self) -> NvmlFns {
        self.fns
    }
}

fn required<T: Copy>(lib: &Library, name: &'static str) -> Result<T, NvmlUnavailable> {
    optional(lib, name).ok_or(NvmlUnavailable::MissingSymbol(name))
}

fn optional<T: Copy>(lib: &Library, name: &str) -> Option<T> {
    let mut sym = Vec::with_capacity(name.len() + 1);
    sym.extend_from_slice(name.as_bytes());
    sym.push(0);
    // 2026-09-26: SAFETY: `resolve` calls this with each `name`'s signature
    // from nvml.h.
    unsafe { lib.get::<T>(&sym).ok().map(|s| *s) }
}

fn resolve(lib: &Library) -> Result<NvmlFns, NvmlUnavailable> {
    Ok(NvmlFns {
        init: required(lib, "nvmlInit_v2")?,
        shutdown: required(lib, "nvmlShutdown")?,
        device_by_index: required(lib, "nvmlDeviceGetHandleByIndex_v2")?,
        power_usage: optional(lib, "nvmlDeviceGetPowerUsage"),
        total_energy: optional(lib, "nvmlDeviceGetTotalEnergyConsumption"),
        clock_info: optional(lib, "nvmlDeviceGetClockInfo"),
        temperature: optional(lib, "nvmlDeviceGetTemperature"),
        clocks_event_reasons: optional(lib, "nvmlDeviceGetCurrentClocksEventReasons")
            .or_else(|| optional(lib, "nvmlDeviceGetCurrentClocksThrottleReasons")),
        memory_info: optional(lib, "nvmlDeviceGetMemoryInfo"),
        pcie_throughput: optional(lib, "nvmlDeviceGetPcieThroughput"),
        nvlink_state: optional(lib, "nvmlDeviceGetNvLinkState"),
    })
}

/// 2026-09-26: An initialised NVML session over a function table.
/// `nvmlShutdown` runs on drop, balancing the successful init.
pub struct Nvml {
    fns: NvmlFns,
    /// 2026-09-26: The opened library that keeps `fns` valid; `None` for a
    /// table of functions linked into the binary.
    _keep: Option<NvmlLibrary>,
}

impl Nvml {
    /// 2026-09-26: Open and initialise the system library.
    pub fn open() -> Result<Self, NvmlUnavailable> {
        let lib = NvmlLibrary::open()?;
        let fns = lib.fns();
        Self::init(fns, Some(lib))
    }

    /// 2026-09-26: Initialise over a table whose functions stay valid for the
    /// life of the process. Tests use it.
    pub fn with_fns(fns: NvmlFns) -> Result<Self, NvmlUnavailable> {
        Self::init(fns, None)
    }

    fn init(fns: NvmlFns, keep: Option<NvmlLibrary>) -> Result<Self, NvmlUnavailable> {
        // 2026-09-26: SAFETY: `init` is `nvmlInit_v2` or a test function of
        // that type.
        let rc = unsafe { (fns.init)() };
        if rc != NVML_SUCCESS {
            return Err(NvmlUnavailable::InitFailed(rc));
        }
        Ok(Self { fns, _keep: keep })
    }

    /// 2026-09-26: The device at NVML index `index`.
    pub fn device(&self, index: u32) -> Result<NvmlDevice<'_>, NvmlUnavailable> {
        let mut handle: NvmlDeviceHandle = std::ptr::null_mut();
        // 2026-09-26: SAFETY: `handle` is a valid out-pointer.
        let rc = unsafe { (self.fns.device_by_index)(index, &mut handle) };
        if rc != NVML_SUCCESS {
            return Err(NvmlUnavailable::NoDevice { index, code: rc });
        }
        Ok(NvmlDevice {
            fns: &self.fns,
            handle,
        })
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        // 2026-09-26: SAFETY: balances the successful init in `Nvml::init`. A
        // failed shutdown is ignored: this side holds nothing to release.
        let _ = unsafe { (self.fns.shutdown)() };
    }
}

/// 2026-09-26: One device's queries. `Ok(None)` when the entry point is absent
/// or answers `NVML_ERROR_NOT_SUPPORTED` or `NVML_ERROR_NO_PERMISSION`;
/// `Err(code)` for any other failure.
pub struct NvmlDevice<'a> {
    fns: &'a NvmlFns,
    handle: NvmlDeviceHandle,
}

// 2026-09-26: SAFETY: nvml.h states that NVML is thread-safe; the table is
// plain function pointers.
unsafe impl Send for NvmlDevice<'_> {}

fn answer<T>(rc: NvmlReturn, v: T) -> Result<Option<T>, NvmlReturn> {
    match rc {
        NVML_SUCCESS => Ok(Some(v)),
        NVML_ERROR_NOT_SUPPORTED | NVML_ERROR_NO_PERMISSION => Ok(None),
        other => Err(other),
    }
}

impl NvmlDevice<'_> {
    fn u32_query(&self, f: Option<FnDevU32>) -> Result<Option<u32>, NvmlReturn> {
        let Some(f) = f else { return Ok(None) };
        let mut v: c_uint = 0;
        // 2026-09-26: SAFETY: `f` has the `(device, unsigned*)` signature; `v`
        // is a valid out-pointer.
        answer(unsafe { f(self.handle, &mut v) }, v)
    }

    fn u64_query(&self, f: Option<FnDevU64>) -> Result<Option<u64>, NvmlReturn> {
        let Some(f) = f else { return Ok(None) };
        let mut v: c_ulonglong = 0;
        // 2026-09-26: SAFETY: `f` has the `(device, unsigned long long*)`
        // signature.
        answer(unsafe { f(self.handle, &mut v) }, v)
    }

    fn arg_query(&self, f: Option<FnDevArgU32>, arg: c_uint) -> Result<Option<u32>, NvmlReturn> {
        let Some(f) = f else { return Ok(None) };
        let mut v: c_uint = 0;
        // 2026-09-26: SAFETY: `f` has the `(device, unsigned, unsigned*)`
        // signature.
        answer(unsafe { f(self.handle, arg, &mut v) }, v)
    }

    pub fn power_mw(&self) -> Result<Option<u32>, NvmlReturn> {
        self.u32_query(self.fns.power_usage)
    }

    pub fn total_energy_mj(&self) -> Result<Option<u64>, NvmlReturn> {
        self.u64_query(self.fns.total_energy)
    }

    pub fn clock_mhz(&self, clock: c_uint) -> Result<Option<u32>, NvmlReturn> {
        self.arg_query(self.fns.clock_info, clock)
    }

    pub fn temperature_c(&self) -> Result<Option<u32>, NvmlReturn> {
        self.arg_query(self.fns.temperature, NVML_TEMPERATURE_GPU)
    }

    pub fn clocks_event_reasons(&self) -> Result<Option<u64>, NvmlReturn> {
        self.u64_query(self.fns.clocks_event_reasons)
    }

    pub fn memory(&self) -> Result<Option<NvmlMemory>, NvmlReturn> {
        let Some(f) = self.fns.memory_info else {
            return Ok(None);
        };
        let mut m = NvmlMemory::default();
        // 2026-09-26: SAFETY: `m` is a valid `nvmlMemory_t` out-pointer.
        answer(unsafe { f(self.handle, &mut m) }, m)
    }

    pub fn pcie_kbps(&self, counter: c_uint) -> Result<Option<u32>, NvmlReturn> {
        self.arg_query(self.fns.pcie_throughput, counter)
    }

    /// 2026-09-26: The number of links in `0..NVML_NVLINK_MAX_LINKS` whose
    /// state query succeeds with state 1; `None` when no link query succeeds.
    /// A failed per-link query counts as no such link, not as an error.
    pub fn nvlink_active_links(&self) -> Option<u32> {
        let f = self.fns.nvlink_state?;
        let mut active = 0u32;
        let mut any = false;
        for link in 0..NVML_NVLINK_MAX_LINKS {
            let mut state: c_uint = 0;
            // 2026-09-26: SAFETY: `f` has the `(device, unsigned, unsigned*)`
            // signature.
            if unsafe { f(self.handle, link, &mut state) } == NVML_SUCCESS {
                any = true;
                active += u32::from(state == 1);
            }
        }
        any.then_some(active)
    }
}

#[cfg(test)]
#[path = "nvml_tests.rs"]
mod tests;
