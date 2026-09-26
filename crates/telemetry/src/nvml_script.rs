// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A scripted NVML for tests: an [`NvmlFns`] table of
//! `extern "C"` functions that answer from the calling thread's [`Script`].
//! `Nvml`, `NvmlDevice`, `NvmlSource` and the hub run over it unchanged.
//!
//! Owner: telemetry.
//! Invariants: the script is thread-local, so parallel tests cannot see one
//! another's scripts.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::{c_uint, c_ulonglong};

use metrale_gpu_sys::nvml::{
    NVML_ERROR_NOT_SUPPORTED, NVML_SUCCESS, NvmlDeviceHandle, NvmlFns, NvmlMemory, NvmlReturn,
};

/// 2026-09-26: What the scripted driver answers.
#[derive(Clone, Debug)]
pub struct Script {
    /// 2026-09-26: Successive energy-counter readings (mJ); the last repeats.
    pub energy_mj: VecDeque<u64>,
    pub power_mw: u32,
    /// 2026-09-26: `false` answers NOT_SUPPORTED to the memory query.
    pub memory_supported: bool,
    /// 2026-09-26: The temperature query's return code.
    pub temperature_rc: NvmlReturn,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            energy_mj: VecDeque::from([0]),
            power_mw: 60_000,
            memory_supported: true,
            temperature_rc: NVML_SUCCESS,
        }
    }
}

thread_local! {
    static SCRIPT: RefCell<Script> = RefCell::new(Script::default());
}

/// 2026-09-26: Install `s` for the calling thread.
pub fn install(s: Script) {
    SCRIPT.with(|c| *c.borrow_mut() = s);
}

unsafe extern "C" fn init() -> NvmlReturn {
    NVML_SUCCESS
}
unsafe extern "C" fn shutdown() -> NvmlReturn {
    NVML_SUCCESS
}
unsafe extern "C" fn by_index(i: c_uint, out: *mut NvmlDeviceHandle) -> NvmlReturn {
    if i != 0 {
        return 2;
    }
    // 2026-09-26: SAFETY: `out` is the caller's valid out-pointer.
    unsafe { *out = 0x10 as NvmlDeviceHandle };
    NVML_SUCCESS
}
unsafe extern "C" fn power(_: NvmlDeviceHandle, out: *mut c_uint) -> NvmlReturn {
    // 2026-09-26: SAFETY: `out` is the caller's valid out-pointer.
    unsafe { *out = SCRIPT.with(|c| c.borrow().power_mw) };
    NVML_SUCCESS
}
unsafe extern "C" fn energy(_: NvmlDeviceHandle, out: *mut c_ulonglong) -> NvmlReturn {
    let v = SCRIPT.with(|c| {
        let mut s = c.borrow_mut();
        if s.energy_mj.len() > 1 {
            s.energy_mj.pop_front().unwrap()
        } else {
            s.energy_mj[0]
        }
    });
    // 2026-09-26: SAFETY: `out` is the caller's valid out-pointer.
    unsafe { *out = v };
    NVML_SUCCESS
}
unsafe extern "C" fn clock(_: NvmlDeviceHandle, kind: c_uint, out: *mut c_uint) -> NvmlReturn {
    // 2026-09-26: SAFETY: `out` is the caller's valid out-pointer.
    unsafe { *out = 1000 + kind };
    NVML_SUCCESS
}
unsafe extern "C" fn temperature(_: NvmlDeviceHandle, _: c_uint, out: *mut c_uint) -> NvmlReturn {
    // 2026-09-26: SAFETY: `out` is the caller's valid out-pointer.
    unsafe { *out = 55 };
    SCRIPT.with(|c| c.borrow().temperature_rc)
}
unsafe extern "C" fn reasons(_: NvmlDeviceHandle, out: *mut c_ulonglong) -> NvmlReturn {
    // 2026-09-26: SAFETY: `out` is the caller's valid out-pointer.
    unsafe { *out = 0x4 };
    NVML_SUCCESS
}
unsafe extern "C" fn memory(_: NvmlDeviceHandle, out: *mut NvmlMemory) -> NvmlReturn {
    if !SCRIPT.with(|c| c.borrow().memory_supported) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    // 2026-09-26: SAFETY: `out` is the caller's valid out-pointer.
    unsafe {
        *out = NvmlMemory {
            total: 8 << 30,
            free: 6 << 30,
            used: 2 << 30,
        }
    };
    NVML_SUCCESS
}

/// 2026-09-26: The scripted table. Ordinal 0 is the only device, and the
/// PCIe and NVLink entry points are left unresolved.
pub fn table() -> NvmlFns {
    NvmlFns {
        init,
        shutdown,
        device_by_index: by_index,
        power_usage: Some(power),
        total_energy: Some(energy),
        clock_info: Some(clock),
        temperature: Some(temperature),
        clocks_event_reasons: Some(reasons),
        memory_info: Some(memory),
        pcie_throughput: None,
        nvlink_state: None,
    }
}
