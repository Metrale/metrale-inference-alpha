// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `Nvml` over a table of `extern "C"` functions defined
//! here. Each function reads the calling thread's thread-local script, so
//! tests running in parallel do not see one another's answers.
//!
//! Owner: metrale-gpu-sys.
//! Invariants: none beyond the types.

use std::cell::Cell;

use super::*;

thread_local! {
    static INIT_RC: Cell<NvmlReturn> = const { Cell::new(NVML_SUCCESS) };
    static SHUTDOWNS: Cell<u32> = const { Cell::new(0) };
    static POWER_RC: Cell<NvmlReturn> = const { Cell::new(NVML_SUCCESS) };
    static LINKS_UP: Cell<u32> = const { Cell::new(0) };
}

unsafe extern "C" fn init() -> NvmlReturn {
    INIT_RC.with(Cell::get)
}
unsafe extern "C" fn shutdown() -> NvmlReturn {
    SHUTDOWNS.with(|c| c.set(c.get() + 1));
    NVML_SUCCESS
}
unsafe extern "C" fn by_index(index: c_uint, out: *mut NvmlDeviceHandle) -> NvmlReturn {
    if index != 0 {
        return 2;
    }
    // 2026-09-26: SAFETY: the caller passes a valid out-pointer.
    unsafe { *out = 0x1000 as NvmlDeviceHandle };
    NVML_SUCCESS
}
unsafe extern "C" fn power(_: NvmlDeviceHandle, out: *mut c_uint) -> NvmlReturn {
    // 2026-09-26: SAFETY: valid out-pointer.
    unsafe { *out = 61_250 };
    POWER_RC.with(Cell::get)
}
unsafe extern "C" fn link(_: NvmlDeviceHandle, l: c_uint, out: *mut c_uint) -> NvmlReturn {
    if l >= LINKS_UP.with(Cell::get) {
        return 2;
    }
    // 2026-09-26: SAFETY: valid out-pointer.
    unsafe { *out = u32::from(l.is_multiple_of(2)) };
    NVML_SUCCESS
}

fn table() -> NvmlFns {
    NvmlFns {
        init,
        shutdown,
        device_by_index: by_index,
        power_usage: Some(power),
        total_energy: None,
        clock_info: None,
        temperature: None,
        clocks_event_reasons: None,
        memory_info: None,
        pcie_throughput: None,
        nvlink_state: Some(link),
    }
}

#[test]
fn a_missing_library_is_unavailable_not_a_crash() {
    let err = NvmlLibrary::open_from(&["/nonexistent/libnvidia-ml.so.1"])
        .err()
        .expect("a path that does not exist cannot load");
    assert!(matches!(err, NvmlUnavailable::NotInstalled(_)), "{err}");
}

#[test]
fn a_failed_init_is_reported_with_its_code() {
    INIT_RC.with(|c| c.set(9));
    let err = Nvml::with_fns(table()).err().expect("init returned 9");
    assert_eq!(err, NvmlUnavailable::InitFailed(9));
}

#[test]
fn shutdown_balances_a_successful_init_exactly_once() {
    let before = SHUTDOWNS.with(Cell::get);
    drop(Nvml::with_fns(table()).expect("scripted init succeeds"));
    assert_eq!(SHUTDOWNS.with(Cell::get), before + 1);
}

#[test]
fn a_device_index_the_driver_refuses_is_no_device() {
    let nvml = Nvml::with_fns(table()).unwrap();
    let err = nvml.device(3).err().expect("index 3 is refused");
    assert_eq!(err, NvmlUnavailable::NoDevice { index: 3, code: 2 });
}

#[test]
fn not_supported_is_absent_and_other_codes_are_errors() {
    let nvml = Nvml::with_fns(table()).unwrap();
    let dev = nvml.device(0).unwrap();
    assert_eq!(dev.power_mw(), Ok(Some(61_250)));
    POWER_RC.with(|c| c.set(NVML_ERROR_NOT_SUPPORTED));
    assert_eq!(dev.power_mw(), Ok(None), "NOT_SUPPORTED reads as absent");
    POWER_RC.with(|c| c.set(999));
    assert_eq!(dev.power_mw(), Err(999), "any other code is an error");
    assert_eq!(
        dev.total_energy_mj(),
        Ok(None),
        "an unresolved symbol is absent"
    );
}

#[test]
fn nvlink_counts_active_links_among_those_that_answer() {
    let nvml = Nvml::with_fns(table()).unwrap();
    let dev = nvml.device(0).unwrap();
    LINKS_UP.with(|c| c.set(0));
    assert_eq!(dev.nvlink_active_links(), None, "no link answers");
    LINKS_UP.with(|c| c.set(5));
    // 2026-09-26: Of the five links that answer, 0, 2 and 4 are active.
    assert_eq!(dev.nvlink_active_links(), Some(3));
}
