// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `cuda_backend` unit tests that need no GPU and no CUDA context.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use std::ffi::c_void;

use crate::registry::RawCudaFunc;

use super::{effective_free_bytes, polled_free_bytes};
use crate::gpu::{DevicePtr, KernelHandle};

#[test]
fn kernel_handle_roundtrip() {
    let fake_ptr = 0xDEAD_BEEF_CAFE_u64;
    let handle = KernelHandle(fake_ptr);
    let raw = RawCudaFunc(handle.0 as *mut c_void);
    let back = raw.0 as u64;
    assert_eq!(back, fake_ptr);
}

#[test]
fn null_free_is_noop() {
    // 2026-09-25: `MetraleCudaBackend::free` returns early on `is_null`; without
    // a GPU this checks only `is_null`.
    assert!(DevicePtr::NULL.is_null());
    assert!(!DevicePtr(0x1000).is_null());
}

// 2026-09-25: `effective_free_bytes` and `polled_free_bytes` are pure, so the
// rule for when host `MemAvailable` may stand in for device free memory is
// tested here without a GPU.

const GIB: usize = 1024 * 1024 * 1024;

/// 2026-09-25: A discrete card with 4 GiB 280 MiB free, on a host whose
/// `MemAvailable` is 1,038,438,936 kB (about 990 GiB).
const DISCRETE_CU_FREE: usize = 4 * GIB + 280 * 1024 * 1024;
const HUGE_HOST_MEM_AVAILABLE: usize = 1_038_438_936 * 1024;

#[test]
fn discrete_device_ignores_host_mem_available() {
    assert_eq!(
        effective_free_bytes(DISCRETE_CU_FREE, Some(HUGE_HOST_MEM_AVAILABLE), false),
        DISCRETE_CU_FREE,
        "a discrete GPU must report the driver's device-free figure verbatim"
    );
}

#[test]
fn integrated_device_takes_the_max() {
    assert_eq!(
        effective_free_bytes(20 * GIB, Some(90 * GIB), true),
        90 * GIB
    );
}

#[test]
fn integrated_device_without_meminfo_uses_driver_figure() {
    // 2026-09-25: `None`: `/proc/meminfo` missing or unparseable.
    assert_eq!(effective_free_bytes(20 * GIB, None, true), 20 * GIB);
}

#[test]
fn integrated_device_keeps_driver_figure_when_it_is_larger() {
    // 2026-09-25: When `MemAvailable` is the smaller figure, the driver's
    // figure stands.
    assert_eq!(
        effective_free_bytes(60 * GIB, Some(10 * GIB), true),
        60 * GIB
    );
}

#[test]
fn watchdog_poll_on_a_discrete_device_reports_device_memory() {
    // 2026-09-25: `polled_free_bytes` backs `cuda_free_memory_bytes`, which the
    // OOM watchdog and the TUI read.
    assert_eq!(
        polled_free_bytes(DISCRETE_CU_FREE, Some(HUGE_HOST_MEM_AVAILABLE), Some(false)),
        DISCRETE_CU_FREE
    );
}

#[test]
fn watchdog_poll_with_unknown_integration_is_treated_as_discrete() {
    // 2026-09-25: `None`: `cuCtxGetDevice` or `cuDeviceGetAttribute` failed.
    // It is treated as discrete, which can only under-report; treating it as
    // integrated would count host RAM as device memory.
    assert_eq!(
        polled_free_bytes(DISCRETE_CU_FREE, Some(HUGE_HOST_MEM_AVAILABLE), None),
        DISCRETE_CU_FREE,
        "an unknown integrated/discrete answer must never inflate device free memory"
    );
}

#[test]
fn watchdog_poll_on_an_integrated_device_still_takes_the_max() {
    assert_eq!(
        polled_free_bytes(20 * GIB, Some(90 * GIB), Some(true)),
        90 * GIB
    );
}
