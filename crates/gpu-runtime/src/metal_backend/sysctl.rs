// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Total system RAM from `sysctlbyname("hw.memsize")`, which
//! `MetalGpuBackend::total_memory` reports.
//!
//! Owner: gpu-runtime (Metal backend).
//! Invariants:
//! - `None` whenever the call fails; nothing is cached.

use std::ffi::c_void;

/// 2026-09-25: `hw.memsize` through `sysctlbyname`: total system RAM in bytes,
/// or `None` when the call fails.
pub(super) fn sysctl_memsize() -> Option<usize> {
    use std::ffi::CString;
    let name = CString::new("hw.memsize").ok()?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut u64 as *mut c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 { Some(value as usize) } else { None }
}
