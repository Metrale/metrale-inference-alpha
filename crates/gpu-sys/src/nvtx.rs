// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: NVTX ranges (feature `nvtx`): `push` and `pop` call
//! `nvtxRangePushA` and `nvtxRangePop` through the C shim `nvtx_shim.c`.
//!
//! Owner: metrale-gpu-sys.
//! Invariants: none beyond the types.

use std::ffi::{CStr, c_char, c_int};

unsafe extern "C" {
    fn metrale_nvtx_range_push(name: *const c_char) -> c_int;
    fn metrale_nvtx_range_pop() -> c_int;
}

/// 2026-09-26: Open a range named `name`.
pub fn push(name: &CStr) {
    // 2026-09-26: SAFETY: `name` is a NUL-terminated string that outlives the
    // call.
    unsafe { metrale_nvtx_range_push(name.as_ptr()) };
}

/// 2026-09-26: Close the innermost open range.
pub fn pop() {
    // 2026-09-26: SAFETY: the call takes no arguments.
    unsafe { metrale_nvtx_range_pop() };
}
