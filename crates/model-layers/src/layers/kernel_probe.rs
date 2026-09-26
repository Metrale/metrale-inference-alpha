// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Optional kernel lookups that return `KernelHandle(0)` instead of an error.
//!
//! Every lookup goes through `GpuBackend::kernel`, which records it in the
//! kernel audit (`metrale_telemetry::kernel_audit::record`). At boot,
//! `serve_phases/kernel_gate.rs` refuses to serve when a failed lookup is not
//! declared in the target's MODEL.toml `[expected_absent]`, unless
//! `--dangerously-allow-unresolved-kernel-lookups` is passed.
//!
//! Owner: model-layers (kernel lookup).
//! Invariants: `try_target_kernel` issues no lookup for a module the backend
//! does not carry (`GpuBackend::has_module`).

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// 2026-09-25: Look up a kernel from a module only some targets compile.
/// Returns `KernelHandle(0)` without a lookup, and so without an audit row,
/// when the backend does not carry `module`; otherwise it is [`try_kernel`].
#[track_caller]
pub fn try_target_kernel(gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    if !gpu.has_module(module) {
        return KernelHandle(0);
    }
    try_kernel(gpu, module, func)
}

/// 2026-09-25: Look up `module::func`, returning `KernelHandle(0)` (and a
/// debug log) when the lookup fails. A zero handle is a slower path or a
/// missing feature, so a caller must check it before launching.
///
/// `#[track_caller]` here and on `GpuBackend::kernel` makes the audit record
/// the caller's `file:line` rather than this function's.
#[track_caller]
pub fn try_kernel(gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    match gpu.kernel(module, func) {
        Ok(h) => h,
        Err(_) => {
            tracing::debug!("Optional kernel '{module}::{func}' not loaded");
            KernelHandle(0)
        }
    }
}

#[cfg(test)]
#[path = "kernel_probe_tests.rs"]
mod kernel_probe_tests;
