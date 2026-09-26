// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Device free-memory queries, which add host `MemAvailable` only
//! on an integrated GPU, and the OOM watchdog that polls them.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - At most one OOM watchdog task is spawned per process (`WATCHDOG_RUNNING`).

use super::*;

/// 2026-09-25: Free device memory without a `GpuBackend`, for the OOM
/// watchdog and the TUI. `None` when `cuMemGetInfo_v2` fails.
///
/// The same rule as `free_memory` (`effective_free_bytes`), except that a
/// failed integrated query counts as discrete (`polled_free_bytes`).
pub fn cuda_free_memory_bytes() -> Option<usize> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
    if status != 0 {
        return None;
    }
    Some(polled_free_bytes(
        free,
        system_available_memory_bytes(),
        current_device_is_integrated(),
    ))
}

/// 2026-09-25: `CU_DEVICE_ATTRIBUTE_INTEGRATED` of the current context's
/// device. Requires a current context (`cuCtxGetDevice`); fails, rather than
/// guessing, when the driver does not answer. Both callers query
/// `cuMemGetInfo_v2` first, which needs the same context.
///
/// Measured 2026-09-04: attribute 18 reads 1 on GB10 and 0 on RTX PRO 6000
/// Blackwell; `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS` (99) reads 1 on
/// both, so it does not tell them apart.
pub(crate) fn device_is_integrated() -> Result<bool> {
    const CU_DEVICE_ATTRIBUTE_INTEGRATED: u32 = 18;
    let mut dev: i32 = 0;
    let status = unsafe { cuCtxGetDevice(&mut dev) };
    if status != 0 {
        bail!("cuCtxGetDevice failed: status {status}");
    }
    let mut integrated: i32 = 0;
    let status =
        unsafe { cuDeviceGetAttribute(&mut integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev) };
    if status != 0 {
        bail!("cuDeviceGetAttribute(INTEGRATED) failed: status {status}");
    }
    Ok(integrated != 0)
}

/// 2026-09-25: [`device_is_integrated`] as an `Option`, for
/// `cuda_free_memory_bytes`, which returns `Option` and cannot report an
/// error.
fn current_device_is_integrated() -> Option<bool> {
    device_is_integrated().ok()
}

/// 2026-09-25: Free device memory to report: the larger of `cu_free` and
/// `mem_available` on an integrated GPU, `cu_free` otherwise or when
/// `mem_available` is `None`. On a discrete GPU host RAM is a separate pool.
/// Pure, so the rule is tested without a GPU.
pub(crate) fn effective_free_bytes(
    cu_free: usize,
    mem_available: Option<usize>,
    integrated: bool,
) -> usize {
    match mem_available {
        Some(avail) if integrated => cu_free.max(avail),
        _ => cu_free,
    }
}

/// 2026-09-25: Whether a watchdog has been spawned. Process-wide, because the
/// watchdog reads the device's free memory, which does not depend on the
/// model: a second model load must not start a second watchdog.
static WATCHDOG_RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 2026-09-25: Spawn the OOM watchdog on the tokio runtime, or return `None`
/// if one was already spawned.
///
/// Every `interval` it reads `cuda_free_memory_bytes`. Below `threshold_mb`
/// MiB it logs an error; at the third consecutive such reading it calls
/// `std::process::exit(1)`. A reading at or above the threshold resets the
/// count, and a `None` reading leaves it unchanged. Dropping the returned
/// handle does not stop the task.
pub fn spawn_oom_watchdog(
    threshold_mb: usize,
    interval: std::time::Duration,
) -> Option<tokio::task::JoinHandle<()>> {
    if WATCHDOG_RUNNING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return None;
    }
    let threshold_bytes = threshold_mb * 1024 * 1024;
    Some(tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        let mut consecutive_low = 0u32;
        loop {
            tick.tick().await;
            if let Some(free) = cuda_free_memory_bytes() {
                if free < threshold_bytes {
                    consecutive_low += 1;
                    let free_mb = free / (1024 * 1024);
                    tracing::error!(
                        "OOM watchdog: GPU free memory critically low: {} MB (threshold: {} MB) [{}/3]",
                        free_mb,
                        threshold_mb,
                        consecutive_low,
                    );
                    if consecutive_low >= 3 {
                        tracing::error!(
                            "OOM watchdog: 3 consecutive readings below threshold. \
                             Terminating to prevent system freeze."
                        );
                        std::process::exit(1);
                    }
                } else {
                    consecutive_low = 0;
                }
            }
        }
    }))
}
