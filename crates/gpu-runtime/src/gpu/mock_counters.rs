// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Readers for the mock backend's call counters and stream logs. A
//! child module of `mock`, so the fields stay private to it.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use std::sync::atomic::Ordering;

use super::MockGpuBackend;

impl MockGpuBackend {
    pub fn alloc_count(&self) -> usize {
        self.allocs.lock().len()
    }

    /// 2026-09-25: Make `alloc` refuse any single allocation above `bytes`.
    pub fn set_max_allocation_bytes(&self, bytes: usize) {
        self.max_allocation_bytes.store(bytes, Ordering::Relaxed);
    }

    pub fn launch_count(&self) -> usize {
        self.launches.lock().len()
    }

    /// 2026-09-25: `synchronize` calls so far.
    pub fn sync_count(&self) -> usize {
        self.syncs.load(Ordering::Relaxed)
    }

    /// 2026-09-25: Blocking `copy_d2h` calls; each one waits for the default
    /// stream on the CUDA backend.
    pub fn d2h_blocking_count(&self) -> usize {
        self.d2h_blocking.load(Ordering::Relaxed)
    }

    /// 2026-09-25: `copy_d2h_async` calls.
    pub fn d2h_async_count(&self) -> usize {
        self.d2h_async.load(Ordering::Relaxed)
    }

    pub fn d2h_async_streams(&self) -> Vec<u64> {
        self.d2h_async_streams.lock().clone()
    }

    pub fn sync_d2h_async_counts(&self) -> Vec<(u64, usize)> {
        self.sync_d2h_async_counts.lock().clone()
    }

    /// 2026-09-25: `copy_d2d` and `copy_d2d_async` calls so far.
    pub fn d2d_count(&self) -> usize {
        self.d2d.load(Ordering::Relaxed)
    }

    /// 2026-09-25: `copy_d2d_2d_async` calls so far.
    pub fn d2d_2d_count(&self) -> usize {
        self.d2d_2d.load(Ordering::Relaxed)
    }

    /// 2026-09-25: Streams passed to `copy_d2d_async`, in call order.
    pub fn d2d_async_streams(&self) -> Vec<u64> {
        self.d2d_async_streams.lock().clone()
    }

    /// 2026-09-25: Streams passed to `copy_d2d_2d_async`, in call order.
    pub fn d2d_2d_async_streams(&self) -> Vec<u64> {
        self.d2d_2d_async_streams.lock().clone()
    }

    /// 2026-09-25: `alloc_host_pinned` calls, so a test can check a staging
    /// buffer is allocated once and reused.
    pub fn host_pinned_alloc_count(&self) -> usize {
        self.host_pinned_allocs.load(Ordering::Relaxed)
    }

    pub fn h2d_count(&self) -> usize {
        self.h2d.load(Ordering::Relaxed)
    }

    pub fn h2d_bytes(&self) -> usize {
        self.h2d_bytes.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{DevicePtr, GpuBackend};

    #[test]
    fn h2d_counters_count_calls_and_bytes() {
        let gpu = MockGpuBackend::new();
        let p = gpu.alloc(16).unwrap();
        gpu.copy_h2d(&[1u8, 2, 3, 4], p).unwrap();
        gpu.copy_h2d(&[5u8, 6], DevicePtr(p.0 + 4)).unwrap();
        assert_eq!(gpu.h2d_count(), 2);
        assert_eq!(gpu.h2d_bytes(), 6);
    }
}
