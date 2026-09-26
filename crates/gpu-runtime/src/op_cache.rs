// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`OpCache`]: per-backend memo of kernel handles, grow-only device
//! scratch, and the once / first-N gates of diagnostics.
//!
//! A kernel handle points into a module owned by the backend's registry, and a
//! scratch pointer is a device allocation of the backend. Keeping both on the
//! backend, not in a process-wide static, stops a later model from reaching
//! handles or scratch of a model that is gone.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - One `OpCache` per backend: `MetraleCudaBackend`, `MetalGpuBackend` and
//!   `MockGpuBackend` each construct their own.
//! - A scratch tag's buffer only grows: a request that fits returns the stored
//!   pointer.

use std::collections::HashMap;
// 2026-09-25: parking_lot locks do not poison, so a panic in one op leaves later
// lookups and teardown usable.
use parking_lot::{Mutex, RwLock};

use crate::gpu::{DevicePtr, GpuBackend, KernelHandle};
use anyhow::Result;

/// 2026-09-25: Memoized kernel handles and scratch allocations for one backend.
#[derive(Default)]
pub struct OpCache {
    /// 2026-09-25: `(module, function)` to resolved handle. An entry is written on
    /// the first lookup of its key and only read after that.
    kernels: RwLock<HashMap<(&'static str, &'static str), KernelHandle>>,
    /// 2026-09-25: Purpose tag to `(pointer, bytes)`. Grow-only.
    scratch: Mutex<HashMap<&'static str, (DevicePtr, usize)>>,
    /// 2026-09-25: Set by `note_alloc_fallback`, after a loader's device allocation failed.
    alloc_fell_back: std::sync::atomic::AtomicBool,
    /// 2026-09-25: `(name hash, m, n, k)` keys already seen by `first_shape`, and by
    /// `once` as `(hash, 0, 0, 0)`. Per backend, so a new model logs its own first
    /// line for each shape.
    logged_shapes: Mutex<std::collections::HashSet<(u64, u32, u32, u32)>>,
    /// 2026-09-25: Per-key call counts for `first_n`.
    counters: Mutex<HashMap<&'static str, u32>>,
}

impl OpCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// 2026-09-25: Resolve `module::func`, memoized: `gpu.kernel(..)` on a miss, a
    /// map read on a hit.
    pub fn kernel(
        &self,
        gpu: &dyn GpuBackend,
        module: &'static str,
        func: &'static str,
    ) -> Result<KernelHandle> {
        if let Some(k) = self.kernels.read().get(&(module, func)) {
            return Ok(*k);
        }
        let handle = gpu.kernel(module, func)?;
        self.kernels.write().insert((module, func), handle);
        Ok(handle)
    }

    /// 2026-09-25: A scratch allocation of at least `bytes`, memoized under `tag`.
    ///
    /// Grow-only: a larger request allocates a new buffer and drops the old pointer
    /// from the cache without freeing it. On the CUDA backend the old buffer stays
    /// on the allocation ledger until `sweep_unreleased` frees it when the backend
    /// drops.
    pub fn scratch(
        &self,
        gpu: &dyn GpuBackend,
        tag: &'static str,
        bytes: usize,
    ) -> Result<DevicePtr> {
        let mut g = self.scratch.lock();
        match g.get(tag) {
            Some(&(p, sz)) if sz >= bytes => Ok(p),
            _ => {
                let p = gpu.alloc(bytes)?;
                g.insert(tag, (p, bytes));
                Ok(p)
            }
        }
    }

    /// 2026-09-25: Whether a loader latched the managed-memory fallback on this
    /// backend. The latch is per backend, so the next model's load tries device
    /// memory again.
    pub fn alloc_fell_back(&self) -> bool {
        self.alloc_fell_back
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// 2026-09-25: Latch the managed-memory fallback for the rest of this backend's life.
    pub fn note_alloc_fallback(&self) {
        self.alloc_fell_back
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// 2026-09-25: `true` for the first `n` times this backend reaches `key`.
    /// Counted per backend, so a second model reports its own.
    pub fn first_n(&self, key: &'static str, n: u32) -> bool {
        let mut g = self.counters.lock();
        let c = g.entry(key).or_insert(0);
        *c += 1;
        *c <= n
    }

    /// 2026-09-25: `true` the first time this backend reaches `key`, `false` after.
    /// Shares `first_shape`'s set, as `(key, 0, 0, 0)`.
    pub fn once(&self, key: &'static str) -> bool {
        self.first_shape(key, 0, 0, 0)
    }

    /// 2026-09-25: `true` the first time this backend sees `(name, m, n, k)`; used to
    /// de-duplicate the GEMM route and shape log lines.
    pub fn first_shape(&self, name: &str, m: u32, n: u32, k: u32) -> bool {
        let mut h: u64 = 1469598103934665603;
        for b in name.bytes() {
            h = (h ^ b as u64).wrapping_mul(1099511628211);
        }
        self.logged_shapes.lock().insert((h, m, n, k))
    }
}

impl std::fmt::Debug for OpCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kernels = self.kernels.read().len();
        let scratch = self.scratch.lock().len();
        f.debug_struct("OpCache")
            .field("kernels", &kernels)
            .field("scratch", &scratch)
            .finish()
    }
}

/// 2026-09-25: Frees every scratch allocation and empties the scratch map.
///
/// Kernel handles need no release here: they point into the modules of the
/// backend's `MetraleRegistry`, which unloads them when it is dropped or passed
/// to `cuda_host::release`.
impl metrale_core::scope::ModelResource<dyn crate::gpu::GpuBackend> for OpCache {
    fn label(&self) -> &'static str {
        "op scratch"
    }

    fn release(&mut self, gpu: &dyn crate::gpu::GpuBackend) -> anyhow::Result<()> {
        let mut first_error = None;
        // 2026-09-25: `drain` makes this idempotent and stops a later lookup from
        // finding a pointer into freed memory.
        let taken: Vec<(DevicePtr, usize)> = self
            .scratch
            .lock()
            .drain()
            .map(|(_, entry)| entry)
            .collect();
        for (ptr, _) in taken {
            if let Err(e) = gpu.free(ptr)
                && first_error.is_none()
            {
                first_error = Some(e);
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpuBackend;

    #[test]
    fn a_kernel_is_resolved_once_and_reused() {
        let gpu = MockGpuBackend::new();
        let c = OpCache::new();
        let a = c.kernel(&gpu, "w4a16", "bf16_to_fp8").expect("resolves");
        let b = c.kernel(&gpu, "w4a16", "bf16_to_fp8").expect("resolves");
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn two_caches_do_not_share_handles_or_scratch() {
        let gpu = MockGpuBackend::new();
        let a = OpCache::new();
        let b = OpCache::new();
        let _ = a.scratch(&gpu, "fp8_activation", 1024).expect("allocs");
        assert!(
            format!("{a:?}").contains("scratch: 1"),
            "the first cache holds it"
        );
        assert!(
            format!("{b:?}").contains("scratch: 0"),
            "the second starts empty"
        );
    }

    #[test]
    fn scratch_grows_but_never_shrinks() {
        let gpu = MockGpuBackend::new();
        let c = OpCache::new();
        let small = c.scratch(&gpu, "act", 64).expect("allocs");
        let same = c.scratch(&gpu, "act", 32).expect("reuses");
        assert_eq!(small.0, same.0, "a smaller request reuses the buffer");
        let bigger = c.scratch(&gpu, "act", 4096).expect("reallocs");
        assert_ne!(small.0, bigger.0, "a larger request gets a new buffer");
    }
}
