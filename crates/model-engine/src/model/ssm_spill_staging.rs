// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`SpillStaging`], the reusable host blob the SSM snapshot tier gathers
//! into (spill) and scatters out of (fault-in); page-locked when the backend allows.
//!
//! Owner: model-engine SSM snapshot pool.
//! Invariants:
//! - Each [`SpillStaging`] has at most one live [`StagingGuard`], and its blob is
//!   reachable only through that guard.
//!
//! A sample `METRALE_SSM_TIER_TIMING=1` spill line, kept as recorded; the date and
//! build of the run are not recorded:
//!
//! ```text
//! SSM spill: 66846720 B  gather+sync=392936us  store.put=19397us  total=412334us
//! ```

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;
use parking_lot::{Mutex, MutexGuard};

/// 2026-09-25: A host staging blob. `pinned` records whether it came from
/// [`GpuBackend::alloc_host_pinned`] or from the heap fallback; `StagingGuard::kind`
/// reports it in the timing log.
struct StagingBlob {
    ptr: *mut u8,
    bytes: usize,
    pinned: bool,
}

// 2026-09-25: SAFETY: the pointer is either `alloc_host_pinned` memory or a heap
// allocation, and it is owned exclusively: it lives in a `Mutex` and is reachable
// only through a `StagingGuard`.
unsafe impl Send for StagingBlob {}

/// 2026-09-25: Reusable staging buffer for the SSM spill tier, allocated on first
/// `acquire`, so a pool that never spills or faults in allocates none. The mutex is
/// what makes handing out `&mut [u8]` from the raw pointer sound. It is a
/// `parking_lot::Mutex`, which is not reentrant: acquiring while this thread holds
/// the guard deadlocks.
#[derive(Default)]
pub(crate) struct SpillStaging {
    slot: Mutex<Option<StagingBlob>>,
}

impl SpillStaging {
    /// 2026-09-25: Borrow the staging blob, allocating it on first use and
    /// reallocating when `bytes` differs from its size. Not zeroed: the caller must
    /// write all `bytes` before reading them.
    pub(crate) fn acquire<'a>(
        &'a self,
        gpu: &dyn GpuBackend,
        bytes: usize,
    ) -> Result<StagingGuard<'a>> {
        let mut slot = self.slot.lock();
        let need_alloc = match slot.as_ref() {
            Some(b) => b.bytes != bytes,
            None => true,
        };
        if need_alloc {
            if let Some(old) = slot.take() {
                free_blob(gpu, old);
            }
            *slot = Some(alloc_blob(gpu, bytes));
        }
        Ok(StagingGuard { slot })
    }

    /// 2026-09-25: Release the buffer. `SsmSnapshotPool::free_staging` calls it from
    /// `TransformerModel::drop`.
    pub(crate) fn free(&self, gpu: &dyn GpuBackend) {
        if let Some(b) = self.slot.lock().take() {
            free_blob(gpu, b);
        }
    }
}

/// 2026-09-25: Page-locked host memory of `bytes`, or a 4096-aligned heap allocation
/// when that fails; panics if the heap allocation fails.
fn alloc_blob(gpu: &dyn GpuBackend, bytes: usize) -> StagingBlob {
    match gpu.alloc_host_pinned(bytes) {
        Ok(ptr) if !ptr.is_null() => StagingBlob {
            ptr,
            bytes,
            pinned: true,
        },
        other => {
            if let Err(e) = other {
                tracing::warn!(
                    "SSM tier: could not page-lock a {bytes} B staging buffer ({e:#}); \
                     using heap memory — spill/fault-in stay correct but lose the DMA \
                     fast path (expect the old ~165 MB/s D2H bandwidth)"
                );
            }
            let layout = std::alloc::Layout::from_size_align(bytes, 4096)
                .expect("staging layout: bytes > 0, align 4096");
            // 2026-09-25: SAFETY: non-zero size: the callers, `spill_slot` and
            // `fault_in_slot`, pass `spill_blob_bytes()` only when the Marconi region
            // exists. `free_blob` frees it with the same layout.
            let ptr = unsafe { std::alloc::alloc(layout) };
            assert!(
                !ptr.is_null(),
                "SSM tier staging heap alloc failed: {bytes} B"
            );
            StagingBlob {
                ptr,
                bytes,
                pinned: false,
            }
        }
    }
}

fn free_blob(gpu: &dyn GpuBackend, b: StagingBlob) {
    if b.pinned {
        if let Err(e) = gpu.free_host_pinned(b.ptr, b.bytes) {
            tracing::warn!("SSM tier: failed to free pinned staging buffer: {e:#}");
        }
        return;
    }
    let layout = std::alloc::Layout::from_size_align(b.bytes, 4096)
        .expect("staging layout: bytes > 0, align 4096");
    // 2026-09-25: SAFETY: allocated by `alloc_blob`'s heap arm with this layout.
    unsafe { std::alloc::dealloc(b.ptr, layout) };
}

/// 2026-09-25: Exclusive borrow of the staging blob. Dropping it releases the lock;
/// the caller must synchronise the stream first, since queued async copies still
/// reference these bytes.
pub(crate) struct StagingGuard<'a> {
    slot: MutexGuard<'a, Option<StagingBlob>>,
}

impl StagingGuard<'_> {
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        let b = self
            .slot
            .as_mut()
            .expect("acquire() installs the blob before returning the guard");
        // 2026-09-25: SAFETY: `b.ptr` owns `b.bytes` of live host memory, and this
        // guard holds the mutex, so it holds the only reference.
        unsafe { std::slice::from_raw_parts_mut(b.ptr, b.bytes) }
    }

    /// 2026-09-25: `"pinned"` or `"heap"`, for the `METRALE_SSM_TIER_TIMING` line.
    pub(crate) fn kind(&self) -> &'static str {
        match self.slot.as_ref() {
            Some(b) if b.pinned => "pinned",
            _ => "heap",
        }
    }
}
