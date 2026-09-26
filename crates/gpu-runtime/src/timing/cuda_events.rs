// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The CUDA event pool behind [`super`]'s span ring: two timing events
//! per slot, created when the recorder is installed and destroyed with it.
//!
//! Owner: gpu-runtime (telemetry).
//! Invariants:
//! - Every event this pool creates is destroyed exactly once, in `Drop`; a
//!   `create` that fails part-way destroys the events it already made.

use anyhow::{Result, bail};
use metrale_telemetry::kernel::{Edge, EventOps, SpanPoll};

use crate::cuda_backend::{cuEventCreate, cuEventDestroy_v2, cuEventRecord};

unsafe extern "C" {
    fn cuEventQuery(hEvent: u64) -> i32;
    fn cuEventElapsedTime(pMilliseconds: *mut f32, hStart: u64, hEnd: u64) -> i32;
}

/// 2026-09-25: `CU_EVENT_DEFAULT`: `CU_EVENT_DISABLE_TIMING` is not set, so the
/// events can be timed.
const CU_EVENT_DEFAULT: u32 = 0;
const CUDA_ERROR_NOT_READY: i32 = 600;

pub(super) struct CudaEvents {
    /// 2026-09-25: `[start_0, end_0, start_1, end_1, ...]`.
    events: Box<[u64]>,
}

// 2026-09-25: SAFETY: `events` is written only in `create`; after that the pool
// only reads the handles and passes them to driver calls.
unsafe impl Send for CudaEvents {}
unsafe impl Sync for CudaEvents {}

impl CudaEvents {
    /// 2026-09-25: Create `slots` pairs in the current context.
    pub(super) fn create(slots: usize) -> Result<Self> {
        let mut events = Vec::with_capacity(slots * 2);
        for _ in 0..slots * 2 {
            let mut ev = 0u64;
            // 2026-09-25: SAFETY: `ev` is a valid out-pointer.
            let rc = unsafe { cuEventCreate(&mut ev, CU_EVENT_DEFAULT) };
            if rc != 0 {
                let made = Self {
                    events: events.into_boxed_slice(),
                };
                drop(made);
                bail!("cuEventCreate failed with status {rc}");
            }
            events.push(ev);
        }
        Ok(Self {
            events: events.into_boxed_slice(),
        })
    }

    fn event(&self, slot: usize, edge: Edge) -> u64 {
        self.events[slot * 2 + usize::from(edge == Edge::End)]
    }
}

impl EventOps for CudaEvents {
    fn record(&self, slot: usize, edge: Edge, stream: u64) -> bool {
        // 2026-09-25: SAFETY: the event belongs to this pool and is live.
        unsafe { cuEventRecord(self.event(slot, edge), stream) == 0 }
    }

    fn poll(&self, slot: usize) -> SpanPoll {
        let (start, end) = (self.event(slot, Edge::Start), self.event(slot, Edge::End));
        // 2026-09-25: SAFETY: both events belong to this pool and are live.
        match unsafe { cuEventQuery(end) } {
            0 => {
                let mut ms = 0f32;
                // 2026-09-25: SAFETY: `ms` is a valid out-pointer; both events completed.
                if unsafe { cuEventElapsedTime(&mut ms, start, end) } == 0 && ms >= 0.0 {
                    SpanPoll::Ready((f64::from(ms) * 1e6) as u64)
                } else {
                    SpanPoll::Failed
                }
            }
            CUDA_ERROR_NOT_READY => SpanPoll::Pending,
            _ => SpanPoll::Failed,
        }
    }
}

impl Drop for CudaEvents {
    fn drop(&mut self) {
        for &ev in self.events.iter() {
            // 2026-09-25: SAFETY: each handle was created by this pool and is
            // destroyed once here. The status is ignored: at teardown there is
            // nothing to undo.
            unsafe { cuEventDestroy_v2(ev) };
        }
    }
}
