// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: L1 GPU spans: CUDA event pairs around each serve lane on every
//! step, and around each eager kernel launch on the steps the telemetry hub arms
//! (one in `kernel_span_every`), at telemetry level `Kernel` only.
//!
//! The bookkeeping is `metrale_telemetry::kernel::SpanRing`; this module is the
//! CUDA half: the event pool (`cuda_events`) and the process-wide recorder that
//! `MetraleCudaBackend::new` installs.
//!
//! Lane spans are recorded on the registry's stream (`MetraleRegistry::raw_stream`),
//! which cudarc creates non-blocking, so a lane pair times the work queued on that
//! stream only.
//!
//! Owner: gpu-runtime (telemetry).
//! Invariants:
//! - No span synchronises: a completed pair is harvested with a non-blocking
//!   `cuEventQuery`, and a pair still in flight is left for a later harvest.
//! - Below level `Kernel` no entry point reaches CUDA: each first reads the hub's
//!   level or armed flag (a relaxed atomic load), and `kernel_end` acts only on a
//!   span `kernel_begin` opened.
//! - No kernel span is opened on a stream that is being captured.

#[cfg(feature = "cuda")]
mod cuda_events;

#[cfg(feature = "cuda")]
pub use imp::*;

/// 2026-09-25: Event pairs in the pool. The ring hands out slots in turn; a span
/// whose slot is still in flight is dropped and counted (`spans_dropped`).
pub const SPAN_SLOTS: usize = 4096;

#[cfg(feature = "cuda")]
mod imp {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, RwLock};

    use metrale_telemetry::kernel::{LANES, SpanKey, SpanRing, SpanToken};
    use metrale_telemetry::{Level, global};

    use super::SPAN_SLOTS;
    use super::cuda_events::CudaEvents;

    /// 2026-09-25: The recorder `install_for_stream` installs; a later install replaces it.
    pub struct GpuSpans {
        ring: SpanRing<CudaEvents>,
        stream: u64,
        open_lanes: [AtomicUsize; LANES.len()],
    }

    const NONE: usize = usize::MAX;

    static INSTALLED: RwLock<Option<Arc<GpuSpans>>> = RwLock::new(None);

    fn installed() -> Option<Arc<GpuSpans>> {
        INSTALLED.read().ok()?.clone()
    }

    /// 2026-09-25: Build and install the recorder for a backend whose stream is
    /// `stream`, when telemetry is at level `Kernel`. Called from
    /// `MetraleCudaBackend::new` after the registry loads. A pool that cannot be
    /// created leaves L1 off and logs why.
    pub fn install_for_stream(stream: u64) {
        if global().level() != Level::Kernel {
            return;
        }
        match CudaEvents::create(SPAN_SLOTS) {
            Ok(events) => {
                let spans = GpuSpans {
                    ring: SpanRing::new(events, SPAN_SLOTS),
                    stream,
                    open_lanes: [const { AtomicUsize::new(NONE) }; LANES.len()],
                };
                if let Ok(mut slot) = INSTALLED.write() {
                    *slot = Some(Arc::new(spans));
                }
            }
            Err(e) => tracing::warn!("telemetry: GPU spans disabled — {e}"),
        }
    }

    /// 2026-09-25: Open lane `lane`'s span; an index outside `LANES` is ignored.
    pub fn lane_begin(lane: usize) {
        if !global().lane_spans() {
            return;
        }
        let Some(s) = installed() else { return };
        let Some(slot) = s.open_lanes.get(lane) else {
            return;
        };
        if let Some(tok) = s
            .ring
            .begin(SpanKey::Lane(lane), s.stream, &global().kernel)
        {
            slot.store(tok.slot(), Ordering::Relaxed);
        }
    }

    /// 2026-09-25: Close lane `lane`'s span, if one is open.
    pub fn lane_end(lane: usize) {
        if !global().lane_spans() {
            return;
        }
        let Some(s) = installed() else { return };
        let Some(slot) = s.open_lanes.get(lane) else {
            return;
        };
        let i = slot.swap(NONE, Ordering::Relaxed);
        if i != NONE {
            s.ring
                .end(SpanToken::from_slot(i), s.stream, &global().kernel);
        }
    }

    /// 2026-09-25: Land every completed span. The scheduler calls it at each step begin.
    pub fn harvest() {
        if global().level() != Level::Kernel {
            return;
        }
        if let Some(s) = installed() {
            s.ring.harvest(&global().kernel);
        }
    }

    /// 2026-09-25: A kernel span, when this step is armed for kernel spans and
    /// `stream` is not being captured (an event recorded into a capture would be a
    /// graph node, not a timestamp).
    pub fn kernel_begin(
        gpu: &dyn crate::gpu::GpuBackend,
        func: u64,
        stream: u64,
    ) -> Option<KernelSpan> {
        if !global().kernel_spans_armed() || gpu.stream_is_capturing(stream) {
            return None;
        }
        let s = installed()?;
        let tok = s
            .ring
            .begin(SpanKey::Kernel(func), stream, &global().kernel)?;
        Some(KernelSpan { spans: s, tok })
    }

    /// 2026-09-25: Close a kernel span opened by [`kernel_begin`].
    pub fn kernel_end(span: Option<KernelSpan>, stream: u64) {
        if let Some(k) = span {
            k.spans.ring.end(k.tok, stream, &global().kernel);
        }
    }

    /// 2026-09-25: An open kernel span.
    pub struct KernelSpan {
        spans: Arc<GpuSpans>,
        tok: SpanToken,
    }
}

/// 2026-09-25: Non-CUDA builds carry no GPU spans. The entry points the scheduler
/// calls stay, so callers need no cfg of their own.
#[cfg(not(feature = "cuda"))]
mod imp_absent {
    pub fn lane_begin(_lane: usize) {}
    pub fn lane_end(_lane: usize) {}
    pub fn harvest() {}
}
#[cfg(not(feature = "cuda"))]
pub use imp_absent::*;
