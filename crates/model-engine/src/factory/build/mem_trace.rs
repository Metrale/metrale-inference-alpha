// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `MemTrace`, the per-step free-memory log of `build_model`.
//!
//! Owner: metrale-model-engine.
//! Invariants:
//! - `MemTrace` only reads `free_memory` and logs; it allocates nothing.

use metrale_gpu_runtime::gpu::GpuBackend;

/// 2026-09-25: Logs the change in `gpu.free_memory()` at each build step.
/// It allocates nothing.
pub(super) struct MemTrace<'a> {
    gpu: &'a dyn GpuBackend,
    last: usize,
    start: usize,
}

impl<'a> MemTrace<'a> {
    pub(super) fn new(gpu: &'a dyn GpuBackend) -> Self {
        let f = gpu.free_memory().unwrap_or(0);
        Self {
            gpu,
            last: f,
            start: f,
        }
    }

    /// 2026-09-25: Logs the free-memory change since the previous mark;
    /// negative means allocated. Does nothing if `free_memory` fails.
    pub(super) fn mark(&mut self, step: &str) {
        let Ok(now) = self.gpu.free_memory() else {
            return;
        };
        let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
        let delta = now as i128 - self.last as i128;
        tracing::info!(target: "metrale_model_engine::factory::build", "build residency: {step:<26} {:+9.3} GiB   (cumulative {:8.3} GiB, free {:7.3} GiB)",
            delta as f64 / (1024.0 * 1024.0 * 1024.0),
            gib(self.start) - gib(now),
            gib(now),
        );
        self.last = now;
    }
}
