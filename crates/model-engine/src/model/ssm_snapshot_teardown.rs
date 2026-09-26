// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Teardown for [`SsmSnapshotPool`].
//!
//! Owner: model-engine SSM snapshot pool.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::GpuBackend;

use super::ssm_snapshot::SsmSnapshotPool;

/// 2026-09-25: Free the Marconi and decode-rollback h and conv regions, then clear the
/// free list and session tags. Every pointer is attempted; the first free error is
/// returned.
impl metrale_core::scope::ModelResource<dyn GpuBackend> for SsmSnapshotPool {
    fn label(&self) -> &'static str {
        "ssm snapshot pool"
    }

    fn release(&mut self, gpu: &dyn GpuBackend) -> anyhow::Result<()> {
        let mut first_error = None;
        for pool in [
            &mut self.h_snapshots,
            &mut self.conv_snapshots,
            &mut self.decode_h_snapshots,
            &mut self.decode_conv_snapshots,
        ] {
            for ptr in pool.drain(..) {
                if let Err(e) = gpu.free(ptr)
                    && first_error.is_none()
                {
                    first_error = Some(e);
                }
            }
        }
        // 2026-09-25: With the regions gone, an empty free list keeps `save` and the
        // other acquire paths from handing out a slot into freed memory.
        self.free_slots.lock().clear();
        self.session_tags.lock().clear();
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
