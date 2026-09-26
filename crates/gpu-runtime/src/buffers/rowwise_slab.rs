// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The row-wise FP8 GDN prefill BF16-weight slab: its size and the bump carve the prefill arms take their per-layer slices from.
//!
//! The slab is one arena allocation, so the preflight reserve (`total_bytes`)
//! counts it.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - A carve never allocates: it returns a slice of the slab or an error (slab
//!   absent, or exhausted).
//! - Successful carves do not overlap and lie inside the slab.

use super::BufferArena;
use crate::gpu::DevicePtr;

impl BufferArena {
    /// 2026-09-25: Allocated byte size of the slab; 0 unless `METRALE_FP8_ROWWISE=1`
    /// when the arena was sized.
    pub fn ssm_rowwise_w_bf16_bytes(&self) -> usize {
        self.sizes.ssm_rowwise_w_bf16
    }

    /// 2026-09-25: Carve the next `bytes` of the slab. Slices are never returned:
    /// a GDN layer carves its `in_proj_qkvz` and `out_proj` copies on its first
    /// prefill and keeps them for the arena's life. The slab holds exactly
    /// `num_ssm_layers` such pairs (`sizes_rowwise.rs`), so exhaustion is a
    /// sizing bug and is returned as an error, not served by a new allocation.
    /// The cursor is not rolled back after that error.
    ///
    /// `Relaxed` ordering: the scheduler runs one forward at a time, the same
    /// assumption `cublaslt::Ctx` states for its `Send`/`Sync`.
    pub fn take_ssm_rowwise_w_bf16(&self, bytes: usize) -> anyhow::Result<DevicePtr> {
        use std::sync::atomic::Ordering;
        if self.ssm_rowwise_w_bf16 == DevicePtr::NULL {
            anyhow::bail!(
                "row-wise GDN prefill BF16-weight slab is absent: the arena was sized without                  METRALE_FP8_ROWWISE=1 but a row-wise prefill arm asked for {bytes} B. Both the                  loader's weight install and this ledger entry read the same lever, so they                  cannot legitimately disagree"
            );
        }
        let base = self
            .ssm_rowwise_w_bf16_used
            .fetch_add(bytes, Ordering::Relaxed);
        let end = base + bytes;
        if end > self.sizes.ssm_rowwise_w_bf16 {
            anyhow::bail!(
                "row-wise GDN prefill BF16-weight slab exhausted: wanted {bytes} B at offset                  {base}, slab is {} B. `sizes_rowwise::ssm_rowwise_w_bf16_bytes` sizes it for                  num_ssm_layers x (in_proj_qkvz + out_proj)",
                self.sizes.ssm_rowwise_w_bf16
            );
        }
        Ok(self.ssm_rowwise_w_bf16.offset(base))
    }
}
