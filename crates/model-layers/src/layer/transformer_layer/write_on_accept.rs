// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `LayerWriteOnAccept`, the GDN write-on-accept hooks: stash sizing, buffer
//! binding, and the fold of the accepted rows after a batched verify.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-09-26: A supertrait of `TransformerLayer`; see the module header.
pub trait LayerWriteOnAccept {
    /// 2026-09-25: Per-sequence write-on-accept stash size in f32 elements when this layer
    /// can run write-on-accept, `None` otherwise (the default). The model sizes the stash
    /// from the largest answer and binds each GDN layer with [`Self::gdn_woa_bind`].
    fn gdn_woa_stash_seq_floats(&self) -> Option<usize> {
        None
    }

    /// 2026-09-25: Bind this layer's write-on-accept flag word and stash slab (`seqs`
    /// sequences of [`Self::gdn_woa_stash_seq_floats`] f32 each). The model calls it
    /// once, on the first write-on-accept request and before its graph decision, and
    /// never moves the buffers afterwards (`gdn_woa.rs`).
    fn gdn_woa_bind(&self, _flag: DevicePtr, _stash: DevicePtr, _seqs: usize) {}

    /// 2026-09-25: Apply the accepted rows of the last batched verify to this layer's h
    /// states. `h_table` is the layer's WY pointer-table slice, `na_tab` a device `u32[n]`
    /// of accepted row counts in batch order. `Ok(false)` when the layer did nothing, as
    /// the default does.
    fn gdn_fold_accepted(
        &self,
        _gpu: &dyn GpuBackend,
        _h_table: DevicePtr,
        _na_tab: DevicePtr,
        _k_rows: usize,
        _n: usize,
        _stream: u64,
    ) -> Result<bool> {
        Ok(false)
    }
}
