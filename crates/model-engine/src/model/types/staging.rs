// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `PinnedMetaStaging::packer_for`, the bounds-checked writer over the pinned
//! metadata staging buffer.
//!
//! Owner: model-engine.
//! Invariants:
//! - The packer's capacity is never more than the pinned allocation (`bytes`).

use super::PinnedMetaStaging;

impl PinnedMetaStaging {
    /// 2026-09-25: A bounds-checked cursor over this buffer, the only way the
    /// model writes it. See [`crate::model::pinned_pack`].
    ///
    /// `dest_bytes` is the room at the device destination the pack is uploaded
    /// to. The packer's capacity is `min(bytes, dest_bytes)`, because a pack
    /// that fits the host buffer can still overrun a device destination that
    /// starts at an offset inside scratch.
    ///
    /// Takes `&self`: the bytes it writes are the separate `alloc_host_pinned`
    /// region `ptr` points to, not this struct, so callers can still read the
    /// reusable `Vec`s alongside it.
    pub(crate) fn packer_for(
        &self,
        dest_bytes: usize,
    ) -> crate::model::pinned_pack::PinnedPacker<'_> {
        // 2026-09-25: SAFETY: `ptr`/`bytes` are the `alloc_host_pinned` region
        // installed in `impl_a1.rs` and freed on drop (`drop_pinned_staging`);
        // it is live for the model's lifetime, zeroed at allocation (the
        // `alloc_host_pinned` contract), and only touched from the scheduler
        // thread, the same assumption the `unsafe impl Sync for
        // TransformerModel` in `types.rs` rests on. The capacity handed over is
        // `min(bytes, dest_bytes)`, never more than the allocation.
        unsafe {
            crate::model::pinned_pack::PinnedPacker::new(self.ptr, self.bytes.min(dest_bytes))
        }
    }
}
