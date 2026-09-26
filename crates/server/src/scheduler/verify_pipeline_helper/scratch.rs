// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: reused host buffer for the verify-time dequantised logits.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

thread_local! {
    /// 2026-09-25: the dequantised logits of one verify position, reused
    /// across calls on the same thread so each position does not allocate a
    /// fresh `Vec<f32>`. `verify_pick_with_pipeline` overwrites every entry
    /// before reading, so leftover contents do not matter.
    pub(super) static DEQUANT_SCRATCH: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// 2026-09-25: puts the dequant buffer back into [`DEQUANT_SCRATCH`] on
/// drop, so every return of `verify_pick_with_pipeline` hands it back.
pub(super) struct ScratchGuard(pub(super) Vec<f32>);

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.0);
        DEQUANT_SCRATCH.with(|s| {
            *s.borrow_mut() = buf;
        });
    }
}

impl std::ops::Deref for ScratchGuard {
    type Target = Vec<f32>;
    fn deref(&self) -> &Vec<f32> {
        &self.0
    }
}

impl std::ops::DerefMut for ScratchGuard {
    fn deref_mut(&mut self) -> &mut Vec<f32> {
        &mut self.0
    }
}
