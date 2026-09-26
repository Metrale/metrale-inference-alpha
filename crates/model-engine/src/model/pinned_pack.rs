// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The bounds-checked way to pack the model's pinned metadata staging buffer
//! (`PinnedMetaStaging`).
//!
//! Bytes enter the buffer only through [`PinnedPacker::put_at`], which checks the
//! destination bound before it writes. The packing sites are `prefill_a`, `prefill_c`,
//! `prefill_b::upload_meta`, `prefill_b::stage_batched` and `decode_b`.
//!
//! ## Why `packed()` may span bytes this call never wrote
//!
//! Callers round field offsets up for alignment, which leaves gaps no write fills, and
//! then form one `&[u8]` over the whole packed range for a single H2D. A slice over an
//! uninitialised byte is UB whatever the device later does with it. It is sound here
//! because [`metrale_gpu_runtime::gpu::GpuBackend::alloc_host_pinned`] is documented to
//! return a zeroed region, so every byte of the buffer is initialised from allocation on.
//!
//! Owner: model-engine.
//! Invariants:
//! - `put_at` and `pad_to` refuse, before writing, anything that would end past
//!   `capacity()`, so `high_water() <= capacity()`.

use std::marker::PhantomData;

use anyhow::{Result, ensure};

/// 2026-09-25: Types that may be reinterpreted as bytes when packed into pinned staging.
///
/// # Safety
///
/// The implementing type must have no padding and no invalid bit patterns, so
/// that every byte of `[T]` is initialised and reading it as `[u8]` is defined.
pub(crate) unsafe trait PinnedPod: Copy {}

// 2026-09-25: SAFETY: fixed-width integers: no padding, every bit pattern valid.
unsafe impl PinnedPod for u8 {}
unsafe impl PinnedPod for u32 {}
unsafe impl PinnedPod for i32 {}
unsafe impl PinnedPod for i64 {}
unsafe impl PinnedPod for u64 {}

/// 2026-09-25: A bounds-checked cursor over the pinned staging allocation.
///
/// Built by [`super::types::PinnedMetaStaging::packer_for`]. Borrows the staging
/// struct so it cannot outlive it. The bytes it writes are the separate
/// `alloc_host_pinned` region the struct points at, not the struct itself, so a
/// shared borrow suffices and the caller can still read its source `Vec`s.
pub(crate) struct PinnedPacker<'a> {
    ptr: *mut u8,
    bytes: usize,
    high_water: usize,
    _staging: PhantomData<&'a ()>,
}

impl<'a> PinnedPacker<'a> {
    /// 2026-09-25: A packer over the `bytes` bytes at `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must be the base of a live, zero-initialised host allocation of at
    /// least `bytes` bytes, and the caller must hold exclusive access to it for
    /// `'a`.
    pub(crate) unsafe fn new(ptr: *mut u8, bytes: usize) -> Self {
        Self {
            ptr,
            bytes,
            high_water: 0,
            _staging: PhantomData,
        }
    }

    /// 2026-09-25: The end of the highest field placed or padded to. Callers use
    /// it to compute the next field's aligned offset.
    pub(crate) fn high_water(&self) -> usize {
        self.high_water
    }

    /// 2026-09-25: The byte bound this packer enforces. `packer_for` passes the
    /// smaller of the host allocation and the device destination room.
    pub(crate) fn capacity(&self) -> usize {
        self.bytes
    }

    /// 2026-09-25: Place `src` at byte offset `at`, refusing before the write if
    /// it would end past `capacity()` or the offset overflows.
    ///
    /// `what` names the field in the error, so an over-run report says which
    /// table was too big.
    pub(crate) fn put_at<T: PinnedPod>(&mut self, what: &str, at: usize, src: &[T]) -> Result<()> {
        let len = std::mem::size_of_val(src);
        let end = at.checked_add(len).ok_or_else(|| {
            anyhow::anyhow!("pinned staging: {what} offset {at} + {len} overflows")
        })?;
        ensure!(
            end <= self.bytes,
            "pinned staging: {what} needs bytes [{at}, {end}) but the buffer is {} B — \
             refusing to pack (a {}-element table at this context length does not fit)",
            self.bytes,
            src.len()
        );
        if len > 0 {
            // 2026-09-25: SAFETY: `end <= self.bytes` was just checked, so
            // `[at, at + len)` is inside the allocation `new`'s contract
            // guarantees. The source is a live `&[T]` and `len` is `size_of_val`
            // of that slice; `T: PinnedPod` makes every one of those bytes
            // initialised. `new`'s contract gives the packer exclusive access to
            // the destination for `'a`, so a live `&[T]` cannot point into it.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, self.ptr.add(at), len);
            }
        }
        self.high_water = self.high_water.max(end);
        Ok(())
    }

    /// 2026-09-25: Place the first `n` elements of `src` at `at`, erroring,
    /// before any write, if `src` is shorter than `n`.
    pub(crate) fn put_prefix_at<T: PinnedPod>(
        &mut self,
        what: &str,
        at: usize,
        src: &[T],
        n: usize,
    ) -> Result<()> {
        let head = src.get(..n).ok_or_else(|| {
            anyhow::anyhow!(
                "pinned staging: {what} has {} elements, needed {n}",
                src.len()
            )
        })?;
        self.put_at(what, at, head)
    }

    /// 2026-09-25: Extend the packed range to `end` without writing, for a
    /// caller whose device-side layout includes trailing alignment padding.
    ///
    /// Sound for the same reason the interior gaps are: the allocation is
    /// zeroed, so those bytes are initialised.
    pub(crate) fn pad_to(&mut self, end: usize) -> Result<()> {
        ensure!(
            end <= self.bytes,
            "pinned staging: padding to {end} exceeds the {} B buffer",
            self.bytes
        );
        self.high_water = self.high_water.max(end);
        Ok(())
    }

    /// 2026-09-25: The packed bytes, `[0, high_water)`, ready for one H2D copy.
    pub(crate) fn packed(self) -> &'a [u8] {
        // 2026-09-25: SAFETY: `high_water` only advances through `put_at` and
        // `pad_to`, both of which check it against `self.bytes` first, so the
        // range is inside the allocation. Every byte in it is initialised: the
        // ones written, plus gaps that `alloc_host_pinned` zeroed (see the
        // module docs). The lifetime is the staging borrow, so the region
        // outlives the slice.
        unsafe { std::slice::from_raw_parts(self.ptr, self.high_water) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: Drives a packer over a real zeroed heap allocation, standing
    /// in for the zeroed pinned region.
    fn with_buf<R>(bytes: usize, f: impl FnOnce(PinnedPacker<'_>) -> R) -> R {
        let mut backing = vec![0u8; bytes];
        // 2026-09-25: SAFETY: `backing` is live, zeroed, `bytes` long, and
        // exclusively borrowed for the duration of the call.
        let packer = unsafe { PinnedPacker::new(backing.as_mut_ptr(), bytes) };
        f(packer)
    }

    /// 2026-09-25: An over-long table is refused before any byte lands.
    #[test]
    fn refuses_before_writing_a_single_byte() {
        let mut backing = vec![0u8; 64];
        let over = vec![0xAAu32; 32]; // 2026-09-25: 128 B into a 64 B buffer
        // 2026-09-25: SAFETY: see `with_buf`.
        let mut packer = unsafe { PinnedPacker::new(backing.as_mut_ptr(), 64) };
        assert!(packer.put_at("block_table", 0, &over).is_err());
        // 2026-09-25: The refusal happened first: nothing was written, so the
        // buffer is untouched.
        assert!(backing.iter().all(|&b| b == 0));
    }

    #[test]
    fn packs_at_offsets_and_reports_the_high_water() {
        with_buf(256, |mut p| {
            p.put_at("positions", 0, &[1u32, 2, 3]).unwrap();
            assert_eq!(p.high_water(), 12);
            // 2026-09-25: A gap at 12..16 that nothing writes: still inside the packed range.
            p.put_at("slots", 16, &[7i64, 8]).unwrap();
            assert_eq!(p.high_water(), 32);
            let packed = p.packed();
            assert_eq!(packed.len(), 32);
            let positions: Vec<u32> = packed[0..12]
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            assert_eq!(positions, [1, 2, 3]);
            assert_eq!(&packed[12..16], &[0, 0, 0, 0], "gap stays zeroed");
            let slots: Vec<i64> = packed[16..32]
                .chunks_exact(8)
                .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            assert_eq!(slots, [7, 8]);
        });
    }

    #[test]
    fn the_last_byte_fits_and_one_more_does_not() {
        with_buf(16, |mut p| {
            p.put_at("exact", 8, &[1u32, 2]).unwrap();
            assert_eq!(p.high_water(), 16);
        });
        with_buf(16, |mut p| {
            assert!(p.put_at("one_over", 9, &[1u32, 2]).is_err());
        });
    }

    #[test]
    fn put_prefix_at_checks_the_source_length_too() {
        with_buf(64, |mut p| {
            let src = vec![1u32, 2, 3, 4];
            p.put_prefix_at("positions", 0, &src, 2).unwrap();
            assert_eq!(p.high_water(), 8);
            let e = p.put_prefix_at("positions", 0, &src, 5).unwrap_err();
            assert!(e.to_string().contains("needed 5"), "{e}");
            let packed = p.packed();
            assert_eq!(
                packed,
                [1u32, 2]
                    .into_iter()
                    .flat_map(u32::to_ne_bytes)
                    .collect::<Vec<_>>()
            );
        });
    }

    #[test]
    fn pad_to_extends_the_range_but_is_still_bounded() {
        with_buf(32, |mut p| {
            p.put_at("f", 0, &[1u32]).unwrap();
            p.pad_to(8).unwrap();
            assert_eq!(p.high_water(), 8);
            // 2026-09-25: pad_to never shrinks.
            p.pad_to(4).unwrap();
            assert_eq!(p.high_water(), 8);
            assert!(p.pad_to(33).is_err());
        });
    }

    /// 2026-09-25: An empty table (no blocks yet) packs nothing: `put_at` skips
    /// the copy and the high-water mark stays 0.
    #[test]
    fn empty_source_is_a_no_op() {
        with_buf(16, |mut p| {
            p.put_at("empty", 0, &[] as &[u32]).unwrap();
            assert_eq!(p.high_water(), 0);
            assert!(p.packed().is_empty());
        });
    }
}
