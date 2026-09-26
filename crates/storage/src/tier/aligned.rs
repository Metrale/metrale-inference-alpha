// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `PageAlignedBuf`, a 4 KiB-aligned heap buffer, so `Residency`'s
//! scratch goes to O_DIRECT without a copy.
//!
//! Owner: storage, tiered-cache core.
//! Invariants:
//! - A non-empty buffer's pointer is 4 KiB-aligned and owns `len` initialised bytes,
//!   zeroed at creation, until drop.
//!
//! Not a `Vec<u8>`: the unix `DirectSwapFile` copies any buffer whose address is not
//! 4 KiB-aligned through its own bounce, and a `Vec` allocation is not guaranteed
//! to be.

/// 2026-09-25: An owned, zeroed, 4 KiB-aligned byte buffer. The zero-length
/// [`Default`] allocates nothing, so `std::mem::take` on a field of this type is
/// cheap.
pub(crate) struct PageAlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// 2026-09-25: SAFETY: exclusive ownership of a plain heap allocation; no interior
// sharing.
unsafe impl Send for PageAlignedBuf {}

const PAGE: usize = 4096;

impl PageAlignedBuf {
    /// 2026-09-25: Zeroed buffer of `len` bytes, 4 KiB-aligned. `len == 0` allocates
    /// nothing; a failed allocation panics.
    pub(crate) fn new(len: usize) -> Self {
        if len == 0 {
            return Self::default();
        }
        let layout = std::alloc::Layout::from_size_align(len, PAGE)
            .expect("PageAlignedBuf: len > 0 and PAGE is a valid power-of-two align");
        // 2026-09-25: SAFETY: non-zero layout size; freed in `Drop` with the same layout.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "PageAlignedBuf: alloc of {len} B failed");
        Self { ptr, len }
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // 2026-09-25: SAFETY: `ptr` owns `len` initialised bytes for the lifetime of
        // `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // 2026-09-25: SAFETY: as above, and `&mut self` guarantees exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Default for PageAlignedBuf {
    fn default() -> Self {
        Self {
            ptr: std::ptr::NonNull::dangling().as_ptr(),
            len: 0,
        }
    }
}

impl Drop for PageAlignedBuf {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        let layout = std::alloc::Layout::from_size_align(self.len, PAGE)
            .expect("PageAlignedBuf: layout was valid at construction");
        // 2026-09-25: SAFETY: allocated by `new` with this exact layout, never freed
        // twice.
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}
