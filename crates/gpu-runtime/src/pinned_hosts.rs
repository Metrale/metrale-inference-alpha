// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Which host buffers are page-locked, so the CUDA backend's
//! `copy_h2d_async` can keep its promise that the caller may drop the source on
//! return.
//!
//! From pageable memory the driver stages the bytes before the call returns; from
//! page-locked memory the DMA engine reads the caller's pages after it returns.
//! `MetraleCudaBackend::alloc_host_pinned` registers each buffer it allocates and
//! `free_host_pinned` unregisters it before the free. `copy_h2d_async` asks
//! [`is_pinned`] and, for a pinned source, synchronises the stream before
//! returning and logs a warning once per process.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Only memory from `alloc_host_pinned` is tracked. Page-locked memory from
//!   another route, such as the `cudaHostAlloc` workspace in `flashinfer.rs`, is
//!   not seen.
//! - Entries stay sorted by base, with at most one entry per base.

use std::sync::RwLock;

/// 2026-09-25: `(base, len)` of each live page-locked host region, sorted by base.
static PINNED: RwLock<Vec<(usize, usize)>> = RwLock::new(Vec::new());

/// 2026-09-25: Record a page-locked region. A repeated base replaces the stored
/// length; a null pointer or a zero length is ignored.
pub fn register(ptr: *const u8, bytes: usize) {
    if ptr.is_null() || bytes == 0 {
        return;
    }
    let base = ptr as usize;
    let mut g = match PINNED.write() {
        Ok(g) => g,
        // 2026-09-25: A poisoned lock only means an earlier holder panicked;
        // recover the table rather than propagate the panic.
        Err(e) => e.into_inner(),
    };
    match g.binary_search_by_key(&base, |&(b, _)| b) {
        Ok(i) => g[i].1 = bytes,
        Err(i) => g.insert(i, (base, bytes)),
    }
}

/// 2026-09-25: Forget a region. `free_host_pinned` calls it before the free, so a
/// reused address is not reported as still pinned.
pub fn unregister(ptr: *const u8) {
    if ptr.is_null() {
        return;
    }
    let base = ptr as usize;
    let mut g = match PINNED.write() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    if let Ok(i) = g.binary_search_by_key(&base, |&(b, _)| b) {
        g.remove(i);
    }
}

/// 2026-09-25: Whether all of `src` lies inside one live registered region. An
/// empty slice is never pinned.
pub fn is_pinned(src: &[u8]) -> bool {
    if src.is_empty() {
        return false;
    }
    let start = src.as_ptr() as usize;
    let g = match PINNED.read() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    // 2026-09-25: Largest base <= start, then a containment test, so a sub-slice
    // of a registered region matches too.
    match g.binary_search_by_key(&start, |&(b, _)| b) {
        Ok(i) => start + src.len() <= g[i].0 + g[i].1,
        Err(0) => false,
        Err(i) => {
            let (b, len) = g[i - 1];
            start < b + len && start + src.len() <= b + len
        }
    }
}

/// 2026-09-25: Number of live registered regions.
pub fn live_count() -> usize {
    match PINNED.read() {
        Ok(g) => g.len(),
        Err(e) => e.into_inner().len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: Serialises this module's tests, since `PINNED` is process-global.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn a_pageable_buffer_is_not_pinned() {
        let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let v = vec![0u8; 128];
        assert!(!is_pinned(&v));
    }

    #[test]
    fn a_registered_region_and_its_subslices_are_pinned() {
        let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut v = vec![0u8; 256];
        register(v.as_ptr(), v.len());

        assert!(is_pinned(&v), "the whole region");
        // 2026-09-25: A base-address-only lookup would miss these sub-slices.
        assert!(is_pinned(&v[..64]), "prefix");
        assert!(is_pinned(&v[64..128]), "interior");
        assert!(is_pinned(&v[255..]), "last byte");

        unregister(v.as_ptr());
        assert!(!is_pinned(&v), "unregistered again");
        v[0] = 1;
        assert_eq!(v[0], 1);
    }

    #[test]
    fn a_neighbouring_pageable_buffer_is_not_caught() {
        let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let pinned = vec![0u8; 128];
        let other = vec![0u8; 128];
        register(pinned.as_ptr(), pinned.len());
        assert!(is_pinned(&pinned));
        assert!(!is_pinned(&other), "a different allocation must not match");
        unregister(pinned.as_ptr());
    }

    #[test]
    fn empty_and_null_are_handled() {
        let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let before = live_count();
        register(std::ptr::null(), 16);
        register(0x1000 as *const u8, 0);
        assert_eq!(live_count(), before, "neither is a real region");
        unregister(std::ptr::null());
        assert!(!is_pinned(&[]));
    }

    #[test]
    fn re_registering_the_same_base_updates_the_length() {
        let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let v = vec![0u8; 256];
        let before = live_count();
        register(v.as_ptr(), 64);
        register(v.as_ptr(), 256);
        assert_eq!(live_count(), before + 1, "one entry, not two");
        assert!(is_pinned(&v[..256]), "the updated length is in effect");
        unregister(v.as_ptr());
    }
}
