// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `MmapSlotArena`.
//!
//! Owner: storage, SSM snapshot tier.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Over a page-aligned heap buffer, a slot round-trips and an
/// out-of-range slot is refused.
#[test]
fn mmap_slot_arena_roundtrips() {
    let slot_bytes = 4096usize;
    let n = 3usize;
    let mut p: *mut libc::c_void = std::ptr::null_mut();
    let rc = unsafe { libc::posix_memalign(&mut p, 4096, slot_bytes * n) };
    assert!(rc == 0 && !p.is_null(), "posix_memalign failed rc={rc}");
    {
        let mut arena = unsafe { MmapSlotArena::new(p as *mut u8, slot_bytes, n) };
        let pat = vec![0x3C_u8; slot_bytes];
        arena.write_slot(1, &pat).unwrap();
        let mut out = vec![0u8; slot_bytes];
        arena.read_slot(1, &mut out).unwrap();
        assert_eq!(out, pat);
        assert!(
            arena.write_slot(3, &pat).is_err(),
            "slot out of range rejected"
        );
    }
    unsafe { libc::free(p) };
}
