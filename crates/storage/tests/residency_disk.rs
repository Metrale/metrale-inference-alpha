// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Integration tests of the store side of [`Residency`]: the first
//! store write happens only when the arena is full, and the buffer blobs move
//! through is 4 KiB-aligned.
//!
//! Owner: storage (tests).
//! Invariants: none beyond the types.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;

use metrale_storage::tier::{MemSwapStore, Residency, SwapStore, VecSlotArena};

const B: usize = 8;

fn blob(tag: u8) -> Vec<u8> {
    vec![tag; B]
}

fn residency(slots: usize) -> Residency<VecSlotArena, MemSwapStore> {
    Residency::new(VecSlotArena::new(B, slots), MemSwapStore::new(B)).unwrap()
}

/// 2026-09-25: The residency's scratch buffer is 4 KiB-aligned. The unix
/// `DirectSwapFile` checks `ptr & 0xfff == 0` and copies any other buffer
/// through its own bounce buffer, which changes no result, only the time.
#[test]
fn scratch_is_page_aligned() {
    let r = residency(2);
    assert_eq!(
        r.scratch_addr() & 0xfff,
        0,
        "residency scratch must be 4 KiB-aligned or every O_DIRECT record bounces"
    );
    let big = Residency::new(VecSlotArena::new(4096, 2), MemSwapStore::new(4096)).unwrap();
    assert_eq!(big.scratch_addr() & 0xfff, 0);
}

/// 2026-09-25: A `MemSwapStore` that counts `write_record` calls.
struct SpySwap {
    inner: MemSwapStore,
    writes: Arc<AtomicUsize>,
}
impl SwapStore for SpySwap {
    fn record_bytes(&self) -> usize {
        self.inner.record_bytes()
    }
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        self.inner.write_record(disk_slot, bytes)
    }
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
        self.inner.read_record(disk_slot, out)
    }
    fn discard_record(&mut self, disk_slot: usize) {
        self.inner.discard_record(disk_slot)
    }
}

/// 2026-09-25: The store is written only when the arena has no free slot: in a
/// one-slot arena the first key and its re-put write nothing, and the second
/// key writes one record.
#[test]
fn disk_write_lands_only_when_hot_arena_full() {
    let writes = Arc::new(AtomicUsize::new(0));
    let spy = SpySwap {
        inner: MemSwapStore::new(B),
        writes: Arc::clone(&writes),
    };
    let mut r = Residency::new(VecSlotArena::new(B, 1), spy).unwrap();

    r.put_blob(1, &blob(1)).unwrap();
    assert_eq!(
        writes.load(Ordering::SeqCst),
        0,
        "the 1-slot hot arena absorbs the first key — a 0-byte swap file here is EXPECTED"
    );
    r.put_blob(1, &blob(9)).unwrap();
    assert_eq!(
        writes.load(Ordering::SeqCst),
        0,
        "re-PUT of a live key overwrites in place, it does not spill"
    );
    r.put_blob(2, &blob(2)).unwrap();
    assert_eq!(
        writes.load(Ordering::SeqCst),
        1,
        "the FIRST key beyond the arena is what finally writes disk record 0"
    );
    assert_eq!(r.stats().spills_to_disk, 1);
    assert_eq!(r.disk_high_water(), 1, "exactly one record exists on disk");
}
