// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Integration tests of [`Residency`] when a spill, fault-in or put
//! fails part way, using an arena and a store that fail on request.
//!
//! Owner: storage (tests).
//! Invariants: none beyond the types.

use metrale_storage::tier::{MemSwapStore, Residency, SlotArena, SwapStore, VecSlotArena};

const B: usize = 8;

fn blob(tag: u8) -> Vec<u8> {
    vec![tag; B]
}

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail};

/// 2026-09-25: A `VecSlotArena` whose next `write_slot` fails once
/// `fail_next_write` is set through `arena_mut`.
struct FaultyArena {
    inner: VecSlotArena,
    fail_next_write: bool,
}
impl FaultyArena {
    fn new(slots: usize) -> Self {
        Self {
            inner: VecSlotArena::new(B, slots),
            fail_next_write: false,
        }
    }
}
impl SlotArena for FaultyArena {
    fn slot_bytes(&self) -> usize {
        self.inner.slot_bytes()
    }
    fn num_slots(&self) -> usize {
        self.inner.num_slots()
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        self.inner.read_slot(slot, out)
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        if self.fail_next_write {
            self.fail_next_write = false;
            bail!("injected write_slot failure");
        }
        self.inner.write_slot(slot, bytes)
    }
}

/// 2026-09-25: One-shot failure switches for [`FaultySwap`], shared through an
/// `Arc` because the store itself moves into the `Residency`.
#[derive(Default)]
struct SwapFaults {
    fail_write: AtomicBool,
    fail_read: AtomicBool,
}
struct FaultySwap {
    inner: MemSwapStore,
    faults: Arc<SwapFaults>,
}
impl FaultySwap {
    fn new() -> (Self, Arc<SwapFaults>) {
        let faults = Arc::new(SwapFaults::default());
        (
            Self {
                inner: MemSwapStore::new(B),
                faults: Arc::clone(&faults),
            },
            faults,
        )
    }
}
impl SwapStore for FaultySwap {
    fn record_bytes(&self) -> usize {
        self.inner.record_bytes()
    }
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
        if self.faults.fail_write.swap(false, Ordering::SeqCst) {
            bail!("injected write_record failure");
        }
        self.inner.write_record(disk_slot, bytes)
    }
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
        if self.faults.fail_read.swap(false, Ordering::SeqCst) {
            bail!("injected read_record failure");
        }
        self.inner.read_record(disk_slot, out)
    }
    fn discard_record(&mut self, disk_slot: usize) {
        self.inner.discard_record(disk_slot);
    }
}

/// 2026-09-25: After a `put_blob` whose arena write fails, the key is absent,
/// its slot is reusable and earlier keys are intact.
#[test]
fn put_blob_rolls_back_reservation_on_arena_write_failure() {
    let (swap, _f) = FaultySwap::new();
    let mut r = Residency::new(FaultyArena::new(2), swap).unwrap();
    r.put_blob(1, &blob(1)).unwrap();

    r.arena_mut().fail_next_write = true;
    assert!(r.put_blob(2, &blob(2)).is_err(), "write failure propagates");

    let mut out = vec![0u8; B];
    assert!(
        !r.get_blob(2, &mut out).unwrap(),
        "rolled-back key 2 misses cleanly (not a torn Reserved slot)"
    );
    assert_eq!(r.total_keys(), 1, "no stranded key-2 entry");
    r.put_blob(3, &blob(3)).unwrap();
    assert!(
        r.get_blob(1, &mut out).unwrap() && out == blob(1),
        "key 1 intact"
    );
    assert!(
        r.get_blob(3, &mut out).unwrap() && out == blob(3),
        "key 3 reuses the reclaimed slot"
    );
}

/// 2026-09-25: After a spill whose store write fails, the victim is still
/// resident and intact, no spill is counted, and a later spill succeeds.
#[test]
fn spill_rolls_back_on_swap_write_failure_victim_stays_resident() {
    let (swap, faults) = FaultySwap::new();
    let mut r = Residency::new(FaultyArena::new(1), swap).unwrap();
    r.put_blob(10, &blob(10)).unwrap();

    faults.fail_write.store(true, Ordering::SeqCst);
    assert!(
        r.put_blob(11, &blob(11)).is_err(),
        "spill write failure propagates"
    );
    assert_eq!(r.stats().spills_to_disk, 0, "failed spill is not counted");
    assert_eq!(r.total_keys(), 1, "failed put left no key 11");

    let mut out = vec![0u8; B];
    assert!(
        r.get_blob(10, &mut out).unwrap() && out == blob(10),
        "victim 10 stayed resident and intact"
    );
    assert!(!r.get_blob(11, &mut out).unwrap(), "key 11 never landed");
    r.put_blob(12, &blob(12)).unwrap();
    assert_eq!(r.stats().spills_to_disk, 1);
    assert!(
        r.get_blob(10, &mut out).unwrap() && out == blob(10),
        "10 faults back from disk byte-identical"
    );
}

/// 2026-09-25: After a fault-in whose store read fails, the key is still on
/// disk and a retry succeeds; the key spilled to make room for the failed
/// fault-in is intact.
#[test]
fn fault_in_read_failure_keeps_key_on_disk_and_frees_slot() {
    let (swap, faults) = FaultySwap::new();
    let mut r = Residency::new(FaultyArena::new(1), swap).unwrap();
    r.put_blob(20, &blob(20)).unwrap();
    r.put_blob(21, &blob(21)).unwrap();
    assert_eq!(r.stats().spills_to_disk, 1);

    faults.fail_read.store(true, Ordering::SeqCst);
    let mut out = vec![0u8; B];
    assert!(
        r.get_blob(20, &mut out).is_err(),
        "fault-in read failure propagates"
    );

    assert!(
        r.get_blob(20, &mut out).unwrap() && out == blob(20),
        "20 faults back on retry"
    );
    assert!(
        r.get_blob(21, &mut out).unwrap() && out == blob(21),
        "bystander 21 (spilled during the failed fault) is intact"
    );
}

/// 2026-09-25: A store record belongs to at most one key. A re-put of an
/// on-disk key whose spill fails must leave the key's record allocated: were it
/// freed, the next spill would write another key's blob into it, and a get of
/// the first key would return those bytes as a hit.
#[test]
fn failed_reput_of_spilled_key_must_not_alias_its_disk_record() {
    let (swap, faults) = FaultySwap::new();
    let mut r = Residency::new(FaultyArena::new(1), swap).unwrap();
    r.put_blob(1, &blob(1)).unwrap();
    r.put_blob(2, &blob(2)).unwrap();
    assert_eq!(r.stats().spills_to_disk, 1, "key 1 is on disk");

    // 2026-09-25: The re-put of key 1 has to spill key 2, and that store write
    // fails.
    faults.fail_write.store(true, Ordering::SeqCst);
    assert!(
        r.put_blob(1, &blob(99)).is_err(),
        "ENOSPC during the re-PUT of a spilled key propagates"
    );

    // 2026-09-25: This spill must not be given key 1's record.
    r.put_blob(3, &blob(3)).unwrap();

    let mut out = vec![0u8; B];
    let hit = r.get_blob(1, &mut out).unwrap();
    assert!(
        hit && out == blob(1),
        "the failed re-PUT must preserve key 1's old record, got hit={hit} {out:?}"
    );
}

/// 2026-09-25: Retrying a failed re-put of an on-disk key frees its record
/// once: had the failed attempt freed it too, the index would be on the free
/// list twice and two later spills would share one record.
#[test]
fn retried_reput_must_not_double_free_the_same_disk_record() {
    let (swap, faults) = FaultySwap::new();
    let mut r = Residency::new(FaultyArena::new(1), swap).unwrap();
    r.put_blob(1, &blob(1)).unwrap();
    r.put_blob(2, &blob(2)).unwrap();

    faults.fail_write.store(true, Ordering::SeqCst);
    assert!(
        r.put_blob(1, &blob(99)).is_err(),
        "first attempt hits ENOSPC"
    );
    r.put_blob(1, &blob(99)).unwrap();
    r.put_blob(4, &blob(4)).unwrap();

    let mut out = vec![0u8; B];
    let hit = r.get_blob(2, &mut out).unwrap();
    assert!(
        hit && out == blob(2),
        "key 2 must remain present with its own bytes after retry, got hit={hit} {out:?}"
    );
}
