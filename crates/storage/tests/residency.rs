// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Integration tests of [`Residency`] over [`VecSlotArena`] and
//! [`MemSwapStore`], through the public `metrale_storage::tier` paths.
//!
//! Owner: storage (tests).
//! Invariants: none beyond the types.

use metrale_storage::tier::{MemSwapStore, Residency, SlotArena, SwapStore, VecSlotArena};

const B: usize = 8;

fn blob(tag: u8) -> Vec<u8> {
    vec![tag; B]
}

/// 2026-09-25: The two-step put: `alloc`, write the slot, `commit`.
fn put(r: &mut Residency<VecSlotArena, MemSwapStore>, key: u64, tag: u8) {
    let slot = r.alloc(key).unwrap();
    r.arena_mut().write_slot(slot, &blob(tag)).unwrap();
    r.commit(key).unwrap();
}
fn get(r: &mut Residency<VecSlotArena, MemSwapStore>, key: u64) -> Option<Vec<u8>> {
    r.locate(key).unwrap().map(|slot| {
        let mut out = vec![0u8; B];
        r.arena().read_slot(slot, &mut out).unwrap();
        out
    })
}

fn residency(slots: usize) -> Residency<VecSlotArena, MemSwapStore> {
    Residency::new(VecSlotArena::new(B, slots), MemSwapStore::new(B)).unwrap()
}

fn residency_capped(slots: usize, max_disk: usize) -> Residency<VecSlotArena, MemSwapStore> {
    Residency::new_capped(VecSlotArena::new(B, slots), MemSwapStore::new(B), max_disk).unwrap()
}

/// 2026-09-25: With 2 slots and a disk cap of 3, ten puts keep five keys: the
/// coldest on-disk keys are dropped, a get of one misses, and the hottest two
/// come back intact.
#[test]
fn disk_cap_bounds_swap_and_drops_coldest() {
    let mut r = residency_capped(2, 3);
    for k in 0..10u64 {
        put(&mut r, k, k as u8);
    }
    assert_eq!(r.stats().spills_to_disk, 8);
    assert_eq!(r.stats().disk_evictions, 5, "only overflow is dropped");
    assert_eq!(r.disk_count(), 3, "the configured disk capacity is usable");
    assert_eq!(r.total_keys(), 2 + 3, "RAM and disk capacities are full");
    // 2026-09-25: The misses come first because a miss moves nothing.
    assert_eq!(
        get(&mut r, 0),
        None,
        "oldest key evicted from the capped disk"
    );
    assert_eq!(get(&mut r, 1), None);
    assert_eq!(get(&mut r, 9).as_deref(), Some(&blob(9)[..]));
    assert_eq!(get(&mut r, 8).as_deref(), Some(&blob(8)[..]));
}

/// 2026-09-25: Under a cap, `disk_count` stays at most the cap and
/// `disk_high_water`, the swap file's size in records, at most the cap plus
/// one: `locate` takes a faulting key out of `disk_lru` while its record is
/// live, so `make_disk_room` counts one record fewer than exist. model-engine's
/// `ssm_tier_disk_slots` passes its record budget minus one as the cap
/// (`ssm_tier/unified.rs`).
#[test]
fn disk_high_water_never_exceeds_cap_plus_one() {
    const CAP: usize = 4;
    let mut r = residency_capped(2, CAP);
    for k in 0..40u64 {
        put(&mut r, k, k as u8);
        // 2026-09-25: Gets of the last three keys fault on-disk records back in
        // under the cap.
        for probe in k.saturating_sub(3)..k {
            let _ = get(&mut r, probe);
        }
        assert!(
            r.disk_count() <= CAP,
            "live on-disk records must stay within the cap (k={k}, {} > {CAP})",
            r.disk_count()
        );
        assert!(
            r.disk_high_water() <= CAP + 1,
            "swap file must never exceed cap+1 records (k={k}, {} > {})",
            r.disk_high_water(),
            CAP + 1
        );
    }
    assert_eq!(r.max_disk_slots(), CAP);
    assert!(
        r.stats().disk_evictions > 0,
        "the cap must actually have been load-bearing in this run"
    );
}

/// 2026-09-25: A fault-in never drops its own record. The get of key 1, the
/// coldest on-disk key with the disk at its cap, spills the resident key inside
/// the same call; `locate` has already taken key 1 out of `disk_lru`, so
/// `make_disk_room` drops nothing and key 1 comes back intact.
#[test]
fn capped_fault_in_never_self_evicts() {
    const CAP: usize = 2;
    let mut r = residency_capped(1, CAP);
    for k in 0..4u64 {
        put(&mut r, k, k as u8);
    }
    assert_eq!(r.disk_count(), CAP, "disk sitting exactly at the cap");
    let evictions_before = r.stats().disk_evictions;
    assert_eq!(
        get(&mut r, 1),
        Some(blob(1)),
        "the faulting record must survive its own fault-in, byte-identical"
    );
    assert_eq!(
        r.stats().disk_evictions,
        evictions_before,
        "the self-pin freed the cap slot, so the fault dropped NOTHING — least \
         of all its own record"
    );
    assert_eq!(
        get(&mut r, 2),
        Some(blob(2)),
        "the bystander on-disk key spilled during the fault is intact too"
    );
    assert!(
        r.disk_high_water() <= CAP + 1,
        "the transient extra live record is the documented +1, not unbounded growth"
    );
}

/// 2026-09-25: Without a cap, 64 keys in 4 slots are all kept and all fault
/// back intact.
#[test]
fn infinite_depth_spill_and_fault_byte_identical() {
    let mut r = residency(4);
    for k in 0..64u64 {
        put(&mut r, k, k as u8);
    }
    assert!(
        r.stats().spills_to_disk >= 60,
        "most keys must have spilled to disk"
    );
    assert_eq!(r.resident_count(), 4, "only 4 slots resident at once");
    assert_eq!(r.total_keys(), 64, "all 64 keys tracked — nothing dropped");
    for k in 0..64u64 {
        assert_eq!(
            get(&mut r, k).as_deref(),
            Some(&blob(k as u8)[..]),
            "key {k}"
        );
    }
    assert!(r.stats().faults_from_disk > 0);
}

/// 2026-09-25: A read-pinned key is not evicted even when it is the coldest;
/// the next unpinned key is spilled instead.
#[test]
fn read_pin_survives_concurrent_eviction() {
    let mut r = residency(2);
    put(&mut r, 0, 0);
    put(&mut r, 1, 1);
    assert!(r.locate(0).unwrap().is_some());
    r.pin_read(0);
    assert_eq!(r.read_pin_count(0), 1);
    let faults_before = r.stats().faults_from_disk;

    put(&mut r, 2, 2);
    assert_eq!(r.stats().spills_to_disk, 1, "exactly one eviction");
    assert_eq!(
        get(&mut r, 1),
        Some(blob(1)),
        "the UNPINNED key 1 was the victim"
    );
    assert_eq!(
        get(&mut r, 0),
        Some(blob(0)),
        "pinned key 0 survived intact"
    );
    assert_eq!(
        r.stats().faults_from_disk,
        faults_before + 1,
        "only key 1 faulted back; key 0 never spilled"
    );
}

/// 2026-09-25: Pins are counted: the key is protected until the last one is
/// released, then rejoins the LRU once.
#[test]
fn refcounted_read_pins_release_to_evictable() {
    let mut r = residency(2);
    put(&mut r, 0, 0);
    put(&mut r, 1, 1);
    r.pin_read(0);
    r.pin_read(0);
    assert_eq!(r.read_pin_count(0), 2);
    r.unpin_read(0);
    assert_eq!(r.read_pin_count(0), 1, "still one reader → still pinned");
    put(&mut r, 2, 2);
    assert_eq!(
        get(&mut r, 1),
        Some(blob(1)),
        "key 1 evicted while key 0 still pinned"
    );
    r.unpin_read(0);
    assert_eq!(r.read_pin_count(0), 0);
    assert_eq!(
        r.resident_count(),
        2,
        "keys 0 and 2 resident; no LRU double-insert"
    );
    put(&mut r, 3, 3);
    put(&mut r, 4, 4);
    assert_eq!(
        get(&mut r, 0),
        Some(blob(0)),
        "unpinned key 0 spilled+faulted byte-identical"
    );
}

#[test]
fn resident_hit_does_not_touch_disk() {
    let mut r = residency(4);
    for k in 0..3u64 {
        put(&mut r, k, k as u8);
    }
    let spills_before = r.stats().spills_to_disk;
    assert_eq!(get(&mut r, 1), Some(blob(1)));
    assert_eq!(
        r.stats().spills_to_disk,
        spills_before,
        "resident hit spills nothing"
    );
    assert!(r.stats().resident_hits >= 1);
}

#[test]
fn lru_evicts_coldest_first() {
    let mut r = residency(2);
    put(&mut r, 10, 10);
    put(&mut r, 11, 11);
    // 2026-09-25: The get makes 10 the hottest, so the put of 12 spills 11.
    get(&mut r, 10);
    put(&mut r, 12, 12);
    assert_eq!(get(&mut r, 11), Some(blob(11)));
    assert!(r.stats().faults_from_disk >= 1);
}

#[test]
fn overwrite_in_place_reuses_slot_no_leak() {
    let mut r = residency(2);
    put(&mut r, 5, 100);
    put(&mut r, 5, 200);
    assert_eq!(get(&mut r, 5), Some(blob(200)));
    assert_eq!(r.total_keys(), 1, "no phantom duplicate");
    assert_eq!(r.resident_count(), 1);
}

#[test]
fn overwrite_spilled_key_reclaims_disk() {
    let mut r = residency(1);
    put(&mut r, 1, 1);
    put(&mut r, 2, 2);
    assert_eq!(r.disk_high_water(), 1);
    put(&mut r, 1, 99);
    assert_eq!(
        r.disk_high_water(),
        2,
        "rewriting secures a slot before reclaiming the old record"
    );
    put(&mut r, 3, 3);
    assert_eq!(
        r.disk_high_water(),
        2,
        "the overwritten key's old disk record is reusable"
    );
    assert_eq!(get(&mut r, 1), Some(blob(99)));
    assert_eq!(get(&mut r, 2), Some(blob(2)));
    assert_eq!(get(&mut r, 3), Some(blob(3)));
}

#[test]
fn remove_frees_resources() {
    let mut r = residency(2);
    put(&mut r, 1, 1);
    put(&mut r, 2, 2);
    put(&mut r, 3, 3);
    r.remove(1);
    r.remove(2);
    assert_eq!(get(&mut r, 1), None, "removed key is a clean miss");
    assert_eq!(get(&mut r, 2), None);
    assert_eq!(get(&mut r, 3), Some(blob(3)));
    assert_eq!(r.total_keys(), 1);

    let spills = r.stats().spills_to_disk;
    let high_water = r.disk_high_water();
    put(&mut r, 4, 4);
    assert_eq!(
        r.stats().spills_to_disk,
        spills,
        "the removed resident key releases its arena slot"
    );
    put(&mut r, 5, 5);
    assert_eq!(
        r.disk_high_water(),
        high_water,
        "the removed spilled key releases its disk record for reuse"
    );
    assert_eq!(get(&mut r, 4), Some(blob(4)));
    assert_eq!(get(&mut r, 5), Some(blob(5)));
}

#[test]
fn unknown_key_is_clean_miss() {
    let mut r = residency(2);
    assert_eq!(r.locate(0xdead).unwrap(), None);
    assert_eq!(r.stats().get_miss, 1);
}

#[test]
fn reserved_slot_pinned_during_put() {
    // 2026-09-25: With the only slot reserved and uncommitted, a second
    // `alloc` errors instead of evicting it.
    let mut r = residency(1);
    let slot = r.alloc(1).unwrap();
    let err = r.alloc(2);
    assert!(err.is_err(), "must not evict an uncommitted reserved slot");
    r.arena_mut().write_slot(slot, &blob(1)).unwrap();
    r.commit(1).unwrap();
    assert_eq!(get(&mut r, 1), Some(blob(1)));
}

#[test]
fn size_mismatch_rejected() {
    let bad = Residency::new(VecSlotArena::new(8, 2), MemSwapStore::new(16));
    assert!(
        bad.is_err(),
        "arena/swap size mismatch must be rejected at construction"
    );
}

/// 2026-09-25: `put_blob`/`get_blob` keep and return all 32 keys through a
/// two-slot arena; a wrong-sized buffer is an error.
#[test]
fn put_get_blob_helpers_never_reject_and_roundtrip() {
    let mut r = residency(2);
    for k in 0..32u64 {
        r.put_blob(k, &blob(k as u8)).unwrap();
    }
    assert_eq!(r.total_keys(), 32, "never-reject: every key tracked");
    assert_eq!(r.resident_count(), 2);
    let mut out = vec![0u8; B];
    for k in 0..32u64 {
        assert!(r.get_blob(k, &mut out).unwrap(), "key {k} present");
        assert_eq!(out, blob(k as u8), "key {k} byte-identical");
    }
    assert!(
        !r.get_blob(999, &mut out).unwrap(),
        "unknown key is a clean miss"
    );
    assert!(r.put_blob(1, &[0u8; B + 1]).is_err());
    let mut short = vec![0u8; B - 1];
    assert!(r.get_blob(1, &mut short).is_err());
}

/// 2026-09-25: `Residency<Box<dyn SlotArena>, Box<dyn SwapStore>>`, the type
/// model-engine's `UnifiedSnapshotStore` holds, round-trips blobs.
#[test]
fn boxed_trait_objects_compose() {
    let arena: Box<dyn SlotArena> = Box::new(VecSlotArena::new(B, 2));
    let swap: Box<dyn SwapStore> = Box::new(MemSwapStore::new(B));
    let mut r = Residency::new(arena, swap).unwrap();
    for k in 0..16u64 {
        r.put_blob(k, &blob(k as u8)).unwrap();
    }
    let mut out = vec![0u8; B];
    for k in 0..16u64 {
        assert!(r.get_blob(k, &mut out).unwrap());
        assert_eq!(out, blob(k as u8));
    }
}
