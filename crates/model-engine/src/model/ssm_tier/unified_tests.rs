// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `UnifiedSnapshotStore` tests: the flag parse, capped and uncapped stores,
//! LRU victim choice, the disk-budget conversion, and the transport adapter.
//!
//! Owner: model-engine (SSM snapshot tier).
//! Invariants: none beyond the types.

use super::super::MockSnapshotTransport;
use super::*;

// 2026-09-25: Blob size for the in-process arena and swap fixtures.
const BLOB: usize = 4;

#[test]
fn unified_flag_accepts_only_documented_truthy_values() {
    for on in ["1", "true", "on", "yes", " 1 ", " true ", " on ", " yes "] {
        assert!(unified_flag_truthy(Some(on)), "{on:?} must engage the flag");
    }
    for off in ["", "0", "false", "off", "no", "TRUE", "2"] {
        assert!(!unified_flag_truthy(Some(off)), "{off:?} must stay off");
    }
    assert!(!unified_flag_truthy(None), "unset = default OFF");
}

fn unified_store(slots: usize) -> UnifiedSnapshotStore {
    UnifiedSnapshotStore::new(
        Box::new(metrale_storage::tier::VecSlotArena::new(BLOB, slots)),
        Box::new(metrale_storage::tier::MemSwapStore::new(BLOB)),
        BLOB,
    )
    .unwrap()
}

fn unified_store_capped(slots: usize, max_disk: usize) -> UnifiedSnapshotStore {
    UnifiedSnapshotStore::new_capped(
        Box::new(metrale_storage::tier::VecSlotArena::new(BLOB, slots)),
        Box::new(metrale_storage::tier::MemSwapStore::new(BLOB)),
        BLOB,
        max_disk,
    )
    .unwrap()
}

/// 2026-09-25: A capped store drops old records instead of refusing a put, so
/// `retire_refused_spill` is never reached through it; a dropped record shows up
/// later as a miss.
#[test]
fn capped_store_bounds_disk_but_never_rejects() {
    const CAP: usize = 3;
    let s = unified_store_capped(2, CAP);
    for k in 0..64u64 {
        assert!(
            s.put(k, &[k as u8; BLOB]).unwrap(),
            "put {k} must never be refused by a CAPPED tier either"
        );
        assert!(
            s.disk_records() <= CAP,
            "on-disk records bounded by the cap (k={k})"
        );
    }
    assert_eq!(s.disk_records(), CAP, "the requested disk cap is usable");
    assert_eq!(s.disk_evictions(), 59, "only overflow is dropped");
    assert_eq!(s.len(), 2 + CAP, "hot and disk capacities are both usable");
    assert_eq!(s.stats.put_rejects.load(Ordering::Relaxed), 0);
}

/// 2026-09-25: A record dropped by the cap reads back as `Ok(false)` with `out`
/// untouched: the miss that `fault_in_for_key` handles by freeing the slot.
#[test]
fn capped_store_miss_is_clean() {
    let s = unified_store_capped(1, 2);
    for k in 0..16u64 {
        assert!(s.put(k, &[k as u8; BLOB]).unwrap());
    }
    assert_eq!(s.disk_records(), 2);
    assert_eq!(s.disk_evictions(), 13);
    assert_eq!(s.len(), 3);
    let mut o = [0xAAu8; BLOB];
    assert!(
        !s.get(0, &mut o).unwrap(),
        "the coldest key was dropped at the cap → clean miss"
    );
    assert_eq!(o, [0xAAu8; BLOB], "out untouched on a miss (no torn bytes)");
    assert!(s.get(15, &mut o).unwrap());
    assert_eq!(o, [15u8; BLOB]);
}

/// 2026-09-25: `UnifiedSnapshotStore::new` must stay uncapped:
/// `build_decode_tier_store` relies on it never dropping a record.
#[test]
fn uncapped_new_never_drops() {
    let s = unified_store(2);
    for k in 0..256u64 {
        assert!(s.put(k, &[k as u8; BLOB]).unwrap());
    }
    assert_eq!(
        s.disk_evictions(),
        0,
        "the decode tier's constructor must never drop a blob — a dropped \
         rollback target is a corrupt restore"
    );
    assert_eq!(s.len(), 256, "every key still tracked");
    let mut o = [0u8; BLOB];
    for k in 0..256u64 {
        assert!(s.get(k, &mut o).unwrap(), "key {k} still present");
        assert_eq!(o, [k as u8; BLOB], "key {k} byte-identical");
    }
}

/// 2026-09-25: The GiB-to-records conversion of `disk_slots_from`, with a
/// 66,846,720 B blob (16,320 × 4 KiB).
#[test]
fn ssm_tier_disk_slots_conversion_and_strictness() {
    const HOLO_BLOB: usize = 66_846_720;
    assert_eq!(
        disk_slots_from(None, HOLO_BLOB).unwrap(),
        0,
        "unset ⇒ unbounded"
    );
    assert_eq!(disk_slots_from(Some(""), HOLO_BLOB).unwrap(), 0);
    assert_eq!(
        disk_slots_from(Some("0"), HOLO_BLOB).unwrap(),
        0,
        "0 is the explicit unbounded sentinel"
    );
    // 2026-09-25: 32 GiB / 66,846,720 B = 514 records; one is held back for the
    // fault-in headroom, so the worst-case file (514 records) still fits 32 GiB.
    assert_eq!(disk_slots_from(Some("32"), HOLO_BLOB).unwrap(), 513);
    assert_eq!(disk_slots_from(Some(" 32 "), HOLO_BLOB).unwrap(), 513);
    assert!(
        (513 + 1) as u64 * HOLO_BLOB as u64 <= 32 * (1u64 << 30),
        "worst-case swap file must fit the operator's budget"
    );
    // 2026-09-25: A malformed value is an error, never uncapped.
    for bad in ["abc", "-1", "32GB", "nan", "inf", "1e400"] {
        assert!(
            disk_slots_from(Some(bad), HOLO_BLOB).is_err(),
            "{bad:?} must be a config error, not a silent unbounded tier"
        );
    }
    // 2026-09-25: A budget too small for two snapshots is an error, not 0, which
    // would mean uncapped.
    let tiny = disk_slots_from(Some("0.0001"), HOLO_BLOB);
    assert!(
        tiny.is_err(),
        "under-sized budget must fail fast, got {tiny:?}"
    );
    assert!(
        disk_slots_from(Some("1"), 0).is_err(),
        "a positive budget requires nonzero snapshot geometry"
    );
}

/// 2026-09-25: The spill victim is the least recently used key, not the oldest
/// inserted. A capped `MemBlobStore` would evict key 1 here, and a 2-slot
/// `RdmaSnapshotStore` would refuse key 3.
#[test]
fn unified_store_victim_is_lru_not_fifo_and_not_a_reject() {
    let s = unified_store(2);
    assert!(s.put(1, &[1; BLOB]).unwrap());
    assert!(s.put(2, &[2; BLOB]).unwrap());
    let mut o = [0u8; BLOB];
    assert!(s.get(1, &mut o).unwrap()); // 2026-09-25: touch 1, so 2 is now the coldest
    assert!(s.put(3, &[3; BLOB]).unwrap(), "no drop-on-full");
    assert_eq!(s.bytes_resident(), 2 * BLOB, "two hot slots resident");
    // 2026-09-25: Key 1 is still in the hot tier (no disk fault), and key 2,
    // the LRU victim, faults back from disk.
    let faults0 = s.inner.lock().stats().faults_from_disk;
    assert!(s.get(1, &mut o).unwrap(), "hot-again key survives");
    assert_eq!(o, [1u8; BLOB]);
    assert_eq!(
        s.inner.lock().stats().faults_from_disk,
        faults0,
        "key 1 was still RESIDENT — the LRU victim was key 2, not the FIFO-oldest"
    );
    assert!(
        s.get(2, &mut o).unwrap(),
        "spilled key faults back, never dropped"
    );
    assert_eq!(o, [2u8; BLOB]);
    assert_eq!(
        s.inner.lock().stats().faults_from_disk,
        faults0 + 1,
        "key 2 came back via a disk fault"
    );
    assert!(s.get(3, &mut o).unwrap());
    assert_eq!(o, [3u8; BLOB]);
}

#[test]
fn unified_store_wrong_size_refused_gracefully() {
    let s = unified_store(2);
    assert!(s.put(1, &[7; BLOB]).unwrap());
    assert!(!s.put(1, &[8; BLOB - 1]).unwrap(), "short put refused");
    assert!(!s.put(1, &[9; BLOB + 1]).unwrap(), "long put refused");
    assert_eq!(s.stats.put_rejects.load(Ordering::Relaxed), 2);
    let mut short = [0xa5u8; BLOB - 1];
    let mut long = [0x5au8; BLOB + 1];
    assert!(!s.get(1, &mut short).unwrap(), "short get refused");
    assert!(!s.get(1, &mut long).unwrap(), "long get refused");
    assert_eq!(short, [0xa5; BLOB - 1]);
    assert_eq!(long, [0x5a; BLOB + 1]);
    assert_eq!(s.stats.get_misses.load(Ordering::Relaxed), 2);
    let mut exact = [0u8; BLOB];
    assert!(s.get(1, &mut exact).unwrap());
    assert_eq!(exact, [7; BLOB], "refused replacements preserve the value");
}

#[test]
fn unified_store_remove_releases_resident_key() {
    let s = unified_store(2);
    assert!(s.put(1, &[1; BLOB]).unwrap());
    s.remove(1);
    let mut o = [0u8; BLOB];
    assert!(!s.get(1, &mut o).unwrap());
    assert_eq!(s.len(), 0);
    assert_eq!(s.bytes_resident(), 0);
}

/// 2026-09-25: Over a 4-slot transport arena, where `RdmaSnapshotStore` would
/// refuse the fifth key, the unified store spills to the swap tier and keeps
/// every key.
#[test]
fn unified_over_transport_never_drops_where_bounded_store_did() {
    const SLOTS: usize = 4;
    let hot = Box::new(TransportSlotArena {
        transport: Box::new(MockSnapshotTransport::new(SLOTS * BLOB)),
        slot_bytes: BLOB,
        num_slots: SLOTS,
    });
    let s = UnifiedSnapshotStore::new(
        hot,
        Box::new(metrale_storage::tier::MemSwapStore::new(BLOB)),
        BLOB,
    )
    .unwrap();
    let mut o = [0u8; BLOB];
    for k in 0..16u64 {
        assert!(
            s.put(k, &[k as u8; BLOB]).unwrap(),
            "arena-full put {k} accepted"
        );
    }
    for k in 0..16u64 {
        assert!(s.get(k, &mut o).unwrap(), "key {k} recoverable");
        assert_eq!(o, [k as u8; BLOB]);
    }
}
