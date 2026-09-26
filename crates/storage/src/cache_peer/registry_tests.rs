// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of the cache-peer paging registry: arenas keyed by
//! (kind, blob_bytes), and the disk-cap precedence of `carve_disk_slots`.
//!
//! Owner: metrale-storage peers.
//! Invariants: none beyond the types.

use super::carve_disk_slots;

/// 2026-09-25: With equal blob_bytes, kind 0 and kind 1 get separate arenas
/// and swap files, so a key put in one is not found in the other; a second
/// kind-1 request returns the same arena. Uses the real registry (anonymous
/// mapping, O_DIRECT swap file) and no RDMA.
#[test]
fn cross_kind_arenas_are_disjoint_in_the_real_registry() {
    // 2026-09-25: The swap file is opened O_DIRECT; the test returns early
    // when the filesystem refuses that.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/metrale-kv-paging-registry-test");
    std::fs::create_dir_all(&dir).unwrap();
    let rdma = super::RdmaConfig {
        swap_dir: Some(dir.clone()),
        ..Default::default()
    };
    let ledger = std::sync::Arc::new(crate::blade_cap::CommitLedger::new(0));
    // 2026-09-25: The registry is a process-wide static, so no other test may
    // use this blob size; `DirectSwapFile` needs a 4 KiB multiple.
    let blob = 8192usize;
    let ssm = match super::get_or_init_shared_paging(&rdma, 0, 4 * blob, blob, &ledger) {
        Ok(sh) => sh,
        Err(e) => {
            eprintln!("skipping registry test (filesystem refused O_DIRECT): {e:#}");
            return;
        }
    };
    let kv = super::get_or_init_shared_paging(&rdma, 1, 4 * blob, blob, &ledger).unwrap();
    assert!(
        !std::sync::Arc::ptr_eq(&ssm, &kv),
        "same blob_bytes, different kind ⇒ different arenas"
    );
    let kv2 = super::get_or_init_shared_paging(&rdma, 1, 4 * blob, blob, &ledger).unwrap();
    assert!(std::sync::Arc::ptr_eq(&kv, &kv2));
    assert!(dir.join(format!("metrale-snap-0-{blob}.swap")).exists());
    assert!(dir.join(format!("metrale-snap-1-{blob}.swap")).exists());
    let key = 0x4B56_4B56_4B56_4B56u64;
    kv.residency
        .lock()
        .unwrap()
        .put_blob(key, &vec![0x4B; blob])
        .unwrap();
    assert_eq!(ssm.residency.lock().unwrap().locate(key).unwrap(), None);
    assert!(kv.residency.lock().unwrap().locate(key).unwrap().is_some());
}

#[test]
fn carve_disk_slots_precedence() {
    let bb = 4u64;
    let (slots, rem) = carve_disk_slots(Some(40), 100, 100, bb);
    assert_eq!(slots, 10);
    assert_eq!(
        rem, 100,
        "per-kind override must not consume the shared remainder"
    );
    assert_eq!(carve_disk_slots(Some(0), 100, 100, bb), (0, 100));
    let (slots, rem) = carve_disk_slots(None, 100, 100, bb);
    assert_eq!(slots, 25);
    assert_eq!(rem, 0, "shared carve consumes the remainder");
    assert_eq!(carve_disk_slots(None, 0, 0, bb), (0, 0));
    assert_eq!(carve_disk_slots(None, 100, 0, bb), (1, 0));
}
