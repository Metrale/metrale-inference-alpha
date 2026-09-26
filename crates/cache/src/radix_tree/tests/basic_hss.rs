// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: HSS disk-block ref tests for `RadixTree::insert` and `lookup`: which disk ids the cache reports as newly owned, and that an HSS-off match reports none.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: With HSS active, `insert` reports in
/// `InsertAcquired::disk_block_ids` each disk id it newly holds, so the caller
/// can take one disk ref per id.
#[test]
fn test_hss_disk_ref_acquisition_cold_insert() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    let block_table = vec![10, 20];
    let disk_ids = vec![100u32, 101u32];

    let acquired = tree.insert(&tokens, &block_table, &disk_ids, 16, 0, 0);
    assert_eq!(acquired.disk_block_ids, vec![100, 101]);
    assert_eq!(acquired.blocks, vec![10, 20]);
}

#[test]
fn test_hss_disk_ref_acquisition_re_insert_no_double() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    let block_table = vec![10, 20];
    let disk_ids = vec![100u32, 101u32];

    let acquired1 = tree.insert(&tokens, &block_table, &disk_ids, 16, 0, 0);
    assert_eq!(acquired1.disk_block_ids, vec![100, 101]);

    // 2026-09-25: The cache already holds these ids; reporting them again
    // would make the caller take a second disk ref that is never dropped.
    let acquired2 = tree.insert(&tokens, &block_table, &disk_ids, 16, 0, 0);
    assert!(
        acquired2.disk_block_ids.is_empty(),
        "re-insert should not re-acquire disk_ids; got {acquired2:?}"
    );
    assert!(
        acquired2.blocks.is_empty(),
        "re-insert should not re-ref blocks; got {acquired2:?}"
    );
}

#[test]
fn test_hss_disk_ref_acquisition_extension() {
    let tree = RadixTree::new();
    let tokens_short: Vec<u32> = (0..32).collect();
    let tokens_long: Vec<u32> = (0..48).collect();

    let acquired1 = tree.insert(&tokens_short, &[10, 20], &[100u32, 101u32], 16, 0, 0);
    assert_eq!(acquired1.disk_block_ids, vec![100, 101]);

    let acquired2 = tree.insert(
        &tokens_long,
        &[10, 20, 30],
        &[100u32, 101u32, 102u32],
        16,
        0,
        0,
    );
    assert_eq!(acquired2.disk_block_ids, vec![102]);
    assert_eq!(acquired2.blocks, vec![30]);
}

#[test]
fn test_hss_disk_ref_acquisition_no_op_when_hss_inactive() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();

    // 2026-09-25: An empty `disk_block_ids` slice means HSS is off.
    let acquired = tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);
    assert!(acquired.disk_block_ids.is_empty());
    // 2026-09-25: KV blocks are owned whether or not HSS is on. Without this
    // check an insert that does nothing when HSS is off would also pass.
    assert_eq!(acquired.blocks, vec![10, 20]);

    // 2026-09-25: `lookup` reports no disk ids either: these nodes carry the
    // `u32::MAX` sentinel, and `lookup` returns an empty list when every id is
    // the sentinel. `reuse_prefix_match_disk_ids` (model-engine
    // `block_mgmt.rs`) returns early on an empty list.
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32);
    assert!(
        m.matched_disk_block_ids.is_empty(),
        "an HSS-off match must not signal disk ids; got {:?}",
        m.matched_disk_block_ids
    );
    tree.release(&tokens, 16, 0);
}
