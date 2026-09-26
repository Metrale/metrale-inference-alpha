// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `RadixTree` tests for intermediate SSM snapshots, the armed sub-block tail arms, and adapter isolation of snapshot keys.
//!
//! The `SsmSnapshotIndex` unit tests are in `snapshot_index.rs`, mounted from
//! `radix_tree/snapshot.rs`.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use crate::radix_tree::RadixTree;
use metrale_telemetry::prefix_cache::PrefixCache;

use super::super::hash_token_prefix;
use super::super::snapshot::SsmSnapshotIndex;
use super::arm_legacy_partial_tail;

#[test]
fn test_insert_without_snapshot() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();

    tree.insert(&tokens, &[10], &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);
    // 2026-09-25: The KV walk must hit first: `lookup` skips the snapshot
    // index when `matched_tokens == 0`, so the snapshot checks below would also
    // pass for a tree that matches nothing.
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    assert_eq!(m.ssm_snapshot, None);
    assert_eq!(m.ssm_snapshot_tokens, 0);
    // 2026-09-25: A spilled anchor is reported in the tier fields, not in
    // `ssm_snapshot`, so they are checked too.
    assert_eq!(m.ssm_snapshot_tier_key, None);
    assert_eq!(m.ssm_snapshot_tier_tokens, 0);
    assert!(!m.ssm_snapshot_is_tail);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_intermediate_snapshot_on_partial_match() {
    let tree = RadixTree::new();

    let tokens: Vec<u32> = (0..64).collect();
    tree.insert(&tokens, &[10, 20, 30, 40], &[], 16, 0, 0);

    let tokens_at_2: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&tokens_at_2, &[10, 20], &[], 16, 50, 0, 0, 0);

    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 64);
    assert_eq!(m.ssm_snapshot, Some(50));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_intermediate_snapshot_deepest_wins() {
    let tree = RadixTree::new();

    let tokens: Vec<u32> = (0..64).collect();
    tree.insert_with_snapshot(&tokens, &[10, 20, 30, 40], &[], 16, 99, 0, 0, 0);

    let tokens_at_2: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&tokens_at_2, &[10, 20], &[], 16, 50, 0, 0, 0);

    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 64);
    assert_eq!(m.ssm_snapshot, Some(99));
    assert_eq!(m.ssm_snapshot_tokens, 64);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_intermediate_snapshot_partial_prefix_hit() {
    let tree = RadixTree::new();

    let tokens: Vec<u32> = (0..64).collect();
    tree.insert(&tokens, &[10, 20, 30, 40], &[], 16, 0, 0);

    let tokens_at_2: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&tokens_at_2, &[10, 20], &[], 16, 50, 0, 0, 0);

    // 2026-09-25: Shares the first 48 tokens and diverges in the fourth
    // block; the snapshot at 32 is within the match.
    let mut tokens_new: Vec<u32> = (0..48).collect();
    tokens_new.extend(200..216);
    let m = tree.lookup(&tokens_new, 16, 0, 0);
    assert_eq!(m.matched_tokens, 48);
    assert_eq!(m.ssm_snapshot, Some(50));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&tokens_new, 16, 0);
}

#[test]
fn test_intermediate_snapshot_survives_tree_eviction() {
    let tree = RadixTree::new();

    let tokens: Vec<u32> = (0..32).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0);

    let tokens_at_1: Vec<u32> = (0..16).collect();
    tree.insert_intermediate_snapshot(&tokens_at_1, &[10], &[], 16, 50, 0, 0, 0);

    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![20]);
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![10]);

    assert_eq!(tree.snapshot_count(), 1);
    let snap = tree.evict_snapshot_lru();
    assert_eq!(snap, Some(50));
}

// 2026-09-25: Partial-suffix tests. The sub-block tail arms are off unless
// `METRALE_PREFIX_SUBBLOCK=1`; `tests::partial_tail` covers that default. The
// tests below that call `arm_legacy_partial_tail` check the armed arms.

#[test]
fn test_partial_suffix_insert_and_lookup() {
    let tree = RadixTree::new();
    arm_legacy_partial_tail(&tree);
    // 2026-09-25: 20 tokens: one full block and a 4-token partial block.
    let tokens: Vec<u32> = (0..20).collect();
    let block_table = vec![10, 20];

    tree.insert(&tokens, &block_table, &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);

    assert_eq!(m.matched_tokens, 20);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_partial_suffix_no_match_different_suffix() {
    let tree = RadixTree::new();
    arm_legacy_partial_tail(&tree);
    let tokens_a: Vec<u32> = (0..20).collect();
    tree.insert(&tokens_a, &[10, 20], &[], 16, 0, 0);

    // 2026-09-25: Same first block, different last 4 tokens: only the full
    // block matches.
    let mut tokens_b: Vec<u32> = (0..16).collect();
    tokens_b.extend(100..104);
    let m = tree.lookup(&tokens_b, 16, 0, 0);

    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&tokens_b, 16, 0);
}

#[test]
fn test_partial_suffix_not_matched_for_full_block_request() {
    let tree = RadixTree::new();
    arm_legacy_partial_tail(&tree);
    let tokens: Vec<u32> = (0..20).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);

    // 2026-09-25: 32 tokens: only the first block matches, and the unmatched
    // remainder is a whole block, so the sub-block arms do not apply.
    let tokens_32: Vec<u32> = (0..32).collect();
    let m = tree.lookup(&tokens_32, 16, 0, 0);

    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&tokens_32, 16, 0);

    // 2026-09-25: A block-aligned request has remainder 0, and the sub-block
    // arms must not run: an empty suffix is a prefix of every stored key, so
    // without the `remainder > 0` guard the partial block would be appended to
    // a 16-token match.
    let tokens_16: Vec<u32> = (0..16).collect();
    let m16 = tree.lookup(&tokens_16, 16, 0, 0);
    assert_eq!(m16.matched_tokens, 16);
    assert_eq!(
        m16.matched_blocks,
        vec![10],
        "the partial slot must not be appended to a block-aligned match"
    );
    assert_eq!(
        m16.matched_blocks.len(),
        m16.matched_tokens / 16,
        "block table must stay aligned with matched_tokens"
    );
    tree.release(&tokens_16, 16, 0);
}

#[test]
fn test_partial_suffix_eviction_frees_both_blocks() {
    let tree = RadixTree::new();
    // 2026-09-25: Not armed: the partial block is still stored on insert, and
    // evicting its node frees both blocks.
    let tokens: Vec<u32> = (0..20).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0);

    let evicted = tree.evict(1);
    assert!(evicted.physical.contains(&10));
    assert!(evicted.physical.contains(&20));
}

/// 2026-09-25: A new child node supersedes the partial-suffix slot on its
/// parent: the slot is cleared and its block comes back in `released_blocks`,
/// so the caller can drop the cache's ref on it. The retired block is never
/// served again. `tests::basic::test_partial_suffix_block_is_owned_and_released`
/// covers a partial slot replaced by another partial.
#[test]
fn test_partial_suffix_cleared_when_extended() {
    let tree = RadixTree::new();
    arm_legacy_partial_tail(&tree);
    let tokens_20: Vec<u32> = (0..20).collect();
    let first = tree.insert(&tokens_20, &[10, 20], &[], 16, 0, 0);
    assert!(
        first.blocks.contains(&20),
        "the partial slot takes a ref on block 20; got {:?}",
        first.blocks
    );

    let tokens_32: Vec<u32> = (0..32).collect();
    let extended = tree.insert(&tokens_32, &[10, 30], &[], 16, 0, 0);
    assert_eq!(
        extended.released_blocks,
        vec![20],
        "the superseded partial block must be handed back, not leaked"
    );
    assert!(
        extended.blocks.contains(&30),
        "the superseding child block is acquired; got {:?}",
        extended.blocks
    );

    // 2026-09-25: Armed, 20 tokens are served by the new child (block 30)
    // through the child-key arm.
    let m = tree.lookup(&tokens_20, 16, 0, 0);
    assert_eq!(m.matched_tokens, 20);
    assert_eq!(m.matched_blocks, vec![10, 30]);
    assert!(
        !m.matched_blocks.contains(&20),
        "the retired partial block must never be served again"
    );
    tree.release(&tokens_20, 16, 0);

    let m = tree.lookup(&tokens_32, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 30]);
    tree.release(&tokens_32, 16, 0);
}

#[test]
fn test_partial_suffix_multi_block_prefix() {
    let tree = RadixTree::new();
    arm_legacy_partial_tail(&tree);
    // 2026-09-25: 396 tokens: 24 full blocks and a 12-token partial block,
    // which is `block_table[24]`.
    let tokens: Vec<u32> = (0..396).collect();
    let block_table: Vec<u32> = (0..25).collect();

    tree.insert(&tokens, &block_table, &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);

    assert_eq!(m.matched_tokens, 396);
    assert_eq!(m.matched_blocks, block_table);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_partial_suffix_prefix_match_shorter_lookup() {
    let tree = RadixTree::new();
    arm_legacy_partial_tail(&tree);
    // 2026-09-25: 31 cached tokens (one full block and a 15-token partial);
    // a 22-token lookup's 6-token remainder is a prefix of that partial.
    let tokens_31: Vec<u32> = (0..31).collect();
    tree.insert(&tokens_31, &[10, 20], &[], 16, 0, 0);

    let tokens_22: Vec<u32> = (0..22).collect();
    let m = tree.lookup(&tokens_22, 16, 0, 0);

    assert_eq!(m.matched_tokens, 22);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens_22, 16, 0);
}

#[test]
fn test_sub_block_match_via_child_key_prefix() {
    let tree = RadixTree::new();
    arm_legacy_partial_tail(&tree);
    // 2026-09-25: 35 cached tokens (two full blocks and a 3-token partial); a
    // 22-token lookup's 6-token remainder is a prefix of the second block's
    // key, so the child-key arm adds that block.
    let tokens_35: Vec<u32> = (0..35).collect();
    tree.insert(&tokens_35, &[10, 20, 30], &[], 16, 0, 0);

    let tokens_22: Vec<u32> = (0..22).collect();
    let m = tree.lookup(&tokens_22, 16, 0, 0);

    assert_eq!(m.matched_tokens, 22);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens_22, 16, 0);
}

#[test]
fn test_partial_suffix_sub_block_only() {
    let tree = RadixTree::new();
    // 2026-09-25: No full block, and a partial suffix is never stored on the
    // root, so nothing is cached.
    let tokens: Vec<u32> = (0..10).collect();
    tree.insert(&tokens, &[42], &[], 16, 0, 0);

    assert_eq!(tree.stats(), (0, 0));
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 0);
}

/// 2026-09-25: With `adapter_id == 0` the snapshot key is the plain FNV-1a
/// hash of the tokens; a non-zero adapter changes it.
#[test]
fn test_hash_token_prefix_base_byte_identical() {
    let tokens: Vec<u32> = vec![7, 42, 1000, 65535, 3, 0, 128];
    let mut expected: u64 = 0xcbf29ce484222325;
    for &t in &tokens {
        expected ^= t as u64;
        expected = expected.wrapping_mul(0x100000001b3);
    }
    assert_eq!(
        hash_token_prefix(&tokens, tokens.len(), 0),
        expected,
        "base (adapter_id=0) hash must be byte-identical to the pre-#24 value"
    );
    assert_ne!(
        hash_token_prefix(&tokens, tokens.len(), 0),
        hash_token_prefix(&tokens, tokens.len(), 99),
    );
    assert_ne!(
        hash_token_prefix(&tokens, tokens.len(), 7),
        hash_token_prefix(&tokens, tokens.len(), 9),
    );
}

#[test]
fn test_snapshot_index_adapter_isolation() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..16).collect();
    const A: u64 = 0xAA;
    const B: u64 = 0xBB;

    let ph_a = hash_token_prefix(&tokens, 16, A);
    idx.insert(ph_a, 42, 0, 16);

    assert_eq!(idx.lookup(&tokens, 16, 0, B), None);
    assert_eq!(idx.lookup(&tokens, 16, 0, A), Some((42, 16)));
    assert_eq!(idx.lookup(&tokens, 16, 0, 0), None);
}

#[test]
fn test_ssm_snapshot_adapter_isolation_via_tree() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    const A: u64 = 0x55;
    const B: u64 = 0x66;

    tree.insert_with_snapshot(&tokens, &[10, 20], &[], 16, 42, 0, 0, A);
    tree.release(&tokens, 16, A);

    let m_b = tree.lookup(&tokens, 16, 0, B);
    assert!(m_b.is_empty());
    assert_eq!(m_b.ssm_snapshot, None);

    let m_a = tree.lookup(&tokens, 16, 0, A);
    assert_eq!(m_a.matched_tokens, 32);
    assert_eq!(m_a.ssm_snapshot, Some(42));
    tree.release(&tokens, 16, A);

    // 2026-09-25: Give B its own KV for the same tokens so B's walk hits.
    // `lookup` skips the snapshot index when `matched_tokens == 0`, so the miss
    // above comes from the per-adapter tree roots alone; this checks that the
    // snapshot key carries the adapter.
    tree.insert(&tokens, &[30, 40], &[], 16, 0, B);
    tree.release(&tokens, 16, B);
    let m_b2 = tree.lookup(&tokens, 16, 0, B);
    assert_eq!(m_b2.matched_tokens, 32, "B now has its own cached KV");
    assert_eq!(m_b2.matched_blocks, vec![30, 40]);
    assert_eq!(
        m_b2.ssm_snapshot, None,
        "A's SSM snapshot must not restore for B even on a B-side KV hit"
    );
    assert_eq!(m_b2.ssm_snapshot_tier_key, None);
    tree.release(&tokens, 16, B);
}
