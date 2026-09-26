// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Radix tree tests of adapter isolation: blocks cached under
//! one `adapter_id` are reused only by lookups with that id.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use crate::radix_tree::RadixTree;
use metrale_telemetry::prefix_cache::PrefixCache;

/// 2026-09-25: A prefix cached under adapter A misses for adapter B and for
/// the base model (id 0), since its K/V carry adapter A's delta, and still
/// hits for A.
#[test]
fn test_kv_cache_adapter_isolation() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    const A: u64 = 0x1111_2222_3333_4444;
    const B: u64 = 0x5555_6666_7777_8888;

    // 2026-09-25: A inserts, then the inserting sequence releases.
    tree.insert(&tokens, &[10, 20], &[], 16, 0, A);
    tree.release(&tokens, 16, A);

    let m_b = tree.lookup(&tokens, 16, 0, B);
    assert!(
        m_b.is_empty(),
        "adapter B must NOT reuse adapter A's KV blocks"
    );

    let m_base = tree.lookup(&tokens, 16, 0, 0);
    assert!(
        m_base.is_empty(),
        "base must NOT reuse an adapter's KV blocks"
    );

    let m_a = tree.lookup(&tokens, 16, 0, A);
    assert_eq!(m_a.matched_tokens, 32);
    assert_eq!(m_a.matched_blocks, vec![10, 20]);
    tree.release(&tokens, 16, A);
}

/// 2026-09-25: Blocks cached by the base model (id 0) miss for an adapter and
/// hit for the base model.
#[test]
fn test_base_blocks_not_reused_by_adapter() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();

    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0);

    let m_adapter = tree.lookup(&tokens, 16, 0, 0xABCD);
    assert!(m_adapter.is_empty());

    let m_base = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m_base.matched_tokens, 32);
    assert_eq!(m_base.matched_blocks, vec![10, 20]);
    tree.release(&tokens, 16, 0);
}

/// 2026-09-25: Adapter B's insert of the same tokens creates its own nodes
/// instead of reusing adapter A's, so each adapter gets back its own blocks.
#[test]
fn test_adapter_insert_does_not_clobber_other_adapter() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    const A: u64 = 0xAAAA;
    const B: u64 = 0xBBBB;

    tree.insert(&tokens, &[10, 20], &[], 16, 0, A);
    tree.release(&tokens, 16, A);
    tree.insert(&tokens, &[30, 40], &[], 16, 0, B);
    tree.release(&tokens, 16, B);

    let m_a = tree.lookup(&tokens, 16, 0, A);
    assert_eq!(
        m_a.matched_blocks,
        vec![10, 20],
        "adapter A kept its blocks"
    );
    tree.release(&tokens, 16, A);

    let m_b = tree.lookup(&tokens, 16, 0, B);
    assert_eq!(
        m_b.matched_blocks,
        vec![30, 40],
        "adapter B kept its blocks"
    );
    tree.release(&tokens, 16, B);
}
