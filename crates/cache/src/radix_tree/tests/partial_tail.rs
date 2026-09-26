// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: With `METRALE_PREFIX_SUBBLOCK` unset, a radix match ends on a block boundary; with the lever armed, the sub-block tail arms still work.
//!
//! A match that ends inside a block hands the requester a block another owner
//! still holds, and the requester then writes its own K/V into that block from
//! `matched_tokens` onward (see `radix_tree::partial_tail`).
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use crate::radix_tree::RadixTree;
use metrale_telemetry::prefix_cache::PrefixCache;

use super::arm_legacy_partial_tail;

const BS: usize = 16;

/// 2026-09-25: Every block the walk hands out is fully covered by
/// `matched_tokens`, so none of them is also the requester's writable tail.
fn assert_no_writable_tail_handed_out(m: &metrale_telemetry::prefix_cache::PrefixMatch) {
    assert_eq!(
        m.matched_tokens % BS,
        0,
        "a match that ends inside a block makes that block the requester's \
         writable tail while its owner still writes it (#1193); matched={}",
        m.matched_tokens
    );
    assert_eq!(
        m.matched_blocks.len(),
        m.matched_tokens / BS,
        "block table must be exactly the fully-matched blocks; got {:?} for \
         matched={}",
        m.matched_blocks,
        m.matched_tokens
    );
}

/// 2026-09-25: The `partial_suffix` arm is off: the donor's partial tail
/// block (11) is not handed to a second sequence with the same prompt.
#[test]
fn a_partial_tail_block_is_never_handed_to_a_second_writer() {
    let tree = RadixTree::new();
    let prompt: Vec<u32> = (0..20).collect();
    tree.insert(&prompt, &[10, 11], &[], BS, 0, 0);

    let m = tree.lookup(&prompt, BS, 0, 0);
    assert_eq!(
        m.matched_tokens, 16,
        "only the donor's FULL blocks may be reused"
    );
    assert_eq!(m.matched_blocks, vec![10]);
    assert!(
        !m.matched_blocks.contains(&11),
        "block 11 is the donor's live tail; got {:?}",
        m.matched_blocks
    );
    assert_no_writable_tail_handed_out(&m);
    tree.release(&prompt, BS, 0);
}

/// 2026-09-25: The child-key arm is off: a remainder that is a prefix of a
/// full cached block does not match that block, since the requester would
/// write offsets `[remainder, block_size)` of a block other sequences read.
#[test]
fn a_committed_full_block_is_never_handed_out_as_a_second_writers_tail() {
    let tree = RadixTree::new();
    // 2026-09-25: 35 tokens: full blocks 10 and 20, and a 3-token tail in 30.
    let donor: Vec<u32> = (0..35).collect();
    tree.insert(&donor, &[10, 20, 30], &[], BS, 0, 0);

    // 2026-09-25: 22 tokens: one full block, then a 6-token remainder that is
    // a strict prefix of block 20's key.
    let requester: Vec<u32> = (0..22).collect();
    let m = tree.lookup(&requester, BS, 0, 0);
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    assert!(
        !m.matched_blocks.contains(&20),
        "block 20 is a committed block other sequences read; got {:?}",
        m.matched_blocks
    );
    assert_no_writable_tail_handed_out(&m);
    tree.release(&requester, BS, 0);
}

/// 2026-09-25: Positive control for the two tests above, which a tree that
/// matches nothing would also pass: a block-aligned lookup on the same donor
/// still reuses both full blocks.
#[test]
fn full_block_reuse_is_untouched_by_the_default() {
    let tree = RadixTree::new();
    let donor: Vec<u32> = (0..35).collect();
    tree.insert(&donor, &[10, 20, 30], &[], BS, 0, 0);

    let aligned: Vec<u32> = (0..32).collect();
    let m = tree.lookup(&aligned, BS, 0, 0);
    assert_eq!(m.matched_tokens, 32, "full-block reuse must still happen");
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&aligned, BS, 0);
}

/// 2026-09-25: Armed with `arm_legacy_partial_tail`, the same fixtures as the
/// two tests above get the non-aligned match back from both arms. This also
/// shows those tests measure the lever and not a broken walk.
#[test]
fn the_legacy_sub_block_arms_stay_reachable_for_ab() {
    let tree = RadixTree::new();
    arm_legacy_partial_tail(&tree);

    let prompt: Vec<u32> = (0..20).collect();
    tree.insert(&prompt, &[10, 11], &[], BS, 0, 0);
    let m = tree.lookup(&prompt, BS, 0, 0);
    assert_eq!(m.matched_tokens, 20, "armed: the partial tail is served");
    assert_eq!(m.matched_blocks, vec![10, 11]);
    tree.release(&prompt, BS, 0);

    // 2026-09-25: The child-key arm, on a second tree so the fixtures do not
    // interact.
    let tree = RadixTree::new();
    arm_legacy_partial_tail(&tree);
    let donor: Vec<u32> = (0..35).collect();
    tree.insert(&donor, &[10, 20, 30], &[], BS, 0, 0);
    let requester: Vec<u32> = (0..22).collect();
    let m = tree.lookup(&requester, BS, 0, 0);
    assert_eq!(
        m.matched_tokens, 22,
        "armed: the child-key sub-block is served"
    );
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&requester, BS, 0);
}
