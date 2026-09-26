// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched-verify ordering and graph key: the dispatch order and
//! depth assignment of one verify batch, and the CUDA-graph cache key built
//! from them.
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - `verify_batch_permutation` returns a permutation of `0..n`, and panics
//!   when `slots` and `ks` differ in length.
//! - With `canonical = true`, `verify_batch_order` places slots in
//!   non-decreasing order and depths in non-increasing order, and returns the
//!   multiset of depths it was given.
//! - With `canonical = false`, each member keeps its own depth, ordered
//!   deepest first, then by slot.
//!
//! The scheduler orders each batch with these functions (`mtp_dcut::plan`,
//! `mtp_step`) and the model keys its batched-verify graphs with
//! [`verify_graph_key`] (`verify_e2::verify_batched_graph_key`), so the
//! dispatch and the key see one order. Under the canonical assignment the key
//! depends on the slot set and the depth multiset, not on which sequence D-Cut
//! gave which depth, which reduces the number of distinct keys. Ascending
//! slots also keep a depth run on consecutive slots when the pool slots are
//! consecutive; the batched GDN path checks the actual pointers and declines
//! otherwise (`trait_decode_batched_conv_gdn_multi.rs`).
//!
//! [`canonical_assignment`] chooses the arm from the batch width.
//! `METRALE_NO_CANONICAL_VERIFY_KEY`, present with any value (`0` included),
//! turns the canonical arm off at every width.

/// 2026-09-25: True unless `METRALE_NO_CANONICAL_VERIFY_KEY` is present. Read
/// once per process.
pub fn canonical_verify_key_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_CANONICAL_VERIFY_KEY").is_none())
}

/// 2026-09-25: Batch width (sequences) at or above which [`canonical_assignment`]
/// selects the canonical arm, unless `METRALE_CANONICAL_KEY_MIN_WIDTH` overrides
/// it. Above the scheduler's D-Cut width cap (`SchedLevers::dcut_width_cap`),
/// `mtp_dcut::plan` leaves every depth equal, and then both arms give the same
/// order (`uniform_depths_are_identical_under_both_arms`).
pub const CANONICAL_KEY_MIN_WIDTH: usize = 8;

/// 2026-09-25: The width threshold: `METRALE_CANONICAL_KEY_MIN_WIDTH` when it
/// parses as a `usize` (0 selects the canonical arm at every width), else
/// [`CANONICAL_KEY_MIN_WIDTH`]. Read once per process.
pub fn canonical_key_min_width() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| min_width_from_env(std::env::var_os(ENV_MIN_WIDTH)))
}

const ENV_MIN_WIDTH: &str = "METRALE_CANONICAL_KEY_MIN_WIDTH";

/// 2026-09-25: Parses a raw [`ENV_MIN_WIDTH`] value; [`canonical_key_min_width`]
/// does the environment read, so tests call this directly.
fn min_width_from_env(raw: Option<std::ffi::OsString>) -> usize {
    raw.and_then(|v| v.into_string().ok())
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(CANONICAL_KEY_MIN_WIDTH)
}

/// 2026-09-25: Whether a batch of `n` sequences gets the canonical depth-to-slot
/// assignment. `mtp_dcut::plan` asks with the batch width and `mtp_step` asks
/// with the same width for each chunk it dispatches, so the order `plan`
/// assigned and the permutation `mtp_step` applies use the same arm.
pub fn canonical_assignment(n: usize) -> bool {
    canonical_assignment_at(n, canonical_key_min_width(), canonical_verify_key_enabled())
}

/// 2026-09-25: The policy of [`canonical_assignment`] with its two
/// environment inputs passed in: the threshold, and whether the kill switch
/// is clear. With the kill switch set, no width or threshold selects the
/// canonical arm.
fn canonical_assignment_at(n: usize, min_width: usize, kill_switch_clear: bool) -> bool {
    kill_switch_clear && n >= min_width
}

/// 2026-09-25: Dispatch order for one verify batch, as a permutation only.
///
/// `slots[i]` and `ks[i]` describe member `i` in the caller's order (`ks[i]`
/// is its row count, drafts + 1). `order[p]` is the input index dispatched at
/// position `p`.
///
/// * `canonical = true`: ascending ssm slot; `ks` is unread.
/// * `canonical = false`: deepest first, then ascending ssm slot.
///
/// Ties break on input index, so the order is a deterministic function of
/// the inputs. Under `canonical = true` a second application to the ordered
/// batch returns the identity (`canonical_order_is_idempotent`).
///
/// Panics when `slots` and `ks` differ in length.
pub fn verify_batch_permutation(slots: &[usize], ks: &[usize], canonical: bool) -> Vec<usize> {
    assert_eq!(
        slots.len(),
        ks.len(),
        "verify_batch_permutation: slots/ks mismatch"
    );
    let n = slots.len().min(ks.len());
    let mut order: Vec<usize> = (0..n).collect();
    if canonical {
        order.sort_by_key(|&i| (slots[i], i));
    } else {
        order.sort_by_key(|&i| (std::cmp::Reverse(ks[i]), slots[i], i));
    }
    order
}

/// 2026-09-25: Orders one verify batch and assigns its depths; the entry point
/// of `mtp_dcut::plan`, which truncates each sequence's drafts to the depth
/// assigned here.
///
/// Returns `(order, depths)`: `order` is [`verify_batch_permutation`] and
/// `depths[p]` is the row count position `p` verifies.
///
/// * `canonical = true`: the depth multiset is sorted descending onto the
///   ordered batch, so `depths[p]` need not be `ks[order[p]]`. Slots are
///   non-decreasing in `p` and depths non-increasing.
/// * `canonical = false`: each member keeps its own depth.
///
/// Stages that only need the dispatch order use [`verify_batch_permutation`],
/// which leaves every member's depth attached to it.
pub fn verify_batch_order(
    slots: &[usize],
    ks: &[usize],
    canonical: bool,
) -> (Vec<usize>, Vec<usize>) {
    let order = verify_batch_permutation(slots, ks, canonical);
    let depths: Vec<usize> = if canonical {
        let mut d: Vec<usize> = ks[..order.len()].to_vec();
        d.sort_unstable_by(|a, b| b.cmp(a));
        d
    } else {
        order.iter().map(|&i| ks[i]).collect()
    };
    (order, depths)
}

/// 2026-09-25: The batched-verify CUDA-graph cache key: the `(ssm slot, row
/// count)` pairs in dispatch order, then one sentinel word,
/// `u32::MAX - wy_tables_null - 2 * write_on_accept`, which differs for each
/// of the four flag combinations. `write_on_accept` is in the key because it
/// can select a different WY launch (the write-on-accept K=4 twin,
/// `trait_decode_batched_conv_gdn_multi.rs`).
pub fn verify_graph_key(
    pairs: &[(u32, u32)],
    wy_tables_null: bool,
    write_on_accept: bool,
) -> Vec<u32> {
    let mut key: Vec<u32> = Vec::with_capacity(2 * pairs.len() + 1);
    for &(slot, k) in pairs {
        key.push(slot);
        key.push(k);
    }
    key.push(u32::MAX - u32::from(wy_tables_null) - 2 * u32::from(write_on_accept));
    key
}

#[cfg(test)]
#[path = "verify_key_tests.rs"]
mod tests;
