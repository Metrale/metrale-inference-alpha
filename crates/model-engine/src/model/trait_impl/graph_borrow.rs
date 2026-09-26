// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Drain-tail CUDA-graph borrowing: replay a wider captured graph instead of capturing one per shrinking batch composition.
//!
//! The batched-decode cache (`decode_graph_key.rs`) and the batched-verify cache
//! (`verify_e2.rs`) key on the per-row SSM slot vector, so a drain step, whose slot vector
//! differs from every captured one, never hits them exactly. Retirement compacts survivors
//! into slots `[0..n)` (`retire_finished_sequences`, except under EP protocol v2), so a drain
//! batch's slot vector is a prefix of a wider canonical key. A captured graph whose key starts
//! with this batch's slots is replayed as is: the active rows sit at the rows the scheduler
//! reads, and the tail rows become padding.
//!
//! A tail row writes into the pool state of the slot baked there, so every tail slot must be
//! the dummy slot or a currently free slot (`SsmStatePool::slot_is_free`); a claimed slot
//! vetoes the borrow. A slot's next owner overwrites its h and conv state: allocation zeroes
//! them (`alloc_sequence_dispatch`) and compaction copies over them (`SsmStatePool::copy_slot`).
//! The scheduler charges a plain decode step's time to `active.len()` rows, not to the
//! borrowed width.
//!
//! Policy: borrow only graphs at most [`borrow_width_cap`] wide. `METRALE_NO_GRAPH_BORROW`
//! (presence; `=0` also disables) turns borrowing off. Read once per process.
//!
//! Owner: model-engine decode graphs.
//! Invariants:
//! - A borrowed key is strictly wider than the batch, at most `borrow_width_cap(n)` wide, and
//!   starts with the batch's own slots (and, for verify, its depths and sentinel).

/// 2026-09-25: Borrowing enabled unless `METRALE_NO_GRAPH_BORROW` is present.
pub(super) fn graph_borrow_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_GRAPH_BORROW").is_none())
}

/// 2026-09-25: Widest captured graph an `n`-active batch may borrow: twice its power-of-two
/// width bucket (`n.next_power_of_two()`, the bucket the `mtp_gate` width regime uses).
/// The padding ladder is not used: 12 is a ladder rung, so twice it (24) would keep n=12
/// from borrowing a 32-wide graph.
pub(super) fn borrow_width_cap(n: usize) -> usize {
    2 * n.next_power_of_two()
}

/// 2026-09-25: Dedup gate for the borrow `info!` logs: a line is logged only when the
/// (exact key, borrowed key) pair differs from the previous borrow, so a drain band that
/// replays one borrowed graph logs once.
pub(super) struct BorrowLogGate(std::sync::atomic::AtomicU64);

impl BorrowLogGate {
    pub(super) const fn new() -> Self {
        Self(std::sync::atomic::AtomicU64::new(0))
    }

    /// 2026-09-25: True when this (exact, borrowed) pair is not the one last logged.
    /// FNV-1a over both keys; 0 is reserved for "nothing logged yet".
    pub(super) fn should_log(&self, exact: &[u32], borrowed: &[u32]) -> bool {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        // 2026-09-25: Lengths preserve the pair boundary. Without them, ([1], [2, 3])
        // and ([1, 2], [3]) hash the same concatenated values and a real
        // transition is suppressed.
        for v in std::iter::once(exact.len() as u64)
            .chain(exact.iter().map(|&v| u64::from(v)))
            .chain(std::iter::once(borrowed.len() as u64))
            .chain(borrowed.iter().map(|&v| u64::from(v)))
        {
            h = (h ^ v).wrapping_mul(0x0000_0100_0000_01b3);
        }
        let h = if h == 0 { 1 } else { h };
        self.0.swap(h, std::sync::atomic::Ordering::Relaxed) != h
    }
}

pub(super) static DECODE_BORROW_LOG: BorrowLogGate = BorrowLogGate::new();
pub(super) static VERIFY_BORROW_LOG: BorrowLogGate = BorrowLogGate::new();

/// 2026-09-25: Find a captured batched-decode key to replay for `active_slots`
/// (this batch's per-row SSM slots, batch order).
///
/// A candidate key `K` (row-slot vector, length = its captured width) is
/// borrowable iff:
///   * `n < K.len() <= borrow_width_cap(n)` — strictly wider than the active
///     batch (a same-length match IS the exact key) and inside the factor-2
///     window;
///   * `K[..n] == active_slots` — active rows land on rows baked with their
///     own slots, in order;
///   * every tail entry `K[n..]` satisfies `tail_ok` (caller passes "is the
///     dummy slot, or a currently-free pool slot").
///
/// Returns the narrowest borrowable key (fewest wasted pad lanes). `None` when fewer than
/// two rows are active.
pub(super) fn find_borrowable_decode_key<'a>(
    active_slots: &[u32],
    keys: impl Iterator<Item = &'a Vec<u32>>,
    tail_ok: impl Fn(u32) -> bool,
) -> Option<Vec<u32>> {
    let n = active_slots.len();
    if n < 2 {
        return None;
    }
    let cap = borrow_width_cap(n);
    let mut best: Option<&'a Vec<u32>> = None;
    for k in keys {
        let m = k.len();
        if m <= n || m > cap {
            continue;
        }
        if best.is_some_and(|b| b.len() <= m) {
            continue;
        }
        if k[..n] != *active_slots {
            continue;
        }
        if k[n..].iter().all(|&s| tail_ok(s)) {
            best = Some(k);
        }
    }
    best.cloned()
}

/// 2026-09-25: A borrowable batched-verify graph: the cached key to replay plus the
/// ghost `(slot, k)` tail, the rows baked beyond the active batch, which the caller
/// stages WY table entries for.
pub(super) struct VerifyBorrow {
    pub(super) key: Vec<u32>,
    pub(super) ghosts: Vec<(u32, u32)>,
}

/// 2026-09-25: Find a captured batched-verify key to replay for this batch's exact key
/// (`verify_e2::verify_batched_graph_key` layout: `n` interleaved
/// `(slot, k)` pairs then one sentinel).
///
/// A candidate `K` with `m` pairs is borrowable iff:
///   * `n < m <= borrow_width_cap(n)`;
///   * the sentinels match (a table-less capture never replays a table-full
///     step or the reverse, the same rule as the exact key);
///   * `K`'s first `n` pairs equal the active pairs: same slots and the
///     same per-row depths, so `off[i]` (the scheduler's logits/stash row
///     base) is identical for every active sequence;
///   * every tail pair satisfies `tail_ok(slot, k)` (the caller passes: the slot is
///     free and, when WY tables are staged, its intermediate pool covers depth k).
///
/// Returns the candidate with the fewest ghost rows (Σ tail k).
pub(super) fn find_borrowable_verify_key<'a>(
    exact_key: &[u32],
    keys: impl Iterator<Item = &'a Vec<u32>>,
    tail_ok: impl Fn(u32, u32) -> bool,
) -> Option<VerifyBorrow> {
    // 2026-09-25: `n` pairs plus a sentinel is odd-length; the borrow needs n >= 2 pairs,
    // so length >= 5. Anything else is refused.
    if exact_key.len() < 5 || exact_key.len().is_multiple_of(2) {
        return None;
    }
    let n = (exact_key.len() - 1) / 2;
    let sentinel = exact_key[exact_key.len() - 1];
    let prefix = &exact_key[..2 * n];
    let cap = borrow_width_cap(n);
    let mut best: Option<(&'a Vec<u32>, usize)> = None;
    for k in keys {
        if k.len() < 5 || k.len().is_multiple_of(2) {
            continue;
        }
        let m = (k.len() - 1) / 2;
        if m <= n || m > cap || k[k.len() - 1] != sentinel {
            continue;
        }
        if k[..2 * n] != *prefix {
            continue;
        }
        let tail = &k[2 * n..k.len() - 1];
        if !tail.chunks_exact(2).all(|p| tail_ok(p[0], p[1])) {
            continue;
        }
        let ghost_rows: usize = tail.chunks_exact(2).map(|p| p[1] as usize).sum();
        if best.is_none_or(|(_, r)| ghost_rows < r) {
            best = Some((k, ghost_rows));
        }
    }
    best.map(|(k, _)| VerifyBorrow {
        key: k.clone(),
        ghosts: k[2 * n..k.len() - 1]
            .chunks_exact(2)
            .map(|p| (p[0], p[1]))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: A batch may borrow up to twice its power-of-two width bucket, not twice
    /// its padding-ladder width: `2 * padded_batch_n(12)` is 24, below a 32-wide graph.
    #[test]
    fn borrow_width_cap_is_twice_the_power_of_two_bucket() {
        assert_eq!(borrow_width_cap(1), 2);
        assert_eq!(borrow_width_cap(2), 4);
        assert_eq!(borrow_width_cap(3), 8);
        assert_eq!(borrow_width_cap(8), 16);
        assert_eq!(borrow_width_cap(9), 32);
        assert_eq!(borrow_width_cap(12), 32);
        assert_eq!(borrow_width_cap(13), 32);
        assert_eq!(borrow_width_cap(16), 32);
        assert_eq!(borrow_width_cap(17), 64);
        assert_eq!(borrow_width_cap(32), 64);
    }

    fn canonical(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    /// 2026-09-25: With a 32-wide canonical graph cached, every drain width from 9 to 31
    /// borrows it. Slots >= n are free in a drain because retirement compacts survivors into
    /// `[0..n)`.
    #[test]
    fn drain_widths_inside_the_bucket_replay_the_steady_state_graph() {
        let k32 = canonical(32);
        let cache = [k32.clone()];
        for n in 9..32usize {
            let active = canonical(n as u32);
            let found = find_borrowable_decode_key(&active, cache.iter(), |s| s >= n as u32);
            assert_eq!(
                found.as_ref(),
                Some(&k32),
                "drain width n={n} must replay the 32-wide graph, not capture"
            );
        }
    }

    /// 2026-09-25: Widths 2 to 8 are outside the 32-wide graph's window, so the borrow
    /// declines.
    #[test]
    fn drain_widths_below_the_window_capture_a_narrow_canonical_graph() {
        let cache = [canonical(32)];
        for n in 2..=8usize {
            let active = canonical(n as u32);
            assert!(
                find_borrowable_decode_key(&active, cache.iter(), |s| s >= n as u32).is_none(),
                "n={n} is outside the 32-wide borrow window"
            );
        }
    }

    /// 2026-09-25: A key captured at n=11 and padded with the dummy slot is borrowed by
    /// n = 6 to 10, whose tails mix free slots and the dummy slot.
    #[test]
    fn a_dummy_padded_canonical_key_serves_the_rest_of_its_bucket() {
        const DUMMY: u32 = 99;
        // 2026-09-25: Captured at n=11, padded to 12: rows [0..11) real, row 11 dummy.
        let mut k11 = canonical(11);
        k11.push(DUMMY);
        let cache = [k11.clone()];
        for n in 6..11usize {
            let active = canonical(n as u32);
            let found =
                find_borrowable_decode_key(&active, cache.iter(), |s| s == DUMMY || s >= n as u32);
            assert_eq!(
                found.as_ref(),
                Some(&k11),
                "n={n} must borrow the 12-row graph"
            );
        }
    }

    /// 2026-09-25: A tail slot claimed by a mid-prefill sequence (not free) blocks the
    /// borrow: pad lanes would write into its SSM state.
    #[test]
    fn a_claimed_tail_slot_vetoes_the_borrow() {
        let cache = [canonical(32)];
        let active = canonical(20);
        // 2026-09-25: Slot 20 is claimed by a prefilling sequence; 21..32 are free.
        let found = find_borrowable_decode_key(&active, cache.iter(), |s| s >= 21);
        assert!(found.is_none(), "claimed tail slot must veto the borrow");
    }

    /// 2026-09-25: A subset batch (such as {0,2,5}) or a permutation must not
    /// borrow: the baked rows would not match the active rows' slots.
    #[test]
    fn a_non_prefix_slot_vector_never_borrows() {
        let cache = [canonical(32)];
        assert!(find_borrowable_decode_key(&[0, 2, 5], cache.iter(), |_| true).is_none());
        assert!(find_borrowable_decode_key(&[1, 0, 2], cache.iter(), |_| true).is_none());
    }

    /// 2026-09-25: The narrowest borrowable graph wins (fewest wasted pad lanes).
    #[test]
    fn the_narrowest_candidate_is_preferred() {
        let cache = [canonical(32), canonical(24)];
        let active = canonical(17);
        let found = find_borrowable_decode_key(&active, cache.iter(), |_| true);
        assert_eq!(found, Some(canonical(24)));
    }

    /// 2026-09-25: A same-length key is the exact key's job, never a borrow; a one-row batch
    /// never borrows; malformed verify keys never match.
    #[test]
    fn invalid_or_exact_width_keys_never_borrow() {
        let cache = [canonical(8)];
        assert!(find_borrowable_decode_key(&canonical(8), cache.iter(), |_| true).is_none());
        assert!(find_borrowable_decode_key(&[0], cache.iter(), |_| true).is_none());

        let valid = [vkey(&[(0, 2), (1, 2)], WY)];
        assert!(find_borrowable_verify_key(&[0, 2, 1, 2], valid.iter(), |_, _| true).is_none());
        assert!(find_borrowable_verify_key(&[0, 2, 1, 2, WY], valid.iter(), |_, _| true).is_none());
        let malformed_candidates = [vec![0, 2, 1, 2], vec![0, 2, 1, 2, 2, 2]];
        assert!(
            find_borrowable_verify_key(&[0, 2, 1, 2, WY], malformed_candidates.iter(), |_, _| true)
                .is_none()
        );
    }

    // 2026-09-25: Batched-verify keys: interleaved (slot, k) pairs, then a sentinel.

    fn vkey(pairs: &[(u32, u32)], sentinel: u32) -> Vec<u32> {
        let mut k: Vec<u32> = pairs.iter().flat_map(|&(s, d)| [s, d]).collect();
        k.push(sentinel);
        k
    }

    fn uniform(n: u32, k: u32, sentinel: u32) -> Vec<u32> {
        vkey(&(0..n).map(|s| (s, k)).collect::<Vec<_>>(), sentinel)
    }

    const WY: u32 = u32::MAX - 1; // 2026-09-25: the sentinel of a step without WY tables

    /// 2026-09-25: A cached n=32, k=2 verify graph serves widths 9 to 31; the ghosts are
    /// the baked tail pairs.
    #[test]
    fn verify_drain_widths_replay_the_steady_state_graph_with_ghost_tails() {
        let k32 = uniform(32, 2, WY);
        let cache = [k32.clone()];
        for n in 9..32u32 {
            let exact = uniform(n, 2, WY);
            let found = find_borrowable_verify_key(&exact, cache.iter(), |s, _| s >= n)
                .unwrap_or_else(|| panic!("verify n={n} must borrow, not capture"));
            assert_eq!(found.key, k32);
            assert_eq!(
                found.ghosts,
                (n..32).map(|s| (s, 2)).collect::<Vec<_>>(),
                "ghost tail must be the baked (slot, k) pairs beyond the batch"
            );
        }
        let exact = uniform(8, 2, WY);
        assert!(
            find_borrowable_verify_key(&exact, cache.iter(), |s, _| s >= 8).is_none(),
            "n=8 is outside the 32-wide window"
        );
    }

    /// 2026-09-25: Borrow logging is transition-deduped: repeats of one pair stay silent,
    /// and every new pair logs.
    #[test]
    fn borrow_log_gate_fires_once_per_transition() {
        let gate = BorrowLogGate::new();
        let k32 = canonical(32);
        let a = canonical(20);
        let b = canonical(19);
        assert!(gate.should_log(&a, &k32), "first borrow must log");
        assert!(!gate.should_log(&a, &k32), "same pair repeats silently");
        assert!(!gate.should_log(&a, &k32));
        assert!(gate.should_log(&b, &k32), "width change logs again");
        assert!(gate.should_log(&a, &k32), "returning to a prior pair logs");

        let boundary = BorrowLogGate::new();
        assert!(boundary.should_log(&[1], &[2, 3]));
        assert!(
            boundary.should_log(&[1, 2], &[3]),
            "moving the exact/borrowed boundary is a distinct transition"
        );
    }

    /// 2026-09-25: Depth is part of the row layout: a prefix whose ks differ must not
    /// borrow (off[i] would shift under the scheduler's row reads), and the
    /// sentinel must match exactly.
    #[test]
    fn verify_borrow_requires_matching_depths_and_sentinel() {
        let cache = [uniform(32, 2, WY)];
        let deeper = uniform(20, 3, WY);
        assert!(find_borrowable_verify_key(&deeper, cache.iter(), |_, _| true).is_none());
        let other_sentinel = uniform(20, 2, u32::MAX);
        assert!(find_borrowable_verify_key(&other_sentinel, cache.iter(), |_, _| true).is_none());
    }

    /// 2026-09-25: The tail check sees each ghost's depth too: a slot whose tiered
    /// intermediate pool is too shallow for the baked k vetoes the borrow.
    #[test]
    fn verify_tail_check_receives_slot_and_depth() {
        let mut pairs: Vec<(u32, u32)> = (0..15).map(|s| (s, 2)).collect();
        pairs.push((15, 4));
        let cache = [vkey(&pairs, WY)];
        let exact = uniform(15, 2, WY);
        // 2026-09-25: Tier check: slot 15 only covers k ≤ 3.
        let found = find_borrowable_verify_key(&exact, cache.iter(), |s, k| s >= 15 && k <= 3);
        assert!(
            found.is_none(),
            "shallow-tier tail slot must veto the borrow"
        );
        let found = find_borrowable_verify_key(&exact, cache.iter(), |s, k| s >= 15 && k <= 4);
        assert!(found.is_some());
    }

    /// 2026-09-25: Fewest ghost rows wins (Σ tail k, not pair count).
    #[test]
    fn verify_prefers_the_fewest_ghost_rows() {
        let mut more_pairs: Vec<(u32, u32)> = (0..16).map(|s| (s, 2)).collect();
        more_pairs.extend((16..24).map(|s| (s, 1)));
        let a = vkey(&more_pairs, WY);
        let mut fewer_pairs: Vec<(u32, u32)> = (0..16).map(|s| (s, 2)).collect();
        fewer_pairs.extend((16..19).map(|s| (s, 4)));
        let b = vkey(&fewer_pairs, WY);
        let cache = [b, a.clone()];
        let exact = uniform(16, 2, WY);
        let found = find_borrowable_verify_key(&exact, cache.iter(), |_, _| true).unwrap();
        assert_eq!(found.key, a);
        assert_eq!(found.ghosts.len(), 8, "pair count is not the cost metric");
        assert_eq!(found.ghosts.iter().map(|&(_, k)| k).sum::<u32>(), 8);
    }
}
