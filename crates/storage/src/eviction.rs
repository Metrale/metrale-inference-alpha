// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Eviction order for the high-speed-swap scratch slots: which
//! resident slot a newly needed block overwrites. `rank` orders the unpinned
//! slots by (recorded predictor score, last access epoch), ascending.
//!
//! Owner: metrale-storage high-speed swap.
//! Invariants: none beyond the types.

use std::collections::HashSet;

#[derive(Debug)]
pub struct EvictionPolicy {
    /// 2026-09-25: Per-slot epoch of the last `touch`; 0 when never touched.
    last_access: Vec<u64>,
    /// 2026-09-25: Per-slot score from `record_score`; -inf until one is
    /// recorded.
    last_score: Vec<f32>,
    /// 2026-09-25: The epoch the next `touch` records; it starts at 1.
    epoch: u64,
}

impl EvictionPolicy {
    pub fn new(num_slots: u32) -> Self {
        Self {
            last_access: vec![0; num_slots as usize],
            last_score: vec![f32::NEG_INFINITY; num_slots as usize],
            epoch: 1,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.last_access.len() as u32
    }

    /// 2026-09-25: Record an access to `slot` at the current epoch.
    pub fn touch(&mut self, slot: u32) {
        self.last_access[slot as usize] = self.epoch;
        self.epoch = self.epoch.wrapping_add(1);
    }

    pub fn record_score(&mut self, slot: u32, score: f32) {
        self.last_score[slot as usize] = score;
    }

    /// 2026-09-25: Return to the state `new` creates.
    pub fn reset(&mut self) {
        for v in self.last_access.iter_mut() {
            *v = 0;
        }
        for v in self.last_score.iter_mut() {
            *v = f32::NEG_INFINITY;
        }
        self.epoch = 1;
    }

    /// 2026-09-25: Slot indices, most evictable first, without `pinned`.
    /// Scores that do not compare (NaN) count as equal.
    pub fn rank(&self, pinned: &[u32]) -> Vec<u32> {
        let pinned_set: HashSet<u32> = pinned.iter().copied().collect();
        let mut candidates: Vec<u32> = (0..self.capacity())
            .filter(|s| !pinned_set.contains(s))
            .collect();
        candidates.sort_by(|a, b| {
            let ai = *a as usize;
            let bi = *b as usize;
            self.last_score[ai]
                .partial_cmp(&self.last_score[bi])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(self.last_access[ai].cmp(&self.last_access[bi]))
        });
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowest_score_evicts_first() {
        let mut p = EvictionPolicy::new(4);
        p.record_score(0, 10.0);
        p.record_score(1, 1.0);
        p.record_score(2, 5.0);
        p.record_score(3, 100.0);
        let order = p.rank(&[]);
        assert_eq!(order[0], 1, "lowest score should evict first: {order:?}");
        assert_eq!(order.last(), Some(&3));
    }

    #[test]
    fn ties_break_to_oldest_access() {
        let mut p = EvictionPolicy::new(3);
        p.record_score(0, 5.0);
        p.record_score(1, 5.0);
        p.record_score(2, 5.0);
        p.touch(2);
        p.touch(0);
        let order = p.rank(&[]);
        assert_eq!(order[0], 1, "oldest should win the tie: {order:?}");
        assert_eq!(order[1], 2);
        assert_eq!(order[2], 0);
    }

    #[test]
    fn pinned_slots_excluded() {
        let mut p = EvictionPolicy::new(4);
        p.record_score(0, 1.0);
        p.record_score(1, 2.0);
        p.record_score(2, 3.0);
        p.record_score(3, 4.0);
        let order = p.rank(&[0, 1]);
        assert_eq!(order, vec![2, 3]);
    }

    #[test]
    fn reset_clears_state() {
        let mut p = EvictionPolicy::new(2);
        p.record_score(0, 1.0);
        p.record_score(1, 2.0);
        p.touch(0);
        p.touch(1);
        p.reset();
        assert_eq!(p.last_score, vec![f32::NEG_INFINITY; 2]);
        assert_eq!(p.last_access, vec![0; 2]);
        assert_eq!(p.epoch, 1);
    }
}
