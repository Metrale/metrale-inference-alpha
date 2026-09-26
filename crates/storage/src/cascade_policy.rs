// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Slot bookkeeping for the cascade's first tier (T1), the
//! write-back group cache of `CascadeBackend`: placement, LRU eviction and the
//! hit/miss split of a read. No I/O and no CUDA.
//!
//! Owner: metrale-storage KV tier.
//! Invariants:
//! - An occupied slot holds one `GroupKey`, recorded in both `lookup` and
//!   `slot_key`; eviction clears both before the slot is reused.

use std::collections::{HashMap, VecDeque};

use crate::group::GroupKey;

/// 2026-09-25: The slot to write a group into and, when a resident group was
/// evicted for it, that victim and its slot. The victim's bytes stay in the
/// slot until the caller overwrites it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WritePlan {
    pub slot: u32,
    pub flush_victim: Option<(GroupKey, u32)>,
}

/// 2026-09-25: A fixed-capacity LRU set of resident groups over `cap_slots`
/// slots.
pub struct SlotCache {
    cap_slots: u32,
    lookup: HashMap<GroupKey, u32>,
    slot_key: Vec<Option<GroupKey>>,
    free: VecDeque<u32>,
    /// 2026-09-25: Front is least recently used (the eviction target), back is
    /// most recently used.
    lru: VecDeque<u32>,
}

impl SlotCache {
    pub fn new(cap_slots: u32) -> Self {
        assert!(cap_slots > 0, "SlotCache needs at least one slot");
        Self {
            cap_slots,
            lookup: HashMap::new(),
            slot_key: vec![None; cap_slots as usize],
            free: (0..cap_slots).collect(),
            lru: VecDeque::with_capacity(cap_slots as usize),
        }
    }

    pub fn capacity(&self) -> u32 {
        self.cap_slots
    }

    fn move_to_back(&mut self, slot: u32) {
        if let Some(pos) = self.lru.iter().position(|&s| s == slot) {
            self.lru.remove(pos);
        }
        self.lru.push_back(slot);
    }

    /// 2026-09-25: Make `slot` the most recently used. Callers pass occupied
    /// slots, from `plan_read` hits.
    pub fn touch(&mut self, slot: u32) {
        self.move_to_back(slot);
    }

    /// 2026-09-25: Plan a write of `key`: its own slot if resident, else a free
    /// slot, else the least recently used slot, whose group is returned as the
    /// victim.
    pub fn plan_write(&mut self, key: GroupKey) -> WritePlan {
        if let Some(&slot) = self.lookup.get(&key) {
            self.move_to_back(slot);
            return WritePlan {
                slot,
                flush_victim: None,
            };
        }
        if let Some(slot) = self.free.pop_front() {
            self.install(key, slot);
            return WritePlan {
                slot,
                flush_victim: None,
            };
        }
        let victim_slot = self.lru.pop_front().expect("full cache has an LRU entry");
        let victim_key = self.slot_key[victim_slot as usize]
            .take()
            .expect("occupied slot has a key");
        self.lookup.remove(&victim_key);
        self.install(key, victim_slot);
        WritePlan {
            slot: victim_slot,
            flush_victim: Some((victim_key, victim_slot)),
        }
    }

    fn install(&mut self, key: GroupKey, slot: u32) {
        self.lookup.insert(key, slot);
        self.slot_key[slot as usize] = Some(key);
        self.lru.push_back(slot);
    }

    /// 2026-09-25: Split `keys` into hits `(index, slot)` and misses `index`,
    /// without changing LRU order; `CascadeBackend` touches each hit after
    /// queueing its copy.
    pub fn plan_read(&self, keys: &[GroupKey]) -> (Vec<(usize, u32)>, Vec<usize>) {
        let mut hits = Vec::new();
        let mut misses = Vec::new();
        for (i, k) in keys.iter().enumerate() {
            match self.lookup.get(k) {
                Some(&slot) => hits.push((i, slot)),
                None => misses.push(i),
            }
        }
        (hits, misses)
    }

    /// 2026-09-25: Every resident `(key, slot)`.
    pub fn residents(&self) -> Vec<(GroupKey, u32)> {
        self.lookup.iter().map(|(&k, &s)| (k, s)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::KvKind;

    fn k(block: u32) -> GroupKey {
        GroupKey::new(0, block, 0, KvKind::K)
    }

    #[test]
    fn free_slots_then_overwrite_in_place() {
        let mut c = SlotCache::new(3);
        let p0 = c.plan_write(k(0));
        assert_eq!(p0.flush_victim, None);
        let p1 = c.plan_write(k(1));
        assert_ne!(p1.slot, p0.slot);
        let p0b = c.plan_write(k(0));
        assert_eq!(p0b.slot, p0.slot);
        assert_eq!(p0b.flush_victim, None);
    }

    #[test]
    fn fill_then_evict_lru_tail() {
        let mut c = SlotCache::new(2);
        let s0 = c.plan_write(k(0)).slot;
        let _s1 = c.plan_write(k(1)).slot;
        let (hits, _) = c.plan_read(&[k(0)]);
        c.touch(hits[0].1);
        let p2 = c.plan_write(k(2));
        assert_eq!(p2.flush_victim.map(|(vk, _)| vk), Some(k(1)));
        assert_ne!(p2.slot, s0);
    }

    #[test]
    fn hit_miss_partition() {
        let mut c = SlotCache::new(4);
        c.plan_write(k(0));
        c.plan_write(k(2));
        let (hits, misses) = c.plan_read(&[k(0), k(1), k(2), k(3)]);
        let hit_idx: Vec<usize> = hits.iter().map(|(i, _)| *i).collect();
        assert_eq!(hit_idx, vec![0, 2]);
        assert_eq!(misses, vec![1, 3]);
    }

    #[test]
    fn residents_lists_all_live_groups() {
        let mut c = SlotCache::new(3);
        c.plan_write(k(0));
        c.plan_write(k(1));
        let mut r: Vec<GroupKey> = c.residents().into_iter().map(|(k, _)| k).collect();
        r.sort_by_key(|g| g.block);
        assert_eq!(r, vec![k(0), k(1)]);
    }

    #[test]
    fn evicted_group_is_no_longer_a_hit() {
        let mut c = SlotCache::new(1);
        c.plan_write(k(0));
        let p = c.plan_write(k(1));
        assert_eq!(p.flush_victim.map(|(vk, _)| vk), Some(k(0)));
        let (hits, misses) = c.plan_read(&[k(0), k(1)]);
        assert_eq!(hits.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![1]);
        assert_eq!(misses, vec![0]);
    }
}
