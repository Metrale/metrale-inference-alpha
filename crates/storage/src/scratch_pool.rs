// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Device scratch pool of the high-speed-swap tier: slots that hold one
//! block's K and V for every KV head, filled from the storage backend.
//!
//! Owner: storage, high-speed swap.
//! Invariants:
//! - `lookup[k] == s` exactly when `residents[s] == Some(k)`.
//! - A slot is in `free_list` only while `residents` holds `None` for it.
//!
//! Slot `S` is `slot_bytes = 2 * num_kv_heads * group_stride` bytes. The K stripe of
//! head `h` starts at `S * slot_bytes + h * group_stride` and its V stripe at
//! `S * slot_bytes + (num_kv_heads + h) * group_stride`. That is the layout
//! `TiledAttention::scratch_pool_strides` describes when `group_stride` is exactly
//! `block_size * head_dim * 2` bytes, with no filesystem padding.

use anyhow::{Result, bail};
use std::collections::{HashMap, VecDeque};

use crate::cuda_min::DeviceBuffer;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResidentKey {
    pub layer: u32,
    pub block: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ScratchDims {
    pub num_slots: u32,
    pub num_kv_heads: u16,
    // 2026-09-25: Bytes per (block, kv_head) stripe.
    pub group_stride: u64,
}

impl ScratchDims {
    pub fn slot_bytes(&self) -> usize {
        (2 * self.num_kv_heads as u64 * self.group_stride) as usize
    }
    pub fn pool_bytes(&self) -> usize {
        self.num_slots as usize * self.slot_bytes()
    }
}

pub struct ScratchPool {
    dims: ScratchDims,
    pool: DeviceBuffer,
    residents: Vec<Option<ResidentKey>>,
    lookup: HashMap<ResidentKey, u32>,
    free_list: VecDeque<u32>,
}

impl ScratchPool {
    pub fn new(dims: ScratchDims) -> Result<Self> {
        if dims.num_slots == 0 {
            bail!("ScratchPool requires at least one slot");
        }
        let pool = DeviceBuffer::new(dims.pool_bytes())?;
        let residents = vec![None; dims.num_slots as usize];
        let free_list = (0..dims.num_slots).collect();
        Ok(Self {
            dims,
            pool,
            residents,
            lookup: HashMap::new(),
            free_list,
        })
    }

    pub fn dims(&self) -> ScratchDims {
        self.dims
    }
    pub fn pool_dev_ptr(&self) -> u64 {
        self.pool.ptr
    }
    pub fn slot_dev_ptr(&self, slot: u32) -> u64 {
        self.pool.ptr + (slot as u64) * (self.dims.slot_bytes() as u64)
    }
    pub fn slot_k_ptr(&self, slot: u32, kv_head: u16) -> u64 {
        self.slot_dev_ptr(slot) + (kv_head as u64) * self.dims.group_stride
    }
    pub fn slot_v_ptr(&self, slot: u32, kv_head: u16) -> u64 {
        self.slot_dev_ptr(slot)
            + (self.dims.num_kv_heads as u64 + kv_head as u64) * self.dims.group_stride
    }

    pub fn lookup(&self, key: ResidentKey) -> Option<u32> {
        self.lookup.get(&key).copied()
    }

    /// 2026-09-25: Free `key`'s slot, if any, so the next `assign` of `key` needs a
    /// fresh read. The offload path calls it after rewriting a block on disk, so
    /// attention does not read the old resident copy.
    pub fn invalidate(&mut self, key: ResidentKey) {
        if let Some(slot) = self.lookup.remove(&key) {
            self.residents[slot as usize] = None;
            self.free_list.push_back(slot);
        }
    }

    pub fn capacity(&self) -> u32 {
        self.dims.num_slots
    }
    pub fn free_count(&self) -> u32 {
        self.free_list.len() as u32
    }

    /// 2026-09-25: Slot for `key`: its current slot if resident, else the first free
    /// slot, else the first resident slot in `evict_candidates`, whose key is evicted.
    /// With no free slot and no resident candidate, an error. The caller fills the
    /// slot.
    pub fn assign(&mut self, key: ResidentKey, evict_candidates: &[u32]) -> Result<u32> {
        if let Some(&slot) = self.lookup.get(&key) {
            return Ok(slot);
        }
        let slot = match self.free_list.pop_front() {
            Some(s) => s,
            None => {
                // 2026-09-25: Pinned slots are the caller's to leave out of
                // `evict_candidates`.
                let mut chosen = None;
                for &c in evict_candidates {
                    if self
                        .residents
                        .get(c as usize)
                        .and_then(|r| r.as_ref())
                        .is_some()
                    {
                        chosen = Some(c);
                        break;
                    }
                }
                let s = chosen.ok_or_else(|| {
                    anyhow::anyhow!("no slot available and no eviction candidate is resident")
                })?;
                if let Some(prev) = self.residents[s as usize].take() {
                    self.lookup.remove(&prev);
                }
                s
            }
        };
        self.residents[slot as usize] = Some(key);
        self.lookup.insert(key, slot);
        Ok(slot)
    }

    /// 2026-09-25: Every resident slot with its key, in slot order.
    pub fn residents(&self) -> Vec<(u32, ResidentKey)> {
        self.residents
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|k| (i as u32, k)))
            .collect()
    }

    pub fn clear(&mut self) {
        self.lookup.clear();
        for r in self.residents.iter_mut() {
            *r = None;
        }
        self.free_list.clear();
        for s in 0..self.dims.num_slots {
            self.free_list.push_back(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> crate::cuda_min::CudaCtx {
        crate::cuda_min::CudaCtx::new(0).expect("cuda init")
    }

    #[test]
    #[ignore = "requires GPU"]
    fn assign_and_lookup() {
        let _ctx = ctx();
        let mut pool = ScratchPool::new(ScratchDims {
            num_slots: 4,
            num_kv_heads: 2,
            group_stride: 4096,
        })
        .unwrap();
        let k0 = ResidentKey { layer: 0, block: 7 };
        let s0 = pool.assign(k0, &[]).unwrap();
        assert_eq!(pool.lookup(k0), Some(s0));
        let s0_again = pool.assign(k0, &[]).unwrap();
        assert_eq!(s0, s0_again);
        for b in 8..11 {
            pool.assign(ResidentKey { layer: 0, block: b }, &[])
                .unwrap();
        }
        assert_eq!(pool.free_count(), 0);
        let evicted = pool
            .assign(
                ResidentKey {
                    layer: 0,
                    block: 99,
                },
                &[s0],
            )
            .unwrap();
        assert_eq!(evicted, s0);
        assert_eq!(pool.lookup(k0), None);
    }

    #[test]
    #[ignore = "requires GPU"]
    fn slot_pointer_layout() {
        let _ctx = ctx();
        let pool = ScratchPool::new(ScratchDims {
            num_slots: 2,
            num_kv_heads: 4,
            group_stride: 4096,
        })
        .unwrap();
        let base = pool.pool_dev_ptr();
        assert_eq!(pool.slot_dev_ptr(0), base);
        assert_eq!(pool.slot_dev_ptr(1), base + 8 * 4096);
        assert_eq!(pool.slot_k_ptr(0, 2), base + 2 * 4096);
        assert_eq!(pool.slot_v_ptr(0, 2), base + (4 + 2) * 4096);
    }
}
