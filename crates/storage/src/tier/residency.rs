// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`Residency`], the page table over a bounded hot [`SlotArena`]
//! and a cold [`SwapStore`].
//!
//! Owner: storage (tier).
//! Invariants, between calls:
//! - A key is in `lru` if and only if it is `Resident` and holds no read-pin.
//! - Every `OnDisk` key is in `disk_lru` exactly once.

use std::collections::{HashMap, VecDeque};

use anyhow::{Result, bail};

use crate::tier::aligned::PageAlignedBuf;
use crate::tier::traits::{SlotArena, SwapStats, SwapStore};

/// 2026-09-25: Where a key's blob is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Loc {
    /// 2026-09-25: An arena slot handed out by `alloc` and not yet committed.
    /// It is not in `lru`, so it is never an eviction victim.
    Reserved(usize),
    /// 2026-09-25: Committed in an arena slot.
    Resident(usize),
    /// 2026-09-25: Spilled to a store record; `locate` faults it back into a
    /// slot.
    OnDisk(usize),
}

/// 2026-09-25: The page table from a `u64` key to a [`SlotArena`] slot or a
/// [`SwapStore`] record, evicting the least recently used resident to the store
/// when the arena is full.
pub struct Residency<A: SlotArena, S: SwapStore> {
    arena: A,
    swap: S,
    blob_bytes: usize,
    map: HashMap<u64, Loc>,
    /// 2026-09-25: Free arena slots, reused last-in first-out.
    free_slots: Vec<usize>,
    /// 2026-09-25: Unpinned resident keys, coldest first: the front is the next
    /// eviction victim.
    lru: VecDeque<u64>,
    /// 2026-09-25: On-disk keys, coldest first: the front is the next record the
    /// disk cap drops.
    disk_lru: VecDeque<u64>,
    /// 2026-09-25: The on-disk record cap, 0 for none. At the cap a spill first
    /// drops the coldest on-disk key from the map, so a later get of it misses.
    max_disk_slots: usize,
    /// 2026-09-25: Free record indices, reused before `next_disk` grows.
    free_disk: Vec<usize>,
    next_disk: usize,
    /// 2026-09-25: The buffer every spill and fault-in moves a blob through,
    /// `blob_bytes` long and 4 KiB-aligned: the unix
    /// [`DirectSwapFile`](crate::tier::DirectSwapFile) copies a buffer that is
    /// not page-aligned through its own bounce buffer first.
    scratch: PageAlignedBuf,
    /// 2026-09-25: Read-pins, key to reader count. A peer hands a client the
    /// slot offset for a one-sided RDMA read that happens after the lock is
    /// released; a pinned key stays out of `lru`, so `evict_coldest_to_disk`
    /// cannot reuse its slot during that read.
    read_pins: HashMap<u64, u32>,
    stats: SwapStats,
}

impl<A: SlotArena, S: SwapStore> Residency<A, S> {
    /// 2026-09-25: [`Residency::new_capped`] with no disk cap.
    pub fn new(arena: A, swap: S) -> Result<Self> {
        Self::new_capped(arena, swap, 0)
    }

    /// 2026-09-25: A residency over `arena` and `swap` with at most
    /// `max_disk_slots` live on-disk records (0: no cap). The file can still
    /// reach `max_disk_slots + 1` records (see [`Residency::disk_high_water`]).
    ///
    /// # Errors
    /// `slot_bytes()` is 0, differs from `swap.record_bytes()`, or the arena has
    /// no slot.
    pub fn new_capped(arena: A, swap: S, max_disk_slots: usize) -> Result<Self> {
        let blob_bytes = arena.slot_bytes();
        if blob_bytes == 0 {
            bail!("Residency: slot_bytes must be > 0");
        }
        if swap.record_bytes() != blob_bytes {
            bail!(
                "Residency: arena slot ({}) and swap record ({}) sizes differ",
                blob_bytes,
                swap.record_bytes()
            );
        }
        let n = arena.num_slots();
        if n == 0 {
            bail!("Residency: arena must have >= 1 slot");
        }
        Ok(Self {
            arena,
            swap,
            blob_bytes,
            map: HashMap::new(),
            free_slots: (0..n).rev().collect(),
            lru: VecDeque::new(),
            disk_lru: VecDeque::new(),
            max_disk_slots,
            free_disk: Vec::new(),
            next_disk: 0,
            scratch: PageAlignedBuf::new(blob_bytes),
            read_pins: HashMap::new(),
            stats: SwapStats::default(),
        })
    }

    pub fn blob_bytes(&self) -> usize {
        self.blob_bytes
    }
    /// 2026-09-25: Address of the scratch buffer, for the alignment test.
    #[doc(hidden)]
    pub fn scratch_addr(&self) -> usize {
        self.scratch.as_slice().as_ptr() as usize
    }
    pub fn stats(&self) -> &SwapStats {
        &self.stats
    }
    pub fn resident_count(&self) -> usize {
        self.lru.len()
    }
    pub fn total_keys(&self) -> usize {
        self.map.len()
    }
    /// 2026-09-25: Keys on disk now; at most `max_disk_slots` when capped.
    pub fn disk_count(&self) -> usize {
        self.disk_lru.len()
    }
    /// 2026-09-25: The highest record index ever allocated, plus one: the swap
    /// file's size in records. [`DirectSwapFile`](crate::tier::DirectSwapFile)
    /// keeps the default no-op `discard_record`, so its file never shrinks and
    /// only index reuse through `free_disk` bounds it. Under a cap this reaches
    /// `max_disk_slots + 1`: [`Residency::locate`]'s `OnDisk` arm takes the
    /// faulting key out of `disk_lru` while its record is still live, so
    /// `make_disk_room` counts one fewer record than exist.
    pub fn disk_high_water(&self) -> usize {
        self.next_disk
    }
    /// 2026-09-25: The on-disk record cap, 0 for none.
    pub fn max_disk_slots(&self) -> usize {
        self.max_disk_slots
    }

    /// 2026-09-25: The arena itself, for callers that move the bytes of a slot
    /// they got from [`Residency::alloc`] or [`Residency::locate`];
    /// [`Residency::put_blob`] and [`Residency::get_blob`] do both steps.
    pub fn arena(&self) -> &A {
        &self.arena
    }
    pub fn arena_mut(&mut self) -> &mut A {
        &mut self.arena
    }

    /// 2026-09-25: Byte offset of `slot` in the arena: `slot * blob_bytes`.
    pub fn slot_offset(&self, slot: usize) -> u64 {
        (slot as u64) * (self.blob_bytes as u64)
    }

    /// 2026-09-25: Put, step 1: reserve an arena slot for `key`, spilling the
    /// coldest unpinned resident to the store when no slot is free. The caller
    /// writes the blob into the slot and then calls [`Residency::commit`]. A
    /// resident or reserved key keeps its slot; an on-disk key gets a new slot
    /// and its record is freed.
    ///
    /// # Errors
    /// Every slot is reserved or read-pinned, or the spill's arena read or
    /// store write fails. On error the key keeps its previous location.
    pub fn alloc(&mut self, key: u64) -> Result<usize> {
        self.stats.puts += 1;
        match self.map.get(&key).copied() {
            Some(Loc::Resident(slot)) => {
                self.lru_remove(key);
                self.map.insert(key, Loc::Reserved(slot));
                return Ok(slot);
            }
            Some(Loc::Reserved(slot)) => return Ok(slot),
            Some(Loc::OnDisk(disk_slot)) => {
                // 2026-09-25: The slot is acquired before the key's record is
                // freed. `acquire_slot` can fail in its spill (a store write
                // error such as ENOSPC); had the record already been freed, it
                // would sit on `free_disk` while `map[key]` still says
                // `OnDisk(disk_slot)`, and a retried put or a `remove` would free
                // it a second time, handing one record to two keys.
                //
                // The key also leaves `disk_lru` first, so `make_disk_room` in
                // that spill cannot pick it and free its record the same way.
                self.disk_lru_remove(key);
                let slot = match self.acquire_slot() {
                    Ok(s) => s,
                    Err(e) => {
                        // 2026-09-25: The key is still `OnDisk` with its own
                        // record; put it back at the cold end.
                        self.disk_lru.push_front(key);
                        return Err(e);
                    }
                };
                self.free_disk.push(disk_slot);
                self.swap.discard_record(disk_slot);
                self.map.insert(key, Loc::Reserved(slot));
                return Ok(slot);
            }
            None => {}
        }
        let slot = self.acquire_slot()?;
        self.map.insert(key, Loc::Reserved(slot));
        Ok(slot)
    }

    /// 2026-09-25: Put, step 2: the blob is in the reserved slot; mark `key`
    /// resident and, unless read-pinned, the hottest in `lru`. Committing a
    /// resident key does nothing.
    ///
    /// # Errors
    /// `key` is neither reserved nor resident.
    pub fn commit(&mut self, key: u64) -> Result<()> {
        match self.map.get(&key).copied() {
            Some(Loc::Reserved(slot)) => {
                self.map.insert(key, Loc::Resident(slot));
                // 2026-09-25: A key pinned while its re-put was in flight stays
                // out of `lru`; `unpin_read` adds it back.
                if !self.read_pins.contains_key(&key) {
                    self.lru.push_back(key);
                }
                Ok(())
            }
            Some(Loc::Resident(_)) => Ok(()),
            _ => bail!("commit({key:#x}): no reserved slot (alloc not called / evicted)"),
        }
    }

    /// 2026-09-25: Get: the arena slot holding `key`, faulting an on-disk key
    /// back into a slot (which may spill another key). `Ok(None)` for an unknown
    /// key. A reserved key returns its slot, whose bytes the caller has not
    /// committed.
    ///
    /// # Errors
    /// No slot can be acquired, or the store read or arena write fails; the key
    /// then stays on disk and the acquired slot is freed.
    pub fn locate(&mut self, key: u64) -> Result<Option<usize>> {
        self.stats.gets += 1;
        match self.map.get(&key).copied() {
            Some(Loc::Resident(slot)) => {
                self.stats.resident_hits += 1;
                // 2026-09-25: A pinned key is not in `lru`, and touching it would
                // put it back there while it is being read.
                if !self.read_pins.contains_key(&key) {
                    self.lru_touch(key);
                }
                Ok(Some(slot))
            }
            Some(Loc::Reserved(slot)) => {
                // 2026-09-25: Not yet committed; the slot is returned as it is.
                Ok(Some(slot))
            }
            Some(Loc::OnDisk(disk_slot)) => {
                // 2026-09-25: Out of `disk_lru` while the fault runs, so the
                // `make_disk_room` in `acquire_slot`'s spill cannot drop it.
                self.disk_lru_remove(key);
                let slot = match self.acquire_slot() {
                    Ok(s) => s,
                    Err(e) => {
                        self.disk_lru.push_front(key);
                        return Err(e);
                    }
                };
                // 2026-09-25: `&mut self` makes this the only move using
                // `scratch`.
                let mut buf = std::mem::take(&mut self.scratch);
                let r = self
                    .swap
                    .read_record(disk_slot, buf.as_mut_slice())
                    .and_then(|_| self.arena.write_slot(slot, buf.as_slice()));
                // 2026-09-25: `scratch` goes back before any return: while it is
                // taken, `self.scratch` is the empty default, and a later move
                // through it would fail its length check.
                self.scratch = buf;
                if let Err(e) = r {
                    self.free_slots.push(slot);
                    self.disk_lru.push_front(key);
                    return Err(e);
                }
                self.free_disk.push(disk_slot);
                self.swap.discard_record(disk_slot);
                self.map.insert(key, Loc::Resident(slot));
                self.lru.push_back(key);
                self.stats.faults_from_disk += 1;
                Ok(Some(slot))
            }
            None => {
                self.stats.get_miss += 1;
                Ok(None)
            }
        }
    }

    /// 2026-09-25: Forget `key`, freeing its arena slot or store record. A read
    /// pin on it stays until `unpin_read`.
    pub fn remove(&mut self, key: u64) {
        match self.map.remove(&key) {
            Some(Loc::Resident(slot)) | Some(Loc::Reserved(slot)) => {
                self.lru_remove(key);
                self.free_slots.push(slot);
            }
            Some(Loc::OnDisk(disk_slot)) => {
                self.disk_lru_remove(key);
                self.free_disk.push(disk_slot);
                self.swap.discard_record(disk_slot);
            }
            None => {}
        }
    }

    /// 2026-09-25: Read-pin `key` so its slot is not an eviction victim while a
    /// client reads it. Counted per reader; the first pin takes the key out of
    /// `lru`. Does nothing unless `key` is resident.
    pub fn pin_read(&mut self, key: u64) {
        if !matches!(self.map.get(&key), Some(Loc::Resident(_))) {
            return;
        }
        let n = self.read_pins.get(&key).copied().unwrap_or(0);
        if n == 0 {
            self.lru_remove(key);
        }
        self.read_pins.insert(key, n + 1);
    }

    /// 2026-09-25: Release one pin taken by [`Residency::pin_read`]. After the
    /// last one, a key that is still resident rejoins `lru` as the hottest. Does
    /// nothing for a key with no pin.
    pub fn unpin_read(&mut self, key: u64) {
        let Some(n) = self.read_pins.get_mut(&key) else {
            return;
        };
        *n -= 1;
        if *n == 0 {
            self.read_pins.remove(&key);
            if matches!(self.map.get(&key), Some(Loc::Resident(_))) && !self.lru.contains(&key) {
                self.lru.push_back(key);
            }
        }
    }

    /// 2026-09-25: The number of read-pins on `key`.
    pub fn read_pin_count(&self, key: u64) -> u32 {
        self.read_pins.get(&key).copied().unwrap_or(0)
    }

    /// 2026-09-25: Put in one call: [`Residency::alloc`], copy `bytes` into the
    /// slot, [`Residency::commit`].
    ///
    /// # Errors
    /// `bytes` is not `blob_bytes` long, `alloc` fails, or the arena write fails,
    /// after which `key` is removed.
    pub fn put_blob(&mut self, key: u64, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.blob_bytes {
            bail!(
                "put_blob({key:#x}): {} bytes, expected {}",
                bytes.len(),
                self.blob_bytes
            );
        }
        let slot = self.alloc(key)?;
        if let Err(e) = self.arena.write_slot(slot, bytes) {
            self.remove(key);
            return Err(e);
        }
        self.commit(key)
    }

    /// 2026-09-25: Get in one call: [`Residency::locate`], then copy the slot
    /// into `out`. `Ok(false)` for an unknown key.
    ///
    /// # Errors
    /// `out` is not `blob_bytes` long, or `locate` or the arena read fails.
    pub fn get_blob(&mut self, key: u64, out: &mut [u8]) -> Result<bool> {
        if out.len() != self.blob_bytes {
            bail!(
                "get_blob({key:#x}): {} bytes, expected {}",
                out.len(),
                self.blob_bytes
            );
        }
        match self.locate(key)? {
            Some(slot) => {
                self.arena.read_slot(slot, out)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// 2026-09-25: A free arena slot, or the slot of the coldest unpinned
    /// resident after spilling it.
    fn acquire_slot(&mut self) -> Result<usize> {
        if let Some(s) = self.free_slots.pop() {
            return Ok(s);
        }
        self.evict_coldest_to_disk()
    }

    /// 2026-09-25: Spill the front of `lru` to a store record and return its
    /// slot. Reserved and read-pinned keys are not in `lru`. On a failed store
    /// write the victim stays resident at the cold end.
    fn evict_coldest_to_disk(&mut self) -> Result<usize> {
        let Some(victim) = self.lru.pop_front() else {
            bail!(
                "Residency: arena exhausted — all {} slots reserved (uncommitted \
                 PUTs) or read-pinned (in-flight RDMA READs)",
                self.arena.num_slots()
            );
        };
        let slot = match self.map.get(&victim).copied() {
            Some(Loc::Resident(slot)) => slot,
            other => bail!("LRU/map desync: victim {victim:#x} is {other:?}, expected Resident"),
        };
        self.make_disk_room();
        let disk_slot = self.alloc_disk_slot();
        let mut buf = std::mem::take(&mut self.scratch);
        let res = self
            .arena
            .read_slot(slot, buf.as_mut_slice())
            .and_then(|_| self.swap.write_record(disk_slot, buf.as_slice()));
        self.scratch = buf;
        if let Err(e) = res {
            self.free_disk.push(disk_slot);
            self.lru.push_front(victim);
            return Err(e);
        }
        self.map.insert(victim, Loc::OnDisk(disk_slot));
        self.disk_lru.push_back(victim);
        self.stats.spills_to_disk += 1;
        Ok(slot)
    }

    /// 2026-09-25: Drop the coldest on-disk keys until `disk_lru` holds fewer
    /// than `max_disk_slots` (nothing without a cap). A dropped key leaves the
    /// map, so a later get of it misses.
    fn make_disk_room(&mut self) {
        if self.max_disk_slots == 0 {
            return;
        }
        while self.disk_lru.len() >= self.max_disk_slots {
            let Some(cold) = self.disk_lru.pop_front() else {
                break;
            };
            if let Some(Loc::OnDisk(ds)) = self.map.remove(&cold) {
                self.free_disk.push(ds);
                self.swap.discard_record(ds);
                self.stats.disk_evictions += 1;
            }
        }
    }

    fn alloc_disk_slot(&mut self) -> usize {
        if let Some(d) = self.free_disk.pop() {
            d
        } else {
            let d = self.next_disk;
            self.next_disk += 1;
            d
        }
    }

    fn disk_lru_remove(&mut self, key: u64) {
        if let Some(pos) = self.disk_lru.iter().position(|&k| k == key) {
            self.disk_lru.remove(pos);
        }
    }

    fn lru_touch(&mut self, key: u64) {
        self.lru_remove(key);
        self.lru.push_back(key);
    }

    fn lru_remove(&mut self, key: u64) {
        if let Some(pos) = self.lru.iter().position(|&k| k == key) {
            self.lru.remove(pos);
        }
    }
}
