// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `CascadeBackend`: a write-back cache of groups in pinned host
//! memory (T1) in front of another `StorageBackend`. Writes land in T1; when
//! T1 is full, the least recently used group is written down to the backing.
//! Reads copy T1 hits to the device and forward misses to the backing. No
//! tier transforms bytes.
//!
//! Owner: metrale-storage KV tier.
//! Invariants:
//! - A read of a group resident in T1 is served from T1; misses are not
//!   copied into T1.
//! - A write waits for the event recorded after the last async hit copies
//!   before it overwrites a slot.

use std::ffi::c_void;

use anyhow::{Context, Result};

use crate::backend::{ReadRequest, StorageBackend};
use crate::cascade_policy::SlotCache;
use crate::cuda_min::{CudaEvent, PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, GroupLayout};

/// 2026-09-25: The T1 bytes: one pinned buffer of `cap_slots * group_bytes`,
/// slot `i` at `ptr + i * group_bytes`.
struct PinnedStore {
    buf: PinnedBuffer,
    group_bytes: usize,
}

impl PinnedStore {
    fn new(cap_slots: u32, group_bytes: usize) -> Result<Self> {
        let bytes = (cap_slots as usize)
            .checked_mul(group_bytes)
            .context("CascadeBackend T1 size overflow")?;
        Ok(Self {
            buf: PinnedBuffer::new(bytes).context("alloc T1 pinned store")?,
            group_bytes,
        })
    }
    #[inline]
    fn slot_host_ptr(&self, slot: u32) -> *const c_void {
        // 2026-09-25: SAFETY: `SlotCache` hands out slots below `cap_slots`, so
        // the offset is inside the buffer.
        unsafe {
            (self.buf.ptr as *const u8).add(slot as usize * self.group_bytes) as *const c_void
        }
    }
    /// 2026-09-25: Copy `slot`'s bytes into a new Vec, so the caller can pass
    /// them to `&mut backing` without borrowing the store.
    fn slot_bytes(&self, slot: u32) -> Vec<u8> {
        // 2026-09-25: SAFETY: as above; a slot spans `group_bytes` bytes.
        unsafe {
            std::slice::from_raw_parts(
                (self.buf.ptr as *const u8).add(slot as usize * self.group_bytes),
                self.group_bytes,
            )
            .to_vec()
        }
    }
    fn write_slot(&mut self, slot: u32, src: &[u8]) {
        debug_assert_eq!(src.len(), self.group_bytes);
        // 2026-09-25: SAFETY: the slot is in range. `src` must be `group_bytes`
        // long, which only the debug assertion above checks.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                (self.buf.ptr as *mut u8).add(slot as usize * self.group_bytes),
                self.group_bytes,
            );
        }
    }
}

pub struct CascadeBackend {
    hot: SlotCache,
    store: PinnedStore,
    backing: Box<dyn StorageBackend>,
    group_bytes: usize,
    /// 2026-09-25: The event `read_async` records after its hit copies. The
    /// next `write_from_host` waits on it before it overwrites a slot, so the
    /// host does not write a slot that a queued copy still reads.
    last_read_event: Option<CudaEvent>,
}

// 2026-09-25: SAFETY: the pinned store's pointer is used only by `&mut self`
// methods; the one `&self` method, `group_layout`, calls the backing, which the
// trait bound makes `Sync`.
unsafe impl Sync for CascadeBackend {}

impl CascadeBackend {
    pub fn new(
        backing: Box<dyn StorageBackend>,
        layout: GroupLayout,
        cap_slots: u32,
    ) -> Result<Self> {
        let group_bytes = layout.group_bytes() as usize;
        tracing::info!(
            "high-speed-swap: T1 cascade cache = {cap_slots} slots × {group_bytes} B = {:.1} GiB local pinned RAM, backing below",
            (cap_slots as f64 * group_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
        );
        Ok(Self {
            hot: SlotCache::new(cap_slots),
            store: PinnedStore::new(cap_slots, group_bytes)?,
            backing,
            group_bytes,
            last_read_event: None,
        })
    }

    /// 2026-09-25: Write every resident T1 group down to the backing.
    fn flush_all(&mut self) -> Result<()> {
        for (key, slot) in self.hot.residents() {
            let bytes = self.store.slot_bytes(slot);
            self.backing.write_from_host(key, &bytes)?;
        }
        Ok(())
    }

    /// 2026-09-25: Queue a copy of each T1 hit on `stream` and forward the
    /// misses to the backing. Sync: misses go through `backing.read`, then the
    /// stream is synced. Async: misses go through `backing.read_async`, an event
    /// is recorded after the hit copies, and the stream is not synced.
    fn read_common(&mut self, requests: &[ReadRequest], stream: u64, is_async: bool) -> Result<()> {
        let keys: Vec<GroupKey> = requests.iter().map(|r| r.group).collect();
        let (hits, misses) = self.hot.plan_read(&keys);
        for (i, slot) in &hits {
            let src = self.store.slot_host_ptr(*slot);
            copy_h_to_d_async(requests[*i].dst_dev_ptr, src, self.group_bytes, stream)?;
            self.hot.touch(*slot);
        }
        if is_async && !hits.is_empty() {
            let ev = CudaEvent::new()?;
            ev.record(stream)?;
            self.last_read_event = Some(ev);
        }
        // 2026-09-25: Misses are not copied into T1; only writes fill it. The
        // final sync also covers the hit copies.
        if !misses.is_empty() {
            let miss_reqs: Vec<ReadRequest> = misses.iter().map(|&i| requests[i]).collect();
            if is_async {
                self.backing.read_async(&miss_reqs, stream)?;
            } else {
                self.backing.read(&miss_reqs, stream)?;
            }
        }
        if !is_async {
            stream_sync(stream)?;
        }
        Ok(())
    }
}

impl StorageBackend for CascadeBackend {
    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        let plan = self.hot.plan_write(key);
        // 2026-09-25: The victim's bytes are still in the slot: write them down
        // first.
        if let Some((victim_key, victim_slot)) = plan.flush_victim {
            let victim_bytes = self.store.slot_bytes(victim_slot);
            self.backing
                .write_from_host(victim_key, &victim_bytes)
                .context("cascade: flush T1 victim to backing")?;
        }
        // 2026-09-25: An async hit copy may still read this slot.
        if let Some(ev) = self.last_read_event.take() {
            ev.sync()?;
        }
        self.store.write_slot(plan.slot, src);
        Ok(())
    }

    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        self.read_common(requests, stream, false)
    }

    fn read_async(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        self.read_common(requests, stream, true)
    }

    fn register_landing_region(&mut self, base: u64, len: usize) -> Result<()> {
        // 2026-09-25: Forwarded, so the backing's zero-copy reads of misses can
        // land in the region; T1 hits are plain copies.
        self.backing.register_landing_region(base, len)
    }

    fn group_layout(&self) -> GroupLayout {
        // 2026-09-25: The backing's geometry. The block methods keep the trait
        // defaults, which fan out through this backend's per-group methods.
        self.backing.group_layout()
    }
}

impl Drop for CascadeBackend {
    fn drop(&mut self) {
        // 2026-09-25: Write T1 down before the pinned store is freed; errors
        // are ignored.
        let _ = self.flush_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::PosixBackend;
    use crate::cuda_min::{CudaCtx, DeviceBuffer, copy_d_to_h_async, stream_sync};
    use crate::group::KvKind;
    use crate::layout::Layout;

    fn tempdir(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("metrale-cascade-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// 2026-09-25: With a 2-slot T1 over 4 groups: write groups 0 and 1,
    /// `read_async` both (two T1 hits), then write groups 2 and 3, which evict
    /// 0 and 1 from the slots the async copies read. The async results, and a
    /// sync read of all four groups (hits 2 and 3, misses 0 and 1 from the
    /// backing), must match what was written.
    #[test]
    #[ignore = "requires GPU"]
    fn read_async_eviction_no_corruption() {
        let _ctx = CudaCtx::new(0).expect("cuda init");
        let dir = tempdir("evict");
        let spec = GroupLayout::new(1, 4, 1, 16, 128, 2, 4096);
        let layout = Layout::create(&dir, spec).unwrap();
        let bytes = spec.group_bytes() as usize;
        let backing = Box::new(PosixBackend::new(layout).unwrap());
        let mut cascade = CascadeBackend::new(backing, spec, 2).unwrap();

        let keys: Vec<GroupKey> = (0..4u32)
            .map(|b| GroupKey::new(0, b, 0, KvKind::K))
            .collect();
        let pat = |b: usize| -> Vec<u8> {
            (0..bytes)
                .map(|i| ((i * 7 + b * 11) & 0xFF) as u8)
                .collect()
        };

        cascade.write_from_host(keys[0], &pat(0)).unwrap();
        cascade.write_from_host(keys[1], &pat(1)).unwrap();

        let d0 = DeviceBuffer::new(bytes).unwrap();
        let d1 = DeviceBuffer::new(bytes).unwrap();
        cascade
            .read_async(
                &[
                    ReadRequest {
                        group: keys[0],
                        dst_dev_ptr: d0.ptr,
                    },
                    ReadRequest {
                        group: keys[1],
                        dst_dev_ptr: d1.ptr,
                    },
                ],
                _ctx.stream,
            )
            .unwrap();
        cascade.write_from_host(keys[2], &pat(2)).unwrap();
        cascade.write_from_host(keys[3], &pat(3)).unwrap();
        stream_sync(_ctx.stream).unwrap();
        let readback = |d: &DeviceBuffer, want: &[u8]| {
            let mut got = vec![0u8; bytes];
            copy_d_to_h_async(got.as_mut_ptr() as *mut c_void, d.ptr, bytes, _ctx.stream).unwrap();
            stream_sync(_ctx.stream).unwrap();
            assert_eq!(&got, want, "cascade async restore corrupted");
        };
        readback(&d0, &pat(0));
        readback(&d1, &pat(1));

        let devs: Vec<DeviceBuffer> = (0..4).map(|_| DeviceBuffer::new(bytes).unwrap()).collect();
        let reqs: Vec<ReadRequest> = keys
            .iter()
            .zip(&devs)
            .map(|(k, d)| ReadRequest {
                group: *k,
                dst_dev_ptr: d.ptr,
            })
            .collect();
        cascade.read(&reqs, _ctx.stream).unwrap();
        for (b, d) in devs.iter().enumerate() {
            readback(d, &pat(b));
        }
        drop(cascade);
        std::fs::remove_dir_all(&dir).ok();
    }
}
