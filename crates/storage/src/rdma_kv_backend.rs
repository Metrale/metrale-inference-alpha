// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `RdmaKvBackend`, the KV `StorageBackend` whose store is a `cache_peer`
//! arena reached over one-sided RDMA.
//!
//! Owner: storage, RDMA KV tier.
//! Invariants:
//! - Group `g` lives at `remote_base + group_id(g) * group_stride` in the peer arena,
//!   which the peer allocates for this connection alone.
//! - `read` and `read_async` drain every rail's pending writes before posting a
//!   read, so a restore sees earlier offloads.
//!
//! `write_from_host` copies a group into a pinned bounce and posts an RDMA WRITE; it
//! is reaped lazily. `read` posts RDMA READs into pinned bounces and copies each to
//! its destination with `copy_h2d`, or with zero-copy on, READs straight into the
//! destination.
//!
//! Each rail is one QP with its own ring of `depth` bounces, so up to `depth` ops are
//! in flight per rail (`METRALE_KV_PIPELINE_DEPTH`, default 16, clamped to 1..=128).
//! `METRALE_KV_DUAL_RAIL=1` opens a second rail; reads are striped across the rails
//! and writes alternate between them. The peer registers the one arena on every rail.

use std::collections::HashMap;
use std::ffi::c_void;
use std::io::Write;
use std::net::TcpStream;

use anyhow::{Context, Result, bail};

use crate::backend::{ReadRequest, StorageBackend};
use crate::cuda_min::{PinnedBuffer, stream_sync};
use crate::group::{GroupKey, GroupLayout};

mod rail;

use rail::{Bounce, Rail};

pub struct RdmaKvBackend {
    rails: Vec<Rail>,
    layout: GroupLayout,
    remote_base: u64,
    // 2026-09-25: Round-robin rail cursor for writes.
    rr: usize,
    /// 2026-09-25: Zero-copy restore (`METRALE_KV_ZERO_COPY=1`): the RDMA READ lands
    /// in the destination itself, with no bounce and no `copy_h2d`. Cleared for good
    /// the first time a destination cannot be registered.
    zero_copy: bool,
    _stream: TcpStream,
}

// 2026-09-25: SAFETY: every method that touches a QP or a bounce takes `&mut self`;
// the `&self` methods read only `layout` and `remote_base`.
unsafe impl Sync for RdmaKvBackend {}

impl RdmaKvBackend {
    /// 2026-09-25: Connect to the peer at `addr`, request an arena that holds every
    /// group of `layout`, bring up the rails with `metrale_gpu_sys::railset::RailSet`,
    /// and register each rail's bounce ring.
    pub fn connect(addr: &str, layout: GroupLayout) -> Result<Self> {
        use metrale_gpu_sys::env::{first_set, first_set_u32};
        use metrale_gpu_sys::railset::{RailSet, RailSpec};

        let group_bytes = layout.group_bytes() as usize;
        let num_groups = (layout.num_layers as u64)
            * 2
            * (layout.num_blocks as u64)
            * (layout.num_kv_heads as u64);
        let total_bytes = num_groups * layout.group_stride;

        // 2026-09-25: Rail 0 from `METRALE_EXPERT_RDMA_DEV`/`GID`, rail 1 (only with
        // `METRALE_KV_DUAL_RAIL=1`) from `METRALE_KV_RAIL2_DEV`/`GID`; a random 24-bit
        // PSN per rail.
        let spec =
            |dev: String, gid: u32| RailSpec::new(dev, gid, rand::random::<u32>() & 0xff_ffff);
        let rail0 = spec(
            first_set(&["METRALE_EXPERT_RDMA_DEV"], "roceP2p1s0f1"),
            first_set_u32(&["METRALE_EXPERT_RDMA_GID"], 3),
        );
        let dual = std::env::var("METRALE_KV_DUAL_RAIL").ok().as_deref() == Some("1");
        let specs: Vec<RailSpec> = if dual {
            let rail1 = spec(
                first_set(&["METRALE_KV_RAIL2_DEV"], "rocep1s0f1"),
                first_set_u32(&["METRALE_KV_RAIL2_GID"], 3),
            );
            vec![rail0, rail1]
        } else {
            vec![rail0]
        };
        let n_rails = specs.len();
        let depth: usize = first_set_u32(&["METRALE_KV_PIPELINE_DEPTH"], 16).clamp(1, 128) as usize;

        let mut stream =
            TcpStream::connect(addr).with_context(|| format!("connect kv peer {addr}"))?;
        stream.set_nodelay(true).ok();
        // 2026-09-25: `blob_bytes == 0` selects the peer's raw mode: a private arena
        // for this connection, with placement decided by the client.
        stream
            .write_all(&crate::snapshot_swap::encode_paging_v2_header(
                crate::snapshot_swap::PagingKind::KV,
                total_bytes,
                0,
            ))
            .context("send kv raw-mode v2 header")?;

        // 2026-09-25: Bounce MRs are registered LOCAL_WRITE only
        // (`remote_read == false`).
        let mut rs = RailSet::begin(&mut stream, &specs)?;
        let mut rings: Vec<Vec<Bounce>> = Vec::with_capacity(n_rails);
        for rail in &mut rs.rails {
            let mut bounces = Vec::with_capacity(depth);
            for _ in 0..depth {
                let buf = PinnedBuffer::new(group_bytes).context("alloc pinned kv bounce")?;
                // 2026-09-25: SAFETY: buf lives as long as the rail, and so the MR.
                let keys = unsafe { rail.verbs.reg_mr(buf.ptr, group_bytes, false)? };
                bounces.push(Bounce {
                    buf,
                    lkey: keys.lkey,
                    copy_done: None,
                });
            }
            rings.push(bounces);
        }

        let server = rs.finish_rw(&mut stream, "kv peer")?;
        // 2026-09-25: Every rail publishes the same arena base; the last is kept.
        let base = server.last().map(|sp| sp.base_addr).unwrap_or(0);
        let rails: Vec<Rail> = rs
            .into_verbs()
            .into_iter()
            .zip(rings)
            .zip(&server)
            .map(|((verbs, bounces), sp)| Rail {
                verbs,
                remote_rkey: sp.rkey,
                free: (0..depth).collect(),
                bounces,
                inflight: HashMap::new(),
                next_wr: 0,
                dst_lkeys: HashMap::new(),
                region: None,
                direct_inflight: 0,
            })
            .collect();
        tracing::info!(
            "RdmaKvBackend connected to {addr}: {:.1} GiB blade, {n_rails} rail(s), \
             group_stride {}, pipeline depth {depth}",
            total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            layout.group_stride,
        );
        Ok(Self {
            rails,
            layout,
            remote_base: base,
            rr: 0,
            zero_copy: std::env::var("METRALE_KV_ZERO_COPY").ok().as_deref() == Some("1"),
            _stream: stream,
        })
    }

    #[inline]
    fn remote_addr(&self, key: GroupKey) -> u64 {
        self.remote_base + self.layout.group_id(key).0 * self.layout.group_stride
    }

    /// 2026-09-25: Zero-copy restore: RDMA READ each group straight into its
    /// destination, registered as the landing MR, with no bounce and no `copy_h2d`.
    /// The destination must be registerable (`Rail::reg_dst`); the caller has
    /// registered every one.
    fn read_zero_copy(
        &mut self,
        requests: &[ReadRequest],
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: The NIC writes the destinations outside `stream`, and kernels
        // already queued on `stream` may still read them, so wait for `stream` first.
        // The bounce path needs no such wait: its `copy_h2d` is itself on `stream`.
        // Each completion below is polled before return, so the data has landed
        // before any later kernel is queued.
        stream_sync(stream)?;
        let n = self.rails.len();
        let depth = self.rails[0].bounces.len();
        let mut pend: Vec<std::collections::VecDeque<usize>> = vec![Default::default(); n];
        for (j, _) in requests.iter().enumerate() {
            pend[j % n].push_back(j);
        }
        loop {
            let mut active = false;
            for (ri, rail) in self.rails.iter_mut().enumerate() {
                while rail.direct_inflight < depth {
                    let Some(j) = pend[ri].pop_front() else { break };
                    let dst = requests[j].dst_dev_ptr;
                    let lkey = rail.reg_dst(dst, bytes)?;
                    let raddr = self.remote_base
                        + self.layout.group_id(requests[j].group).0 * self.layout.group_stride;
                    let wr = rail.fresh_wr();
                    // 2026-09-25: SAFETY: dst is a registered MR (lkey) of `bytes`, and
                    // raddr/rkey address the peer arena.
                    unsafe {
                        rail.verbs.post_read(
                            dst as *mut c_void,
                            lkey,
                            raddr,
                            rail.remote_rkey,
                            bytes as u32,
                            wr,
                        )?;
                    }
                    rail.direct_inflight += 1;
                }
                if rail.direct_inflight > 0 {
                    rail.verbs.poll()?;
                    rail.direct_inflight -= 1;
                    active = true;
                }
            }
            if !active && pend.iter().all(|q| q.is_empty()) {
                break;
            }
        }
        Ok(())
    }
}

impl RdmaKvBackend {
    /// 2026-09-25: Body of `read` (`is_async == false`) and `read_async` (`true`).
    /// On the bounce path:
    ///   * sync: reuses the most recently freed bounce, records no copy events, and
    ///     ends with `stream_sync`;
    ///   * async: reuses the oldest freed bounce, whose copy has had the longest to
    ///     finish, records a copy event per reaped READ (`Rail::reap_one`), and
    ///     returns without a host sync.
    ///
    /// Both take the zero-copy branch when it is on, which starts with its own
    /// `stream_sync`.
    fn read_common(&mut self, requests: &[ReadRequest], stream: u64, is_async: bool) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        // 2026-09-25: Pending offloads land first, so a restore sees them.
        for rail in &mut self.rails {
            rail.drain(bytes, stream)?;
        }
        if self.zero_copy {
            if requests.is_empty() {
                return Ok(());
            }
            // 2026-09-25: Register every destination on every rail before posting
            // any READ, so a registration failure falls back to the bounce path with
            // nothing half-posted. Each rail is its own device, so each is probed.
            // `reg_dst` caches, so `read_zero_copy` reuses these lkeys.
            let mut all_ok = true;
            'reg: for req in requests {
                for rail in &mut self.rails {
                    if let Err(e) = rail.reg_dst(req.dst_dev_ptr, bytes) {
                        tracing::warn!(
                            "kv restore dst not UMA-registerable ({e:#}); \
                             permanently using bounce restore"
                        );
                        all_ok = false;
                        break 'reg;
                    }
                }
            }
            if all_ok {
                return self.read_zero_copy(requests, bytes, stream);
            }
            // 2026-09-25: A destination failed to register: this and every later read
            // takes the bounce path.
            self.zero_copy = false;
        }
        let n = self.rails.len();
        // 2026-09-25: Per-rail queues of pending request indices, striped
        // round-robin.
        let mut pend: Vec<std::collections::VecDeque<usize>> = vec![Default::default(); n];
        for (j, _) in requests.iter().enumerate() {
            pend[j % n].push_back(j);
        }
        // 2026-09-25: Each outer pass fills every rail's free bounces with new READs,
        // then reaps one completion from each rail that has work in flight.
        loop {
            let mut active = false;
            for (ri, rail) in self.rails.iter_mut().enumerate() {
                while !rail.free.is_empty() {
                    let Some(j) = pend[ri].pop_front() else { break };
                    let b = if is_async {
                        rail.free.pop_front()
                    } else {
                        rail.free.pop_back()
                    }
                    .unwrap();
                    // 2026-09-25: Wait for an async `copy_h2d` still reading this
                    // bounce before the NIC refills it; a no-op unless an earlier
                    // `read_async` left a copy event on `b`.
                    rail.wait_bounce_free(b)?;
                    let raddr = self.remote_base
                        + self.layout.group_id(requests[j].group).0 * self.layout.group_stride;
                    // 2026-09-25: SAFETY: bounce b is a live MR; raddr/rkey address the
                    // peer arena.
                    unsafe { rail.post_read(b, raddr, bytes, requests[j].dst_dev_ptr)? };
                }
                if !rail.inflight.is_empty() {
                    rail.reap_one(bytes, stream, is_async)?;
                    active = true;
                }
            }
            if !active && pend.iter().all(|q| q.is_empty()) {
                break;
            }
        }
        if !is_async {
            // 2026-09-25: `read` returns with `stream` synchronised, as
            // `StorageBackend::read` requires; `read_async` does not wait.
            stream_sync(stream)?;
        }
        Ok(())
    }
}

impl StorageBackend for RdmaKvBackend {
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        self.read_common(requests, stream, false)
    }

    fn read_async(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        self.read_common(requests, stream, true)
    }

    fn register_landing_region(&mut self, base: u64, len: usize) -> Result<()> {
        // 2026-09-25: One MR per rail over the whole region, so every destination
        // inside it reuses that lkey (`Rail::reg_dst`).
        for rail in &mut self.rails {
            rail.register_region(base, len)?;
        }
        tracing::info!(
            "RdmaKvBackend: registered UMA landing region {:.1} MiB on {} rail(s) — zero-copy restore live",
            len as f64 / (1024.0 * 1024.0),
            self.rails.len(),
        );
        Ok(())
    }

    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        if src.len() != bytes {
            bail!(
                "write_from_host: src len {} != group bytes {bytes}",
                src.len()
            );
        }
        let raddr = self.remote_addr(key);
        let n = self.rails.len();
        let ri = self.rr % n;
        self.rr = self.rr.wrapping_add(1);
        let rail = &mut self.rails[ri];
        // 2026-09-25: Take a free bounce on this rail, reaping a completion if none is
        // free.
        if rail.free.is_empty() {
            // 2026-09-25: A `read` that succeeded left no READ in flight, so this
            // reaps a WRITE and issues no copy on stream 0.
            rail.reap_one(bytes, 0, false)?;
        }
        let b = rail.free.pop_back().expect("free bounce after reap");
        // 2026-09-25: Do not overwrite a bounce an async `copy_h2d` may still be
        // reading; a no-op unless an earlier `read_async` left a copy event on `b`.
        rail.wait_bounce_free(b)?;
        // 2026-09-25: SAFETY: bounce b holds `bytes`; copy the group in, then post the
        // RDMA WRITE.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), rail.bounces[b].buf.ptr as *mut u8, bytes);
            rail.post_write(b, raddr, bytes)?;
        }
        // 2026-09-25: The WRITE is reaped lazily, or drained before the next read.
        Ok(())
    }

    fn group_layout(&self) -> GroupLayout {
        // 2026-09-25: Block reads and writes use the trait defaults, which fan out
        // per head through this layout.
        self.layout
    }
}

impl Drop for RdmaKvBackend {
    fn drop(&mut self) {
        let bytes = self.layout.group_bytes() as usize;
        for rail in &mut self.rails {
            let _ = rail.drain(bytes, 0);
        }
    }
}

#[cfg(test)]
mod tests;
