// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `KvPagingBackend`, the KV tier as a paging client of a peer.
//! Every op is synchronous, with one control op in flight:
//!   PUT = ALLOC (control) → RDMA WRITE of the block → poll → COMMIT (control)
//!   GET = GET (control) → RDMA READ of the block → poll
//! With `METRALE_KV_ZERO_COPY=1`, a GET whose destination lies inside every
//! rail's registered landing region is read straight into it.
//!
//! Owner: metrale-storage KV tier.
//! Invariants:
//! - A GET's RDMA READ is polled to completion before the next control op
//!   is sent, unless the poll itself fails.
//! - A GET miss is an error ([`super::kv_miss_error`]).
//! - The per-head `write_from_host` always fails.

use std::ffi::c_void;
use std::io::Write;
use std::net::TcpStream;
use std::num::NonZeroU64;

use anyhow::{Context, Result, bail};

use super::ns;
use crate::backend::{BlockReadRequest, ReadRequest, StorageBackend};
use crate::cuda_min::{PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, GroupLayout, KvKind};
use crate::snapshot_swap::{
    PagingKind, client_alloc, client_bye, client_commit, client_get, encode_paging_v2_header,
};
use metrale_gpu_sys::verbs::Verbs;

/// 2026-09-25: Connect parameters. [`super::connect_kv_peer_backend`] fills
/// them from the environment; `examples/snapshot_paging_smoke.rs` builds them
/// directly.
#[derive(Clone, Copy, Debug)]
pub struct KvPagingConnect {
    /// 2026-09-25: Peer arena bytes: a non-zero multiple of
    /// `layout.block_bytes()`, which `connect` checks.
    pub arena_bytes: u64,
    /// 2026-09-25: The namespace folded into every wire key
    /// ([`ns::wire_key`]).
    pub ns: NonZeroU64,
}

/// 2026-09-25: One rail: a verbs connection and one registered block-sized
/// bounce buffer.
struct PagingRail {
    verbs: Verbs,
    bounce: PinnedBuffer,
    bounce_lkey: u32,
    remote_rkey: u32,
    /// 2026-09-25: The landing region `(base, len, lkey)` from
    /// `register_landing_region`, used by zero-copy reads.
    region: Option<(u64, u64, u32)>,
}

pub struct KvPagingBackend {
    rails: Vec<PagingRail>,
    layout: GroupLayout,
    remote_base: u64,
    ns: NonZeroU64,
    zero_copy: bool,
    rr: usize,
    next_wr: u64,
    /// 2026-09-25: The control connection: ALLOC, COMMIT, GET and BYE go
    /// here; block data moves over the rails.
    ctrl: TcpStream,
}

// 2026-09-25: SAFETY: every method that touches a QP or a bounce buffer takes
// `&mut self`; the `&self` methods read only `layout`, `ns` and the rails'
// `region`.
unsafe impl Sync for KvPagingBackend {}

impl KvPagingBackend {
    /// 2026-09-25: Connect to the paging peer at `addr`: send the v2 paging
    /// header (`PagingKind::KV`, `arena_bytes`, `blob_bytes = block_bytes()`),
    /// bring up rail 0 from `METRALE_EXPERT_RDMA_DEV`/`GID` and, with
    /// `METRALE_KV_DUAL_RAIL=1`, rail 1 from `METRALE_KV_RAIL2_DEV`/`GID`, and
    /// register one block-sized bounce buffer per rail with `remote_read = false`.
    /// Fails unless `cfg.arena_bytes` is a non-zero multiple of `block_bytes()`.
    pub fn connect(addr: &str, layout: GroupLayout, cfg: KvPagingConnect) -> Result<Self> {
        use metrale_gpu_sys::env::{first_set, first_set_u32};
        use metrale_gpu_sys::railset::{RailSet, RailSpec};

        let block_bytes = layout.block_bytes();
        if cfg.arena_bytes == 0 || !cfg.arena_bytes.is_multiple_of(block_bytes) {
            bail!(
                "kv-paging: arena_bytes {} must be a non-zero multiple of block_bytes {block_bytes}",
                cfg.arena_bytes
            );
        }
        let spec =
            |dev: String, gid: u32| RailSpec::new(dev, gid, rand::random::<u32>() & 0xff_ffff);
        let rail0 = spec(
            first_set(&["METRALE_EXPERT_RDMA_DEV"], "roceP2p1s0f1"),
            first_set_u32(&["METRALE_EXPERT_RDMA_GID"], 3),
        );
        let dual = std::env::var("METRALE_KV_DUAL_RAIL").ok().as_deref() == Some("1");
        let specs: Vec<RailSpec> = if dual {
            vec![
                rail0,
                spec(
                    first_set(&["METRALE_KV_RAIL2_DEV"], "rocep1s0f1"),
                    first_set_u32(&["METRALE_KV_RAIL2_GID"], 3),
                ),
            ]
        } else {
            vec![rail0]
        };

        let mut stream =
            TcpStream::connect(addr).with_context(|| format!("connect kv paging peer {addr}"))?;
        stream.set_nodelay(true).ok();
        stream
            .write_all(&encode_paging_v2_header(
                PagingKind::KV,
                cfg.arena_bytes,
                block_bytes,
            ))
            .context("send kv paging v2 header")?;

        let bb = block_bytes as usize;
        // 2026-09-25: `parts` (the pinned bounce buffers) is declared before
        // `rs` (the verbs holding their memory registrations), so on an early
        // return the locals drop in reverse order and the registrations go
        // before the memory. On success, `PagingRail` declares `verbs` before
        // `bounce` for the same order.
        let mut parts: Vec<(PinnedBuffer, u32)> = Vec::new();
        let mut rs = RailSet::begin(&mut stream, &specs)?;
        parts.reserve(rs.n_rails());
        for rail in &mut rs.rails {
            let bounce = PinnedBuffer::new(bb).context("alloc pinned kv paging bounce")?;
            // 2026-09-25: SAFETY: the bounce buffer outlives its registration
            // on both paths (see above).
            let keys = unsafe { rail.verbs.reg_mr(bounce.ptr, bb, false)? };
            parts.push((bounce, keys.lkey));
        }
        let server = rs.finish_rw(&mut stream, "kv paging peer")?;
        let base = server.last().map(|sp| sp.base_addr).unwrap_or(0);
        let rails: Vec<PagingRail> = rs
            .into_verbs()
            .into_iter()
            .zip(parts)
            .zip(&server)
            .map(|((verbs, (bounce, bounce_lkey)), sp)| PagingRail {
                verbs,
                bounce,
                bounce_lkey,
                remote_rkey: sp.rkey,
                region: None,
            })
            .collect();
        let zero_copy = std::env::var("METRALE_KV_ZERO_COPY").ok().as_deref() == Some("1");
        tracing::info!(
            "KvPagingBackend connected to {addr}: kind=KV, blob {block_bytes} B, arena {:.3} GiB, \
             ns {:#018x}, {} rail(s), zero_copy={zero_copy} (strictly synchronous MVP: 1 control \
             op in flight)",
            cfg.arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            cfg.ns.get(),
            rails.len(),
        );
        Ok(Self {
            rails,
            layout,
            remote_base: base,
            ns: cfg.ns,
            zero_copy,
            rr: 0,
            // 2026-09-25: Work-request ids start at 1; `pick_rail` never issues 0.
            next_wr: 1,
            ctrl: stream,
        })
    }

    /// 2026-09-25: The wire key of one KV block: its K-head-0 group id folded
    /// with the namespace.
    fn block_key(&self, layer: u32, block: u32) -> u64 {
        let base = self
            .layout
            .group_id(GroupKey::new(layer, block, 0, KvKind::K))
            .0;
        ns::wire_key(self.ns, base)
    }

    /// 2026-09-25: Control GET: the peer offset of `(layer, block)`. A miss is
    /// `kv_miss_error`.
    fn get_block_offset(&mut self, layer: u32, block: u32) -> Result<u64> {
        let key = self.block_key(layer, block);
        match client_get(&mut self.ctrl, key)? {
            Some(off) => Ok(off),
            None => Err(super::kv_miss_error(layer, block)),
        }
    }

    fn pick_rail(&mut self) -> (usize, u64) {
        let ri = self.rr % self.rails.len();
        self.rr = self.rr.wrapping_add(1);
        let wr = self.next_wr;
        self.next_wr = self.next_wr.wrapping_add(1).max(1);
        (ri, wr)
    }

    /// 2026-09-25: RDMA READ `len` bytes into the rail's bounce buffer, poll,
    /// copy to `dst` on `stream` and sync. Each rail has one bounce buffer, so
    /// the sync keeps the next op from refilling it under the copy.
    fn rdma_read_bounce(&mut self, raddr: u64, len: usize, dst: u64, stream: u64) -> Result<()> {
        let (ri, wr) = self.pick_rail();
        let rail = &mut self.rails[ri];
        // 2026-09-25: SAFETY: the bounce is a registered block-sized buffer and
        // callers pass `len` of at most one block; raddr/rkey address the peer
        // arena.
        unsafe {
            rail.verbs.post_read(
                rail.bounce.ptr,
                rail.bounce_lkey,
                raddr,
                rail.remote_rkey,
                len as u32,
                wr,
            )?;
        }
        while rail.verbs.poll()? != wr {}
        copy_h_to_d_async(dst, rail.bounce.ptr as *const c_void, len, stream)?;
        stream_sync(stream)
    }

    /// 2026-09-25: RDMA READ straight into `dst` through the rail's landing
    /// region. The caller has checked coverage and synced `stream`.
    fn rdma_read_direct(&mut self, raddr: u64, len: usize, dst: u64) -> Result<()> {
        let (ri, wr) = self.pick_rail();
        let rail = &mut self.rails[ri];
        let (_, _, lkey) = rail.region.expect("caller verified region coverage");
        // 2026-09-25: SAFETY: `[dst, dst + len)` is inside the landing region
        // registered under `lkey`; raddr/rkey address the peer arena.
        unsafe {
            rail.verbs.post_read(
                dst as *mut c_void,
                lkey,
                raddr,
                rail.remote_rkey,
                len as u32,
                wr,
            )?;
        }
        while rail.verbs.poll()? != wr {}
        Ok(())
    }

    /// 2026-09-25: Whether every rail's landing region covers `[dst, dst+len)`.
    fn all_regions_cover(&self, dst: u64, len: usize) -> bool {
        self.rails.iter().all(|r| match r.region {
            Some((base, rlen, _)) => dst >= base && dst + len as u64 <= base + rlen,
            None => false,
        })
    }
}

impl StorageBackend for KvPagingBackend {
    /// 2026-09-25: Per-head restore: GET the block, then RDMA READ the one
    /// `group_stride` stripe at `offset + (kind·nkv + kv_head)·gs`.
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        let gs = self.layout.group_stride;
        let nkv = self.layout.num_kv_heads as u64;
        for req in requests {
            let g = req.group;
            let off = self.get_block_offset(g.layer, g.block)?;
            let idx = (g.kv_kind as u64) * nkv + g.kv_head as u64;
            let raddr = self.remote_base + off + idx * gs;
            self.rdma_read_bounce(raddr, gs as usize, req.dst_dev_ptr, stream)?;
        }
        Ok(())
    }

    // 2026-09-25: `read_async` and `read_blocks_async` keep the synchronous
    // trait defaults.

    /// 2026-09-25: Block restore: one control GET and one RDMA READ per block.
    fn read_blocks(&mut self, requests: &[BlockReadRequest], stream: u64) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }
        let bb = self.layout.block_bytes() as usize;
        let zc = self.zero_copy
            && requests
                .iter()
                .all(|r| self.all_regions_cover(r.dst_dev_ptr, bb));
        if zc {
            // 2026-09-25: The NIC writes the slots outside `stream`, so work
            // still reading them on `stream` is drained first, as in
            // `RdmaKvBackend::read_zero_copy`.
            stream_sync(stream)?;
        }
        for req in requests {
            let off = self.get_block_offset(req.base_key.layer, req.base_key.block)?;
            let raddr = self.remote_base + off;
            if zc {
                self.rdma_read_direct(raddr, bb, req.dst_dev_ptr)?;
            } else {
                self.rdma_read_bounce(raddr, bb, req.dst_dev_ptr, stream)?;
            }
        }
        Ok(())
    }

    /// 2026-09-25: Always fails. ALLOC hands out a whole-block slot, so
    /// committing after one `group_stride` stripe would mark the other
    /// `2·nkv − 1` stripes of the slot as valid.
    fn write_from_host(&mut self, key: GroupKey, _src: &[u8]) -> Result<()> {
        bail!(
            "kv-paging: per-head write_from_host (layer {}, block {}) is unsupported — the \
             paging record is one whole KV block; run with METRALE_HSS_COALESCE_BLOCKS on \
             (default) so offload uses write_block_from_host",
            key.layer,
            key.block
        )
    }

    /// 2026-09-25: Offload one whole block: ALLOC, RDMA WRITE through the
    /// rail's bounce buffer, poll, COMMIT. `src` must be `block_bytes()` long.
    fn write_block_from_host(&mut self, base_key: GroupKey, src: &[u8]) -> Result<()> {
        let bb = self.layout.block_bytes() as usize;
        if src.len() != bb {
            bail!(
                "kv-paging write_block_from_host: src len {} != block bytes {bb}",
                src.len()
            );
        }
        let key = self.block_key(base_key.layer, base_key.block);
        let off = client_alloc(&mut self.ctrl, key).with_context(|| {
            format!(
                "kv-paging ALLOC (layer {}, block {}) refused — peer arena exhausted by \
                 reservations/read-pins? Grow METRALE_KV_PAGING_ARENA_GB (and the peer's \
                 --max-blade-gb)",
                base_key.layer, base_key.block
            )
        })?;
        let raddr = self.remote_base + off;
        let (ri, wr) = self.pick_rail();
        let rail = &mut self.rails[ri];
        // 2026-09-25: SAFETY: the bounce is a registered buffer of
        // `block_bytes()` and `src` was checked to that length; the write
        // targets the slot ALLOC returned.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), rail.bounce.ptr as *mut u8, bb);
            rail.verbs.post_write(
                rail.bounce.ptr,
                rail.bounce_lkey,
                raddr,
                rail.remote_rkey,
                bb as u32,
                wr,
            )?;
        }
        while rail.verbs.poll()? != wr {}
        client_commit(&mut self.ctrl, key)
    }

    /// 2026-09-25: `false`: the peer assigns each block's slot, so a run of
    /// blocks has no contiguous remote range. `write_blocks_run` keeps the
    /// per-block default.
    fn supports_write_run_coalescing(&self) -> bool {
        false
    }

    fn group_layout(&self) -> GroupLayout {
        self.layout
    }

    /// 2026-09-25: Register `[base, base + len)` as one landing region per
    /// rail, with `remote_read = false`, for zero-copy reads.
    fn register_landing_region(&mut self, base: u64, len: usize) -> Result<()> {
        for rail in &mut self.rails {
            // 2026-09-25: SAFETY: relies on the caller keeping
            // `[base, base + len)` allocated until this backend drops; the
            // safe trait method cannot enforce it.
            let keys = unsafe { rail.verbs.reg_mr(base as *mut c_void, len, false) }
                .context("kv-paging: register UMA landing region")?;
            rail.region = Some((base, len as u64, keys.lkey));
        }
        tracing::info!(
            "KvPagingBackend: registered UMA landing region {:.1} MiB on {} rail(s) — \
             zero-copy restore live",
            len as f64 / (1024.0 * 1024.0),
            self.rails.len(),
        );
        Ok(())
    }
}

impl Drop for KvPagingBackend {
    fn drop(&mut self) {
        let _ = client_bye(&mut self.ctrl);
    }
}
