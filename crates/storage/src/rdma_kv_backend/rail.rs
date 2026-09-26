// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-rail state of `RdmaKvBackend`: the registered bounce ring, the
//! in-flight work requests, and the post, poll and drain operations.
//!
//! Owner: storage, RDMA KV tier.
//! Invariants:
//! - A bounce named by an `inflight` entry is not in `free`: `reap_one` removes the
//!   entry before it frees the bounce. A failed post or copy can leave a bounce in
//!   neither.

use std::collections::HashMap;
use std::ffi::c_void;

use anyhow::{Context, Result};

use crate::cuda_min::{CudaEvent, PinnedBuffer, copy_h_to_d_async};
use metrale_gpu_sys::verbs::Verbs;

pub(super) struct Bounce {
    pub(super) buf: PinnedBuffer,
    pub(super) lkey: u32,
    /// 2026-09-25: Event recorded after the async `copy_h_to_d_async` out of this
    /// bounce (`reap_one` with `track`). `wait_bounce_free` syncs and clears it before
    /// the bounce is refilled. Always `None` when only synchronous reads have run.
    pub(super) copy_done: Option<CudaEvent>,
}

pub(super) enum InFlight {
    /// 2026-09-25: A restore: once the READ lands, copy the bounce to `dst`.
    Read { bounce: usize, dst: u64 },
    /// 2026-09-25: An offload WRITE from this bounce; freed on completion.
    Write { bounce: usize },
}

/// 2026-09-25: One QP on one RDMA device, with its own bounce ring and completion
/// tracking.
pub(super) struct Rail {
    pub(super) verbs: Verbs,
    pub(super) remote_rkey: u32,
    pub(super) bounces: Vec<Bounce>,
    pub(super) free: std::collections::VecDeque<usize>,
    pub(super) inflight: HashMap<u64, InFlight>,
    pub(super) next_wr: u64,
    /// 2026-09-25: Zero-copy restore: lkeys of destination MRs registered on demand
    /// outside `region`, by destination address.
    pub(super) dst_lkeys: HashMap<u64, u32>,
    /// 2026-09-25: Pre-registered landing region `(base, len, lkey)`; a destination
    /// inside it reuses this lkey.
    pub(super) region: Option<(u64, u64, u32)>,
    /// 2026-09-25: In-flight zero-copy reads on this rail; they hold no bounce.
    pub(super) direct_inflight: usize,
}

impl Rail {
    #[inline]
    pub(super) fn fresh_wr(&mut self) -> u64 {
        let w = self.next_wr;
        self.next_wr = self.next_wr.wrapping_add(1);
        w
    }

    /// 2026-09-25: Register `[base, base+len)` as one landing MR on this rail,
    /// replacing any earlier region.
    pub(super) fn register_region(&mut self, base: u64, len: usize) -> Result<()> {
        // 2026-09-25: SAFETY: the caller of `register_landing_region` passes a live
        // allocation that outlives this rail.
        let keys = unsafe { self.verbs.reg_mr(base as *mut c_void, len, false) }
            .context("register UMA landing region")?;
        self.region = Some((base, len as u64, keys.lkey));
        Ok(())
    }

    /// 2026-09-25: The lkey for a zero-copy READ of `bytes` into `addr`: the
    /// landing region's when `addr` lies inside it, else a cached or newly registered
    /// MR. A registration failure is an error.
    pub(super) fn reg_dst(&mut self, addr: u64, bytes: usize) -> Result<u32> {
        if let Some((base, len, lkey)) = self.region
            && addr >= base
            && addr + bytes as u64 <= base + len
        {
            return Ok(lkey);
        }
        if let Some(&lk) = self.dst_lkeys.get(&addr) {
            return Ok(lk);
        }
        // 2026-09-25: SAFETY: in zero-copy mode the caller's destination is a live
        // buffer of at least `bytes`.
        let keys = unsafe { self.verbs.reg_mr(addr as *mut c_void, bytes, false) }
            .context("zero-copy restore needs a UMA (GPU-addressable) dst; reg_mr failed")?;
        self.dst_lkeys.insert(addr, keys.lkey);
        Ok(keys.lkey)
    }

    /// 2026-09-25: Reap one completion on this rail and free its bounce. For a READ,
    /// first enqueue the copy of the bounce to its destination on `stream`; with
    /// `track`, also record an event after that copy in the bounce's `copy_done`.
    /// Only `read_async` passes `track`. A completion with an unknown `wr_id` is an
    /// error.
    pub(super) fn reap_one(&mut self, group_bytes: usize, stream: u64, track: bool) -> Result<()> {
        let wr = self.verbs.poll()?;
        let op = self
            .inflight
            .remove(&wr)
            .with_context(|| format!("kv: completion for unknown wr_id {wr:#x}"))?;
        let bounce = match op {
            InFlight::Read { bounce, dst } => {
                copy_h_to_d_async(
                    dst,
                    self.bounces[bounce].buf.ptr as *const _,
                    group_bytes,
                    stream,
                )?;
                if track {
                    let ev = CudaEvent::new()?;
                    ev.record(stream)?;
                    self.bounces[bounce].copy_done = Some(ev);
                }
                bounce
            }
            InFlight::Write { bounce } => bounce,
        };
        self.free.push_back(bounce);
        Ok(())
    }

    /// 2026-09-25: Before reusing bounce `b`, wait for the async copy out of it that
    /// a `read_async` reap recorded, if any.
    pub(super) fn wait_bounce_free(&mut self, b: usize) -> Result<()> {
        if let Some(ev) = self.bounces[b].copy_done.take() {
            ev.sync()?;
        }
        Ok(())
    }

    pub(super) fn drain(&mut self, group_bytes: usize, stream: u64) -> Result<()> {
        while !self.inflight.is_empty() {
            // 2026-09-25: Callers drain before a read or on drop; no later reuse
            // waits on these copies' events, so none is recorded.
            self.reap_one(group_bytes, stream, false)?;
        }
        Ok(())
    }

    /// 2026-09-25: Post an RDMA READ of `bytes` from `raddr` into bounce `bounce`,
    /// to be copied to `dst` when reaped.
    ///
    /// # Safety
    /// `bounce` must be a free, registered bounce of at least `bytes`, and `raddr`
    /// must lie in the peer arena.
    pub(super) unsafe fn post_read(
        &mut self,
        bounce: usize,
        raddr: u64,
        bytes: usize,
        dst: u64,
    ) -> Result<()> {
        let wr = self.fresh_wr();
        unsafe {
            self.verbs.post_read(
                self.bounces[bounce].buf.ptr,
                self.bounces[bounce].lkey,
                raddr,
                self.remote_rkey,
                bytes as u32,
                wr,
            )?;
        }
        self.inflight.insert(wr, InFlight::Read { bounce, dst });
        Ok(())
    }

    /// 2026-09-25: Post an RDMA WRITE of `bytes` from bounce `bounce` to `raddr`.
    ///
    /// # Safety
    /// As [`Self::post_read`]; the bounce already holds the bytes to write.
    pub(super) unsafe fn post_write(
        &mut self,
        bounce: usize,
        raddr: u64,
        bytes: usize,
    ) -> Result<()> {
        let wr = self.fresh_wr();
        unsafe {
            self.verbs.post_write(
                self.bounces[bounce].buf.ptr,
                self.bounces[bounce].lkey,
                raddr,
                self.remote_rkey,
                bytes as u32,
                wr,
            )?;
        }
        self.inflight.insert(wr, InFlight::Write { bounce });
        Ok(())
    }
}
