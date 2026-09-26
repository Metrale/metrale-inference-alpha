// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `RdmaSnapshotArena`, the synchronous RDMA transport of the SSM snapshot
//! spill tier, addressed by byte offset or, in paging mode, by key.
//!
//! Owner: storage, SSM snapshot tier.
//! Invariants:
//! - Every transfer holds the arena's lock throughout, and returns `Ok` only after
//!   its RDMA work has completed.
//! - Without the `cuda` feature and `metrale_rdma_verbs`, both constructors fail.
//!
//! Both modes send the v2 paging header with kind `SSM`. Raw mode (`connect`, which
//! sends `blob_bytes == 0`) gets a private arena and the caller chooses offsets.
//! Paging mode (`connect_paging`) lets the peer own residency, driven by ALLOC,
//! COMMIT, GET and REMOVE on the TCP stream. This transport moves host bytes only;
//! `SsmSnapshotPool::spill_slot` and `fault_in_slot` in metrale-model-engine gather
//! the state and order the device streams.

// 2026-09-25: Without `cuda` and verbs a stub whose constructors fail takes the real
// type's place, so callers compile unchanged.
#[cfg(all(feature = "cuda", metrale_rdma_verbs))]
pub use imp::RdmaSnapshotArena;
#[cfg(not(all(feature = "cuda", metrale_rdma_verbs)))]
pub use stub::RdmaSnapshotArena;

#[cfg(not(all(feature = "cuda", metrale_rdma_verbs)))]
mod stub {
    use anyhow::{Result, bail};
    /// 2026-09-25: Stand-in when RDMA verbs or CUDA are not built. Both constructors
    /// fail; the other methods panic.
    pub struct RdmaSnapshotArena;
    impl RdmaSnapshotArena {
        pub fn connect(_addr: &str, _arena_bytes: u64, _blob_bytes: usize) -> Result<Self> {
            bail!("RDMA snapshot tier not built (needs feature `cuda` + metrale_rdma_verbs)")
        }
        pub fn connect_paging(_addr: &str, _arena_bytes: u64, _blob_bytes: usize) -> Result<Self> {
            bail!("RDMA snapshot tier not built (needs feature `cuda` + metrale_rdma_verbs)")
        }
        pub fn write(&self, _offset: u64, _bytes: &[u8]) -> Result<()> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
        pub fn read(&self, _offset: u64, _out: &mut [u8]) -> Result<()> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
        pub fn paging_put(&self, _key: u64, _bytes: &[u8]) -> Result<()> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
        pub fn paging_get(&self, _key: u64, _out: &mut [u8]) -> Result<bool> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
        pub fn paging_remove(&self, _key: u64) -> Result<()> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
    }
}

#[cfg(all(feature = "cuda", metrale_rdma_verbs))]
mod imp {
    use std::io::Write;
    use std::net::TcpStream;
    use std::sync::Mutex;

    use anyhow::{Result, bail};

    use crate::cuda_min::PinnedBuffer;
    use metrale_gpu_sys::env::{first_set, first_set_u32};
    use metrale_gpu_sys::railset::{RailSet, RailSpec};
    use metrale_gpu_sys::verbs::Verbs;

    /// 2026-09-25: One rail: its QP and one registered `blob_bytes` bounce.
    struct SnapRail {
        verbs: Verbs,
        bounce: PinnedBuffer,
        lkey: u32,
        remote_rkey: u32,
        /// 2026-09-25: This rail's lkey for the shared staging buffer; 0 when staging
        /// is off. Every rail registers the same buffer, so a chunk lands at its own
        /// offset whichever rail moves it.
        staging_lkey: u32,
    }

    /// 2026-09-25: Mutable transport state, behind one lock because the public
    /// methods take `&self`.
    struct ArenaInner {
        rails: Vec<SnapRail>,
        /// 2026-09-25: One contiguous `blob_bytes` staging buffer for the striped path
        /// (`METRALE_SSM_STAGING=1`); `None` moves each blob as one WR through a
        /// rail's bounce.
        staging: Option<PinnedBuffer>,
        rr: usize,
        next_wr: u64,
        /// 2026-09-25: Raw mode: held open, idle, for the connection's lifetime.
        /// Paging mode: the control channel for ALLOC, COMMIT, GET and REMOVE.
        stream: TcpStream,
    }

    /// 2026-09-25: RDMA snapshot arena. `write` and `read` move one `blob_bytes`
    /// blob to or from `base + offset`; the `paging_*` methods address blobs by key.
    pub struct RdmaSnapshotArena {
        inner: Mutex<ArenaInner>,
        remote_base: u64,
        blob_bytes: usize,
    }

    // 2026-09-25: SAFETY: every access to the verbs and bounce state goes through
    // `inner`'s Mutex. `Verbs` and `PinnedBuffer` are `Send`.
    unsafe impl Send for RdmaSnapshotArena {}
    unsafe impl Sync for RdmaSnapshotArena {}

    impl RdmaSnapshotArena {
        /// 2026-09-25: Raw-mode connect to the peer at `addr` for an arena of
        /// `arena_bytes`, registering `blob_bytes` bounces. Rails as `RdmaKvBackend`:
        /// rail 0 from `METRALE_EXPERT_RDMA_DEV`/`GID`, rail 1 (only with
        /// `METRALE_KV_DUAL_RAIL=1`) from `METRALE_KV_RAIL2_DEV`/`GID`.
        pub fn connect(addr: &str, arena_bytes: u64, blob_bytes: usize) -> Result<Self> {
            Self::connect_inner(addr, arena_bytes, blob_bytes, false)
        }

        /// 2026-09-25: Paging-mode connect: the peer owns residency, caching blobs in
        /// its arena over an NVMe swap file, and the client uses `paging_put`,
        /// `paging_get` and `paging_remove`. The peer refuses it without a swap
        /// directory (`RdmaConfig::swap_dir`).
        pub fn connect_paging(addr: &str, arena_bytes: u64, blob_bytes: usize) -> Result<Self> {
            Self::connect_inner(addr, arena_bytes, blob_bytes, true)
        }

        fn connect_inner(
            addr: &str,
            arena_bytes: u64,
            blob_bytes: usize,
            paging: bool,
        ) -> Result<Self> {
            // 2026-09-25: A random 24-bit PSN per rail.
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

            let mut stream = TcpStream::connect(addr)
                .map_err(|e| anyhow::anyhow!("connect snapshot peer {addr}: {e}"))?;
            stream.set_nodelay(true).ok();
            // 2026-09-25: Paging mode sends the blob size; raw mode sends
            // `blob_bytes == 0`, and the peer gives it a private arena.
            stream.write_all(&crate::snapshot_swap::encode_paging_v2_header(
                crate::snapshot_swap::PagingKind::SSM,
                arena_bytes,
                if paging { blob_bytes as u64 } else { 0 },
            ))?;
            let mut rs = RailSet::begin(&mut stream, &specs)?;

            // 2026-09-25: One staging buffer, registered on every rail below.
            let staging_on = std::env::var("METRALE_SSM_STAGING").ok().as_deref() == Some("1");
            let staging = if staging_on {
                Some(PinnedBuffer::new(blob_bytes)?)
            } else {
                None
            };

            // 2026-09-25: Bounce and staging MRs are LOCAL_WRITE only
            // (`remote_read == false`).
            let mut parts: Vec<(PinnedBuffer, u32, u32)> = Vec::with_capacity(n_rails);
            for rail in &mut rs.rails {
                let bounce = PinnedBuffer::new(blob_bytes)?;
                // 2026-09-25: SAFETY: the bounce lives as long as the rail, and so the MR.
                let keys = unsafe { rail.verbs.reg_mr(bounce.ptr, blob_bytes, false)? };
                // 2026-09-25: SAFETY: `staging` is declared after `rails` in
                // `ArenaInner`, so it is dropped after them.
                let staging_lkey = match &staging {
                    Some(s) => unsafe { rail.verbs.reg_mr(s.ptr, blob_bytes, false)?.lkey },
                    None => 0,
                };
                parts.push((bounce, keys.lkey, staging_lkey));
            }

            // 2026-09-25: Every rail publishes the same arena base; the last is kept.
            let server = rs.finish_rw(&mut stream, "snapshot peer")?;
            let base = server.last().map(|sp| sp.base_addr).unwrap_or(0);
            let rails: Vec<SnapRail> = rs
                .into_verbs()
                .into_iter()
                .zip(parts)
                .zip(&server)
                .map(|((verbs, (bounce, lkey, staging_lkey)), sp)| SnapRail {
                    verbs,
                    bounce,
                    lkey,
                    remote_rkey: sp.rkey,
                    staging_lkey,
                })
                .collect();
            tracing::info!(
                "RdmaSnapshotArena connected to {addr}: {:.1} GiB arena, {n_rails} rail(s), blob {blob_bytes} B",
                arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            Ok(Self {
                inner: Mutex::new(ArenaInner {
                    rails,
                    staging,
                    rr: 0,
                    next_wr: 1,
                    stream,
                }),
                remote_base: base,
                blob_bytes,
            })
        }

        #[inline]
        pub fn blob_bytes(&self) -> usize {
            self.blob_bytes
        }

        /// 2026-09-25: RDMA WRITE one `blob_bytes` blob to `base + offset` and wait for
        /// its completion.
        pub fn write(&self, offset: u64, bytes: &[u8]) -> Result<()> {
            if bytes.len() != self.blob_bytes {
                bail!(
                    "snapshot write: {} != blob_bytes {}",
                    bytes.len(),
                    self.blob_bytes
                );
            }
            let mut g = self.inner.lock().expect("snapshot arena mutex");
            self.rdma_write_locked(&mut g, self.remote_base + offset, bytes)
        }

        /// 2026-09-25: RDMA READ one `blob_bytes` blob from `base + offset` into `out`
        /// and wait for its completion.
        pub fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
            if out.len() != self.blob_bytes {
                bail!(
                    "snapshot read: {} != blob_bytes {}",
                    out.len(),
                    self.blob_bytes
                );
            }
            let mut g = self.inner.lock().expect("snapshot arena mutex");
            self.rdma_read_locked(&mut g, self.remote_base + offset, out)
        }

        /// 2026-09-25: The next rail, round-robin, and a fresh non-zero wr id.
        fn rail_and_wr(g: &mut ArenaInner) -> (usize, u64) {
            let n = g.rails.len();
            let ri = g.rr % n;
            g.rr = g.rr.wrapping_add(1);
            let wr = g.next_wr;
            g.next_wr = g.next_wr.wrapping_add(1).max(1);
            (ri, wr)
        }

        fn rdma_write_locked(&self, g: &mut ArenaInner, raddr: u64, bytes: &[u8]) -> Result<()> {
            if g.staging.is_some() {
                return self.rdma_staged(g, raddr, Some(bytes), None);
            }
            let (ri, wr) = Self::rail_and_wr(g);
            let rail = &mut g.rails[ri];
            // 2026-09-25: SAFETY: the bounce is a live registered MR of `blob_bytes`,
            // and `bytes` has that length.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    rail.bounce.ptr as *mut u8,
                    self.blob_bytes,
                );
                rail.verbs.post_write(
                    rail.bounce.ptr,
                    rail.lkey,
                    raddr,
                    rail.remote_rkey,
                    self.blob_bytes as u32,
                    wr,
                )?;
            }
            while rail.verbs.poll()? != wr {}
            Ok(())
        }

        fn rdma_read_locked(&self, g: &mut ArenaInner, raddr: u64, out: &mut [u8]) -> Result<()> {
            if g.staging.is_some() {
                return self.rdma_staged(g, raddr, None, Some(out));
            }
            let (ri, wr) = Self::rail_and_wr(g);
            let rail = &mut g.rails[ri];
            // 2026-09-25: SAFETY: the bounce is a live registered MR of `blob_bytes`, and
            // `out` has that length.
            unsafe {
                rail.verbs.post_read(
                    rail.bounce.ptr,
                    rail.lkey,
                    raddr,
                    rail.remote_rkey,
                    self.blob_bytes as u32,
                    wr,
                )?;
            }
            while rail.verbs.poll()? != wr {}
            unsafe {
                std::ptr::copy_nonoverlapping(
                    rail.bounce.ptr as *const u8,
                    out.as_mut_ptr(),
                    self.blob_bytes,
                );
            }
            Ok(())
        }

        /// 2026-09-25: Move one blob through the staging buffer in chunks
        /// (`stripe_plan`), round-robin across rails with at most `staging_depth()` in
        /// flight per rail. `write_src` set: copy in, then WRITE; `read_dst` set: READ,
        /// then copy out. Every chunk sits at its blob offset in both the staging
        /// buffer and the peer arena, so one copy assembles the blob.
        fn rdma_staged(
            &self,
            g: &mut ArenaInner,
            raddr: u64,
            write_src: Option<&[u8]>,
            read_dst: Option<&mut [u8]>,
        ) -> Result<()> {
            let staging = g.staging.as_ref().expect("staging present").ptr as *mut u8;
            let is_read = read_dst.is_some();
            if let Some(src) = write_src {
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), staging, self.blob_bytes) };
            }
            let n = g.rails.len();
            let chunk = crate::snapshot_swap::staging_chunk_bytes();
            let depth = crate::snapshot_swap::staging_depth();
            let plan = crate::snapshot_swap::stripe_plan(self.blob_bytes, chunk, n);
            let total: usize = plan.iter().map(|w| w.len()).sum();
            let mut posted = vec![0usize; n];
            let mut reaped = vec![0usize; n];
            let mut done = 0usize;
            while done < total {
                for ri in 0..n {
                    while posted[ri] < plan[ri].len() && (posted[ri] - reaped[ri]) < depth {
                        let (off, len) = plan[ri][posted[ri]];
                        let wr = g.next_wr;
                        g.next_wr = g.next_wr.wrapping_add(1).max(1);
                        let rail = &mut g.rails[ri];
                        let local = unsafe { staging.add(off) } as *mut _;
                        let raddr_chunk = raddr + off as u64;
                        unsafe {
                            if is_read {
                                rail.verbs.post_read(
                                    local,
                                    rail.staging_lkey,
                                    raddr_chunk,
                                    rail.remote_rkey,
                                    len as u32,
                                    wr,
                                )?;
                            } else {
                                rail.verbs.post_write(
                                    local,
                                    rail.staging_lkey,
                                    raddr_chunk,
                                    rail.remote_rkey,
                                    len as u32,
                                    wr,
                                )?;
                            }
                        }
                        posted[ri] += 1;
                    }
                }
                for ri in 0..n {
                    if posted[ri] > reaped[ri] {
                        g.rails[ri].verbs.poll()?;
                        reaped[ri] += 1;
                        done += 1;
                    }
                }
            }
            if let Some(dst) = read_dst {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        staging as *const u8,
                        dst.as_mut_ptr(),
                        self.blob_bytes,
                    )
                };
            }
            Ok(())
        }

        /// 2026-09-25: Store `key`'s blob: ALLOC a slot on the control channel, RDMA
        /// WRITE the blob there, then COMMIT, all under the lock.
        pub fn paging_put(&self, key: u64, bytes: &[u8]) -> Result<()> {
            if bytes.len() != self.blob_bytes {
                bail!(
                    "paging_put: {} != blob_bytes {}",
                    bytes.len(),
                    self.blob_bytes
                );
            }
            let mut g = self.inner.lock().expect("snapshot arena mutex");
            let off = crate::snapshot_swap::client_alloc(&mut g.stream, key)?;
            self.rdma_write_locked(&mut g, self.remote_base + off, bytes)?;
            crate::snapshot_swap::client_commit(&mut g.stream, key)
        }

        /// 2026-09-25: Read `key`'s blob into `out`; `Ok(false)` when the peer does not
        /// have it.
        pub fn paging_get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
            if out.len() != self.blob_bytes {
                bail!(
                    "paging_get: {} != blob_bytes {}",
                    out.len(),
                    self.blob_bytes
                );
            }
            let mut g = self.inner.lock().expect("snapshot arena mutex");
            match crate::snapshot_swap::client_get(&mut g.stream, key)? {
                Some(off) => {
                    self.rdma_read_locked(&mut g, self.remote_base + off, out)?;
                    Ok(true)
                }
                None => Ok(false),
            }
        }

        /// 2026-09-25: Ask the peer to drop `key`.
        pub fn paging_remove(&self, key: u64) -> Result<()> {
            let mut g = self.inner.lock().expect("snapshot arena mutex");
            crate::snapshot_swap::client_remove(&mut g.stream, key)
        }
    }
}
