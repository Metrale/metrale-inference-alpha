// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `RdmaTier`: fetches expert records from an expert peer
//! (`expert_peer`) straight into the pinned arena and returns residency
//! addresses inside it, as `UmaArenaTier` does from local files. Transports:
//! TCP, where the peer reads each record and sends it, and one-sided RDMA
//! READ from the peer's registered store (`cfg(metrale_rdma_verbs)` only).
//! Verbs rail 0 comes from `METRALE_EXPERT_RDMA_DEV` / `METRALE_EXPERT_RDMA_GID`;
//! `METRALE_EXPERT_DUAL_RAIL=1` adds rail 1 from `METRALE_EXPERT_RAIL2_DEV` /
//! `METRALE_EXPERT_RAIL2_GID`.
//!
//! Owner: metrale-storage experts.
//! Invariants:
//! - A fetched record passes `residency_from`'s identity check before its
//!   addresses are returned.
//! - The rails' registrations of the arena are dropped before the arena.

use std::io::{Read, Write};
use std::net::TcpStream;

use anyhow::{Context, Result, bail};

use crate::expert::{ExpertKey, ExpertLayout, ExpertRecordSpec};
use crate::expert_arena::ExpertArena;
use crate::expert_peer::{MODE_TCP, STATUS_OK, encode_request, read_manifest};
use crate::expert_tier::{ArenaSlot, ExpertResidency, ExpertTier, TierKind, residency_from};

/// 2026-09-25: The peer transport. `Verbs` exists only with
/// `cfg(metrale_rdma_verbs)`; a verbs fetch uses rail `expert % n_rails`.
enum Transport {
    Tcp,
    #[cfg(metrale_rdma_verbs)]
    Verbs(Vec<Rail>),
}

/// 2026-09-25: One verbs rail: its connection, the arena's lkey on it, and the
/// peer's per-layer `(base, rkey)` table for it.
#[cfg(metrale_rdma_verbs)]
struct Rail {
    verbs: metrale_gpu_sys::verbs::Verbs,
    arena_lkey: u32,
    layers: Vec<(u64, u32)>,
}

pub struct RdmaTier {
    stream: TcpStream,
    // 2026-09-25: Declared before `arena`, so it drops first: the verbs rails
    // hold registrations of the arena's pinned pages, which must go before the
    // pages are freed.
    transport: Transport,
    arena: ExpertArena,
    spec: ExpertRecordSpec,
    layout: ExpertLayout,
    healthy: bool,
}

impl RdmaTier {
    /// 2026-09-25: Connect to the peer at `addr`, read its manifest, allocate
    /// the arena with the manifest's record stride, and bring up the transport.
    pub fn connect(
        addr: &str,
        num_slabs: u32,
        slots_per_slab: u32,
        use_verbs: bool,
    ) -> Result<Self> {
        let mut stream =
            TcpStream::connect(addr).with_context(|| format!("connect expert peer {addr}"))?;
        stream.set_nodelay(true).ok();
        let index = read_manifest(&mut stream)?;
        let spec = index.spec();
        let layout = index.layout();
        let arena = ExpertArena::new(num_slabs, slots_per_slab, layout.record_stride as usize)?;

        let transport = if use_verbs {
            #[cfg(metrale_rdma_verbs)]
            {
                connect_verbs(&mut stream, &arena, index.num_moe_layers)?
            }
            // 2026-09-25: Without `cfg(metrale_rdma_verbs)` only TCP exists.
            #[cfg(not(metrale_rdma_verbs))]
            {
                let _ = &arena;
                bail!(
                    "--expert-backend rdma-verbs needs a build with rdma-core \
                     (metrale_rdma_verbs cfg); use --expert-backend rdma (TCP) instead"
                );
            }
        } else {
            stream
                .write_all(&[MODE_TCP])
                .context("send TCP transport mode")?;
            Transport::Tcp
        };

        let label = match &transport {
            Transport::Tcp => "TCP".to_string(),
            #[cfg(metrale_rdma_verbs)]
            Transport::Verbs(rails) => {
                format!("verbs (one-sided RDMA READ, {} rail(s))", rails.len())
            }
        };
        tracing::info!(
            "RdmaTier[{label}] connected to {addr}: {} layers, {} experts, stride {}",
            index.num_moe_layers,
            index.num_experts,
            layout.record_stride
        );
        Ok(Self {
            stream,
            arena,
            spec,
            layout,
            transport,
            healthy: true,
        })
    }

    pub fn arena(&self) -> &ExpertArena {
        &self.arena
    }

    /// 2026-09-25: TCP fetch: send the request, read a status byte, then read
    /// `stride` bytes straight into the pinned slot.
    fn fetch_tcp(&mut self, key: ExpertKey, host: *mut u8, stride: usize) -> Result<()> {
        if let Err(e) = self
            .stream
            .write_all(&encode_request(key.layer, key.expert))
        {
            self.healthy = false;
            return Err(e).with_context(|| format!("peer request {:?}", key));
        }
        let mut status = [0u8; 1];
        if let Err(e) = self.stream.read_exact(&mut status) {
            self.healthy = false;
            return Err(e).with_context(|| format!("peer status {:?}", key));
        }
        if status[0] != STATUS_OK {
            bail!("peer returned error status {} for {:?}", status[0], key);
        }
        // 2026-09-25: SAFETY: `host` points at a `stride`-byte slot inside the
        // pinned arena.
        let dst = unsafe { std::slice::from_raw_parts_mut(host, stride) };
        if let Err(e) = self.stream.read_exact(dst) {
            self.healthy = false;
            return Err(e).with_context(|| format!("peer payload {:?}", key));
        }
        Ok(())
    }
}

/// 2026-09-25: Bring up the verbs transport through `RailSet`: one rail, or two
/// with `METRALE_EXPERT_DUAL_RAIL=1`; register the whole arena on each; check
/// the peer's per-rail layer tables against the manifest; connect the rails and
/// wait for the peer's ack.
#[cfg(metrale_rdma_verbs)]
fn connect_verbs(
    stream: &mut TcpStream,
    arena: &ExpertArena,
    num_layers: u32,
) -> Result<Transport> {
    use crate::expert_peer::MODE_VERBS;
    use metrale_gpu_sys::env::{first_set, first_set_u32};
    use metrale_gpu_sys::railset::{RailSet, RailSpec};

    stream
        .write_all(&[MODE_VERBS])
        .context("send verbs transport mode")?;

    // 2026-09-25: Each rail gets a random 24-bit PSN.
    let spec = |dev: String, gid: u32| RailSpec::new(dev, gid, rand::random::<u32>() & 0xff_ffff);
    let rail0 = spec(
        first_set(&["METRALE_EXPERT_RDMA_DEV"], "roceP2p1s0f1"),
        first_set_u32(&["METRALE_EXPERT_RDMA_GID"], 3),
    );
    let dual = std::env::var("METRALE_EXPERT_DUAL_RAIL").ok().as_deref() == Some("1");
    let specs: Vec<RailSpec> = if dual {
        let rail1 = spec(
            first_set(&["METRALE_EXPERT_RAIL2_DEV"], "rocep1s0f1"),
            first_set_u32(&["METRALE_EXPERT_RAIL2_GID"], 3),
        );
        vec![rail0, rail1]
    } else {
        vec![rail0]
    };

    // 2026-09-25: Each rail registers the whole arena with `remote_read = false`.
    let mut rs = RailSet::begin(stream, &specs)?;
    let mut arena_lkeys: Vec<u32> = Vec::with_capacity(rs.n_rails());
    for rail in &mut rs.rails {
        // 2026-09-25: SAFETY: the arena outlives every registration (see
        // `RdmaTier::transport`); base_ptr()/total_bytes() describe exactly its
        // allocation.
        let keys = unsafe {
            rail.verbs
                .reg_mr(arena.base_ptr(), arena.total_bytes(), false)?
        };
        arena_lkeys.push(keys.lkey);
    }

    // 2026-09-25: Each rail's layer table must match the manifest before any
    // client parameters are sent.
    let server = rs
        .read_server_ro(stream)
        .context("read verbs server params")?;
    for sp in &server {
        if sp.layers.len() != num_layers as usize {
            bail!(
                "verbs peer published {} layer MRs but manifest has {num_layers} MoE layers",
                sp.layers.len()
            );
        }
    }

    rs.complete(stream, &server, "verbs peer")?;
    let rails: Vec<Rail> = rs
        .into_verbs()
        .into_iter()
        .zip(arena_lkeys)
        .zip(server)
        .map(|((verbs, arena_lkey), sp)| Rail {
            verbs,
            arena_lkey,
            layers: sp.layers,
        })
        .collect();
    Ok(Transport::Verbs(rails))
}

impl ExpertTier for RdmaTier {
    fn fetch(&mut self, key: ExpertKey, slot: ArenaSlot, _stream: u64) -> Result<ExpertResidency> {
        let stride = self.layout.record_stride as usize;
        let host = self.arena.slot_host_ptr(slot.slab, slot.slot)?;
        let dev_va = self.arena.slot_dev_va(slot.slab, slot.slot)?;
        // 2026-09-25: A copy, taken before `self.transport` is borrowed mutably.
        let spec = self.spec;

        match &mut self.transport {
            Transport::Tcp => {
                self.fetch_tcp(key, host, stride)?;
            }
            #[cfg(metrale_rdma_verbs)]
            Transport::Verbs(rails) => {
                let ri = (key.expert as usize) % rails.len();
                let rail = &mut rails[ri];
                let (base, rkey) = *rail.layers.get(key.layer as usize).with_context(|| {
                    format!("verbs: no layer MR for layer {} ({:?})", key.layer, key)
                })?;
                let remote_addr = base + (key.expert as u64) * (stride as u64);
                let wr_id = ((key.layer as u64) << 32) | (key.expert as u64);
                // 2026-09-25: SAFETY: `host` is a `stride`-byte slot inside the
                // arena registered under `arena_lkey`; remote_addr/rkey address
                // the peer's layer on the same rail.
                let post = unsafe {
                    rail.verbs.post_read(
                        host as *mut std::ffi::c_void,
                        rail.arena_lkey,
                        remote_addr,
                        rkey,
                        stride as u32,
                        wr_id,
                    )
                };
                if let Err(e) = post {
                    self.healthy = false;
                    return Err(e).with_context(|| format!("verbs post_read {:?}", key));
                }
                match rail.verbs.poll() {
                    Ok(got) if got == wr_id => {}
                    Ok(got) => {
                        self.healthy = false;
                        bail!("verbs completion wr_id {got:#x} != expected {wr_id:#x} ({key:?})");
                    }
                    Err(e) => {
                        self.healthy = false;
                        return Err(e).with_context(|| format!("verbs poll {:?}", key));
                    }
                }
            }
        }

        // 2026-09-25: SAFETY: the slot now holds the `stride` bytes fetched above.
        let record = unsafe { std::slice::from_raw_parts(host, stride) };
        residency_from(&spec, record, dev_va, key)
    }

    fn kind(&self) -> TierKind {
        TierKind::Rdma
    }

    /// 2026-09-25: `false` after a failed send, receive, post or poll, or an
    /// unexpected completion; a peer error status leaves it `true`.
    fn healthy(&self) -> bool {
        self.healthy
    }
}
