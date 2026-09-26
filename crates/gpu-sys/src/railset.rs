// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `RailSet`, the client-side bring-up of a connection's RDMA
//! rails, shared by the expert, weight, LoRA, KV and snapshot tiers.
//!
//! The caller writes its own preamble to the stream first, and keeps the
//! stream: every method borrows it. `RailSet` has no registration method:
//! callers register memory on `rail.verbs` themselves, with their own access
//! flags. The PSN comes from the caller in `RailSpec`.
//!
//! Order: `begin` (writes the rail count, creates one QP per rail); the caller
//! registers its buffers; `read_server_ro` or `read_server_rw`; an RO caller
//! checks the server table against its manifest before anything is written
//! back; `complete` (writes the client params, connects each rail
//! INIT -> RTR -> RTS, reads the ack); `into_verbs`.
//!
//! Owner: metrale-gpu-sys.
//! Invariants: none beyond the types.

use anyhow::{Context, Result};
use std::io::{Read, Write};

use crate::handshake;
use crate::rail_count::check_rail_count;
use crate::verbs::Verbs;
use crate::wire::{CacheServerParams, RemoteQp, VerbsServerParams, read_server_rails};

/// 2026-09-26: One rail: device name, GID index and the QP's send PSN. The
/// tiers pass a random 24-bit PSN (`rand::random::<u32>() & 0xff_ffff`).
#[derive(Clone, Debug)]
pub struct RailSpec {
    pub dev: String,
    pub gid_idx: u32,
    pub psn: u32,
}

impl RailSpec {
    pub fn new(dev: String, gid_idx: u32, psn: u32) -> Self {
        Self { dev, gid_idx, psn }
    }
}

/// 2026-09-26: One rail. `verbs` is public: registration and the data plane
/// (`post_read`, `post_write`, `poll`) belong to the caller.
pub struct Rail {
    pub verbs: Verbs,
}

/// 2026-09-26: The rails of one client connection.
pub struct RailSet {
    pub rails: Vec<Rail>,
}

impl RailSet {
    /// 2026-09-26: Write `[u8 n_rails]` to the stream, then `Verbs::create`
    /// one RC QP per spec.
    pub fn begin<W: Write>(stream: &mut W, specs: &[RailSpec]) -> Result<Self> {
        handshake::write_n_rails(stream, specs.len())?;
        let rails = specs
            .iter()
            .map(|s| Verbs::create(&s.dev, s.gid_idx, s.psn).map(|verbs| Rail { verbs }))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { rails })
    }

    pub fn n_rails(&self) -> usize {
        self.rails.len()
    }

    /// 2026-09-26: RO dialect (expert, weight and LoRA tiers): `[u8 n]`, which
    /// must equal the rail count and lie in 1..=8, then n `VerbsServerParams`.
    /// The caller can check them before `complete` writes anything back.
    pub fn read_server_ro<R: Read>(&self, stream: &mut R) -> Result<Vec<VerbsServerParams>> {
        read_server_rails(stream, self.rails.len())
    }

    /// 2026-09-26: RW dialect (KV and snapshot tiers): `[u8 n]`, then n
    /// `CacheServerParams`.
    pub fn read_server_rw<R: Read>(
        &self,
        stream: &mut R,
        peer: &str,
    ) -> Result<Vec<CacheServerParams>> {
        handshake::read_rw_server_params(stream, self.rails.len(), peer)
    }

    /// 2026-09-26: Refuse a server param count that differs from the rail
    /// count, write `[u8 n_rails]` again and each rail's client params, connect
    /// each rail to its server rail (INIT -> RTR -> RTS), then read the ack.
    pub fn complete<S: Read + Write, P: RemoteQp>(
        &mut self,
        stream: &mut S,
        server: &[P],
        peer: &str,
    ) -> Result<()> {
        check_rail_count(server.len(), self.rails.len(), peer)?;
        let ids: Vec<(u32, u32, [u8; 16])> = self
            .rails
            .iter()
            .map(|r| (r.verbs.qpn(), r.verbs.psn(), r.verbs.gid()))
            .collect();
        handshake::write_client_params(stream, &ids)?;
        for (rail, sp) in self.rails.iter_mut().zip(server) {
            let (qpn, psn, gid) = sp.qp_identity();
            rail.verbs
                .connect(qpn, psn, &gid)
                .with_context(|| format!("connect {peer} rail"))?;
        }
        handshake::read_ack(stream, peer)
    }

    /// 2026-09-26: `read_server_rw` then `complete`; returns the server params.
    pub fn finish_rw<S: Read + Write>(
        &mut self,
        stream: &mut S,
        peer: &str,
    ) -> Result<Vec<CacheServerParams>> {
        let server = self.read_server_rw(stream, peer)?;
        self.complete(stream, &server, peer)?;
        Ok(server)
    }

    /// 2026-09-26: The rails' `Verbs`, in rail order, for the caller to own.
    pub fn into_verbs(self) -> Vec<Verbs> {
        self.rails.into_iter().map(|r| r.verbs).collect()
    }
}
