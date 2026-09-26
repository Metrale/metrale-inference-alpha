// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: RDMA handshake wire codecs, used by the RDMA clients and
//! re-exported by the storage peer daemons, so both ends share one codec.
//!
//! Owner: metrale-gpu-sys.
//! Invariants:
//! - Every integer is little-endian.
//! - The byte layouts are pinned by `tests/wire_roundtrip.rs` and
//!   `tests/transcript_golden.rs`, the constant values by `frozen_wire_constants`.
//! - Only `std::io` and `anyhow`, with no `cfg` gate, so it builds without rdma-core.

use anyhow::{Context, Result, bail};

pub const STATUS_OK: u8 = 0;
pub const STATUS_ERR: u8 = 1;
/// 2026-09-26: Transport byte for two-sided TCP record streaming from the expert peer.
pub const MODE_TCP: u8 = 0;
/// 2026-09-26: Transport byte for one-sided RDMA READ: the server sends its
/// MRs as `VerbsServerParams` and the client reads records from them.
pub const MODE_VERBS: u8 = 1;

/// 2026-09-26: The server's half of the read-only verbs handshake (expert,
/// weight and LoRA tiers): its QP identity plus one registered MR per MoE layer
/// (expert tier) or per shard (weight and LoRA tiers).
/// The expert tier reads at `layers[layer].0 + expert * record_stride`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerbsServerParams {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
    /// 2026-09-26: `(mr_base_addr, rkey)` per MoE layer or per shard, by index.
    pub layers: Vec<(u64, u32)>,
}

/// 2026-09-26: The client's half of the verbs handshake: its QP identity only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerbsClientParams {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
}

/// 2026-09-26: The peer's half of the read-write handshake (KV paging and SSM
/// snapshots): its QP identity plus the base address and rkey of its RW MR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheServerParams {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
    pub base_addr: u64,
    pub rkey: u32,
}

/// 2026-09-26: A remote QP identity a client rail can `connect` to. Both
/// server-param types implement it, so `RailSet::complete` serves both dialects.
pub trait RemoteQp {
    fn qp_identity(&self) -> (u32, u32, [u8; 16]);
}

impl RemoteQp for VerbsServerParams {
    fn qp_identity(&self) -> (u32, u32, [u8; 16]) {
        (self.qpn, self.psn, self.gid)
    }
}

impl RemoteQp for CacheServerParams {
    fn qp_identity(&self) -> (u32, u32, [u8; 16]) {
        (self.qpn, self.psn, self.gid)
    }
}

impl VerbsServerParams {
    /// 2026-09-26: Wire form: `[u32 qpn][u32 psn][16 gid][u32 n_layers]{[u64 base][u32 rkey]}*`.
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.qpn.to_le_bytes())?;
        w.write_all(&self.psn.to_le_bytes())?;
        w.write_all(&self.gid)?;
        w.write_all(&(self.layers.len() as u32).to_le_bytes())?;
        for (base, rkey) in &self.layers {
            w.write_all(&base.to_le_bytes())?;
            w.write_all(&rkey.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn read_from<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let qpn = read_u32(r)?;
        let psn = read_u32(r)?;
        let mut gid = [0u8; 16];
        r.read_exact(&mut gid).context("read server gid")?;
        let n = read_u32(r)? as usize;
        if n == 0 || n > 4096 {
            bail!("implausible verbs layer count: {n}");
        }
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            let mut b8 = [0u8; 8];
            r.read_exact(&mut b8).context("read mr base")?;
            let base = u64::from_le_bytes(b8);
            let rkey = read_u32(r)?;
            layers.push((base, rkey));
        }
        Ok(Self {
            qpn,
            psn,
            gid,
            layers,
        })
    }
}

impl VerbsClientParams {
    /// 2026-09-26: Wire form: `[u32 qpn][u32 psn][16 gid]`.
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.qpn.to_le_bytes())?;
        w.write_all(&self.psn.to_le_bytes())?;
        w.write_all(&self.gid)?;
        Ok(())
    }

    pub fn read_from<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let qpn = read_u32(r)?;
        let psn = read_u32(r)?;
        let mut gid = [0u8; 16];
        r.read_exact(&mut gid).context("read client gid")?;
        Ok(Self { qpn, psn, gid })
    }
}

impl CacheServerParams {
    /// 2026-09-26: Wire form: `[u32 qpn][u32 psn][16 gid][u64 base_addr][u32 rkey]`.
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.qpn.to_le_bytes())?;
        w.write_all(&self.psn.to_le_bytes())?;
        w.write_all(&self.gid)?;
        w.write_all(&self.base_addr.to_le_bytes())?;
        w.write_all(&self.rkey.to_le_bytes())?;
        Ok(())
    }

    pub fn read_from<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let mut b4 = [0u8; 4];
        let mut b8 = [0u8; 8];
        let mut gid = [0u8; 16];
        r.read_exact(&mut b4).context("kv qpn")?;
        let qpn = u32::from_le_bytes(b4);
        r.read_exact(&mut b4).context("kv psn")?;
        let psn = u32::from_le_bytes(b4);
        r.read_exact(&mut gid).context("kv gid")?;
        r.read_exact(&mut b8).context("kv base")?;
        let base_addr = u64::from_le_bytes(b8);
        r.read_exact(&mut b4).context("kv rkey")?;
        let rkey = u32::from_le_bytes(b4);
        Ok(Self {
            qpn,
            psn,
            gid,
            base_addr,
            rkey,
        })
    }
}

fn read_u32<R: std::io::Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).context("read u32")?;
    Ok(u32::from_le_bytes(b))
}

/// 2026-09-26: Write `[u8 n_rails]` followed by each rail's `VerbsServerParams`.
/// Fails before writing anything when the count is 0 or above 8.
pub fn write_server_rails<W: std::io::Write>(w: &mut W, rails: &[VerbsServerParams]) -> Result<()> {
    if rails.is_empty() || rails.len() > 8 {
        bail!("implausible server rail count: {}", rails.len());
    }
    w.write_all(&[rails.len() as u8])?;
    for sp in rails {
        sp.write_to(w)?;
    }
    Ok(())
}

/// 2026-09-26: Read `want` rails of `VerbsServerParams` framed by a leading
/// `[u8 n_rails]`. Fails when the framed count is 0, above 8, or not `want`.
pub fn read_server_rails<R: std::io::Read>(
    r: &mut R,
    want: usize,
) -> Result<Vec<VerbsServerParams>> {
    let mut b1 = [0u8; 1];
    r.read_exact(&mut b1).context("read server rail count")?;
    let n = b1[0] as usize;
    if n == 0 || n > 8 {
        bail!("implausible server rail count: {n}");
    }
    if n != want {
        bail!("server framed {n} rails but client negotiated {want}");
    }
    let mut rails = Vec::with_capacity(n);
    for _ in 0..n {
        rails.push(VerbsServerParams::read_from(r)?);
    }
    Ok(rails)
}
