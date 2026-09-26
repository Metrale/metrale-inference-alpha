// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The paging control protocol on the TCP stream between a paging client
//! and the peer: the handshake header, the request loop, and the client calls.
//!
//! Owner: storage, SSM snapshot tier.
//! Invariants:
//! - A connection holds at most one read pin, taken on a GET hit. It is released at
//!   the connection's next request, or when the loop ends on BYE or hangup; a failed
//!   key read or reply write returns without releasing it.
//!
//! Every client first sends `[u64 PAGING_MAGIC_V2][u8 kind][u64 arena_bytes]
//! [u64 blob_bytes]` on the stream the RDMA handshake then uses. `blob_bytes == 0`
//! selects raw mode (a private arena, placement by the client, no residency); any
//! other value a paging arena. After the rail handshake the client sends
//! `[op][u64 key]` and the peer replies `[status]`, plus a `[u64 offset]` for ALLOC
//! and a GET hit. Blobs move over one-sided RDMA to and from that offset; only
//! control messages cross TCP. The peer and client halves live here together, and
//! `wire_tests.rs` pins the byte layout.

use std::io::{Read, Write};

use anyhow::{Context, Result, bail};

use super::{Residency, SlotArena, SwapStore};

/// 2026-09-25: ASCII "PAGE" + 1, recognised only to be refused with its own error.
const PAGING_MAGIC_V1_RETIRED: u64 = 0x5041_4745_0000_0001;

/// 2026-09-25: The only accepted first u64 on the paging port, ASCII "PAGE" + 2. The
/// `[u8 kind]` that follows lets one peer keep an arena per `(kind, blob_bytes)`.
pub const PAGING_MAGIC_V2: u64 = 0x5041_4745_0000_0002;

/// 2026-09-25: The tier a paging arena serves. Only `SSM` and `KV` are accepted
/// (`parse_paging_header`); the read-only expert and weight peers use their own
/// handshake.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PagingKind(pub u8);
impl PagingKind {
    pub const SSM: PagingKind = PagingKind(0);
    pub const KV: PagingKind = PagingKind(1);
    /// 2026-09-25: `true` for `SSM` and `KV`.
    pub fn is_paging_rw(self) -> bool {
        self.0 <= 1
    }
}

/// 2026-09-25: Parse the rest of the handshake header, given the first u64 the caller
/// has read: `[u8 kind][u64 arena_bytes][u64 blob_bytes]`. A first u64 other than
/// [`PAGING_MAGIC_V2`] (with its own message for the "PAGE" + 1 value), or a kind
/// other than `SSM`/`KV`, is an error.
pub fn parse_paging_header<R: Read>(first: u64, r: &mut R) -> Result<(PagingKind, u64, u64)> {
    if first == PAGING_MAGIC_V1_RETIRED {
        bail!(
            "paging: v1 client no longer supported (magic {first:#x}); rebuild the client — \
             every connect now sends [u64 PAGING_MAGIC_V2][u8 kind][u64 arena_bytes][u64 blob_bytes]"
        );
    }
    if first != PAGING_MAGIC_V2 {
        bail!(
            "paging: first u64 {first:#x} is not PAGING_MAGIC_V2 (the bare legacy total_bytes \
             handshake was retired; RAW one-sided clients send the v2 header with \
             blob_bytes == 0)"
        );
    }
    let mut kb = [0u8; 1];
    r.read_exact(&mut kb).context("read paging kind")?;
    let kind = PagingKind(kb[0]);
    if !kind.is_paging_rw() {
        bail!(
            "paging: unsupported kind {} (only SSM/KV ride this handshake)",
            kb[0]
        );
    }
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b8).context("read paging arena_bytes")?;
    let arena_bytes = u64::from_le_bytes(b8);
    r.read_exact(&mut b8).context("read paging blob_bytes")?;
    let blob_bytes = u64::from_le_bytes(b8);
    Ok((kind, arena_bytes, blob_bytes))
}

/// 2026-09-25: The 25-byte header every client sends first, in paging and raw mode:
/// `[u64 PAGING_MAGIC_V2 LE][u8 kind][u64 arena_bytes LE][u64 blob_bytes LE]`. The
/// `RailSet` exchange follows it.
pub fn encode_paging_v2_header(kind: PagingKind, arena_bytes: u64, blob_bytes: u64) -> [u8; 25] {
    let mut w = [0u8; 25];
    w[0..8].copy_from_slice(&PAGING_MAGIC_V2.to_le_bytes());
    w[8] = kind.0;
    w[9..17].copy_from_slice(&arena_bytes.to_le_bytes());
    w[17..25].copy_from_slice(&blob_bytes.to_le_bytes());
    w
}

/// 2026-09-25: Split a `blob_bytes` transfer into `chunk_bytes` chunks dealt
/// round-robin to `n_rails`, as per-rail lists of `(offset, len)`. The offset is the
/// chunk's position in both the staging buffer and the peer slot, so one copy
/// assembles the blob whichever rail moved each chunk. Only the last chunk may be
/// short. Zero `chunk_bytes` or `n_rails` counts as 1.
pub fn stripe_plan(
    blob_bytes: usize,
    chunk_bytes: usize,
    n_rails: usize,
) -> Vec<Vec<(usize, usize)>> {
    let n = n_rails.max(1);
    let cb = chunk_bytes.max(1);
    let mut rails: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut off = 0usize;
    let mut j = 0usize;
    while off < blob_bytes {
        let len = cb.min(blob_bytes - off);
        rails[j % n].push((off, len));
        off += len;
        j += 1;
    }
    rails
}

/// 2026-09-25: Chunk size of the striped snapshot transfer: `METRALE_SSM_CHUNK_BYTES`,
/// or 1 MiB when it is unset, unparsable or below 4096.
pub fn staging_chunk_bytes() -> usize {
    std::env::var("METRALE_SSM_CHUNK_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= 4096)
        .unwrap_or(1024 * 1024)
}
/// 2026-09-25: Chunks in flight per rail: `METRALE_SSM_PIPELINE_DEPTH`, or 16 when it
/// is unset or unparsable, clamped to 1..=128.
pub fn staging_depth() -> usize {
    std::env::var("METRALE_SSM_PIPELINE_DEPTH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(16)
        .clamp(1, 128)
}

pub const OP_BYE: u8 = 0;
pub const OP_ALLOC: u8 = 1;
pub const OP_COMMIT: u8 = 2;
pub const OP_GET: u8 = 3;
pub const OP_REMOVE: u8 = 4;

pub const ST_OK: u8 = 0;
pub const ST_MISS: u8 = 1;
pub const ST_ERR: u8 = 2;

/// 2026-09-25: Result of one control request, ready to serialise.
#[derive(Debug, PartialEq, Eq)]
pub enum PagingReply {
    /// 2026-09-25: `ST_OK` and a u64 arena offset (ALLOC, and a GET hit).
    Located(u64),
    /// 2026-09-25: `ST_OK` alone (COMMIT, REMOVE).
    Ok,
    /// 2026-09-25: `ST_MISS`: GET of an unknown key.
    Miss,
    /// 2026-09-25: `ST_ERR`: the operation failed, or the op is unknown.
    Err,
    /// 2026-09-25: The client asked to close; nothing is written.
    Bye,
}

/// 2026-09-25: Execute one control op against `res` and return the reply; no I/O, so
/// the protocol is testable without a socket or RDMA.
pub fn dispatch<A: SlotArena, S: SwapStore>(
    res: &mut Residency<A, S>,
    op: u8,
    key: u64,
) -> PagingReply {
    match op {
        OP_BYE => PagingReply::Bye,
        OP_ALLOC => match res.alloc(key) {
            Ok(slot) => PagingReply::Located(res.slot_offset(slot)),
            Err(e) => {
                tracing::warn!("paging ALLOC {key:#x} failed: {e:#}");
                PagingReply::Err
            }
        },
        OP_COMMIT => match res.commit(key) {
            Ok(()) => PagingReply::Ok,
            Err(e) => {
                tracing::warn!("paging COMMIT {key:#x} failed: {e:#}");
                PagingReply::Err
            }
        },
        OP_GET => match res.locate(key) {
            Ok(Some(slot)) => PagingReply::Located(res.slot_offset(slot)),
            Ok(None) => PagingReply::Miss,
            Err(e) => {
                tracing::warn!("paging GET {key:#x} failed: {e:#}");
                PagingReply::Err
            }
        },
        OP_REMOVE => {
            res.remove(key);
            PagingReply::Ok
        }
        other => {
            tracing::warn!("paging: unknown op {other}");
            PagingReply::Err
        }
    }
}

fn write_reply<W: Write>(w: &mut W, reply: &PagingReply) -> Result<()> {
    match reply {
        PagingReply::Located(off) => {
            w.write_all(&[ST_OK])?;
            w.write_all(&off.to_le_bytes())?;
        }
        PagingReply::Ok => w.write_all(&[ST_OK])?,
        PagingReply::Miss => w.write_all(&[ST_MISS])?,
        PagingReply::Err => w.write_all(&[ST_ERR])?,
        PagingReply::Bye => {}
    }
    w.flush()?;
    Ok(())
}

/// 2026-09-25: One control op with the connection's read pin: release the pin of the
/// previous GET, dispatch, and pin a GET hit so an ALLOC on another connection cannot
/// evict the slot while this client reads it. Releasing on the next op is sound only
/// for clients that finish each READ before sending another op, as
/// `RdmaSnapshotArena` and `KvPagingBackend` do.
fn handle_paging_op<A: SlotArena, S: SwapStore>(
    res: &mut Residency<A, S>,
    op: u8,
    key: u64,
    pinned: &mut Option<u64>,
) -> PagingReply {
    if let Some(prev) = pinned.take() {
        res.unpin_read(prev);
    }
    let reply = dispatch(res, op, key);
    if op == OP_GET && matches!(reply, PagingReply::Located(_)) {
        res.pin_read(key);
        *pinned = Some(key);
    }
    reply
}

/// 2026-09-25: The peer's control loop over one residency: read `[op][u64 key]`,
/// dispatch, reply, until BYE or a failed read of the op byte.
pub fn run_paging_loop<T: Read + Write, A: SlotArena, S: SwapStore>(
    stream: &mut T,
    res: &mut Residency<A, S>,
) -> Result<()> {
    let mut pinned: Option<u64> = None;
    loop {
        let mut op = [0u8; 1];
        if stream.read_exact(&mut op).is_err() {
            break;
        }
        if op[0] == OP_BYE {
            break;
        }
        let mut kb = [0u8; 8];
        stream.read_exact(&mut kb).context("read paging key")?;
        let key = u64::from_le_bytes(kb);
        let reply = handle_paging_op(res, op[0], key, &mut pinned);
        write_reply(stream, &reply)?;
    }
    if let Some(pk) = pinned {
        res.unpin_read(pk);
    }
    Ok(())
}

// 2026-09-25: Client side. Each call sends `[op][u64 key]` and reads the reply. A PUT
// is `client_alloc`, the RDMA WRITE, then `client_commit`; a GET is `client_get`,
// then the RDMA READ.

fn send_req<T: Write>(s: &mut T, op: u8, key: u64) -> Result<()> {
    let mut buf = [0u8; 9];
    buf[0] = op;
    buf[1..].copy_from_slice(&key.to_le_bytes());
    s.write_all(&buf)?;
    s.flush()?;
    Ok(())
}

fn read_status<T: Read>(s: &mut T) -> Result<u8> {
    let mut st = [0u8; 1];
    s.read_exact(&mut st).context("read paging status")?;
    Ok(st[0])
}

fn read_offset<T: Read>(s: &mut T) -> Result<u64> {
    let mut b = [0u8; 8];
    s.read_exact(&mut b).context("read paging offset")?;
    Ok(u64::from_le_bytes(b))
}

/// 2026-09-25: Reserve a slot for `key` and return its arena offset; a non-OK status
/// is an error.
pub fn client_alloc<T: Read + Write>(s: &mut T, key: u64) -> Result<u64> {
    send_req(s, OP_ALLOC, key)?;
    match read_status(s)? {
        ST_OK => read_offset(s),
        st => bail!("paging ALLOC {key:#x} refused (status {st})"),
    }
}

/// 2026-09-25: After the WRITE has completed, mark `key` resident.
pub fn client_commit<T: Read + Write>(s: &mut T, key: u64) -> Result<()> {
    send_req(s, OP_COMMIT, key)?;
    match read_status(s)? {
        ST_OK => Ok(()),
        st => bail!("paging COMMIT {key:#x} failed (status {st})"),
    }
}

/// 2026-09-25: `Some(offset)` to READ from, or `None` when the peer has no such key.
pub fn client_get<T: Read + Write>(s: &mut T, key: u64) -> Result<Option<u64>> {
    send_req(s, OP_GET, key)?;
    match read_status(s)? {
        ST_OK => Ok(Some(read_offset(s)?)),
        ST_MISS => Ok(None),
        st => bail!("paging GET {key:#x} error (status {st})"),
    }
}

/// 2026-09-25: Ask the peer to drop `key`.
pub fn client_remove<T: Read + Write>(s: &mut T, key: u64) -> Result<()> {
    send_req(s, OP_REMOVE, key)?;
    match read_status(s)? {
        ST_OK => Ok(()),
        st => bail!("paging REMOVE {key:#x} failed (status {st})"),
    }
}

/// 2026-09-25: Tell the peer to end the paging loop; no reply is read.
pub fn client_bye<T: Write>(s: &mut T) -> Result<()> {
    send_req(s, OP_BYE, 0)
}

/// 2026-09-25: [`run_paging_loop`] over a residency shared by many connections, so a
/// blob one client PUTs another can GET under the same key. The lock is held for
/// each dispatch, which may move bytes to or from swap, and never across a TCP read
/// or write.
pub fn run_paging_loop_shared<T: Read + Write, A: SlotArena, S: SwapStore>(
    stream: &mut T,
    res: &std::sync::Mutex<Residency<A, S>>,
) -> Result<()> {
    // 2026-09-25: This connection's read pin (`handle_paging_op`), taken under the
    // same lock as the dispatch.
    let mut pinned: Option<u64> = None;
    loop {
        let mut op = [0u8; 1];
        if stream.read_exact(&mut op).is_err() {
            break;
        }
        if op[0] == OP_BYE {
            break;
        }
        let mut kb = [0u8; 8];
        stream.read_exact(&mut kb).context("read paging key")?;
        let key = u64::from_le_bytes(kb);
        let reply = {
            let mut g = res.lock().expect("shared residency mutex poisoned");
            handle_paging_op(&mut g, op[0], key, &mut pinned)
        };
        write_reply(stream, &reply)?;
    }
    if let Some(pk) = pinned {
        res.lock()
            .expect("shared residency mutex poisoned")
            .unpin_read(pk);
    }
    Ok(())
}

#[cfg(test)]
#[path = "wire_tests.rs"]
mod tests;
