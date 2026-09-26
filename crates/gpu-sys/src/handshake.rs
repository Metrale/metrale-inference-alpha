// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The client side of the rail handshake as byte functions over
//! `Read`/`Write` that take QP identities as `(qpn, psn, gid)` tuples, so they
//! need no `Verbs`. `railset::RailSet` calls them; tests/transcript_golden.rs
//! drives them against a scripted peer.
//!
//! Wire order (the params' integers are little-endian, see `wire`):
//! - client: `[u8 n_rails]` (`write_n_rails`);
//! - server, RO dialect: `[u8 n]`, then n `VerbsServerParams`
//!   (`wire::read_server_rails`); RW dialect: `[u8 n]`, then n
//!   `CacheServerParams` (`read_rw_server_params`);
//! - client: `[u8 n_rails]` a second time, then n `VerbsClientParams`
//!   (`write_client_params`);
//! - server: `[u8 ack]`, which must be `STATUS_OK` (`read_ack`).
//!
//! Owner: metrale-gpu-sys.
//! Invariants: none beyond the types.

use anyhow::{Context, Result, bail};
use std::io::{Read, Write};

use crate::wire::{CacheServerParams, STATUS_OK, VerbsClientParams};

/// 2026-09-26: Write the rail count as one byte.
pub fn write_n_rails<W: Write>(w: &mut W, n: usize) -> Result<()> {
    w.write_all(&[n as u8]).context("send n_rails")?;
    Ok(())
}

/// 2026-09-26: RW dialect (the KV and snapshot tiers): read the peer's
/// `[u8 n]`, which must equal `want`, then `want` `CacheServerParams`. Unlike
/// `wire::read_server_rails`, the count is not limited to 1..=8.
pub fn read_rw_server_params<R: Read>(
    r: &mut R,
    want: usize,
    peer: &str,
) -> Result<Vec<CacheServerParams>> {
    let mut b1 = [0u8; 1];
    if let Err(e) = r.read_exact(&mut b1) {
        // 2026-09-26: A clean EOF here means the peer refused the client: the
        // cache peer logs the reason (e.g. "paging blade cap") and closes, and
        // the wire has no error frame to carry it.
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            bail!(
                "{peer} closed the connection during the rail handshake without sending rail \
                 params. The peer REJECTED this client — read ITS log for the reason. Most \
                 common: the arena this client requested exceeds the peer's --max-blade-gb \
                 cap (peer logs \"paging blade cap\"); also possible: a blob_bytes/kind \
                 mismatch against an arena a prior client already fixed."
            );
        }
        return Err(e).context("read peer n_rails");
    }
    if b1[0] as usize != want {
        bail!("{peer} granted {} rails, wanted {want}", b1[0]);
    }
    let mut server = Vec::with_capacity(want);
    for _ in 0..want {
        server
            .push(CacheServerParams::read_from(r).with_context(|| format!("read {peer} params"))?);
    }
    Ok(server)
}

/// 2026-09-26: Write the rail count again, then each rail's
/// `VerbsClientParams`.
pub fn write_client_params<W: Write>(w: &mut W, ids: &[(u32, u32, [u8; 16])]) -> Result<()> {
    w.write_all(&[ids.len() as u8])
        .context("send client n_rails")?;
    for &(qpn, psn, gid) in ids {
        VerbsClientParams { qpn, psn, gid }
            .write_to(w)
            .context("send verbs client params")?;
    }
    Ok(())
}

/// 2026-09-26: Read the peer's one-byte ack, which must be `STATUS_OK`.
pub fn read_ack<R: Read>(r: &mut R, peer: &str) -> Result<()> {
    let mut ack = [0u8; 1];
    r.read_exact(&mut ack).context("read verbs ready ack")?;
    if ack[0] != STATUS_OK {
        bail!("{peer} refused connection (ack {})", ack[0]);
    }
    Ok(())
}
