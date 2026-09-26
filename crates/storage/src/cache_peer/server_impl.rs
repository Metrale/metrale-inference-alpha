// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The cache peer's accept loop and connection handler: the v2
//! handshake, the server side of the rail handshake, and the two data planes
//! (the shared paging control loop, or RAW mode idling until hangup).
//!
//! Owner: metrale-storage peers.
//! Invariants:
//! - `handle_conn` holds the crate's only `reg_mr_rw` call (remote read and
//!   write access).
//! - A connection's memory registrations are dropped before the arena they
//!   cover, on every path.

use std::net::{TcpListener, TcpStream, ToSocketAddrs};

use anyhow::{Context, Result, bail};

/// 2026-09-25: The cache peer's settings. A client asks for up to
/// `rails.len()` rails, and the peer registers the arena on each.
#[derive(Clone, Debug)]
pub struct RdmaConfig {
    /// 2026-09-25: `(device, gid_idx)` per rail; a client asking for n rails
    /// gets the first n.
    pub rails: Vec<(String, u32)>,
    /// 2026-09-25: Ceiling on the arena bytes reserved across this `serve`'s
    /// connections; 0 means none.
    pub max_blade_bytes: u64,
    /// 2026-09-25: Directory of the paging arenas' O_DIRECT swap files. With
    /// `None`, paging clients are refused.
    pub swap_dir: Option<std::path::PathBuf>,
    /// 2026-09-25: Swap-disk budget shared by the paging arenas of kinds
    /// without a per-kind cap (`registry::carve_disk_slots`); 0 = unbounded.
    pub swap_cap_bytes: u64,
    /// 2026-09-25: Per-kind disk caps (`kind.0` → bytes). A listed kind gets
    /// this budget instead of a share of `swap_cap_bytes`; 0 = unbounded.
    pub per_kind_swap_cap_bytes: std::collections::HashMap<u8, u64>,
}

impl Default for RdmaConfig {
    fn default() -> Self {
        Self {
            rails: vec![("roceP2p1s0f1".into(), 3), ("rocep1s0f1".into(), 3)],
            max_blade_bytes: 0,
            swap_dir: None,
            swap_cap_bytes: 50 * 1024 * 1024 * 1024,
            per_kind_swap_cap_bytes: std::collections::HashMap::new(),
        }
    }
}

/// 2026-09-25: Serve on `addr`, one thread per connection. Accept errors are
/// logged and skipped; only a bind error returns.
pub fn serve<A: ToSocketAddrs>(addr: A, rdma: RdmaConfig) -> Result<()> {
    let listener = TcpListener::bind(addr).context("bind cache-peer listener")?;
    let local = listener.local_addr().ok();
    // 2026-09-25: One ledger per `serve`, shared by its connection threads.
    let ledger = std::sync::Arc::new(crate::blade_cap::CommitLedger::new(rdma.max_blade_bytes));
    tracing::info!(
        "cache-peer (RW RDMA overflow blade) listening on {:?} (rails {:?}, cap {})",
        local,
        rdma.rails,
        if rdma.max_blade_bytes == 0 {
            "unlimited".to_string()
        } else {
            format!(
                "{:.1} GiB",
                rdma.max_blade_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            )
        },
    );
    // 2026-09-25: Paging arenas are never freed, so without a ceiling every
    // new (kind, blob_bytes) adds RAM that stays mapped.
    if rdma.max_blade_bytes == 0 && rdma.swap_dir.is_some() {
        tracing::warn!(
            "cache-peer paging registry active with NO blade ceiling (--max-blade-gb 0 = \
             unlimited): each new (kind, shape) arena pins RDMA-registered RAM without bound. \
             Set --max-blade-gb <G> to cap total memlocked blade RAM."
        );
    }
    for conn in listener.incoming() {
        let stream = match conn {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("cache-peer accept error: {e}");
                continue;
            }
        };
        let rdma = rdma.clone();
        let ledger = ledger.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle_conn(stream, &rdma, &ledger) {
                tracing::warn!("cache-peer connection ended: {e}");
            }
        });
    }
    Ok(())
}

#[cfg(not(metrale_rdma_verbs))]
fn handle_conn(
    _stream: TcpStream,
    _rdma: &RdmaConfig,
    _ledger: &std::sync::Arc<crate::blade_cap::CommitLedger>,
) -> Result<()> {
    bail!("cache-peer needs a build with rdma-core (metrale_rdma_verbs)");
}

#[cfg(metrale_rdma_verbs)]
fn handle_conn(
    mut stream: TcpStream,
    rdma: &RdmaConfig,
    ledger: &std::sync::Arc<crate::blade_cap::CommitLedger>,
) -> Result<()> {
    use super::registry::{self, Mmap, SharedPaging};
    use metrale_gpu_sys::verbs::Verbs;
    use metrale_gpu_sys::wire::{CacheServerParams, STATUS_OK, VerbsClientParams};
    use std::io::{Read, Write};
    stream.set_nodelay(true).ok();

    // 2026-09-25: Every client sends
    // `[u64 PAGING_MAGIC_V2][u8 kind][u64 arena_bytes][u64 blob_bytes]`, and
    // `parse_paging_header` rejects any other first word. blob_bytes > 0
    // selects paging over the shared (kind, blob_bytes) arena; 0 selects RAW
    // mode, a per-connection arena that the client allocates in.
    let mut b8 = [0u8; 8];
    stream.read_exact(&mut b8).context("read paging magic")?;
    let first = u64::from_le_bytes(b8);
    let (kind, arena_bytes, blob) = crate::snapshot_swap::parse_paging_header(first, &mut stream)?;
    let total = arena_bytes as usize;
    let blob = blob as usize;
    // 2026-09-25: The ledger has no ceiling unless one is configured, so the
    // arena size is bounded here too.
    if total == 0 || total > (1usize << 42) {
        bail!("implausible blade arena size: {total}");
    }
    let paging: Option<(u8, usize)> = if blob == 0 {
        None
    } else {
        if !total.is_multiple_of(blob) {
            bail!("paging: arena_bytes {total} not a multiple of blob_bytes {blob}");
        }
        // 2026-09-25: Refused before the rail handshake.
        if rdma.swap_dir.is_none() {
            bail!("paging client but peer started without --swap-dir; refusing");
        }
        Some((kind.0, blob))
    };
    let mut b1 = [0u8; 1];
    stream.read_exact(&mut b1).context("read n_rails")?;
    let n_rails = b1[0] as usize;
    if n_rails == 0 || n_rails > rdma.rails.len() {
        bail!(
            "client asked for {n_rails} rails; peer has {}",
            rdma.rails.len()
        );
    }

    // 2026-09-25: RAW mode maps and charges an arena per connection. Paging
    // uses the shared arena of (kind, blob_bytes), charged once when it was
    // created, so every client's rails point at the same slots.
    let pid = std::process::id();
    let shared: Option<std::sync::Arc<SharedPaging>> = match paging {
        Some((kind, blob)) => Some(registry::get_or_init_shared_paging(
            rdma, kind, total, blob, ledger,
        )?),
        None => None,
    };
    // 2026-09-25: RAW mode only; the shared arena's reservation lives in the
    // registry.
    let local: Option<(crate::blade_cap::Reservation, Mmap)> = if shared.is_none() {
        let reservation = ledger.try_reserve(total as u64).context("kv blade cap")?;
        let arena = Mmap::anon(total).context("mmap kv blade arena")?;
        Some((reservation, arena))
    } else {
        None
    };
    let (arena_base, arena_len): (*mut libc::c_void, usize) = match (&shared, &local) {
        (Some(sh), _) => (sh.arena.addr, sh.arena.len),
        (None, Some((_, arena))) => (arena.addr, arena.len),
        _ => unreachable!("exactly one of shared/local is set"),
    };
    // 2026-09-25: Each rail registers the same arena: n rails make n
    // registrations of one memory range.
    let mut rails: Vec<Verbs> = Vec::with_capacity(n_rails);
    let mut rkeys: Vec<u32> = Vec::with_capacity(n_rails);
    for (i, (dev, gid)) in rdma.rails.iter().take(n_rails).enumerate() {
        let psn = (0x5a5a5a ^ pid ^ ((i as u32) << 20)) & 0xff_ffff;
        let mut v = Verbs::create(dev, *gid, psn)?;
        // 2026-09-25: SAFETY: the arena (shared or local) outlives every rail;
        // see the drops at the end.
        let keys = unsafe { v.reg_mr_rw(arena_base as *mut _, arena_len)? };
        rkeys.push(keys.rkey);
        rails.push(v);
    }

    stream.write_all(&[n_rails as u8]).context("send n_rails")?;
    for (v, rkey) in rails.iter().zip(&rkeys) {
        CacheServerParams {
            qpn: v.qpn(),
            psn: v.psn(),
            gid: v.gid(),
            base_addr: arena_base as u64,
            rkey: *rkey,
        }
        .write_to(&mut stream)
        .context("send kv server params")?;
    }

    stream.read_exact(&mut b1).context("read client n_rails")?;
    if b1[0] as usize != n_rails {
        bail!("client rail count mismatch");
    }
    for v in rails.iter_mut() {
        let cp = VerbsClientParams::read_from(&mut stream).context("read kv client params")?;
        v.connect(cp.qpn, cp.psn, &cp.gid)?;
    }
    stream
        .write_all(&[STATUS_OK])
        .context("send kv ready ack")?;
    let mode = if paging.is_some() {
        "paging"
    } else {
        "raw one-sided"
    };
    tracing::info!(
        "cache-peer client connected: kind {}, {n_rails} rail(s), {:.1} GiB RW blade ({mode})",
        kind.0,
        total as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    if let Some(sh) = shared {
        // 2026-09-25: Paging: control messages on this stream drive the
        // shared residency; record bytes move over RDMA through the
        // registrations made above.
        tracing::info!("cache-peer PAGING client joined shared arena ({n_rails} rail(s))");
        let r = crate::snapshot_swap::run_paging_loop_shared(&mut stream, &sh.residency);
        drop(rails);
        return r;
    }

    // 2026-09-25: RAW mode: the client allocates in the arena; the peer
    // waits for hangup.
    let mut sink = [0u8; 8];
    loop {
        match stream.read(&mut sink) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
    }
    // 2026-09-25: Registrations go before the mapping they cover.
    drop(rails);
    drop(local);
    Ok(())
}
