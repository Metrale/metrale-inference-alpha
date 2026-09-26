// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The expert peer, a process that serves an expert store's
//! records to `RdmaTier` clients, and its wire format (little-endian):
//!   1. On accept the peer sends the manifest, `[u32 len][len bytes of JSON]`.
//!   2. The client sends one transport byte, `MODE_TCP` or `MODE_VERBS`.
//!   3. TCP: the client sends `[u32 layer][u32 expert]`; the peer answers
//!      `[STATUS_OK][record_stride bytes]` or `[STATUS_ERR]`, until the client
//!      hangs up or sends `SHUTDOWN_MARKER` in both fields. Verbs: the peer maps
//!      and registers every layer file for remote read, publishes its rails, and
//!      waits for hangup while the client reads records itself.
//!
//! The peer uses no CUDA.
//!
//! Owner: metrale-storage peers.
//! Invariants:
//! - A verbs connection reserves `index.total_bytes()` in the ledger before it
//!   maps or registers anything.

// 2026-09-25: Used only by unix code, under the same cfg, so a Windows build
// has no unused imports.
#[cfg(unix)]
use anyhow::{Context, Result, bail};

// 2026-09-25: The handshake codecs are metrale-gpu-sys `wire`, re-exported for
// this module's users; crates/gpu-sys/tests/wire_roundtrip.rs tests them.
pub use metrale_gpu_sys::wire::{
    MODE_TCP, MODE_VERBS, STATUS_ERR, STATUS_OK, VerbsClientParams, VerbsServerParams,
    read_server_rails, write_server_rails,
};

/// 2026-09-25: In both fields of a TCP request, asks the peer to close the
/// connection.
pub const SHUTDOWN_MARKER: u32 = u32::MAX;

/// 2026-09-25: Encode a TCP request, `[u32 layer][u32 expert]` little-endian.
pub fn encode_request(layer: u32, expert: u32) -> [u8; 8] {
    let mut b = [0u8; 8];
    b[0..4].copy_from_slice(&layer.to_le_bytes());
    b[4..8].copy_from_slice(&expert.to_le_bytes());
    b
}

/// 2026-09-25: Decode a TCP request; the inverse of `encode_request`.
pub fn decode_request(b: &[u8; 8]) -> (u32, u32) {
    let layer = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let expert = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    (layer, expert)
}

#[cfg(unix)]
pub use server_impl::{RdmaConfig, serve};

#[cfg(unix)]
mod server_impl {
    use super::*;
    use crate::expert::ExpertKey;
    use crate::expert_pack::ExpertFileReader;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream, ToSocketAddrs};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    /// 2026-09-25: Settings of the verbs transport, unused by TCP clients. A
    /// client asking for n rails gets the first n `(device, gid_idx)` entries;
    /// the peer registers every layer file on each.
    #[derive(Clone, Debug)]
    pub struct RdmaConfig {
        pub rails: Vec<(String, u32)>,
        /// 2026-09-25: Ceiling on store bytes registered across concurrent verbs
        /// connections; 0 means none. A connection reserves the store size once,
        /// whatever its rail count.
        pub max_blade_bytes: u64,
    }

    impl Default for RdmaConfig {
        fn default() -> Self {
            Self {
                rails: vec![("roceP2p1s0f1".into(), 3)],
                max_blade_bytes: 0,
            }
        }
    }

    /// 2026-09-25: Serve the store in `dir` on `addr`, one thread per
    /// connection. Accept errors are logged and skipped; only opening the store
    /// or binding returns an error.
    pub fn serve<A: ToSocketAddrs>(dir: &Path, addr: A, rdma: RdmaConfig) -> Result<()> {
        let reader = Arc::new(ExpertFileReader::open(dir)?);
        let manifest = serde_json::to_vec(reader.index())?;
        let dir: Arc<PathBuf> = Arc::new(dir.to_path_buf());
        let rdma = Arc::new(rdma);
        // 2026-09-25: One ledger per `serve`, shared by its connection threads.
        let ledger = Arc::new(crate::blade_cap::CommitLedger::new(rdma.max_blade_bytes));
        let listener = TcpListener::bind(addr).context("bind expert-peer listener")?;
        let local = listener.local_addr().ok();
        tracing::info!(
            "expert-peer serving {} ({} layers, {} experts, stride {}) on {:?} \
             (verbs rails {:?}, cap {})",
            dir.display(),
            reader.index().num_moe_layers,
            reader.index().num_experts,
            reader.index().record_stride,
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
        for conn in listener.incoming() {
            let stream = match conn {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("expert-peer accept error: {e}");
                    continue;
                }
            };
            let reader = reader.clone();
            let manifest = manifest.clone();
            let dir = dir.clone();
            let rdma = rdma.clone();
            let ledger = ledger.clone();
            std::thread::spawn(move || {
                if let Err(e) = handle_conn(stream, &reader, &manifest, &dir, &rdma, &ledger) {
                    tracing::warn!("expert-peer connection ended: {e}");
                }
            });
        }
        Ok(())
    }

    fn handle_conn(
        mut stream: TcpStream,
        reader: &ExpertFileReader,
        manifest: &[u8],
        dir: &Path,
        rdma: &RdmaConfig,
        ledger: &Arc<crate::blade_cap::CommitLedger>,
    ) -> Result<()> {
        stream.set_nodelay(true).ok();
        stream.write_all(&(manifest.len() as u32).to_le_bytes())?;
        stream.write_all(manifest)?;

        // 2026-09-25: TCP reads records on demand and pins nothing, so only the
        // verbs transport, which maps and registers the store, is charged.
        let mut mode = [0u8; 1];
        stream
            .read_exact(&mut mode)
            .context("read transport mode")?;
        match mode[0] {
            MODE_TCP => serve_tcp(stream, reader),
            MODE_VERBS => serve_verbs(stream, reader, dir, rdma, ledger),
            other => bail!("client requested unknown transport mode {other}"),
        }
    }

    /// 2026-09-25: Answer each request with the record, or `STATUS_ERR` when it
    /// cannot be read, until the client hangs up or sends `SHUTDOWN_MARKER`.
    fn serve_tcp(mut stream: TcpStream, reader: &ExpertFileReader) -> Result<()> {
        let stride = reader.index().record_stride as usize;
        let mut req = [0u8; 8];
        loop {
            if stream.read_exact(&mut req).is_err() {
                break;
            }
            let (layer, expert) = decode_request(&req);
            if layer == SHUTDOWN_MARKER && expert == SHUTDOWN_MARKER {
                break;
            }
            match reader.read_record_raw(ExpertKey::new(layer, expert)) {
                Ok(rec) => {
                    debug_assert_eq!(rec.len(), stride);
                    stream.write_all(&[STATUS_OK])?;
                    stream.write_all(&rec)?;
                }
                Err(e) => {
                    tracing::warn!("expert-peer read {layer}/{expert}: {e}");
                    stream.write_all(&[STATUS_ERR])?;
                }
            }
        }
        Ok(())
    }

    #[cfg(not(metrale_rdma_verbs))]
    fn serve_verbs(
        _stream: TcpStream,
        _reader: &ExpertFileReader,
        _dir: &Path,
        _rdma: &RdmaConfig,
        _ledger: &Arc<crate::blade_cap::CommitLedger>,
    ) -> Result<()> {
        bail!("client requested verbs transport but this peer was built without rdma-core");
    }

    /// 2026-09-25: Map and register each layer file for remote read, publish
    /// each rail's QP and per-layer `(base, rkey)`, connect to the client's QPs,
    /// then wait for hangup; the client reads the records itself.
    #[cfg(metrale_rdma_verbs)]
    fn serve_verbs(
        mut stream: TcpStream,
        reader: &ExpertFileReader,
        dir: &Path,
        rdma: &RdmaConfig,
        ledger: &Arc<crate::blade_cap::CommitLedger>,
    ) -> Result<()> {
        use metrale_gpu_sys::verbs::Verbs;

        let index = reader.index();
        let num_layers = index.num_moe_layers;

        let mut b1 = [0u8; 1];
        stream.read_exact(&mut b1).context("read n_rails")?;
        let n_rails = b1[0] as usize;
        if n_rails == 0 || n_rails > rdma.rails.len() {
            bail!(
                "client asked for {n_rails} rails; peer has {}",
                rdma.rails.len()
            );
        }

        // 2026-09-25: The rails register the same mapped pages, so the store is
        // charged once, before anything is mapped; the reservation is released
        // on every return.
        let _reservation = ledger
            .try_reserve(index.total_bytes())
            .context("expert blade cap")?;

        // 2026-09-25: The PSN differs per rail and per process.
        let pid = std::process::id();
        let mut rails: Vec<Verbs> = Vec::with_capacity(n_rails);
        for (i, (dev, gid)) in rdma.rails.iter().take(n_rails).enumerate() {
            let psn = (0x424242 ^ pid ^ ((i as u32) << 20)) & 0xff_ffff;
            rails.push(Verbs::create(dev, *gid, psn)?);
        }

        // 2026-09-25: Each layer file is mapped once and registered on every
        // rail: one rkey per (rail, layer), one base address per layer.
        let mut mmaps: Vec<Mmap> = Vec::with_capacity(num_layers as usize);
        let mut per_rail_layers: Vec<Vec<(u64, u32)>> = (0..n_rails)
            .map(|_| Vec::with_capacity(num_layers as usize))
            .collect();
        for l in 0..num_layers {
            let path = dir.join(index.file_name(l));
            let m = Mmap::open_ro(&path).with_context(|| format!("mmap {}", path.display()))?;
            for (ri, v) in rails.iter_mut().enumerate() {
                // 2026-09-25: SAFETY: the mapping covers `m.len` bytes at `m.addr`.
                // The explicit drops at the end release `rails` first; an early
                // return drops `mmaps` first (reverse declaration order).
                let keys = unsafe { v.reg_mr(m.addr as *mut _, m.len, true)? };
                per_rail_layers[ri].push((m.addr as u64, keys.rkey));
            }
            mmaps.push(m);
        }

        let sp: Vec<VerbsServerParams> = rails
            .iter()
            .enumerate()
            .map(|(ri, v)| VerbsServerParams {
                qpn: v.qpn(),
                psn: v.psn(),
                gid: v.gid(),
                layers: std::mem::take(&mut per_rail_layers[ri]),
            })
            .collect();
        write_server_rails(&mut stream, &sp).context("send verbs server params")?;

        stream.read_exact(&mut b1).context("read client n_rails")?;
        if b1[0] as usize != n_rails {
            bail!("client rail count mismatch");
        }
        for v in rails.iter_mut() {
            let cp =
                VerbsClientParams::read_from(&mut stream).context("read verbs client params")?;
            v.connect(cp.qpn, cp.psn, &cp.gid)?;
        }
        stream
            .write_all(&[STATUS_OK])
            .context("send verbs ready ack")?;
        tracing::info!(
            "expert-peer verbs client connected ({n_rails} rail(s), {} layer MRs/rail)",
            num_layers,
        );

        // 2026-09-25: The client reads every record itself; the peer holds the
        // registrations until hangup.
        let mut sink = [0u8; 8];
        loop {
            match stream.read(&mut sink) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        // 2026-09-25: Registrations go before the mappings they cover.
        drop(rails);
        drop(mmaps);
        Ok(())
    }

    /// 2026-09-25: A read-only shared mapping of a whole file, unmapped on drop.
    #[cfg(metrale_rdma_verbs)]
    struct Mmap {
        addr: *mut libc::c_void,
        len: usize,
    }

    #[cfg(metrale_rdma_verbs)]
    impl Mmap {
        fn open_ro(path: &Path) -> Result<Self> {
            use std::os::fd::AsRawFd;
            let f = std::fs::File::open(path)?;
            let len = f.metadata()?.len() as usize;
            if len == 0 {
                bail!("empty layer file {}", path.display());
            }
            // 2026-09-25: SAFETY: `f` is an open read-only file of `len` bytes. The
            // mapping stays valid after the file is closed.
            let addr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    f.as_raw_fd(),
                    0,
                )
            };
            if addr == libc::MAP_FAILED {
                bail!(
                    "mmap {} failed: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                );
            }
            Ok(Self { addr, len })
        }
    }

    #[cfg(metrale_rdma_verbs)]
    impl Drop for Mmap {
        fn drop(&mut self) {
            // 2026-09-25: SAFETY: addr/len come from a successful mmap; drop runs once.
            unsafe { libc::munmap(self.addr, self.len) };
        }
    }
}

/// 2026-09-25: Read and parse the length-prefixed manifest at the start of a
/// connection (used by `expert_tier_rdma`). Fails on a length of 0 or above
/// 16 MiB.
#[cfg(unix)]
pub fn read_manifest<R: std::io::Read>(stream: &mut R) -> Result<crate::expert_pack::ExpertIndex> {
    let mut lenb = [0u8; 4];
    stream
        .read_exact(&mut lenb)
        .context("read manifest length")?;
    let len = u32::from_le_bytes(lenb) as usize;
    if len == 0 || len > 16 * 1024 * 1024 {
        bail!("implausible peer manifest length: {len}");
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).context("read manifest json")?;
    let index: crate::expert_pack::ExpertIndex =
        serde_json::from_slice(&buf).context("parse peer manifest")?;
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips() {
        let b = encode_request(7, 42);
        assert_eq!(decode_request(&b), (7, 42));
        let s = encode_request(SHUTDOWN_MARKER, SHUTDOWN_MARKER);
        assert_eq!(decode_request(&s), (SHUTDOWN_MARKER, SHUTDOWN_MARKER));
    }
}
