// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The weight peer server (unix). Per connection it reads a model
//! request, stages that model (`shard`: the manifest and read-only mmaps of its
//! shards), sends the manifest (`wire`), and registers every shard mmap for
//! remote read on each rail (`reg_mr(.., true)`) so the client reads the
//! tensors with one-sided RDMA.
//!
//! Owner: storage (weight peer).
//! Invariants: every `StagedModel` holds a `CommitLedger` reservation of its
//! shard bytes, taken before its shards are mapped and released when it drops.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::io::Read;
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::manifest::WeightManifest;
use super::shard::{Mmap, build_manifest};
use super::wire::{read_model_request, write_weight_manifest};
use crate::expert_peer::MODE_VERBS;

/// 2026-09-25: The peer's rails, its staging ceiling, and which model
/// directories a client may request.
#[derive(Clone, Debug)]
pub struct WeightPeerConfig {
    /// 2026-09-25: `(device, gid_idx)` per rail; a client asking for `n` rails
    /// gets the first `n`.
    pub rails: Vec<(String, u32)>,
    /// 2026-09-25: Ceiling, in bytes, on the shard bytes of all staged models
    /// together, 0 for none. Each model is charged once, when it is first
    /// staged; per-connection registrations are not charged.
    pub max_blade_bytes: u64,
    /// 2026-09-25: Directories staged at startup. Unless `allow_any_path` is
    /// set, a request must name one of them by full path or by final component.
    pub staged_dirs: Vec<PathBuf>,
    /// 2026-09-25: When true, a request that names any existing directory is
    /// staged on demand.
    pub allow_any_path: bool,
}

impl Default for WeightPeerConfig {
    fn default() -> Self {
        Self {
            rails: vec![("roceP2p1s0f1".into(), 3)],
            max_blade_bytes: 0,
            staged_dirs: Vec::new(),
            allow_any_path: false,
        }
    }
}

/// 2026-09-25: A staged model: its shard mmaps, which stay mapped across
/// connections, its manifest, and its ledger reservation, released on drop.
/// Each connection registers the same mappings on its own rails.
struct StagedModel {
    // 2026-09-25: Read only by the `metrale_rdma_verbs` `serve_verbs`.
    #[cfg_attr(not(metrale_rdma_verbs), allow(dead_code))]
    shard_mmaps: Vec<Mmap>,
    manifest: WeightManifest,
    _reservation: crate::blade_cap::Reservation,
}

type StagedMap = Arc<Mutex<HashMap<String, Arc<StagedModel>>>>;

/// 2026-09-25: Stage `cfg.staged_dirs`, then accept connections on `addr`, one
/// thread each, until the listener ends. A directory that fails to stage is
/// logged and skipped.
///
/// # Errors
/// Binding `addr` fails.
pub fn serve<A: ToSocketAddrs>(addr: A, cfg: WeightPeerConfig) -> Result<()> {
    let cfg = Arc::new(cfg);
    let ledger = Arc::new(crate::blade_cap::CommitLedger::new(cfg.max_blade_bytes));
    let staged: StagedMap = Arc::new(Mutex::new(HashMap::new()));

    for dir in &cfg.staged_dirs {
        match stage_model(&staged, &ledger, dir) {
            Ok(m) => tracing::info!(
                "weight-peer pre-staged {} ({} shards, {} tensors, {:.1} GiB)",
                m.manifest.model_id,
                m.manifest.num_shards(),
                m.manifest.tensors.len(),
                m.manifest.total_shard_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
            ),
            Err(e) => tracing::warn!("weight-peer pre-stage {} failed: {e}", dir.display()),
        }
    }

    let listener = TcpListener::bind(addr).context("bind weight-peer listener")?;
    let local = listener.local_addr().ok();
    tracing::info!(
        "weight-peer serving on {:?} (verbs rails {:?}, cap {}, allow_any_path {})",
        local,
        cfg.rails,
        if cfg.max_blade_bytes == 0 {
            "unlimited".to_string()
        } else {
            format!(
                "{:.1} GiB",
                cfg.max_blade_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            )
        },
        cfg.allow_any_path,
    );

    for conn in listener.incoming() {
        let stream = match conn {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("weight-peer accept error: {e}");
                continue;
            }
        };
        let cfg = cfg.clone();
        let ledger = ledger.clone();
        let staged = staged.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle_conn(stream, &cfg, &ledger, &staged) {
                tracing::warn!("weight-peer connection ended: {e}");
            }
        });
    }
    Ok(())
}

fn handle_conn(
    mut stream: TcpStream,
    cfg: &WeightPeerConfig,
    ledger: &Arc<crate::blade_cap::CommitLedger>,
    staged: &StagedMap,
) -> Result<()> {
    stream.set_nodelay(true).ok();

    let request = read_model_request(&mut stream)?;
    let dir = resolve_request(cfg, &request)?;

    let model = stage_model(staged, ledger, &dir)?;
    write_weight_manifest(&mut stream, &model.manifest).context("send manifest")?;

    // 2026-09-25: The client then names its transport; only verbs is served.
    let mut mode = [0u8; 1];
    stream
        .read_exact(&mut mode)
        .context("read transport mode")?;
    match mode[0] {
        MODE_VERBS => serve_verbs(stream, &model, cfg),
        other => bail!("weight-peer only serves verbs; client asked for mode {other}"),
    }
}

/// 2026-09-25: The staged directory a request names (by full path or by final
/// component), or with `allow_any_path` any existing directory it names.
fn resolve_request(cfg: &WeightPeerConfig, request: &str) -> Result<PathBuf> {
    let req = Path::new(request);
    for d in &cfg.staged_dirs {
        if d == req
            || d.file_name().and_then(|n| n.to_str()) == Some(request)
            || d.to_string_lossy() == request
        {
            return Ok(d.clone());
        }
    }
    if cfg.allow_any_path && req.is_dir() {
        return Ok(req.to_path_buf());
    }
    bail!(
        "model '{request}' is not staged (and allow_any_path is off); \
         pass it to the peer with --stage <dir>"
    );
}

/// 2026-09-25: The staged model for `dir`, staging it first if needed: build
/// the manifest, reserve its shard bytes on the ledger, then mmap every shard.
/// Keyed by the directory string, so a second call returns the same model.
fn stage_model(
    staged: &StagedMap,
    ledger: &Arc<crate::blade_cap::CommitLedger>,
    dir: &Path,
) -> Result<Arc<StagedModel>> {
    let key = dir.to_string_lossy().into_owned();
    {
        let map = staged.lock().unwrap();
        if let Some(m) = map.get(&key) {
            return Ok(m.clone());
        }
    }

    let (shard_paths, manifest) = build_manifest(dir, &key)?;
    // 2026-09-25: The reservation is taken before any shard is mapped, and is
    // released if a mapping below fails.
    let reservation = ledger
        .try_reserve(manifest.total_shard_bytes())
        .context("weight blade cap")?;

    let mut shard_mmaps = Vec::with_capacity(shard_paths.len());
    for p in &shard_paths {
        shard_mmaps.push(Mmap::open_ro(p).with_context(|| format!("mmap {}", p.display()))?);
    }

    let model = Arc::new(StagedModel {
        shard_mmaps,
        manifest,
        _reservation: reservation,
    });
    let mut map = staged.lock().unwrap();
    // 2026-09-25: If another thread staged the same directory meanwhile, its
    // model is kept and this one, with its reservation, is dropped.
    Ok(map.entry(key).or_insert(model).clone())
}

/// 2026-09-25: Without `metrale_rdma_verbs` a verbs request is refused.
#[cfg(not(metrale_rdma_verbs))]
fn serve_verbs(
    _stream: TcpStream,
    _model: &Arc<StagedModel>,
    _cfg: &WeightPeerConfig,
) -> Result<()> {
    bail!("client requested verbs transport but this peer was built without rdma-core");
}

#[cfg(metrale_rdma_verbs)]
fn serve_verbs(
    mut stream: TcpStream,
    model: &Arc<StagedModel>,
    cfg: &WeightPeerConfig,
) -> Result<()> {
    use crate::expert_peer::{STATUS_OK, VerbsClientParams, VerbsServerParams, write_server_rails};
    use metrale_gpu_sys::verbs::Verbs;
    use std::io::Write;

    let num_shards = model.shard_mmaps.len();

    let mut b1 = [0u8; 1];
    stream.read_exact(&mut b1).context("read n_rails")?;
    let n_rails = b1[0] as usize;
    if n_rails == 0 || n_rails > cfg.rails.len() {
        bail!(
            "client asked for {n_rails} rails; peer has {}",
            cfg.rails.len()
        );
    }

    // 2026-09-25: One QP per rail, each with its own PSN. Nothing is charged
    // to the ledger here: `stage_model` charged the model once.
    let pid = std::process::id();
    let mut rails: Vec<Verbs> = Vec::with_capacity(n_rails);
    for (i, (dev, gid)) in cfg.rails.iter().take(n_rails).enumerate() {
        let psn = (0x77_7777 ^ pid ^ ((i as u32) << 20)) & 0xff_ffff;
        rails.push(Verbs::create(dev, *gid, psn)?);
    }

    // 2026-09-25: Every shard mmap is registered for remote read on every rail:
    // one rkey per (rail, shard), all at the mapping's own address.
    let mut per_rail_shards: Vec<Vec<(u64, u32)>> = (0..n_rails)
        .map(|_| Vec::with_capacity(num_shards))
        .collect();
    for m in &model.shard_mmaps {
        for (ri, v) in rails.iter_mut().enumerate() {
            // 2026-09-25: SAFETY: the mapping covers `m.len` bytes at `m.addr` and
            // belongs to `model`, which outlives `rails` (dropped at the end).
            let keys = unsafe { v.reg_mr(m.addr as *mut _, m.len, true)? };
            per_rail_shards[ri].push((m.addr as u64, keys.rkey));
        }
    }

    // 2026-09-25: One `VerbsServerParams` per rail; its `layers` holds each
    // shard's `(base, rkey)` in shard order.
    let sp: Vec<VerbsServerParams> = rails
        .iter()
        .enumerate()
        .map(|(ri, v)| VerbsServerParams {
            qpn: v.qpn(),
            psn: v.psn(),
            gid: v.gid(),
            layers: std::mem::take(&mut per_rail_shards[ri]),
        })
        .collect();
    write_server_rails(&mut stream, &sp).context("send verbs server params")?;

    stream.read_exact(&mut b1).context("read client n_rails")?;
    if b1[0] as usize != n_rails {
        bail!("client rail count mismatch");
    }
    for v in rails.iter_mut() {
        let cp = VerbsClientParams::read_from(&mut stream).context("read verbs client params")?;
        v.connect(cp.qpn, cp.psn, &cp.gid)?;
    }
    stream
        .write_all(&[STATUS_OK])
        .context("send verbs ready ack")?;
    tracing::info!(
        "weight-peer verbs client connected to {} ({n_rails} rail(s), {num_shards} shard MRs/rail)",
        model.manifest.model_id,
    );

    // 2026-09-25: The client reads the tensors itself; wait for it to close
    // the connection.
    let mut sink = [0u8; 8];
    loop {
        match stream.read(&mut sink) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
    }
    // 2026-09-25: The rails, and with them the MRs, are dropped while `model`
    // still holds the mappings.
    drop(rails);
    Ok(())
}
