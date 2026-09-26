// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Library tab data: list the `models--*` directories of the HF
//! cache and read each one's `config.json`, without loading weights. The cache
//! root comes from `model_resolver::resolve_cache_root` and the snapshot from
//! `model_resolver::find_snapshot_with_weights`.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use std::path::{Path, PathBuf};

/// 2026-09-26: One locally cached model.
#[derive(Clone, Debug)]
pub struct LibraryEntry {
    /// 2026-09-26: HF id, un-mangled (`org/name`).
    pub id: String,
    pub snapshot_dir: PathBuf,
    pub size_bytes: u64,
    pub has_weights: bool,
    /// 2026-09-26: From config.json; `?` when it is missing or does not parse
    /// (the entry is still listed).
    pub model_type: String,
    pub quant: String,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub experts: usize,
    pub context: usize,
    /// 2026-09-26: `ptx_for_config` returned a target for this model's
    /// `(model_type, hidden_size)` with the HF id as the tie-break reference.
    /// False when it returned none or an error.
    pub optimized: bool,
}

/// 2026-09-26: Total size of the files under `dir`, recursively; 0 when `dir`
/// cannot be read. `is_dir` and `metadata` follow symlinks, so a symlinked
/// file counts at its target's size.
fn dir_size(dir: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            total += dir_size(&p);
        } else if let Ok(md) = std::fs::metadata(&p) {
            total += md.len();
        }
    }
    total
}

/// 2026-09-26: Scan the HF cache, largest first. `cache_dir` is the
/// `--cache-dir` override, if any.
pub fn scan(cache_dir: Option<&Path>) -> Vec<LibraryEntry> {
    let Ok(root) = crate::model_resolver::resolve_cache_root(cache_dir) else {
        return Vec::new();
    };
    let Ok(rd) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(mangled) = name.strip_prefix("models--") else {
            continue;
        };
        let id = mangled.replace("--", "/");
        let snapshots = e.path().join("snapshots");
        let Some(snap) =
            crate::model_resolver::find_snapshot_with_weights(&snapshots).or_else(|| {
                // 2026-09-26: No snapshot with weights: list the first
                // snapshot directory `read_dir` yields, if any.
                std::fs::read_dir(&snapshots)
                    .ok()?
                    .flatten()
                    .map(|s| s.path())
                    .find(|p| p.is_dir())
            })
        else {
            continue;
        };
        // 2026-09-26: `has_weights` needs a `.safetensors` file in the snapshot
        // and a `refs/main`. `snapshot_has_weights` alone also accepts
        // `model.safetensors.index.json`, and the downloader writes `refs/main`
        // only when a download is published.
        let has_shard = std::fs::read_dir(&snap)
            .map(|rd| {
                rd.filter_map(|e| e.ok()).any(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.ends_with(".safetensors"))
                })
            })
            .unwrap_or(false);
        let published = e.path().join("refs/main").exists();
        let has_weights = has_shard && published;
        let mut entry = LibraryEntry {
            id,
            // 2026-09-26: The size of `blobs/`, or of the snapshot directory
            // when `blobs/` is missing or empty.
            size_bytes: match dir_size(&e.path().join("blobs")) {
                0 => dir_size(&snap),
                n => n,
            },
            snapshot_dir: snap.clone(),
            has_weights,
            model_type: "?".into(),
            quant: "-".into(),
            layers: 0,
            hidden: 0,
            heads: 0,
            experts: 0,
            context: 0,
            optimized: false,
        };
        if let Ok(json) = std::fs::read_to_string(snap.join("config.json"))
            && let Ok(cfg) = metrale_config::parse_config(&json)
        {
            entry.model_type = cfg.model_type.clone();
            entry.layers = cfg.num_hidden_layers;
            entry.hidden = cfg.hidden_size;
            entry.heads = cfg.num_attention_heads;
            entry.experts = cfg.num_experts;
            entry.context = cfg.max_position_embeddings;
            if let Some(q) = &cfg.quantization_config {
                entry.quant = if q.quant_algo.is_empty() {
                    q.quant_method.clone()
                } else {
                    q.quant_algo.to_lowercase()
                };
            }
            // 2026-09-26: The HF id is passed as the tie-break reference; an
            // `Err` counts as not optimized.
            entry.optimized = matches!(
                metrale_kernels::ptx_for_config(
                    &cfg.model_type,
                    cfg.hidden_size,
                    &[entry.id.as_str()],
                    None,
                ),
                Ok(Some(_))
            );
        }
        out.push(entry);
    }
    out.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    out
}

/// 2026-09-26: Human size, via `format::bytes`.
pub fn human_size(bytes: u64) -> String {
    crate::tui::format::bytes(bytes)
}

/// 2026-09-26: Run [`scan`] on a `worker::spawn` thread; the receiver gets the
/// result, or an empty list if the thread cannot start (the same answer `scan`
/// gives for an unreadable cache).
pub fn scan_in_background(
    cache_dir: Option<&Path>,
) -> std::sync::mpsc::Receiver<Vec<LibraryEntry>> {
    let owned = cache_dir.map(|p| p.to_path_buf());
    crate::tui::worker::spawn(
        "metrale-libscan",
        move || scan(owned.as_deref()),
        |_| Vec::new(),
    )
}
