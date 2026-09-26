// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The weight peer's staging (unix): find a model directory's shard
//! files, build its [`WeightManifest`] from their safetensors headers, and map
//! each shard read-only ([`Mmap`]) for `serve` to register.
//!
//! Owner: storage (weight peer).
//! Invariants: every published tensor span lies inside its shard file
//! (`metrale_core::safetensors::tensor_span` refuses one that does not).

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::manifest::{WeightManifest, WeightTensorRecord};

/// 2026-09-25: The shard paths of `dir` and their [`WeightManifest`], with
/// `extra_weights.safetensors`, when present, as the last shard.
///
/// # Errors
/// No shard is found, a file cannot be read, or a header is malformed, holds
/// an unsupported dtype or a span outside its file.
pub(super) fn build_manifest(dir: &Path, model_id: &str) -> Result<(Vec<PathBuf>, WeightManifest)> {
    let (shard_paths, weight_map) = resolve_shards(dir)?;

    let mut shard_files = Vec::with_capacity(shard_paths.len());
    let mut shard_lens = Vec::with_capacity(shard_paths.len());
    let mut tensors: Vec<WeightTensorRecord> = Vec::new();

    for (idx, path) in shard_paths.iter().enumerate() {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        let len = std::fs::metadata(path)
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        // 2026-09-25: With an index, only the tensors its `weight_map` lists are
        // published; a header may also list tensors the index leaves out.
        // Without an index every header tensor is published.
        parse_shard_header(path, idx as u32, false, weight_map.as_ref(), &mut tensors)?;
        shard_files.push(name);
        shard_lens.push(len);
    }

    let extra = dir.join("extra_weights.safetensors");
    if extra.exists() {
        let idx = shard_paths.len();
        let mut shard_paths = shard_paths.clone();
        let len = std::fs::metadata(&extra)?.len();
        // 2026-09-25: Every tensor of the extra shard is published, with
        // `extra` set; no `weight_map` filter applies to it.
        parse_shard_header(&extra, idx as u32, true, None, &mut tensors)?;
        shard_files.push("extra_weights.safetensors".to_string());
        shard_lens.push(len);
        shard_paths.push(extra);
        let manifest = WeightManifest {
            version: WeightManifest::VERSION,
            model_id: model_id.to_string(),
            shard_files,
            shard_lens,
            tensors,
        };
        return Ok((shard_paths, manifest));
    }

    let manifest = WeightManifest {
        version: WeightManifest::VERSION,
        model_id: model_id.to_string(),
        shard_files,
        shard_lens,
        tensors,
    };
    Ok((shard_paths, manifest))
}

/// 2026-09-25: The shard paths in shard order, and the index `weight_map`
/// (tensor name to shard file) when the directory has an index.
type ShardResolution = (Vec<PathBuf>, Option<HashMap<String, String>>);

/// 2026-09-25: Shard discovery, first match wins: (1) the files named by
/// `model.safetensors.index.json`, else by `consolidated.safetensors.index.json`,
/// sorted; (2) `model.safetensors`; (3) `adapter_model.safetensors`; (4) the
/// sorted `model.safetensors-*` / `consolidated-*` files ending `.safetensors`.
/// model-weights `fast_weights::header::resolve_shards` has the same order
/// without step 3.
pub(super) fn resolve_shards(dir: &Path) -> Result<ShardResolution> {
    let index = dir.join("model.safetensors.index.json");
    let consolidated = dir.join("consolidated.safetensors.index.json");
    let actual = if index.exists() {
        Some(index)
    } else if consolidated.exists() {
        Some(consolidated)
    } else {
        None
    };

    if let Some(ip) = actual {
        let json =
            std::fs::read_to_string(&ip).with_context(|| format!("read {}", ip.display()))?;
        let v: Value = serde_json::from_str(&json)?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .context("index json missing weight_map object")?;
        let weight_map: HashMap<String, String> = map
            .iter()
            .filter_map(|(k, val)| val.as_str().map(|s| (k.clone(), s.to_string())))
            .collect();
        let mut shards: Vec<String> = weight_map
            .values()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        shards.sort();
        let files = shards.iter().map(|s| dir.join(s)).collect();
        return Ok((files, Some(weight_map)));
    }

    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok((vec![single], None));
    }

    // 2026-09-25: A PEFT adapter directory, read by the LoRA client
    // (model-weights `weight_lora_rdma.rs`).
    let adapter = dir.join("adapter_model.safetensors");
    if adapter.exists() {
        return Ok((vec![adapter], None));
    }

    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                (n.starts_with("model.safetensors-") || n.starts_with("consolidated-"))
                    && n.ends_with(".safetensors")
            })
        })
        .collect();
    shards.sort();
    if shards.is_empty() {
        bail!("no safetensor files found in {}", dir.display());
    }
    Ok((shards, None))
}

/// 2026-09-25: Push a [`WeightTensorRecord`] for each tensor of one shard's
/// header (`[u64 LE header_size][header JSON]`, data from `8 + header_size`),
/// with the file offset of its first byte, skipping `__metadata__` and, given
/// a `weight_map`, the tensors it does not list.
///
/// # Errors
/// A header over 64 MiB or not a JSON object, a dtype [`validate_dtype`]
/// refuses, or a span `tensor_span` refuses.
fn parse_shard_header(
    path: &Path,
    shard_index: u32,
    extra: bool,
    weight_map: Option<&HashMap<String, String>>,
    out: &mut Vec<WeightTensorRecord>,
) -> Result<()> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_len = f
        .metadata()
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    let mut size_buf = [0u8; 8];
    f.read_exact(&mut size_buf)
        .with_context(|| format!("read header size of {}", path.display()))?;
    let header_size = u64::from_le_bytes(size_buf) as usize;
    if header_size > 64 * 1024 * 1024 {
        bail!(
            "{}: safetensor header too large ({header_size} bytes)",
            path.display()
        );
    }
    let mut header_buf = vec![0u8; header_size];
    f.read_exact(&mut header_buf)
        .with_context(|| format!("read header of {}", path.display()))?;
    let data_start = 8 + header_size as u64;

    let json: Value = serde_json::from_slice(&header_buf)?;
    let obj = json
        .as_object()
        .with_context(|| format!("{}: header is not a JSON object", path.display()))?;

    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        if let Some(map) = weight_map
            && !map.contains_key(name)
        {
            continue;
        }
        let dtype = info["dtype"].as_str().unwrap_or("BF16").to_string();
        validate_dtype(&dtype, name)?;
        let shape: Vec<u64> = info["shape"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_default();
        // 2026-09-25: The same span check as model-weights
        // `fast_weights/header.rs`; the span becomes a client's remote read.
        let span = metrale_core::safetensors::tensor_span(
            name,
            &info["data_offsets"],
            data_start,
            file_len,
        )
        .with_context(|| format!("{}", path.display()))?;
        out.push(WeightTensorRecord {
            name: name.clone(),
            dtype,
            shape,
            offset_in_shard: span.abs_offset,
            len: span.len,
            shard_index,
            extra,
        });
    }
    Ok(())
}

/// 2026-09-25: The dtypes a manifest may carry: those
/// `WeightDtype::from_safetensors_str` maps, plus `F16`. Anything else fails
/// staging.
fn validate_dtype(dtype: &str, tensor: &str) -> Result<()> {
    match dtype {
        // 2026-09-25: `F16` is for adapter shards: the LoRA client converts
        // F16 and F32 to BF16 (`weight_lora_rdma.rs`).
        "F32" | "F16" | "BF16" | "U8" | "I8" | "F8_E4M3" | "F8_E8M0" | "I64" => Ok(()),
        other => bail!("unsupported safetensors dtype '{other}' for tensor {tensor}"),
    }
}

/// 2026-09-25: A read-only, shared, whole-file `mmap`, advised
/// `POSIX_MADV_WILLNEED` and unmapped on drop. No Rust code reads through
/// `addr`; `serve` passes it to `reg_mr` as a base address.
pub(super) struct Mmap {
    pub(super) addr: *mut libc::c_void,
    pub(super) len: usize,
}

// 2026-09-25: SAFETY: `addr` and `len` describe a read-only mapping that no Rust
// code dereferences, so sharing them across threads races on nothing.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    pub(super) fn open_ro(path: &Path) -> Result<Self> {
        use std::os::fd::AsRawFd;
        let f = std::fs::File::open(path)?;
        let len = f.metadata()?.len() as usize;
        if len == 0 {
            bail!("empty shard file {}", path.display());
        }
        // 2026-09-25: SAFETY: `f` is an open read-only file of `len` bytes, and a
        // mapping stays valid after its file descriptor is closed.
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
        // 2026-09-25: A hint to read the pages in; its result is ignored.
        // SAFETY: addr/len came from the successful mmap above.
        unsafe { libc::posix_madvise(addr, len, libc::POSIX_MADV_WILLNEED) };
        Ok(Self { addr, len })
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // 2026-09-25: SAFETY: `addr`/`len` came from a successful mmap, and drop
        // runs once.
        unsafe { libc::munmap(self.addr, self.len) };
    }
}

#[cfg(test)]
#[path = "shard_tests.rs"]
mod tests;
