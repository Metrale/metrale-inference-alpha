// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The weight peer's length-prefixed framing: the model request
//! `[u32 len][len bytes UTF-8]` and the manifest `[u32 len][len bytes JSON]`,
//! lengths little-endian. Compiled on every platform and independent of
//! `serve` and `shard`.
//!
//! Owner: storage (weight peer).
//! Invariants: none beyond the types.

use anyhow::{Context, Result, bail};

use super::manifest::WeightManifest;

fn read_u32<R: std::io::Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).context("read u32")?;
    Ok(u32::from_le_bytes(b))
}

/// 2026-09-25: The longest model request, in bytes, either side accepts; a
/// longer length prefix is refused before any buffer is allocated.
pub const MODEL_REQUEST_MAX: usize = 8192;

/// 2026-09-25: Write the model request `id`.
///
/// # Errors
/// `id` is empty or longer than [`MODEL_REQUEST_MAX`], or the write fails.
pub fn write_model_request<W: std::io::Write>(w: &mut W, id: &str) -> Result<()> {
    let bytes = id.as_bytes();
    if bytes.is_empty() || bytes.len() > MODEL_REQUEST_MAX {
        bail!("implausible model request length: {}", bytes.len());
    }
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(bytes)?;
    Ok(())
}

/// 2026-09-25: Read a model request written by [`write_model_request`].
///
/// # Errors
/// A length of 0 or above [`MODEL_REQUEST_MAX`], a short read, or a body that
/// is not UTF-8.
pub fn read_model_request<R: std::io::Read>(r: &mut R) -> Result<String> {
    let len = read_u32(r)? as usize;
    if len == 0 || len > MODEL_REQUEST_MAX {
        bail!("implausible model request length: {len}");
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).context("read model request body")?;
    String::from_utf8(buf).context("model request is not valid UTF-8")
}

/// 2026-09-25: Serialize `m` to JSON and write it with its length prefix.
pub fn write_weight_manifest<W: std::io::Write>(w: &mut W, m: &WeightManifest) -> Result<()> {
    let json = serde_json::to_vec(m).context("serialize weight manifest")?;
    w.write_all(&(json.len() as u32).to_le_bytes())?;
    w.write_all(&json)?;
    Ok(())
}

/// 2026-09-25: Read and parse a manifest written by [`write_weight_manifest`].
///
/// # Errors
/// A length of 0 or above 256 MiB, a short read, bad JSON, a `version` other
/// than [`WeightManifest::VERSION`], or `shard_files` and `shard_lens` of
/// different lengths.
pub fn read_weight_manifest<R: std::io::Read>(r: &mut R) -> Result<WeightManifest> {
    let len = read_u32(r)? as usize;
    if len == 0 || len > 256 * 1024 * 1024 {
        bail!("implausible weight manifest length: {len}");
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .context("read weight manifest json")?;
    let m: WeightManifest = serde_json::from_slice(&buf).context("parse weight manifest json")?;
    if m.version != WeightManifest::VERSION {
        bail!(
            "weight manifest version {} != supported {}",
            m.version,
            WeightManifest::VERSION
        );
    }
    if m.shard_files.len() != m.shard_lens.len() {
        bail!(
            "manifest shard_files ({}) / shard_lens ({}) length mismatch",
            m.shard_files.len(),
            m.shard_lens.len()
        );
    }
    Ok(m)
}

#[cfg(test)]
#[path = "wire_tests.rs"]
mod tests;
