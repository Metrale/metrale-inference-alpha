// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Build a [`ModelConfig`] from the metadata of the `.gguf` in a model
//! directory, so a directory with no `config.json` or `params.json` can be served.
//!
//! `config_from_gguf` is in metrale-config, which does not depend on this crate's
//! GGUF parser. This module implements metrale-config's [`GgufMeta`] over
//! [`GgufFile`] and reads the two tensor-directory facts the builder takes: the
//! vocab dim of `token_embd.weight` and whether `output.weight` exists.
//!
//! Owner: model-weights (GGUF).
//! Invariants: none beyond the types.

use std::path::Path;

use anyhow::{Context, Result};
use metrale_config::{GgufConfigInputs, GgufMeta, ModelConfig, config_from_gguf};

use super::container::GgufFile;
use super::find_gguf;

impl GgufMeta for GgufFile {
    fn get_u64(&self, key: &str) -> Option<u64> {
        GgufFile::get_u64(self, key)
    }
    fn get_f64(&self, key: &str) -> Option<f64> {
        GgufFile::get_f64(self, key)
    }
    fn get_str(&self, key: &str) -> Option<&str> {
        GgufFile::get_str(self, key)
    }
    fn get_arr_len(&self, key: &str) -> Option<usize> {
        GgufFile::arr_len(self, key)
    }
    /// 2026-09-26: Integer array, every element widened to u64 whatever its
    /// integer type (a bool reads as 0 or 1). `None` when the key is absent, the
    /// value is not an array, or any element is negative or not an integer; no
    /// element is dropped.
    fn get_u64_arr(&self, key: &str) -> Option<Vec<u64>> {
        let arr = GgufFile::get(self, key)?.as_array()?;
        arr.iter()
            .map(|v| {
                v.as_u64()
                    .or_else(|| v.as_i64().and_then(|i| u64::try_from(i).ok()))
            })
            .collect()
    }
    /// 2026-09-26: Float array, every element widened to f64 by
    /// `MetaValue::as_f64`, which also converts integers and bools. `None` when
    /// the key is absent, the value is not an array, or any element does not
    /// convert.
    fn get_f64_arr(&self, key: &str) -> Option<Vec<f64>> {
        let arr = GgufFile::get(self, key)?.as_array()?;
        arr.iter().map(|v| v.as_f64()).collect()
    }
}

/// 2026-09-26: Build a [`ModelConfig`] from the `.gguf` that `find_gguf` picks in
/// `model_dir`.
///
/// Passes `config_from_gguf` the trailing ggml dim of `token_embd.weight` (the
/// vocab) and whether `output.weight` exists; `config_from_gguf` treats a missing
/// `output.weight` as tied embeddings. Errors when there is no `.gguf`, when the
/// file cannot be opened, mapped or parsed, or when the config cannot be built.
pub fn config_from_gguf_dir(model_dir: &Path) -> Result<ModelConfig> {
    let path = find_gguf(model_dir)
        .with_context(|| format!("no .gguf file in {}", model_dir.display()))?;
    let file =
        std::fs::File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
    // 2026-09-26: SAFETY: `GgufFile::parse` reads the map while `mmap` is alive.
    // As in `sidecar::open_gguf`, nothing stops another process from modifying the
    // file while it is mapped; this assumes none does.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let gguf = GgufFile::parse(&mmap)
        .with_context(|| format!("failed to parse GGUF metadata: {}", path.display()))?;

    let token_embd_vocab = gguf
        .tensor("token_embd.weight")
        .and_then(|t| t.dims.last().copied());
    let has_output_weight = gguf.tensor("output.weight").is_some();

    let inputs = GgufConfigInputs {
        meta: &gguf,
        token_embd_vocab,
        has_output_weight,
    };
    config_from_gguf(&inputs).context("failed to build ModelConfig from GGUF metadata")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-26: The `GgufMeta` impl reaches the inherent getters (a call that
    /// resolved to the trait method would recurse) and maps `get_arr_len` to
    /// `arr_len`.
    #[test]
    fn gguf_meta_bridge_forwards() {
        // 2026-09-26: A minimal GGUF v3 built by hand: no tensors, one UINT32 KV
        // (type 4) and one STRING-array KV (type 9, element type 8).
        let mut b: Vec<u8> = Vec::new();
        let push_u32 = |b: &mut Vec<u8>, v: u32| b.extend_from_slice(&v.to_le_bytes());
        let push_u64 = |b: &mut Vec<u8>, v: u64| b.extend_from_slice(&v.to_le_bytes());
        let push_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        push_u32(&mut b, 0x4655_4747);
        push_u32(&mut b, 3);
        push_u64(&mut b, 0);
        push_u64(&mut b, 2);
        push_str(&mut b, "qwen3.block_count");
        push_u32(&mut b, 4);
        push_u32(&mut b, 28);
        push_str(&mut b, "tokenizer.ggml.tokens");
        push_u32(&mut b, 9);
        push_u32(&mut b, 8);
        push_u64(&mut b, 3);
        for s in ["a", "bb", "ccc"] {
            push_str(&mut b, s);
        }
        // 2026-09-26: Pad to the default 32-byte alignment, so the empty
        // tensor-data section starts inside the buffer, as `GgufFile::parse`
        // requires.
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        let gguf = GgufFile::parse(&b).unwrap();
        let m: &dyn GgufMeta = &gguf;
        assert_eq!(m.get_u64("qwen3.block_count"), Some(28));
        assert_eq!(m.get_arr_len("tokenizer.ggml.tokens"), Some(3));
        assert_eq!(m.get_str("nonexistent"), None);
    }
}
