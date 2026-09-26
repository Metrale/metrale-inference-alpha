// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `build_manifest` over small safetensors files written to the
//! temp directory.
//!
//! Owner: storage (weight peer).
//! Invariants: none beyond the types.

use super::*;
use std::io::Write;

/// 2026-09-25: Write `[u64 LE header length][header][data]` to `path`.
fn write_st(path: &Path, header: &str, data: &[u8]) {
    let hb = header.as_bytes();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
    f.write_all(hb).unwrap();
    f.write_all(data).unwrap();
}

/// 2026-09-25: A header tensor that the index `weight_map` does not list is not
/// published.
#[test]
fn build_manifest_filters_orphan_tensors() {
    let dir = std::env::temp_dir().join(format!("wpeer-orphan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let shard = "model-00001.safetensors";
    let header = r#"{"a.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"orphan.weight":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
    write_st(&dir.join(shard), header, &[0u8; 16]);
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        format!(r#"{{"weight_map":{{"a.weight":"{shard}"}}}}"#),
    )
    .unwrap();

    let (_paths, manifest) = build_manifest(&dir, "test").unwrap();
    let names: Vec<&str> = manifest.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["a.weight"],
        "orphan tensor (not in weight_map) must not be published"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// 2026-09-25: Without an index (a single `model.safetensors`) every header
/// tensor is published.
#[test]
fn build_manifest_keeps_all_when_no_index() {
    let dir = std::env::temp_dir().join(format!("wpeer-single-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let header = r#"{"a.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b.weight":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
    write_st(&dir.join("model.safetensors"), header, &[0u8; 16]);

    let (_paths, manifest) = build_manifest(&dir, "test").unwrap();
    let mut names: Vec<&str> = manifest.tensors.iter().map(|t| t.name.as_str()).collect();
    names.sort();
    assert_eq!(names, vec!["a.weight", "b.weight"]);

    std::fs::remove_dir_all(&dir).ok();
}

/// 2026-09-25: A reversed `data_offsets` pair fails staging, naming the shard.
#[test]
fn build_manifest_rejects_reversed_data_offsets() {
    let dir = std::env::temp_dir().join(format!("wpeer-rev-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let header = r#"{"a.weight":{"dtype":"F32","shape":[2],"data_offsets":[16,0]}}"#;
    write_st(&dir.join("model.safetensors"), header, &[0u8; 16]);

    let err = build_manifest(&dir, "test").unwrap_err().to_string();
    assert!(
        err.contains("model.safetensors"),
        "error should name the shard: {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// 2026-09-25: A span past the end of the file fails staging, naming the shard;
/// published, it would be a remote read past the end of the registered mapping.
#[test]
fn build_manifest_rejects_span_past_end_of_shard() {
    let dir = std::env::temp_dir().join(format!("wpeer-trunc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let header = r#"{"a.weight":{"dtype":"F32","shape":[1024],"data_offsets":[0,4096]}}"#;
    write_st(&dir.join("model.safetensors"), header, &[0u8; 16]);

    let err = build_manifest(&dir, "test").unwrap_err().to_string();
    assert!(
        err.contains("model.safetensors"),
        "error should name the shard: {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
