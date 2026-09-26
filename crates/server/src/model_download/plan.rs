// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Which files of a Hub repo to fetch, in what order, and how many bytes that is. No network and no filesystem access.
//!
//! Owner: server (model download).
//! Invariants:
//! - Every name `select` returns passes `is_contained`: it is not empty, does
//!   not start with `/`, has no `\` and no drive prefix, and has no empty,
//!   `.` or `..` component.

/// 2026-09-26: One file as the Hub lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    pub name: String,
    /// 2026-09-26: Bytes, when known. `hf::repo_info` leaves it `None`; `run`
    /// fills it from `hf::sizes`.
    pub size: Option<u64>,
}

/// 2026-09-26: Top-level metadata files fetched by exact name. A fixed list,
/// not every `.json`, so evaluation results and other extra files are skipped.
const METADATA: &[&str] = &[
    "config.json",
    "params.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
];

/// 2026-09-26: Non-safetensors weight formats, and directories (`original/`,
/// `onnx/`, ...) that hold other copies of the model.
fn is_excluded(name: &str) -> bool {
    const SKIP_DIRS: &[&str] = &["original/", "onnx/", "openvino/", "coreml/", "tflite/"];
    const SKIP_EXT: &[&str] = &[
        ".bin", ".pth", ".pt", ".msgpack", ".h5", ".onnx", ".gguf", ".tflite",
    ];
    SKIP_DIRS.iter().any(|d| name.starts_with(d)) || SKIP_EXT.iter().any(|e| name.ends_with(e))
}

fn is_weight(name: &str) -> bool {
    name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json")
}

/// 2026-09-26: Does this name stay inside the snapshot directory it is joined
/// onto?
///
/// The name is the Hub's `rfilename`, chosen by the repo's publisher, and the
/// downloader joins it onto the cache path. `Path::join` keeps `..` components
/// and lets an absolute name replace the base, so an unchecked name could write
/// anywhere the process can. Weights may sit in subdirectories, so `/` itself
/// is allowed.
fn is_contained(name: &str) -> bool {
    // 2026-09-26: `\` is a Windows separator, and a `C:` prefix is a Windows
    // drive.
    if name.is_empty() || name.starts_with('/') || name.contains('\\') {
        return false;
    }
    let b = name.as_bytes();
    if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
        return false;
    }
    // 2026-09-26: An empty component comes from `//`.
    name.split('/')
        .all(|c| !c.is_empty() && c != "." && c != "..")
}

/// 2026-09-26: Is this file part of the plan?
pub fn wanted(name: &str) -> bool {
    if !is_contained(name) || is_excluded(name) {
        return false;
    }
    // 2026-09-26: Metadata only at the top level; `subdir/config.json` is
    // skipped.
    is_weight(name) || (!name.contains('/') && METADATA.contains(&name))
}

/// 2026-09-26: The files to fetch, in fetch order: metadata first, then weight
/// files (shards and index) by ascending size, ties by name. An unknown size
/// sorts as 0. So `config.json` is on disk before any shard.
pub fn select(files: &[RemoteFile]) -> Vec<RemoteFile> {
    let mut out: Vec<RemoteFile> = files.iter().filter(|f| wanted(&f.name)).cloned().collect();
    out.sort_by(|a, b| {
        let key = |f: &RemoteFile| (is_weight(&f.name), f.size.unwrap_or(0), f.name.clone());
        key(a).cmp(&key(b))
    });
    out
}

/// 2026-09-26: Does the plan contain a `.safetensors` file? A GGUF-only repo
/// does not, and `run` refuses it before fetching anything.
pub fn has_weights(plan: &[RemoteFile]) -> bool {
    plan.iter().any(|f| f.name.ends_with(".safetensors"))
}

/// 2026-09-26: Total bytes of a plan, counting only files whose size is known.
pub fn total_bytes(plan: &[RemoteFile]) -> u64 {
    plan.iter().filter_map(|f| f.size).sum()
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
