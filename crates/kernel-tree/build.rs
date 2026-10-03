// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Packs every `kernels/**/*.{toml,cu,cuh,h}` file into `OUT_DIR/kernel_tree.bin`
//! (records of `u32` path length, path, `u64` length, bytes; sorted by path; deflated) and its
//! SHA-256 into `OUT_DIR/kernel_tree.sha256`.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const EXTENSIONS: [&str; 4] = ["toml", "cu", "cuh", "h"];

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk(&p, out);
        } else if p
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| EXTENSIONS.contains(&x))
        {
            out.push(p);
        }
    }
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let root = manifest
        .join("../..")
        .canonicalize()
        .expect("repository root");
    let kernels = root.join("kernels");
    println!("cargo:rerun-if-changed={}", kernels.display());
    let mut files = Vec::new();
    walk(&kernels, &mut files);
    let mut rels: Vec<(String, &PathBuf)> = files
        .iter()
        .map(|f| {
            let rel = f.strip_prefix(&root).expect("under the root");
            (rel.to_string_lossy().replace('\\', "/"), f)
        })
        .collect();
    rels.sort();
    let mut raw = Vec::new();
    for (rel, f) in rels {
        let bytes = std::fs::read(f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        raw.extend_from_slice(&(rel.len() as u32).to_le_bytes());
        raw.extend_from_slice(rel.as_bytes());
        raw.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        raw.extend_from_slice(&bytes);
    }
    let digest: String = Sha256::digest(&raw)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let mut z = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::best());
    z.write_all(&raw).expect("deflate");
    let packed = z.finish().expect("deflate");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    std::fs::write(out.join("kernel_tree.bin"), packed).expect("write kernel_tree.bin");
    std::fs::write(out.join("kernel_tree.sha256"), digest).expect("write kernel_tree.sha256");
}
