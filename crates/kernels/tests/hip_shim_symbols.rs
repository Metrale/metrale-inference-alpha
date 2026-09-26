// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Checks that the HIP shims define every CUDA driver, runtime and
//! cuBLASLt entry point the workspace declares in an `extern "C"` block.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! On the `hip` vendor, build.rs builds `hip/*.cpp` as the `cuda`, `cudart`
//! and `cublasLt` libraries the workspace links (a `cuda.dll` whose export
//! list is exactly the symbols the objects define, on Windows). A declaration
//! with no shim definition therefore fails only at that target's link step,
//! which no Linux job runs.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .to_path_buf()
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for e in read.flatten() {
        let path = e.path();
        if path.is_dir() {
            if e.file_name() != "target" {
                rust_files(&path, out);
            }
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

/// 2026-09-26: A name the shims own: `cu*` driver, `cuda*` runtime or
/// `cublasLt*` entry points, each followed by an upper-case letter.
fn is_shimmed(name: &str) -> bool {
    ["cublasLt", "cuda", "cu"].iter().any(|p| {
        name.strip_prefix(p)
            .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_uppercase()))
    })
}

fn ident_at(text: &str) -> &str {
    let end = text
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(text.len());
    &text[..end]
}

/// 2026-09-26: Names declared as `fn NAME(...) -> ...;`: a signature that
/// ends at `;` before any `{` is a foreign declaration.
fn declared(text: &str, out: &mut Vec<(String, String)>, file: &str) {
    for (i, _) in text.match_indices("fn ") {
        if i > 0 && text[..i].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        let name = ident_at(&text[i + 3..]);
        if !is_shimmed(name) {
            continue;
        }
        let rest = &text[i..];
        let semi = rest.find(';');
        let brace = rest.find('{');
        if semi.is_some_and(|s| brace.is_none_or(|b| s < b)) {
            out.push((name.to_string(), file.to_string()));
        }
    }
}

/// 2026-09-26: Names the shims define at top level as `int NAME(`.
fn defined(shim: &str) -> Vec<String> {
    shim.lines()
        .filter_map(|l| l.strip_prefix("int "))
        .map(|rest| ident_at(rest.trim_start()).to_string())
        .filter(|n| is_shimmed(n))
        .collect()
}

#[test]
fn every_declared_cuda_entry_point_has_a_hip_shim_definition() {
    let root = workspace_root();
    let hip = root.join("crates/kernels/hip");
    let mut defs = Vec::new();
    for name in [
        "libcuda_hip_shim.cpp",
        "libcudart_hip_shim.cpp",
        "libcublaslt_stub.cpp",
    ] {
        let path = hip.join(name);
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        defs.extend(defined(&text));
    }
    assert!(
        defs.len() > 50,
        "only {} shim definitions parsed",
        defs.len()
    );

    let mut files = Vec::new();
    rust_files(&root.join("crates"), &mut files);
    let mut decls = Vec::new();
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        let rel = f.strip_prefix(&root).unwrap().display().to_string();
        declared(&text, &mut decls, &rel);
    }
    assert!(decls.len() > 50, "only {} declarations found", decls.len());

    let mut missing: Vec<String> = decls
        .iter()
        .filter(|(n, _)| !defs.contains(n))
        .map(|(n, f)| format!("{n} (declared in {f})"))
        .collect();
    missing.sort();
    missing.dedup();
    assert!(
        missing.is_empty(),
        "{} CUDA entry point(s) the workspace declares have no definition in \
         crates/kernels/hip/*.cpp, so the HIP target cannot link:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}
