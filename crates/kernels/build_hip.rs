// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The native ROCm/HIP helpers for build.rs: the mask-widened
//! source mirror and the Windows CUDA-library shim built on HIP.
//!
//! Owner: metrale-kernels build.
//! Invariants:
//! - The source tree is never modified; only the mirror under OUT_DIR is
//!   written.
//!
//! Included via `#[path = "build_hip.rs"] mod build_hip;`.

use std::path::PathBuf;

use super::content_hash;

// 2026-09-25: HIP (vendor "hip") helpers. Only the native ROCm/HIP path calls them.

/// 2026-09-25: Mirror `src` into `mirror_root` with [`widen_warp_masks`]
/// applied, and return the mirrored path to compile.
///
/// Every source and header in `src`'s directory is mirrored on each call, so a
/// sibling `#include "foo.cuh"` resolves next to the mirrored `.cu`; a file is
/// rewritten only when its widened text differs from the mirror. The source
/// tree is never modified.
pub(super) fn hip_mirror_source(
    src: &std::path::Path,
    mirror_root: &std::path::Path,
    source_ext: &str,
) -> PathBuf {
    let src_dir = src.parent().expect("kernel source has no parent dir");
    // 2026-09-25: One mirror subdir per source dir, keyed by a hash of its path,
    // so same-named files in different directories do not collide.
    let dir_key = content_hash(&src_dir.to_string_lossy());
    let mirror_dir = mirror_root.join(dir_key);
    std::fs::create_dir_all(&mirror_dir)
        .unwrap_or_else(|e| panic!("create hip mirror subdir: {e}"));

    // 2026-09-25: Transform every source, `.cuh` and `.h` file in the source dir
    // into the mirror, so headers stay in lockstep with their `.cu`.
    if let Ok(entries) = std::fs::read_dir(src_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            let is_kernel_src = p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e == source_ext || e == "cuh" || e == "h")
                .unwrap_or(false);
            if !p.is_file() || !is_kernel_src {
                continue;
            }
            let name = p.file_name().unwrap();
            let dst = mirror_dir.join(name);
            let text = std::fs::read_to_string(&p)
                .unwrap_or_else(|e| panic!("read {} for HIP mirror: {e}", p.display()));
            let widened = widen_warp_masks(&text);
            // 2026-09-25: Only rewrite when content changed (keeps mtimes stable across
            // incremental builds → no needless hipcc recompiles).
            let needs_write = match std::fs::read_to_string(&dst) {
                Ok(existing) => existing != widened,
                Err(_) => true,
            };
            if needs_write {
                std::fs::write(&dst, &widened)
                    .unwrap_or_else(|e| panic!("write HIP mirror {}: {e}", dst.display()));
            }
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }

    mirror_dir.join(src.file_name().unwrap())
}

/// 2026-09-25: Append `ULL` to the hex warp-mask literal that is the first
/// argument of every `__shfl*_sync(` / `__ballot_sync(`, and widen
/// `unsigned int <v> = __activemask();` to `unsigned long long` (e.g.
/// `__shfl_down_sync(0xFFFFFFFF, ...)` -> `__shfl_down_sync(0xFFFFFFFFULL, ...)`).
/// Anything that is not a sync-call mask literal (e.g. a byte-extraction
/// `& 0xFFFFFFFF`) is left alone.
fn widen_warp_masks(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len() + 256);
    // 2026-09-25: Sync-call prefixes whose first argument is a warp mask.
    const SYNC_CALLS: &[&str] = &[
        "__shfl_sync(",
        "__shfl_up_sync(",
        "__shfl_down_sync(",
        "__shfl_xor_sync(",
        "__ballot_sync(",
    ];
    let mut i = 0usize;
    while i < bytes.len() {
        let mut matched = None;
        for call in SYNC_CALLS {
            if src[i..].starts_with(call) {
                matched = Some(call.len());
                break;
            }
        }
        if let Some(call_len) = matched {
            out.push_str(&src[i..i + call_len]);
            i += call_len;
            while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
                out.push(bytes[i] as char);
                i += 1;
            }
            // 2026-09-25: If a hex literal follows, emit it with a 64-bit suffix
            // (an existing LL/ULL suffix is kept). Decimal/identifier masks are
            // left untouched.
            if i + 1 < bytes.len()
                && bytes[i] == b'0'
                && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X')
            {
                let start = i;
                i += 2;
                while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                    i += 1;
                }
                let lit = &src[start..i];
                // 2026-09-25: Consume any existing integer suffix (u/U/l/L) so it is not doubled.
                let mut suffix_end = i;
                while suffix_end < bytes.len()
                    && matches!(bytes[suffix_end], b'u' | b'U' | b'l' | b'L')
                {
                    suffix_end += 1;
                }
                let existing_suffix = &src[i..suffix_end];
                out.push_str(lit);
                if existing_suffix.to_ascii_uppercase().contains("LL") {
                    out.push_str(existing_suffix);
                } else {
                    out.push_str("ULL");
                }
                i = suffix_end;
            }
            continue;
        }
        // 2026-09-25: Widen an `unsigned int` declaration whose statement calls
        // `__activemask()` to `unsigned long long`.
        if src[i..].starts_with("unsigned int ") {
            let stmt_end = src[i..].find(';').map(|o| i + o).unwrap_or(bytes.len());
            if src[i..stmt_end].contains("__activemask()") {
                out.push_str("unsigned long long ");
                i += "unsigned int ".len();
                continue;
            }
        }
        // 2026-09-25: Otherwise copy one UTF-8 character.
        let ch_len = src[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
        out.push_str(&src[i..i + ch_len]);
        i += ch_len;
    }
    out
}

/// 2026-09-25: Windows native-HIP runtime shim. Builds one `cuda.dll` from the
/// same three sources as the Linux shims (libcuda_hip_shim.cpp,
/// libcudart_hip_shim.cpp, libcublaslt_stub.cpp), linked against
/// `amdhip64.lib`, plus its import library. cudarc loads `cuda` or `nvcuda` at
/// runtime, so the DLL is also copied to `nvcuda.dll`; the workspace links
/// `cuda`, `cudart` and `cublasLt`, so the import lib is copied to all three
/// names (each carries every export). `nvcuda.dll` and any `amdhip64*.dll`
/// found in the SDK's `bin` are copied into OUT_DIR.
pub(super) fn build_hip_shim_windows(manifest_dir: &std::path::Path, out_dir: &std::path::Path) {
    let hipcc = std::env::var("METRALE_HIPCC")
        .expect("METRALE_HIPCC must point at the Windows HIP SDK hipcc for the hip target");
    let hip_path =
        std::env::var("HIP_PATH").expect("HIP_PATH must be set for the windows hip build");
    let hip_root = std::path::Path::new(&hip_path);

    // 2026-09-25: 1. Compile the three shims to objects (host C++ over HIP; no -fPIC on MSVC).
    let sources = [
        "libcuda_hip_shim.cpp",
        "libcudart_hip_shim.cpp",
        "libcublaslt_stub.cpp",
    ];
    let mut objs = Vec::new();
    for name in sources {
        let src = manifest_dir.join("hip").join(name);
        assert!(src.exists(), "HIP shim source missing at {}", src.display());
        println!("cargo:rerun-if-changed={}", src.display());
        let obj = out_dir.join(format!("{name}.obj"));
        let status = std::process::Command::new(&hipcc)
            .args(["-c", "-O2"])
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .status()
            .unwrap_or_else(|e| panic!("hipcc -c failed for {name} ({e})"));
        assert!(status.success(), "hipcc failed compiling {name}");
        objs.push(obj);
    }

    // 2026-09-25: 2. Export list: exactly the extern symbols the objects define,
    // read back with MSVC `dumpbin /SYMBOLS`. Reading the objects means the .def
    // cannot drift from the sources. Defined externals are
    // `SECTn ... External | <name>`; undefined imports (the hip* the shim calls)
    // are `UNDEF` and start with `hip`, so filtering on `External`, not `UNDEF`,
    // and a `cu` name prefix keeps exactly cu*/cudart/cublasLt.
    let mut exports = Vec::new();
    for obj in &objs {
        let out = std::process::Command::new("dumpbin")
            .arg("/SYMBOLS")
            .arg(obj)
            .output()
            .unwrap_or_else(|e| panic!("dumpbin /SYMBOLS failed on {} ({e})", obj.display()));
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            if line.contains("External")
                && !line.contains("UNDEF")
                && let Some(name) = line.split_whitespace().last()
                && name.starts_with("cu")
            {
                exports.push(name.to_string());
            }
        }
    }
    exports.sort();
    exports.dedup();
    assert!(
        !exports.is_empty(),
        "no cu*/cudart/cublasLt exports found in shim objects"
    );
    let def = out_dir.join("metrale_hip_cuda.def");
    std::fs::write(&def, format!("EXPORTS\n{}\n", exports.join("\n"))).expect("write cuda.def");

    // 2026-09-25: 3. Link cuda.dll + its import lib against amdhip64.
    let dll = out_dir.join("cuda.dll");
    let implib = out_dir.join("cuda.lib");
    let amdhip = hip_root.join("lib").join("amdhip64.lib");
    let mut link = std::process::Command::new(&hipcc);
    link.arg("-shared");
    for obj in &objs {
        link.arg(obj);
    }
    let status = link
        .arg(&amdhip)
        .arg("-o")
        .arg(&dll)
        .arg(format!("-Wl,/DEF:{}", def.display()))
        .arg(format!("-Wl,/IMPLIB:{}", implib.display()))
        .status()
        .unwrap_or_else(|e| panic!("hipcc -shared (cuda.dll) failed ({e})"));
    assert!(status.success(), "linking cuda.dll failed");

    // 2026-09-25: 4. cudarc loads nvcuda.dll; the workspace links
    // cuda/cudart/cublasLt.lib. One DLL, one import lib carrying every export,
    // copied to each needed name.
    std::fs::copy(&dll, out_dir.join("nvcuda.dll")).expect("copy cuda.dll -> nvcuda.dll");
    for lib in ["cudart.lib", "cublasLt.lib"] {
        std::fs::copy(&implib, out_dir.join(lib))
            .unwrap_or_else(|e| panic!("copy import lib -> {lib}: {e}"));
    }

    // 2026-09-25: 5. Copy the HIP runtime DLL into OUT_DIR: every
    // `amdhip64*.dll` under the SDK's `bin`. If there is none, the build emits
    // a cargo warning and continues: the DLL is an AMD driver component, so an
    // AMD Windows host supplies it at runtime.
    let mut staged_runtime = false;
    if let Ok(entries) = std::fs::read_dir(hip_root.join("bin")) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("amdhip64") && name.ends_with(".dll") {
                std::fs::copy(entry.path(), out_dir.join(&*name))
                    .unwrap_or_else(|e| panic!("stage {name}: {e}"));
                staged_runtime = true;
            }
        }
    }
    if !staged_runtime {
        println!(
            "cargo:warning=metrale-kernels: no amdhip64*.dll in {}\\bin to bundle — it is an AMD-driver component, present on real AMD Windows hosts at runtime.",
            hip_root.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // 2026-09-25: Expose the dir holding nvcuda.dll/amdhip64*.dll as
    // METRALE_HIP_RUNTIME_DIR; nothing in the tree reads it.
    println!(
        "cargo:rustc-env=METRALE_HIP_RUNTIME_DIR={}",
        out_dir.display()
    );
    println!(
        "cargo:warning=metrale-kernels: built Windows HIP runtime shim (cuda.dll/nvcuda.dll + import libs, {} exports) at {}",
        exports.len(),
        out_dir.display()
    );
}
