// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Build script for metrale-kernels: resolves the targets `METRALE_TARGET_{HW,MODEL,QUANT}` select, compiles their kernel sources and generates `OUT_DIR/target_ptx.rs`.
//!
//! Owner: metrale-kernels build.
//! Invariants:
//! - Both paths, the skip stub (`METRALE_SKIP_BUILD`, or macOS without an
//!   explicit `METRALE_TARGET_HW`) and the compiling build, write
//!   `target_ptx.rs` with `TARGET_DEFAULTS` and `TARGET_SM_COUNT` appended and
//!   emit `METRALE_KERNEL_SET_HASH` over that file's content.
//! - A failed kernel compile panics the build, after every compile job has run.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

/// 2026-09-26: One unique compile in the build's work plan: `main` queues it
/// and `build_plan::run_compile_jobs` runs it.
struct CompileJob {
    cu_file: std::path::PathBuf,
    arch: String,
    extra_flags: Vec<String>,
    out_file: std::path::PathBuf,
    /// 2026-09-25: The layer directory that resolved the source, passed as
    /// `-I` so a quoted include that escapes the stage still resolves.
    /// `gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu` includes
    /// `../../../gb10/qwen3.6-27b/nvfp4/q4k_vendor/*.cuh`, a path that only
    /// resolves from a directory three levels below `kernels/`.
    include_dir: std::path::PathBuf,
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // 2026-09-25: Two levels up from `crates/kernels`. Resolved before the skip
    // branch because `target_defaults_literal` runs on both paths.
    let workspace_root_owned = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/kernels is two levels below the workspace root")
        .to_path_buf();

    println!("cargo:rerun-if-env-changed=METRALE_SKIP_BUILD");
    // 2026-09-25: Without these, cargo reuses the previous build-script output
    // when only the target-selection env vars change, and the binary keeps an
    // out-of-date kernel registry.
    println!("cargo:rerun-if-env-changed=METRALE_TARGET_HW");
    println!("cargo:rerun-if-env-changed=METRALE_TARGET_MODEL");
    println!("cargo:rerun-if-env-changed=METRALE_TARGET_QUANT");
    // 2026-09-25: METRALE_EXTRA_NVCC_FLAGS: extra nvcc flags appended by
    // `build_target::NvidiaTarget::compile`, for kernel bisection (e.g.
    // `-DMETRALE_FAST_SOFTMAX_EXP=1`, an `#ifdef` in prefill_paged_compute.cuh).
    println!("cargo:rerun-if-env-changed=METRALE_EXTRA_NVCC_FLAGS");
    // 2026-09-25: Skip the kernel build on macOS unless `METRALE_TARGET_HW` is
    // set: the default `gb10` target needs nvcc, which a Mac does not have.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let hw_explicit = env::var("METRALE_TARGET_HW").is_ok();
    let auto_skip_macos = target_os == "macos" && !hw_explicit;
    let skip_env = matches!(
        env::var("METRALE_SKIP_BUILD").as_deref(),
        Ok("1") | Ok("true")
    );
    if skip_env || auto_skip_macos {
        write_skip_stub(&out_dir, &workspace_root_owned);
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

    let targets = resolve_targets(workspace_root);

    assert!(
        !targets.is_empty(),
        "{}",
        build_diagnose::no_targets(
            &workspace_root.join("kernels"),
            &env::var("METRALE_TARGET_HW").unwrap_or_else(|_| build_diagnose::DEFAULT_HW.into()),
            &env::var("METRALE_TARGET_MODEL").unwrap_or_else(|_| "*".into()),
            &env::var("METRALE_TARGET_QUANT").unwrap_or_else(|_| "nvfp4".into()),
        )
    );

    // 2026-09-25: HARDWARE.toml `vendor` selects the compiler
    // (`build_target::resolve_compute_target`): nvidia/cuda -> nvcc,
    // apple/metal -> xcrun, amd/rocm/scale -> SCALE's nvcc, hip -> hipcc.
    let hw_dir = workspace_root
        .join("kernels")
        .join(env::var("METRALE_TARGET_HW").unwrap_or_else(|_| build_diagnose::DEFAULT_HW.into()));
    let hw_toml_path = hw_dir.join("HARDWARE.toml");
    let hw_toml: toml::Value = {
        let content = std::fs::read_to_string(&hw_toml_path)
            .unwrap_or_else(|_| panic!("Cannot read {}", hw_toml_path.display()));
        toml::from_str(&content).unwrap_or_else(|e| panic!("Invalid HARDWARE.toml: {e}"))
    };
    let vendor_str = hw_toml
        .get("hardware")
        .and_then(|h| h.get("vendor"))
        .and_then(|v| v.as_str());
    let compute_target = resolve_compute_target(vendor_str);
    let output_ext = compute_target.output_extension();
    let uses_cuda_api = compute_target.uses_cuda_module_api();

    let mut all_target_modules: Vec<Vec<(String, String)>> = Vec::new();
    // 2026-09-25: Per target, the (module, kernel) pairs its leaf-role files
    // drop by shadowing their `common/` namesakes, baked into
    // `TargetPtxSet::shadowed_dropped`. See `shadowed_dropped_pairs`.
    let mut all_target_drops: Vec<Vec<(String, String)>> = Vec::new();

    // 2026-09-25: Pre-walk every (target, cu_file) pair into two queues:
    //   - `compile_jobs`: unique (source, arch, sorted_flags) tuples that
    //     need nvcc, run in parallel via thread::scope;
    //   - `copy_jobs`: cache hits, where another target already produced
    //     this exact binary, copied sequentially after the compiles.
    let mut compile_jobs: Vec<CompileJob> = Vec::new();
    let mut copy_jobs: Vec<(std::path::PathBuf, std::path::PathBuf)> = Vec::new();
    let mut compile_cache: std::collections::HashMap<
        (std::path::PathBuf, String, Vec<String>),
        std::path::PathBuf,
    > = std::collections::HashMap::new();

    let source_ext = compute_target.source_extension();

    // 2026-09-25: Native ROCm/HIP path (vendor "hip"). Every block gated on
    // `is_hip` is a no-op for the other vendors.
    //   (a) Point METRALE_HIP_COMPAT_INCLUDE at the CUDA->HIP compat headers
    //       (`hip/compat`): HipTarget::compile force-includes
    //       hip/hip_runtime.h and `-I`s this dir, so the unmodified `.cu` find
    //       cuda_runtime.h / cuda_bf16.h / cuda_fp8.h.
    //   (b) Mask widening: HIP's `__shfl_*_sync` / `__ballot_sync` masks and
    //       `__activemask()` are 64-bit, so each source directory is mirrored
    //       into OUT_DIR with `widen_warp_masks` applied (kernels/ is never
    //       modified) and the mirror is compiled. The other vendors compile the
    //       staged copies directly.
    let is_hip = vendor_str == Some("hip");
    let hip_mirror_dir = out_dir.join("hip_mirror");
    if is_hip {
        let compat_dir = manifest_dir.join("hip").join("compat");
        assert!(
            compat_dir.join("cuda_runtime.h").exists(),
            "HIP compat headers missing at {} — expected the staged \
             metrale-kernels/hip/compat dir (cuda_runtime.h, cuda_bf16.h, cuda_fp8.h).",
            compat_dir.display()
        );
        println!(
            "cargo:rustc-env=METRALE_HIP_COMPAT_INCLUDE={}",
            compat_dir.display()
        );
        // 2026-09-25: HipTarget::compile reads this via std::env::var in this
        // build-script process (the `cargo:rustc-env` directive only reaches the
        // compiled crate). Set before the parallel compile scope is spawned, so
        // no other thread is running.
        unsafe {
            std::env::set_var("METRALE_HIP_COMPAT_INCLUDE", &compat_dir);
        }
        println!("cargo:rerun-if-changed={}", compat_dir.display());
        std::fs::create_dir_all(&hip_mirror_dir)
            .unwrap_or_else(|e| panic!("create hip_mirror dir: {e}"));
    }

    // 2026-09-25: Stage every target's layers, then plan the compiles.
    // The stage is what the compiler sees: `common/` and `<source model>/
    // <quant>/`, each holding the files its layer resolves to, so a source a
    // directory `use`s compiles with that directory's headers beside it.
    let stage_root = out_dir.join("stage");
    build_stage::reset(&stage_root);
    let mut staged: Vec<build_stage::Staged> = Vec::new();

    for (idx, target) in targets.iter().enumerate() {
        let layout = &target.layout;
        let (st, resolved, cu_files) =
            build_plan::stage_target(target, &stage_root, source_ext, compute_target.as_ref());

        // 2026-09-25: Gather work for this target; nothing compiles until the
        // full work plan is built.
        for (cu_file, home) in &cu_files {
            let stem = cu_file.file_stem().unwrap().to_str().unwrap().to_string();
            let out_file = out_dir.join(format!("t{idx}__{stem}.{output_ext}"));

            // 2026-09-25: HIP compiles a mask-widened mirror of the source
            // (`hip_mirror_source`); the other vendors compile the staged copy.
            let compile_source = if is_hip {
                hip_mirror_source(cu_file, &hip_mirror_dir, source_ext)
            } else {
                cu_file.clone()
            };

            // 2026-09-25: Dedup key: same (source, arch, sorted flags) -> identical
            // binary output. Sorting keeps flag order from splitting the cache.
            // Staged paths are shared by every target that resolves the same
            // role directory, so those targets share compiles.
            let mut sorted_flags = target.extra_flags.clone();
            sorted_flags.sort();
            let key = (compile_source.clone(), target.arch.clone(), sorted_flags);

            if let Some(existing) = compile_cache.get(&key) {
                copy_jobs.push((existing.clone(), out_file));
            } else {
                compile_jobs.push(CompileJob {
                    cu_file: compile_source.clone(),
                    arch: target.arch.clone(),
                    extra_flags: target.extra_flags.clone(),
                    out_file: out_file.clone(),
                    include_dir: home.clone(),
                });
                compile_cache.insert(key, out_file);
            }
        }

        build_plan::emit_layout_rerun(layout);

        let modules = build_plan::module_names(&cu_files, target);

        all_target_modules.push(modules);

        let drops = build_plan::target_drops(layout, &st, target);
        all_target_drops.push(drops);

        build_plan::print_summary(&resolved, layout, &cu_files, target);
        staged.push(st);
    }

    build_plan::run_plan(
        compute_target.as_ref(),
        &compile_jobs,
        &copy_jobs,
        is_hip,
        &manifest_dir,
        &out_dir,
    );

    emit_generated(
        &targets,
        &all_target_modules,
        &all_target_drops,
        output_ext,
        uses_cuda_api,
        workspace_root,
        &out_dir,
    );

    // 2026-09-25: Bake what each target was compiled from. Read back as
    // `metrale_kernels::TARGET_CLOSURES` by the benchmark record
    // (`bench_record.rs`), so a record attests to the binary's sources rather
    // than to whatever the tree held when the record was written.
    println!(
        "cargo:rustc-env=METRALE_TARGET_CLOSURES={}",
        closure_attestation(workspace_root, &targets, compute_target.as_ref())
    );
}

/// 2026-09-25: FNV-1a 64-bit content fingerprint, truncated to its low 48 bits
/// (12 hex chars). Used for `METRALE_KERNEL_SET_HASH` and the HIP mirror
/// directory names.
fn content_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in s.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:012x}", h & 0xffff_ffff_ffff)
}

/// 2026-09-25: Compile the HIP shims (`libcuda.so`, `libcudart.so`,
/// `libcublasLt.so`) into `out_dir` and emit the link search-path directive.
/// The runtime keeps importing the CUDA APIs; the shims implement them on HIP.
fn build_hip_shim(manifest_dir: &std::path::Path, out_dir: &std::path::Path) {
    // 2026-09-25: Windows builds one cuda.dll + import libs instead and copies
    // the runtime DLLs into OUT_DIR. See build_hip_shim_windows.
    if cfg!(windows) {
        build_hip_shim_windows(manifest_dir, out_dir);
        return;
    }
    let hipcc = std::env::var("METRALE_HIPCC").unwrap_or_else(|_| "/opt/rocm/bin/hipcc".into());
    // 2026-09-25: One shim per CUDA library the workspace links:
    //   libcuda.so     — cu* driver API (libcuda_hip_shim.cpp)
    //   libcudart.so   — cudart runtime API (libcudart_hip_shim.cpp)
    //   libcublasLt.so — cuBLASLt stub (libcublaslt_stub.cpp), for the
    //                    `METRALE_CUBLAS_GEMM` path
    for (src_name, so_name) in [
        ("libcuda_hip_shim.cpp", "libcuda.so"),
        ("libcudart_hip_shim.cpp", "libcudart.so"),
        ("libcublaslt_stub.cpp", "libcublasLt.so"),
    ] {
        let src = manifest_dir.join("hip").join(src_name);
        assert!(src.exists(), "HIP shim source missing at {}", src.display());
        println!("cargo:rerun-if-changed={}", src.display());
        let out = out_dir.join(so_name);
        let status = std::process::Command::new(&hipcc)
            .args([
                "-shared",
                "-fPIC",
                src.to_str().unwrap(),
                "-o",
                out.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("failed to run hipcc for {so_name} ({hipcc}): {e}"));
        assert!(
            status.success(),
            "hipcc failed building {so_name} from {}",
            src.display()
        );
    }
    // 2026-09-25: OUT_DIR on the link search path so `-lcuda`/`-lcudart`/`-lcublasLt`
    // resolve to the shims.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!(
        "cargo:warning=metrale-kernels: built HIP shims (libcuda/libcudart/libcublasLt) in {}",
        out_dir.display()
    );
}

/// 2026-09-25: Fail the build when differently-named targets collide on a
/// `(model_type, hidden_size)` declaration without every participant
/// declaring `[model] match_names`, or when one participant's needles can
/// never win against another's.
///
/// Runtime resolution (`metrale_kernels::resolve`) breaks such a collision by
/// matching the needles against the checkpoint reference and errors when it
/// cannot, so a colliding target with no needles could never be selected.
fn validate_collision_match_names(targets: &[Target]) {
    let mut by_pair: HashMap<(&str, Option<usize>), Vec<&Target>> = HashMap::new();
    for t in targets {
        for m in &t.model_type_matches {
            by_pair
                .entry((m.model_type.as_str(), m.hidden_size))
                .or_default()
                .push(t);
        }
    }
    for ((model_type, hidden), group) in by_pair {
        let mut names: Vec<&str> = group.iter().map(|t| t.model.as_str()).collect();
        names.sort();
        names.dedup();
        if names.len() < 2 {
            continue; // 2026-09-25: no collision (multi-quant variants of one target are fine)
        }
        for t in &group {
            assert!(
                !t.match_names.is_empty(),
                "kernel targets {names:?} all declare (model_type \"{model_type}\", \
                 hidden_size {hidden:?}) in [[model_types]], but target \"{}\" has no \
                 [model] match_names — runtime resolution could never select it \
                 unambiguously. Add e.g. `match_names = [\"{}\"]` to \
                 kernels/<hw>/{}/MODEL.toml (needles are case-insensitive substrings \
                 of the checkpoint's HF id / --model-name / model dir).",
                t.model,
                t.model,
                t.model,
            );
        }
        // 2026-09-25: Presence is not enough: a target whose every needle
        // contains one of a colliding sibling's needles can never be the unique
        // match, so every reference it exists to route hits the Ambiguous
        // startup error. This rejects that at build time.
        for a in &group {
            for b in &group {
                if a.model == b.model {
                    continue;
                }
                let winnable = a.match_names.iter().any(|na| {
                    let na = na.to_lowercase();
                    !b.match_names
                        .iter()
                        .any(|nb| na.contains(&nb.to_lowercase()))
                });
                assert!(
                    winnable,
                    "kernel target \"{}\" collides with \"{}\" on (model_type \
                     \"{model_type}\", hidden_size {hidden:?}), but every one of its \
                     match_names {:?} contains one of \"{}\"'s needles {:?} — any \
                     reference matching \"{}\" also matches \"{}\", so \"{}\" can \
                     never win the tie and the tier always hard-errors Ambiguous. \
                     Give \"{}\" a needle the sibling does not shadow, or remove \
                     the colliding [[model_types]] entry.",
                    a.model,
                    b.model,
                    a.match_names,
                    b.model,
                    b.match_names,
                    a.model,
                    b.model,
                    a.model,
                    a.model,
                );
            }
        }
    }
}

#[path = "build_parse.rs"]
mod build_parse;

// 2026-09-25: Entry-point resolution for `shadowed_dropped_pairs`. Its own file
// so `tests/kernel_shadow_detector.rs` can compile the same code: `cargo test`
// never runs a build script's `#[cfg(test)]` modules.
#[path = "build_shadow.rs"]
mod build_shadow;

// 2026-09-25: The HARDWARE.toml `arch` -> `KernelTarget.arch` mapping. Its own file, with
// no `super::` dependencies, so `tests/kernel_target_arch.rs` can compile the
// same code — cargo never runs a build script's `#[cfg(test)]` modules.
#[path = "build_arch.rs"]
mod build_arch;

// 2026-09-25: The extra-compiler-flag layers and their merge rule. Same reason for its own
// file as `build_arch.rs`: `tests/kernel_build_flags.rs` compiles it directly.
#[path = "build_flags.rs"]
mod build_flags;

// 2026-09-25: The per-target serving defaults (`[defaults]` in HARDWARE.toml) and the
// `[hardware] sm_count`. Same reason for its own file:
// `tests/target_defaults.rs` compiles it against the real kernels/ tree.
#[path = "build_defaults.rs"]
mod build_defaults;

// 2026-09-25: The one summary line a build prints per kernel target. Same reason for its
// own file: `tests/build_summary.rs` compiles it directly.
#[path = "build_summary.rs"]
mod build_summary;

#[path = "build_codegen.rs"]
mod build_codegen;

#[path = "build_target.rs"]
mod build_target;
use build_target::resolve_compute_target;

#[path = "build_diagnose.rs"]
mod build_diagnose;

#[path = "build_types.rs"]
mod build_types;
use build_types::{DflashRaw, ModelTypeMatch, SamplingCat, Target};

#[path = "build_resolve.rs"]
mod build_resolve;
use build_resolve::resolve_targets;

#[path = "build_plan.rs"]
mod build_plan;

#[path = "build_emit.rs"]
mod build_emit;
use build_emit::{closure_attestation, emit_generated, write_skip_stub};

#[path = "build_hip.rs"]
mod build_hip;
use build_hip::{build_hip_shim_windows, hip_mirror_source};

// 2026-09-25: Staging of the resolved layers into OUT_DIR, so a `use`d source compiles
// beside the headers of the directory that uses it.
#[path = "build_stage.rs"]
mod build_stage;
