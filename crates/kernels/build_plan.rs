// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The per-target compile plan for build.rs and its execution.
//!
//! Owner: metrale-kernels build.
//! Invariants:
//! - `main` calls these once per target in the order it prints and plans;
//!   no helper reorders a `cargo:` line or a compile.
//! - `run_compile_jobs` panics only after every compile job has run.
//!
//! Included via `#[path = "build_plan.rs"] mod build_plan;` so types from
//! build.rs (`Target`, `CompileJob`) are reachable via `super::`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use metrale_closure::layout::{Entry, Layout};

use super::build_shadow::shadowed_missing_symbols;
use super::build_target::ComputeTarget;
use super::{CompileJob, Target, build_hip_shim, build_stage, build_summary};

/// 2026-09-26: Stage `target`'s layers under `stage_root` and list its
/// modules: each staged source with the directory of the layer that
/// resolved it, sorted. Also returns `Layout::modules`.
pub(super) fn stage_target<'a>(
    target: &'a Target,
    stage_root: &Path,
    source_ext: &str,
    compute_target: &dyn ComputeTarget,
) -> (
    build_stage::Staged,
    Vec<(String, &'a Entry)>,
    Vec<(PathBuf, PathBuf)>,
) {
    let layout = &target.layout;
    assert_eq!(
        layout.hardware.source_ext, source_ext,
        "({}, {}, {}): the resolver and the compute target disagree on the source extension",
        target.hw, target.model, target.quant,
    );
    let st = build_stage::stage(stage_root, layout);
    let resolved = layout.modules();
    assert!(
        !resolved.is_empty(),
        "No .{} files found for target ({}, {}, {})",
        compute_target.source_extension(),
        target.hw,
        target.model,
        target.quant,
    );
    let mut cu_files: Vec<(PathBuf, PathBuf)> = resolved
        .iter()
        .map(|(stem, entry)| {
            (
                st.module_path(layout, stem),
                layout.layers[entry.layer].dir.clone(),
            )
        })
        .collect();
    cu_files.sort();
    (st, resolved, cu_files)
}

/// 2026-09-26: Print the `rerun-if-changed` lines for everything `layout` read.
pub(super) fn emit_layout_rerun(layout: &Layout) {
    // 2026-09-25: What a rebuild must watch: every real input the resolution read —
    // the sources and headers wherever they live, the vendored
    // subdirectories, the manifests — and each layer directory, so a
    // file added to or removed from one is seen too.
    for input in layout.inputs() {
        println!("cargo:rerun-if-changed={}", input.display());
    }
    for l in &layout.layers {
        if l.dir.is_dir() {
            println!("cargo:rerun-if-changed={}", l.dir.display());
        }
    }
}

/// 2026-09-26: `(stem, module name)` for each compiled file, sorted by module
/// name. `[modules]` in KERNEL.toml renames a module.
pub(super) fn module_names(
    cu_files: &[(PathBuf, PathBuf)],
    target: &Target,
) -> Vec<(String, String)> {
    let mut modules: Vec<(String, String)> = cu_files
        .iter()
        .map(|(f, _)| {
            let stem = f.file_stem().unwrap().to_str().unwrap().to_string();
            let module_name = target
                .module_overrides
                .get(&stem)
                .cloned()
                .unwrap_or_else(|| stem.clone());
            (stem, module_name)
        })
        .collect();
    modules.sort_by(|a, b| a.1.cmp(&b.1));
    modules
}

/// 2026-09-26: The `(module, kernel)` pairs `target` drops by shadowing
/// `common/`, after printing a warning for those `[shadow_exempt]` does not
/// exempt.
pub(super) fn target_drops(
    layout: &Layout,
    st: &build_stage::Staged,
    target: &Target,
) -> Vec<(String, String)> {
    let drops = shadowed_dropped_pairs(layout, st, &target.module_overrides);
    // 2026-09-25: Warning-visible subset: everything except the pairs a
    // KERNEL.toml exempts in `[shadow_exempt]`. The full `drops` list still
    // goes into `TargetPtxSet::shadowed_dropped`. An exemption silences the
    // build warning only: the boot gate (`kernel_gate::audit_and_gate`)
    // fails on any unresolved lookup not declared in `[expected_absent]`.
    let reportable: Vec<&(String, String)> = drops
        .iter()
        .filter(|(m, f)| {
            !target
                .shadow_exempt
                .iter()
                .any(|(em, ef)| em == m && ef == f)
        })
        .collect();
    if !reportable.is_empty() {
        println!(
            "cargo:warning=metrale-kernels: ({}, {}) drops {} kernel(s) by shadowing common/: {}. \
             A dropped kernel fails CLOSED (try_kernel -> handle 0). If the model genuinely \
             cannot use it this is fine; the startup audit will only hard-error if the model's \
             dispatch actually requests it.",
            target.model,
            target.quant,
            reportable.len(),
            reportable
                .iter()
                .map(|(m, f)| format!("{m}::{f}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
    }
    drops
}

/// 2026-09-26: Print the one summary line for `target`.
pub(super) fn print_summary(
    resolved: &[(String, &Entry)],
    layout: &Layout,
    cu_files: &[(PathBuf, PathBuf)],
    target: &Target,
) {
    // 2026-09-25: Three counts, formatted by `build_summary::summary`, which
    // `tests/build_summary.rs` tests: the modules compiled, how many the
    // leaf role supplied, and how many `common/` files this hardware holds
    // itself rather than inheriting.
    let n_leaf = resolved
        .iter()
        .filter(|(_, e)| layout.layers[e.layer].role == metrale_closure::layout::Role::Leaf)
        .count();
    println!(
        "cargo:warning={}",
        build_summary::summary(
            cu_files.len(),
            &target.hw,
            &target.model,
            &target.quant,
            n_leaf,
            build_summary::overlay_owned(layout),
        )
    );
}

/// 2026-09-26: Run every unique compile on `n_threads` workers; panic with
/// every error once all jobs have run.
fn run_compile_jobs(compute: &dyn ComputeTarget, compile_jobs: &[CompileJob], n_threads: usize) {
    let errors_mutex: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let next_idx = std::sync::atomic::AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..n_threads {
            let next_idx = &next_idx;
            let compile_jobs = &compile_jobs;
            let errors_mutex = &errors_mutex;
            s.spawn(move || {
                loop {
                    let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= compile_jobs.len() {
                        break;
                    }
                    let job = &compile_jobs[i];
                    // 2026-09-25: `-I<home>` is a compile-line fact, not a
                    // target flag: it stays out of `extra_flags`, which the
                    // closure attestation hashes and records.
                    let mut flags = job.extra_flags.clone();
                    flags.push(format!("-I{}", job.include_dir.display()));
                    if let Err(e) = compute.compile(&job.cu_file, &job.out_file, &job.arch, &flags)
                    {
                        errors_mutex.lock().unwrap().push(e);
                    }
                }
            });
        }
    });

    let errors = errors_mutex.into_inner().unwrap();
    if !errors.is_empty() {
        panic!("Kernel compilation failed:\n{}", errors.join("\n"));
    }
}

/// 2026-09-26: Copy each cache hit from the output that already holds it.
fn run_copy_jobs(copy_jobs: &[(PathBuf, PathBuf)]) {
    for (src, dst) in copy_jobs {
        std::fs::copy(src, dst).unwrap_or_else(|e| {
            panic!(
                "Failed to copy cached {} → {}: {e}",
                src.display(),
                dst.display(),
            )
        });
    }
}

/// 2026-09-26: Run the work plan: the unique compiles, then the cache-hit
/// copies, then the HIP shims when `is_hip`, then the dedup summary line.
pub(super) fn run_plan(
    compute: &dyn ComputeTarget,
    compile_jobs: &[CompileJob],
    copy_jobs: &[(PathBuf, PathBuf)],
    is_hip: bool,
    manifest_dir: &Path,
    out_dir: &Path,
) {
    let nvcc_invocations = compile_jobs.len();
    let cache_hits = copy_jobs.len();
    let total = nvcc_invocations + cache_hits;

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(nvcc_invocations.max(1));

    if nvcc_invocations > 0 {
        run_compile_jobs(compute, compile_jobs, n_threads);
    }

    run_copy_jobs(copy_jobs);

    // 2026-09-25: HIP: build the shims (`build_hip_shim`) that provide the
    // CUDA libraries the runtime links, implemented on top of HIP.
    if is_hip {
        build_hip_shim(manifest_dir, out_dir);
    }

    if total > 0 {
        println!(
            "cargo:warning=metrale-kernels: dedup+parallel: {nvcc_invocations}/{total} unique nvcc \
             invocations ({cache_hits} cache hits, {:.1}× dedup), {n_threads} parallel workers",
            total as f64 / nvcc_invocations.max(1) as f64,
        );
    }
}

/// 2026-09-25: Every `(module, kernel)` this target's leaf-role files drop
/// relative to their common-role namesakes.
///
/// Shadowing is whole-file, not per-symbol: a model file replaces its common
/// namesake entirely, so a kernel `common/` has and the fork lacks is not
/// compiled for this model, and `try_kernel` returns handle 0.
///
/// The list is baked into the binary (`TargetPtxSet::shadowed_dropped`). The
/// boot audit marks an unresolved lookup found in it as `SHADOW-DROPPED` in
/// its report (`kernel_audit::unresolved_report`); whether the lookup is fatal
/// depends only on `[expected_absent]`.
///
/// Compared on the staged copies, because `build_shadow` follows quoted
/// includes from the file it reads: a fork that includes
/// `../../common/x.cu` must resolve that against this target's common/.
fn shadowed_dropped_pairs(
    layout: &Layout,
    staged: &build_stage::Staged,
    module_overrides: &HashMap<String, String>,
) -> Vec<(String, String)> {
    use metrale_closure::layout::Role;
    let ext = layout.hardware.source_ext;
    let mut out = Vec::new();
    for (stem, entry) in layout.modules() {
        if layout.layers[entry.layer].role != Role::Leaf {
            continue;
        }
        let Some((common_name, _)) = layout.common.iter().find(|(n, _)| {
            n.rsplit_once('.')
                .is_some_and(|(s, e)| s == stem && e == ext)
        }) else {
            continue;
        };
        let common_f = staged.common.join(common_name);
        let model_f = staged.leaf.join(&entry.name);
        let module = module_overrides.get(&stem).cloned().unwrap_or(stem);
        for func in shadowed_missing_symbols(&common_f, &model_f) {
            out.push((module.clone(), func));
        }
    }
    out.sort();
    out
}
