// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Compiles the RDMA verbs C shim (`src/rdma_shim.c`) and, with
//! the `nvtx` feature, the NVTX shim, and decides `cfg(metrale_rdma_verbs)`.
//!
//! The verbs shim is compiled, and the cfg emitted, only when the target OS is
//! Linux, `METRALE_NO_RDMA` is not `1`, and neither `METRALE_SKIP_BUILD` nor
//! `SKIP_METRALE_BUILD` is `1`, `true` or `TRUE`. The rdma-core headers are not
//! probed: when they are missing, the `cc` compile fails the build.
//!
//! `rustc-cfg` does not reach other crates, so the script also prints
//! `cargo:has_verbs=1`. Through `links = "metrale_rdma_shim"`, cargo shows it
//! to the build scripts of direct dependents only, as
//! `DEP_METRALE_RDMA_SHIM_HAS_VERBS`; metrale-storage and metrale-model-weights
//! re-emit the cfg from it.
//!
//! Owner: metrale-gpu-sys.
//! Invariants:
//! - The check-cfg line is printed on every path.
//! - `cfg(metrale_rdma_verbs)` and `has_verbs` are printed together or not at
//!   all.

fn main() {
    // 2026-09-26: Before any early return: the cfg name must be declared even
    // when the cfg is off, because `unexpected_cfgs` is an error under the
    // workspace's `warnings = "deny"`.
    println!("cargo:rustc-check-cfg=cfg(metrale_rdma_verbs)");
    println!("cargo:rerun-if-env-changed=METRALE_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=SKIP_METRALE_BUILD");
    println!("cargo:rerun-if-changed=src/rdma_shim.c");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=METRALE_NO_RDMA");

    // 2026-09-26: Before the early returns below, which concern only the RDMA
    // shim.
    if std::env::var_os("CARGO_FEATURE_NVTX").is_some() {
        build_nvtx_shim();
    }

    // 2026-09-26: Leaves out only the verbs shim and the cfg, so dependents
    // build their `not(metrale_rdma_verbs)` code. `METRALE_SKIP_BUILD` also
    // skips metrale-storage's kernel compile.
    if std::env::var("METRALE_NO_RDMA").as_deref() == Ok("1") {
        return;
    }

    // 2026-09-26: rdma-core (libibverbs) is a Linux library; every other target
    // builds without the shim.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }
    if skip_build() {
        return;
    }

    cc::Build::new()
        .file("src/rdma_shim.c")
        .opt_level(2)
        .warnings(true)
        .compile("metrale_rdma_shim");
    println!("cargo:rustc-link-lib=dylib=ibverbs");
    println!("cargo:rustc-cfg=metrale_rdma_verbs");
    println!("cargo:has_verbs=1");
}

/// 2026-09-26: Compile `src/nvtx_shim.c` against `$CUDA_HOME/include`.
/// Panics when `CUDA_HOME` is unset.
fn build_nvtx_shim() {
    println!("cargo:rerun-if-changed=src/nvtx_shim.c");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    let cuda = std::env::var("CUDA_HOME")
        .expect("the `nvtx` feature compiles against the NVTX3 headers: set CUDA_HOME");
    cc::Build::new()
        .file("src/nvtx_shim.c")
        .include(format!("{cuda}/include"))
        .opt_level(2)
        .warnings(true)
        .compile("metrale_nvtx_shim");
    // 2026-09-26: The NVTX3 headers load the profiler's injection library with
    // `dlopen` (nvtx3/nvtxDetail/nvtxInit.h).
    println!("cargo:rustc-link-lib=dylib=dl");
}

/// 2026-09-26: Whether `METRALE_SKIP_BUILD` or `SKIP_METRALE_BUILD` is `1`,
/// `true` or `TRUE`: the same test as metrale-storage's build script.
fn skip_build() -> bool {
    let truthy = |key: &str| {
        matches!(
            std::env::var(key).ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        )
    };
    truthy("METRALE_SKIP_BUILD") || truthy("SKIP_METRALE_BUILD")
}
