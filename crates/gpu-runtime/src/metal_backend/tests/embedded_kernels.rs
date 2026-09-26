// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The check that the `metal_backend` parity suite runs against built
//! kernels.
//!
//! The other tests build their backend with `helpers::maybe_backend()`, which
//! passes `metrale_kernels::metallib_modules()` to `MetalGpuBackend::new`. An
//! empty slice is not an error there: the backend starts with no libraries and
//! every kernel lookup then fails with `Metal: unknown module '<name>'`. The
//! kernels build emits an empty `metallib_modules()` when it takes its skip
//! branch (`METRALE_SKIP_BUILD` set to `1`/`true`, or macOS without
//! `METRALE_TARGET_HW`). This test reports that state as one named failure, and
//! has no skip path.
//!
//! Owner: gpu-runtime (metal tests).
//! Invariants: none beyond the types.

/// 2026-09-25: Fails when the build embedded no metallib.
#[test]
fn embedded_metallib_set_is_not_empty() {
    let modules = metrale_kernels::metallib_modules();
    assert!(
        !modules.is_empty(),
        "metrale_kernels::metallib_modules() is empty -- no Metal kernels were \
         compiled into this binary, so every parity test below would fail with \
         `Metal: unknown module`, and a suite that cannot reach a kernel cannot \
         report on one.\n\
         \n\
         Cause: metrale-kernels/build.rs took its skip branch. Either \
         METRALE_SKIP_BUILD is set to 1/true, or the build is on macOS with no \
         METRALE_TARGET_HW set (the auto-skip).\n\
         \n\
         Build the kernels instead of skipping them:\n\
         \x20 METRALE_SKIP_BUILD=0 METRALE_TARGET_HW=metal \\\n\
         \x20 METRALE_TARGET_MODEL=qwen3-5-4b-vlm-mlx-int8 METRALE_TARGET_QUANT=mlx_int8 \\\n\
         \x20 cargo test -p metrale-gpu-runtime --no-default-features --features metal metal_backend"
    );
}
