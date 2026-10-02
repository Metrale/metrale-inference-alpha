// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: `w4a16_tc_rows_{16,32,64}` (`kernels/gb10/common/w4a16_tc_rows.cu`): the NVFP4
//! W4A16 projection of 1..=64 rows with the rows as the tensor-core MMA's N columns, the NVFP4
//! point of the row-tile family whose FP8 point is `w8a16_tc_rows`. A row's bits do not depend
//! on the row count or the entry point, so callers chunk wider row counts in 64-row calls.
//!
//! Owner: model-layers ops.
//! Invariants: [`w4a16_tc_rows`] launches only when [`w4a16_tc_rows_shape_ok`] holds.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::weight_map::QuantizedWeight;

/// 2026-10-02: Output columns per CTA (`TR_COLS`).
pub const W4A16_TC_ROWS_COLS: u32 = 64;

/// 2026-10-02: Widest row count of one launch (`8 * NT` of `w4a16_tc_rows_64`).
pub const W4A16_TC_ROWS_MAX_M: u32 = 64;

/// 2026-10-02: The kernel module.
pub const W4A16_TC_ROWS_MODULE: &str = "w4a16_tc_rows";

/// 2026-10-02: The kernel's shape contract, without a GPU: 1..=64 rows, any positive N (the
/// entry points are ragged: a partial last CTA loads zero weight rows and stores nothing past N),
/// K a positive multiple of 256 (whole load groups of every entry point), and an A pitch that
/// keeps rows 16-byte aligned and covers K; the C pitch covers N.
pub fn w4a16_tc_rows_shape_ok(m: u32, n: u32, k: u32, lda: u32, ldc: u32) -> bool {
    (1..=W4A16_TC_ROWS_MAX_M).contains(&m)
        && n > 0
        && k > 0
        && k.is_multiple_of(256)
        && lda >= k
        && lda.is_multiple_of(8)
        && ldc >= n
}

/// 2026-10-02: `output [m, ldc] = input [m, lda] x W^T` for the row-major NVFP4 `weight` `[n, k]`
/// (packed E2M1, E4M3 scales of 16, `weight_scale_2`). Refuses a shape
/// [`w4a16_tc_rows_shape_ok`] refuses.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_tc_rows(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        w4a16_tc_rows_shape_ok(m, n, k, lda, ldc),
        "w4a16_tc_rows: m={m} n={n} k={k} lda={lda} ldc={ldc} outside the kernel's contract"
    );
    let entry = if m <= 16 {
        "w4a16_tc_rows_16"
    } else if m <= 32 {
        "w4a16_tc_rows_32"
    } else {
        "w4a16_tc_rows_64"
    };
    let kernel = gpu.op_cache().kernel(gpu, W4A16_TC_ROWS_MODULE, entry)?;
    KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(W4A16_TC_ROWS_COLS), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(lda)
        .arg_u32(ldc)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CU: &str = include_str!("../../../../../kernels/gb10/common/w4a16_tc_rows.cu");
    const ROWS: &str = include_str!("../../../../../kernels/gb10/common/tc_rows.cuh");

    /// 2026-10-02: The grid and the widest entry point match the kernel (`TR_WARPS` warps of 16
    /// columns, `tr_block<Nvfp4G16, 8, ...>` row tiles of 8), or columns or rows go unwritten.
    #[test]
    fn launch_matches_the_kernel() {
        assert!(
            ROWS.contains("#define TR_WARPS 4\n")
                && ROWS.contains("#define TR_COLS (TR_WARPS * 16)\n")
        );
        assert_eq!(W4A16_TC_ROWS_COLS, 4 * 16);
        assert!(CU.contains(
            "tr_block<Nvfp4G16, 8, 1, true>(A, {packed, scale, s2}, C, M, N, K, lda, ldc, blockIdx.x);"
        ));
        assert_eq!(W4A16_TC_ROWS_MAX_M, 8 * 8);
        for entry in [
            "w4a16_tc_rows_16(",
            "w4a16_tc_rows_32(",
            "w4a16_tc_rows_64(",
        ] {
            assert!(CU.contains(entry), "{entry} missing from the kernel");
        }
    }

    /// 2026-10-02: The shape contract refuses each bound it names and admits the 35B head.
    #[test]
    fn shape_contract() {
        assert!(w4a16_tc_rows_shape_ok(1, 248320, 2048, 2048, 248320));
        assert!(w4a16_tc_rows_shape_ok(64, 248320, 2048, 2048, 248320));
        assert!(!w4a16_tc_rows_shape_ok(0, 248320, 2048, 2048, 248320));
        assert!(!w4a16_tc_rows_shape_ok(65, 248320, 2048, 2048, 248320));
        // 2026-10-02: Ragged N: the checkpoint's 248070-entry vocab.
        assert!(w4a16_tc_rows_shape_ok(8, 248070, 2048, 2048, 248070));
        assert!(!w4a16_tc_rows_shape_ok(8, 248320, 2048 + 128, 2176, 248320));
        assert!(!w4a16_tc_rows_shape_ok(8, 248320, 2048, 2044, 248320));
        assert!(!w4a16_tc_rows_shape_ok(8, 248320, 2048, 2048, 248319));
    }
}
