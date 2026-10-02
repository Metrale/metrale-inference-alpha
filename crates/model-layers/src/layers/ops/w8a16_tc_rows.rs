// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `w8a16_tc_rows_{16,32,64}` (`kernels/gb10/common/w8a16_tc_rows.cu`): the
//! block-scaled W8A16 projection of 1..=64 decode rows with the rows as the tensor-core MMA's
//! N columns, and the rule for when it stands in for the 32- and 64-row tile twins.
//!
//! Under `RowTiers::Canonical` every W8A16 projection of 1..=64 rows goes through those twins
//! ([`super::w8a16_gemm_pipelined_m32_strided`], [`super::w8a16_gemm_pipelined_m64_strided`]);
//! they launch this kernel instead (`METRALE_NO_W8A16_TC_ROWS` keeps the twins). Same precision (FP8 weights,
//! BF16 activations, FP32 accumulation, the 128 x 128 block scales), another summation order,
//! so the output bits differ from the twins'; a row's bits still do not depend on the row
//! count, so the canonical policy holds from 1 to 64 rows. 2026-09-30: above 64 rows
//! `w8a16_tc_rows_64c` runs the same 64-row body per 64-row chunk, so the policy holds at every
//! row count (the 128-row tile it replaces there sums in another order).
//!
//! Owner: model-layers ops.
//! Invariants: [`w8a16_tc_rows_launch`] launches only when [`w8a16_tc_rows_shape_ok`] holds.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-28: Output columns per CTA (`TR_COLS`).
pub const W8A16_TC_ROWS_COLS: u32 = 64;

/// 2026-09-28: Widest row count of one row chunk (`8 * NT` of `w8a16_tc_rows_64`); more rows
/// run as chunks of it (`w8a16_tc_rows_64c`).
pub const W8A16_TC_ROWS_MAX_M: u32 = 64;

const MODULE: &str = "w8a16_tc_rows";

/// 2026-09-28: On unless `METRALE_NO_W8A16_TC_ROWS` is present (a debugging kill switch: the
/// tile twins). Read once per process.
fn w8a16_tc_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_W8A16_TC_ROWS").is_none())
}

/// 2026-09-28: The kernel's shape contract, without a GPU: at least one row, N a positive
/// multiple of 128 (whole scale blocks, whole CTAs), K a positive multiple of 128, and an A
/// pitch that keeps rows 16-byte aligned and covers K; the C pitch covers N.
pub fn w8a16_tc_rows_shape_ok(
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    c_row_stride: u32,
) -> bool {
    m >= 1
        && n > 0
        && n.is_multiple_of(128)
        && k > 0
        && k.is_multiple_of(128)
        && a_row_stride >= k
        && a_row_stride.is_multiple_of(8)
        && c_row_stride >= n
}

/// 2026-09-28: Launches `[m, k] x [n, k]^T` through `w8a16_tc_rows_{16,32,64}` and returns true
/// when the lever is on, the process runs `RowTiers::Canonical`, the shape passes
/// [`w8a16_tc_rows_shape_ok`] and the target ships the module; otherwise returns false and
/// launches nothing.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_tc_rows_launch(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    c_row_stride: u32,
    stream: u64,
) -> Result<bool> {
    if !w8a16_tc_rows_enabled()
        || crate::layers::row_tiers() != crate::layers::RowTiers::Canonical
        || !w8a16_tc_rows_shape_ok(m, n, k, a_row_stride, c_row_stride)
        || !gpu.has_module(MODULE)
    {
        return Ok(false);
    }
    w8a16_tc_rows(
        gpu,
        input,
        weight,
        block_scale,
        output,
        m,
        n,
        k,
        a_row_stride,
        c_row_stride,
        stream,
    )?;
    Ok(true)
}

/// 2026-09-28: The launch itself, with no policy check: `input` `[m, a_row_stride]` BF16,
/// `weight` `[n, k]` E4M3, `block_scale` `[n / 128, k / 128]` FP32, `output` `[m,
/// c_row_stride]` BF16. Refuses a shape [`w8a16_tc_rows_shape_ok`] refuses.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_tc_rows(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    c_row_stride: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        w8a16_tc_rows_shape_ok(m, n, k, a_row_stride, c_row_stride),
        "w8a16_tc_rows: m={m} n={n} k={k} lda={a_row_stride} ldc={c_row_stride} outside the kernel's contract"
    );
    let (entry, chunks) = if m <= 16 {
        ("w8a16_tc_rows_16", 1)
    } else if m <= 32 {
        ("w8a16_tc_rows_32", 1)
    } else if m <= W8A16_TC_ROWS_MAX_M {
        ("w8a16_tc_rows_64", 1)
    } else {
        ("w8a16_tc_rows_64c", m.div_ceil(W8A16_TC_ROWS_MAX_M))
    };
    let kernel = gpu.op_cache().kernel(gpu, MODULE, entry)?;
    KernelLaunch::new(gpu, kernel)
        .grid([n / W8A16_TC_ROWS_COLS * chunks, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(a_row_stride)
        .arg_u32(c_row_stride)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CU: &str = include_str!("../../../../../kernels/gb10/common/w8a16_tc_rows.cu");
    const ROWS: &str = include_str!("../../../../../kernels/gb10/common/tc_rows.cuh");

    /// 2026-09-28: The grid covers `n / W8A16_TC_ROWS_COLS` CTAs of 128 threads and the widest
    /// entry point takes 64 rows: both must match the kernel (`TR_WARPS` warps of 16 columns,
    /// `tr_block<8, ...>` row tiles of 8), or columns or rows go unwritten.
    #[test]
    fn launch_matches_the_kernel() {
        assert!(
            ROWS.contains("#define TR_WARPS 4\n")
                && ROWS.contains("#define TR_COLS (TR_WARPS * 16)\n")
        );
        assert_eq!(W8A16_TC_ROWS_COLS, 4 * 16);
        assert!(CU.contains(
            "tr_block<Fp8Block128, 8, 2>(A, {B, block_scale}, C, M, N, K, lda, ldc, blockIdx.x);"
        ));
        // 2026-09-30: The chunked entry runs the 64-row body per chunk of `W8A16_TC_ROWS_MAX_M`.
        assert!(CU.contains("const unsigned int chunks = (M + 63) / 64;"));
        assert!(
            CU.contains("tr_block<Fp8Block128, 8, 2>(A + (unsigned long long)chunk * 64 * lda")
        );
        assert_eq!(W8A16_TC_ROWS_MAX_M, 8 * 8);
        for entry in [
            "w8a16_tc_rows_16(",
            "w8a16_tc_rows_32(",
            "w8a16_tc_rows_64(",
            "w8a16_tc_rows_64c(",
        ] {
            assert!(CU.contains(entry), "{entry} missing from the kernel");
        }
    }

    /// 2026-09-28: The shape contract refuses each bound it names and admits the 35B shapes.
    #[test]
    fn shape_contract() {
        assert!(w8a16_tc_rows_shape_ok(1, 12288, 2048, 2048, 12288));
        assert!(w8a16_tc_rows_shape_ok(64, 512, 2048, 2048, 512));
        assert!(!w8a16_tc_rows_shape_ok(0, 512, 2048, 2048, 512));
        assert!(w8a16_tc_rows_shape_ok(65, 512, 2048, 2048, 512));
        assert!(w8a16_tc_rows_shape_ok(256, 12288, 2048, 2048, 12288));
        assert!(!w8a16_tc_rows_shape_ok(8, 576, 2048, 2048, 576));
        assert!(!w8a16_tc_rows_shape_ok(8, 512, 2000, 2048, 512));
        assert!(!w8a16_tc_rows_shape_ok(8, 512, 2048, 2044, 512));
        assert!(!w8a16_tc_rows_shape_ok(8, 512, 2048, 2052, 512));
        assert!(!w8a16_tc_rows_shape_ok(8, 512, 2048, 2048, 511));
    }
}
