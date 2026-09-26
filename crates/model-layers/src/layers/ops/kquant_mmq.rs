// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: Launchers for the DeepSeek-V4.1 K-quant kernels (`kquant_moe` module, kernels/gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu), which read raw Q2_K / Q3_K / Q6_K blocks in place.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - Every GEMV launcher refuses `m` outside `1..=8` (the kernels' `KQ_MAX_M`) and a `k` that is
//!   not a multiple of `QK_K` (256).
//! - Decode quantizes the activation with [`kquant_q8_1_rows`] and runs a `kquant_mmvq*`
//!   GEMV; prefill quantizes it into the type's MMQ q8_1 layout through
//!   [`super::quantize_act_q8_1`] (D2S6 for Q2_K, D4 for Q3_K) and runs [`kquant_mmq_gemm`].

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::q4k_mmq::QK_K;

pub const KQUANT_MODULE: &str = "kquant_moe";
/// 2026-09-25: Bytes per 256-value super-block: `block_q2_K`, `block_q3_K`, `block_q6_K`.
pub const Q2K_BLOCK_BYTES: usize = 84;
pub const Q3K_BLOCK_BYTES: usize = 110;
pub const Q6K_BLOCK_BYTES: usize = 210;

/// 2026-09-25: A resident `[N, K]` row-major projection from the GGUF path: expanded bf16, or the
/// raw `Q2_K` / `Q3_K` blocks (`WeightDtype::Q2K` / `Q3K`) that the K-quant GEMV and MMQ
/// kernels read.
#[derive(Clone, Copy, Debug)]
pub enum ResidentMat {
    Bf16(DevicePtr),
    Q2K(DevicePtr),
    Q3K(DevicePtr),
}

impl ResidentMat {
    /// 2026-09-25: The sub-matrix starting `rows` rows in (each row `k` weights long).
    pub fn at_rows(self, rows: usize, k: usize) -> ResidentMat {
        let at = |p: DevicePtr, off: usize| DevicePtr(p.0 + off as u64);
        match self {
            ResidentMat::Bf16(p) => ResidentMat::Bf16(at(p, rows * k * 2)),
            ResidentMat::Q2K(p) => ResidentMat::Q2K(at(p, rows * (k / 256) * Q2K_BLOCK_BYTES)),
            ResidentMat::Q3K(p) => ResidentMat::Q3K(at(p, rows * (k / 256) * Q3K_BLOCK_BYTES)),
        }
    }
}
/// 2026-09-25: Dynamic shared memory of the 128x128 MMQ tile on the MMA path, as the vendored
/// `mmq_get_nbytes_shared` computes it with `MMQ_TILE_NE_K = 32`: ids 512 B + tile_x (128 rows x
/// tile_x_k ints) + tile_y (128 x 144 B). `MMQ_MMA_TILE_X_K_Q2_K` = 2*32 + 32 + 4 = 100;
/// `MMQ_MMA_TILE_X_K_Q3_K` = 2*32 + 16 + 4 = 84.
pub const Q2K_MMQ_SMEM: u32 = 512 + 128 * 100 * 4 + 128 * 144;
pub const Q3K_MMQ_SMEM: u32 = 512 + 128 * 84 * 4 + 128 * 144;
/// 2026-09-25: `block_q8_1`: 32 int8 + half2 (d, sum).
pub const Q8_1_BLOCK_BYTES: usize = 36;

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// 2026-09-25: Raw bytes of an `[n, k]` K-quant weight.
pub fn kquant_weight_bytes(n: u32, k: u32, block_bytes: usize) -> usize {
    (n as usize) * (k as usize / QK_K as usize) * block_bytes
}

/// 2026-09-25: Bytes of the plain `block_q8_1` buffer for `[m, k]` activations.
pub fn kquant_q8_1_rows_bytes(m: u32, k: u32) -> usize {
    (m as usize) * (k as usize / 32) * Q8_1_BLOCK_BYTES
}

/// 2026-09-25: Bytes of the MMQ-layout q8_1 buffer for `[m, k]` activations: 144 B per 128
/// values, with `m` rounded up to the 128-row tile and `k` to 256.
pub fn kquant_mmq_act_bytes(m: u32, k: u32) -> usize {
    let mpad = div_ceil(m, 128) * 128;
    let kpad = div_ceil(k, QK_K) * QK_K;
    (mpad as usize) * (kpad as usize / 128) * 144
}

/// 2026-09-25: bf16 `[m, k]` -> `block_q8_1 [m, k/32]` (`kquant_q8_1_rows_bf16`), one warp per
/// block. Fails unless `k % 32 == 0`.
pub fn kquant_q8_1_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x_bf16: DevicePtr,
    out_q8: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        k.is_multiple_of(32),
        "kquant_q8_1_rows: k={k} is not a multiple of 32"
    );
    let warps = m * (k / 32);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(warps * 32, 128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(x_bf16)
        .arg_ptr(out_q8)
        .arg_u32(k)
        .arg_u32(m)
        .launch(stream)
}

/// 2026-09-25: `moe_v41_swiglu` and [`kquant_q8_1_rows`] in one launch
/// (`kquant_swiglu_q8_1_rows_bf16`): `h[rows, inter] = bf16(swiglu(gate, up) * w[row])`, then
/// the q8_1 blocks of those bf16 rows. `kquant_fold_tests` compares both outputs byte for byte
/// with the two launches. `w` may be null. Fails unless `inter % 32 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn kquant_swiglu_q8_1_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    w: DevicePtr,
    h_bf16: DevicePtr,
    out_q8: DevicePtr,
    rows: u32,
    inter: u32,
    limit: f32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        inter.is_multiple_of(32),
        "kquant_swiglu_q8_1_rows: inter={inter} is not a multiple of 32"
    );
    let warps = rows * (inter / 32);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(warps * 32, 128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(w)
        .arg_ptr(h_bf16)
        .arg_ptr(out_q8)
        .arg_u32(rows)
        .arg_u32(inter)
        .arg_f32(limit)
        .launch(stream)
}

/// 2026-09-25: Decode GEMV on raw blocks: `out[m][n] = sum_k xq8[m][k] * W[n][k]`
/// (`kquant_mmvq_q2_k` / `kquant_mmvq_q3_k`), one 4-warp block per output row. `w_blocks` is
/// `[n][k/256]` raw super-blocks; `y_q8` is the `block_q8_1 [m][k/32]` buffer; `out_bf16` is
/// `[m][n]`.
pub fn kquant_mmvq(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    w_blocks: DevicePtr,
    y_q8: DevicePtr,
    out_bf16: DevicePtr,
    n: u32,
    k: u32,
    m: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!((1..=8).contains(&m), "kquant_mmvq: m={m} outside 1..=8");
    anyhow::ensure!(
        k.is_multiple_of(QK_K),
        "kquant_mmvq: k={k} is not a multiple of {QK_K}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([32, 4, 1])
        .arg_ptr(w_blocks)
        .arg_ptr(y_q8)
        .arg_ptr(out_bf16)
        .arg_u32(k)
        .arg_u32(n)
        .arg_u32(m)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
/// 2026-09-25: `kquant_mmvq_*_experts`: one launch over `n_experts` experts whose block
/// pointers sit in the device table `vxs`. The activation for expert `e` is
/// `y_q8 + e * y_stride_bytes` (0 = shared), its output `out_bf16 + e * m * n`. Each expert's
/// rows go through the same device function as [`kquant_mmvq`].
pub fn kquant_mmvq_experts(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    vxs: DevicePtr,
    y_q8: DevicePtr,
    out_bf16: DevicePtr,
    n: u32,
    k: u32,
    m: u32,
    n_experts: u32,
    y_stride_bytes: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        (1..=8).contains(&m),
        "kquant_mmvq_experts: m={m} outside 1..=8"
    );
    anyhow::ensure!(
        k.is_multiple_of(QK_K),
        "kquant_mmvq_experts: k={k} is not a multiple of {QK_K}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([n, n_experts, 1])
        .block([32, 4, 1])
        .arg_ptr(vxs)
        .arg_ptr(y_q8)
        .arg_ptr(out_bf16)
        .arg_u32(k)
        .arg_u32(n)
        .arg_u32(m)
        .arg_u32(y_stride_bytes)
        .launch(stream)
}

/// 2026-09-25: `kquant_mmvq_*_w`: the warp-per-row GEMV (four rows a block, reduced by warp
/// shuffle), with the same arguments as [`kquant_mmvq`].
pub fn kquant_mmvq_w(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    w_blocks: DevicePtr,
    y_q8: DevicePtr,
    out_bf16: DevicePtr,
    n: u32,
    k: u32,
    m: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!((1..=8).contains(&m), "kquant_mmvq_w: m={m} outside 1..=8");
    anyhow::ensure!(
        k.is_multiple_of(QK_K),
        "kquant_mmvq_w: k={k} is not a multiple of {QK_K}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([32, 4, 1])
        .arg_ptr(w_blocks)
        .arg_ptr(y_q8)
        .arg_ptr(out_bf16)
        .arg_u32(k)
        .arg_u32(n)
        .arg_u32(m)
        .launch(stream)
}

/// 2026-09-25: `kquant_mmvq_q2_k_groups_w`: `n_groups` independent `[n, k]` GEMVs in one launch.
/// Group `g` takes weight rows `[g*n, (g+1)*n)` (`w_blocks` is `[n_groups * n][k/256]`) against
/// activation columns `[g*k, (g+1)*k)` of the `block_q8_1 [m][n_groups * k / 32]` buffer
/// `y_q8`, and writes columns `[g*n, (g+1)*n)` of `out_bf16` (`[m][n_groups * n]`). Each group
/// runs the per-row code of [`kquant_mmvq_w`] with offsets.
#[allow(clippy::too_many_arguments)]
pub fn kquant_mmvq_groups_w(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    w_blocks: DevicePtr,
    y_q8: DevicePtr,
    out_bf16: DevicePtr,
    n: u32,
    k: u32,
    m: u32,
    n_groups: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        (1..=8).contains(&m),
        "kquant_mmvq_groups_w: m={m} outside 1..=8"
    );
    anyhow::ensure!(
        k.is_multiple_of(QK_K),
        "kquant_mmvq_groups_w: k={k} is not a multiple of {QK_K}"
    );
    anyhow::ensure!(n_groups >= 1, "kquant_mmvq_groups_w: no groups");
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), n_groups, 1])
        .block([32, 4, 1])
        .arg_ptr(w_blocks)
        .arg_ptr(y_q8)
        .arg_ptr(out_bf16)
        .arg_u32(k)
        .arg_u32(n)
        .arg_u32(m)
        .arg_u32(n_groups)
        .launch(stream)
}

/// 2026-09-25: `kquant_mmvq_q2_k_pair_w`: two `[n_i, k]` Q2_K projections of one activation
/// (`y_q8`, `block_q8_1 [m][k/32]`) in one launch, `out_i` = `[m][n_i]` bf16. Grid y picks the
/// projection, and each runs the same device function as [`kquant_mmvq_w`].
#[allow(clippy::too_many_arguments)]
pub fn kquant_mmvq_pair_w(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    (w0, out0, n0): (DevicePtr, DevicePtr, u32),
    (w1, out1, n1): (DevicePtr, DevicePtr, u32),
    y_q8: DevicePtr,
    k: u32,
    m: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        (1..=8).contains(&m),
        "kquant_mmvq_pair_w: m={m} outside 1..=8"
    );
    anyhow::ensure!(
        k.is_multiple_of(QK_K),
        "kquant_mmvq_pair_w: k={k} is not a multiple of {QK_K}"
    );
    anyhow::ensure!(n0 >= 1 && n1 >= 1, "kquant_mmvq_pair_w: empty projection");
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n0.max(n1), 4), 2, 1])
        .block([32, 4, 1])
        .arg_ptr(w0)
        .arg_ptr(out0)
        .arg_u32(n0)
        .arg_ptr(w1)
        .arg_ptr(out1)
        .arg_u32(n1)
        .arg_ptr(y_q8)
        .arg_u32(k)
        .arg_u32(m)
        .launch(stream)
}

/// 2026-09-25: [`kquant_mmvq_experts_w`] with `nwarps` rows a block (the `_w2` / `_w8` entries
/// for 2 / 8; the plain `_w` entry is 4). One warp computes each row in the same way for any
/// `nwarps`. Fails unless `nwarps` is 2, 4 or 8.
#[allow(clippy::too_many_arguments)]
pub fn kquant_mmvq_experts_wn(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    w_tables: DevicePtr,
    y_q8: DevicePtr,
    out_bf16: DevicePtr,
    n: u32,
    k: u32,
    m: u32,
    n_experts: u32,
    y_stride_bytes: u32,
    nwarps: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        (1..=8).contains(&m),
        "kquant_mmvq_experts_wn: m={m} outside 1..=8"
    );
    anyhow::ensure!(
        k.is_multiple_of(QK_K),
        "kquant_mmvq_experts_wn: k={k} is not a multiple of {QK_K}"
    );
    anyhow::ensure!(
        matches!(nwarps, 2 | 4 | 8),
        "kquant_mmvq_experts_wn: nwarps={nwarps} not in 2/4/8"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, nwarps), n_experts, 1])
        .block([32, nwarps, 1])
        .arg_ptr(w_tables)
        .arg_ptr(y_q8)
        .arg_ptr(out_bf16)
        .arg_u32(k)
        .arg_u32(n)
        .arg_u32(m)
        .arg_u32(y_stride_bytes)
        .launch(stream)
}

/// 2026-09-25: `kquant_mmvq_*_experts_w`: the warp-per-row expert batch, with the same arguments
/// as [`kquant_mmvq_experts`].
pub fn kquant_mmvq_experts_w(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    vxs: DevicePtr,
    y_q8: DevicePtr,
    out_bf16: DevicePtr,
    n: u32,
    k: u32,
    m: u32,
    n_experts: u32,
    y_stride_bytes: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        (1..=8).contains(&m),
        "kquant_mmvq_experts_w: m={m} outside 1..=8"
    );
    anyhow::ensure!(
        k.is_multiple_of(QK_K),
        "kquant_mmvq_experts_w: k={k} is not a multiple of {QK_K}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), n_experts, 1])
        .block([32, 4, 1])
        .arg_ptr(vxs)
        .arg_ptr(y_q8)
        .arg_ptr(out_bf16)
        .arg_u32(k)
        .arg_u32(n)
        .arg_u32(m)
        .arg_u32(y_stride_bytes)
        .launch(stream)
}

/// 2026-09-25: Prefill GEMM on raw blocks through the vendored MMQ engine:
/// `out[m][n]` (bf16) = `A_q8[m][k]` x `W[n][k]`. `a_q8` must be in the type's MMQ q8_1 layout
/// (D2S6 for Q2_K, D4 for Q3_K), and `smem` is the matching `*_MMQ_SMEM`. `kernel_nc` serves
/// `n % 128 == 0`, and `kernel_wc` (bounds-checked) any other `n`. The kernel's arguments are
/// `(x = weights, y = activations, dst, nrows_x = n, ncols_dst = m, ncols_x = k,
/// stride_row_x = k / 256, ncols_y = m, stride_col_dst = n)`.
pub fn kquant_mmq_gemm(
    gpu: &dyn GpuBackend,
    kernel_nc: KernelHandle,
    kernel_wc: KernelHandle,
    a_q8: DevicePtr,
    w_blocks: DevicePtr,
    out_bf16: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    smem: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        k.is_multiple_of(QK_K),
        "kquant_mmq_gemm: k={k} is not a multiple of {QK_K}"
    );
    let kernel = if n.is_multiple_of(128) {
        kernel_nc
    } else {
        kernel_wc
    };
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([32, 8, 1])
        .shared_mem(smem)
        .arg_ptr(w_blocks)
        .arg_ptr(a_q8)
        .arg_ptr(out_bf16)
        .arg_u32(n)
        .arg_u32(m)
        .arg_u32(k)
        .arg_u32(k / QK_K)
        .arg_u32(m)
        .arg_u32(n)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_the_block_geometry() {
        // 2026-09-25: A DeepSeek-V4.1 expert's gate (2304 x 5120, Q2_K) and down (5120 x 2304,
        // Q3_K) projections (`moe_intermediate_size`, `hidden_dim` in its MODEL.toml).
        assert_eq!(kquant_weight_bytes(2304, 5120, Q2K_BLOCK_BYTES), 3_870_720);
        assert_eq!(kquant_weight_bytes(5120, 2304, Q3K_BLOCK_BYTES), 5_068_800);
        assert_eq!(kquant_q8_1_rows_bytes(1, 5120), 160 * 36);
        assert_eq!(kquant_mmq_act_bytes(200, 5120), 256 * 40 * 144);
        assert_eq!((Q2K_MMQ_SMEM, Q3K_MMQ_SMEM), (70_144, 61_952));
    }
}
