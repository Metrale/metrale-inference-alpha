// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the vectorized packed Q2_0 decode GEMVs (kernels/gb10/common/q2_0_gemv_vec.cu), which `dense_ffn` loads for decode.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - The call surface matches [`super::gemv_q2`]: the fp16 scale is inside each `block_q2_0`,
//!   so there is no scale pointer.
//! - One warp per output row and 8 rows per 256-thread CTA, so every grid is `ceil(N/8)`
//!   (`WARPS_PER_BLOCK` in the kernel file).
//! - No batched launch passes more than `Q2_BATCHM_MAX_M` rows to the kernel.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::PackedQ2Weight;

/// 2026-09-25: Q2_0 GEMV for M=1 decode: `C[1,N] = A[1,K] @ dequant(B)^T`. Each lane reads one
/// `uint32` of 16 codes, and the CTA stages the activation tile in shared memory. `A` and `C`
/// are BF16; `B` is the raw `block_q2_0` buffer; each weight is `(code-1)*d`.
pub fn q2_0_gemv_vec(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &PackedQ2Weight,
    output: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(weight.n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(weight.n)
        .arg_u32(weight.k)
        .arg_u32(weight.group as u32)
        .launch(stream)
}

/// 2026-09-25: Row cap of the `q2_0_gemv_vec_batchm` kernel (`MAX_M` in the .cu). The kernel
/// stages `M` activation rows into `s_A[MAX_M * TILE_K]`, so `M > MAX_M` writes past that
/// tile, and its compute and store loops stop at `MAX_M`, so rows from 8 on are never
/// written. [`q2_0_gemv_vec_batchm`] chunks larger `m` itself.
pub const Q2_BATCHM_MAX_M: u32 = 8;

/// 2026-09-25: Row groups for driving the batchm kernel at any `m`: yields `(r0, m_chunk)`
/// with `1 <= m_chunk <= Q2_BATCHM_MAX_M`, contiguous from 0 and summing to `m` (empty when
/// `m == 0`). The kernel computes each row with its own accumulator, so a row's result does
/// not depend on which chunk it is in.
fn batchm_row_chunks(m: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..m)
        .step_by(Q2_BATCHM_MAX_M as usize)
        .map(move |r0| (r0, (m - r0).min(Q2_BATCHM_MAX_M)))
}

/// 2026-09-25: `(input_byte_offset, output_byte_offset, rows)` for each kernel launch, with
/// BF16 rows of `k` and `n` elements.
fn batchm_launch_chunks(m: u32, k: u32, n: u32) -> impl Iterator<Item = (usize, usize, u32)> {
    batchm_row_chunks(m).map(move |(r0, rows)| {
        (
            r0 as usize * k as usize * 2,
            r0 as usize * n as usize * 2,
            rows,
        )
    })
}

/// 2026-09-25: Batched Q2_0 GEMV for any `m`: `C[M,N] = A[M,K] @ dequant(B)^T`, `A` and `C`
/// row-major BF16. Each weight word is read once per launch and applied to every staged row.
///
/// `m` is split into launches of at most `Q2_BATCHM_MAX_M` rows, with the input and output
/// pointers advanced by whole rows; `m <= 8` is one launch, and `m == 0` launches nothing.
#[allow(clippy::too_many_arguments)]
pub fn q2_0_gemv_vec_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &PackedQ2Weight,
    output: DevicePtr,
    m: u32,
    stream: u64,
) -> Result<()> {
    for (input_offset, output_offset, m_chunk) in batchm_launch_chunks(m, weight.k, weight.n) {
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(weight.n, 8), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input.offset(input_offset))
            .arg_ptr(weight.weight)
            .arg_ptr(output.offset(output_offset))
            .arg_u32(weight.n)
            .arg_u32(weight.k)
            .arg_u32(weight.group as u32)
            .arg_u32(m_chunk)
            .launch(stream)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{batchm_launch_chunks, batchm_row_chunks};

    use half::{bf16, f16};

    // 2026-09-25: The scale `d` is stored as fp16 in each block, so the model rounds it
    // through fp16 as the kernel's scale read does.
    fn f16_rt(x: f32) -> f32 {
        f16::from_f32(x).to_f32()
    }

    /// 2026-09-25: Host model of one activation row through the Q2_0 GEMV at group 128:
    /// `out[n] = sum_k a[k] * (code(n,k)-1) * d(n, k/128)` in fp32, returned as BF16 bits.
    fn gemv_row(a: &[f32], codes: &[u8], scales: &[f32], n: usize, k: usize) -> Vec<u16> {
        let bpr = k / 128; // 2026-09-25: group-128 blocks per weight row
        let mut out = vec![0u16; n];
        for (col, o) in out.iter_mut().enumerate() {
            let mut acc = 0f32;
            for ki in 0..k {
                let d = f16_rt(scales[col * bpr + ki / 128]);
                let w = ((codes[col * k + ki] & 0x3) as i32 - 1) as f32 * d;
                acc += a[ki] * w;
            }
            *o = bf16::from_f32(acc).to_bits();
        }
        out
    }

    fn gen_inputs(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
        let mut act = vec![0f32; m * k];
        for (i, a) in act.iter_mut().enumerate() {
            *a = (i as f32 * 0.09131).sin() * 1.3;
        }
        let mut codes = vec![0u8; n * k];
        for (i, c) in codes.iter_mut().enumerate() {
            *c = (((i * 2654435761usize) >> 6) % 3) as u8;
        }
        let mut scales = vec![0f32; n * (k / 128)];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 0.015 + 0.01 * ((i % 5) as f32);
        }
        (act, codes, scales)
    }

    #[test]
    fn batchm_row_chunks_cover_all_rows_exactly() {
        for &m in &[0u32, 1, 2, 3, 7, 8, 9, 15, 16, 17] {
            let chunks: Vec<(u32, u32)> = batchm_row_chunks(m).collect();
            let mut next_r0 = 0u32;
            let mut total = 0u32;
            for &(r0, mc) in &chunks {
                assert_eq!(r0, next_r0, "M={m}: chunk starts must be contiguous");
                assert!(
                    (1..=super::Q2_BATCHM_MAX_M).contains(&mc),
                    "M={m}: bad chunk {mc}"
                );
                next_r0 += mc;
                total += mc;
            }
            assert_eq!(total, m, "M={m}: chunks must cover every row once");
            assert_eq!(chunks.is_empty(), m == 0);
        }
    }

    #[test]
    fn launch_chunks_advance_bf16_input_and_output_rows_independently() {
        assert_eq!(
            batchm_launch_chunks(17, 256, 5).collect::<Vec<_>>(),
            vec![(0, 0, 8), (4096, 80, 8), (8192, 160, 1)]
        );
    }

    /// 2026-09-25: Walks the launch chunks of `q2_0_gemv_vec_batchm` on the host and checks that
    /// the input and output offsets address every row exactly once, across the 8-row chunk
    /// boundaries, by comparing against a per-row pass of the host model.
    #[test]
    fn chunked_batchm_equals_per_row_gemv() {
        let (n, k) = (5usize, 256usize);
        for &m in &[1usize, 2, 3, 8, 9, 16] {
            let (act, codes, scales) = gen_inputs(m, n, k);

            let mut reference = vec![0u16; m * n];
            for row in 0..m {
                let r = gemv_row(&act[row * k..(row + 1) * k], &codes, &scales, n, k);
                reference[row * n..(row + 1) * n].copy_from_slice(&r);
            }

            let mut got = vec![0u16; m * n];
            for (input_offset, output_offset, mc) in
                batchm_launch_chunks(m as u32, k as u32, n as u32)
            {
                let r0 = input_offset / (k * 2);
                assert_eq!(output_offset, r0 * n * 2);
                for i in 0..mc as usize {
                    let row = r0 + i;
                    let r = gemv_row(&act[row * k..(row + 1) * k], &codes, &scales, n, k);
                    got[row * n..(row + 1) * n].copy_from_slice(&r);
                }
            }
            assert_eq!(
                got, reference,
                "M={m}: chunked batchm must equal per-row M=1 GEMV bit-for-bit"
            );
        }
    }
}
