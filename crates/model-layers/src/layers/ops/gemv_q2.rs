// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the packed Q2_0 GEMVs, which keep the 2-bit weight packed, and for its dequant to BF16.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - A Q2_0 weight carries its fp16 scale inside each `block_q2_0`, so no launcher here takes
//!   a separate scale pointer (kernels/gb10/common/q2_0_gemv.cu).
//! - The GEMV grids are `ceil(N/4)` blocks of 256 threads: 4 outputs per block, 64 threads per
//!   output (`N_PER_BLOCK`, `BLOCK_SIZE` in that file).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::PackedQ2Weight;

/// 2026-09-25: Q2_0 GEMV for M=1 decode: `C[1,N] = A[1,K] @ dequant(B)^T`. `A` and `C` are
/// BF16; `B` is the raw `block_q2_0` buffer, and each weight is dequantized as `(code-1)*d`
/// inside the dot product.
pub fn q2_0_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &PackedQ2Weight,
    output: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(weight.n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(weight.n)
        .arg_u32(weight.k)
        .arg_u32(weight.group as u32)
        .launch(stream)
}

/// 2026-09-25: Batched Q2_0 GEMV: `C[M,N] = A[M,K] @ dequant(B)^T`, `A` and `C` row-major
/// BF16. Each weight block is read once and applied to all `m` rows.
///
/// `m` must be in `1..=8`: the kernel keeps `acc[MAX_M]` with `MAX_M = 8` and indexes it up to
/// `M`. This launcher does not check it.
#[allow(clippy::too_many_arguments)]
pub fn q2_0_gemv_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &PackedQ2Weight,
    output: DevicePtr,
    m: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(weight.n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(weight.n)
        .arg_u32(weight.k)
        .arg_u32(weight.group as u32)
        .arg_u32(m)
        .launch(stream)
}

/// 2026-09-25: Dequantize a packed Q2_0 weight `[N, K]` into the caller's BF16 buffer `[N, K]`
/// on `stream`, with no allocation and no host sync (kernel `dequant_q2_0_gn_to_bf16` in
/// kernels/gb10/common/dequant_gguf_bf16.cu). Packed-Q2 prefill uses it to feed a BF16 GEMM
/// while the resident weight stays packed (`dense_ffn.rs`, `qwen3_attention/prefill_weights.rs`).
///
/// `k` must be a multiple of `group`. One CTA per `block_q2_0`: `n * (k / group)` blocks of
/// `2 + group/4` bytes, each expanding to `group` BF16 values.
#[allow(clippy::too_many_arguments)]
pub fn dequant_q2_0_gn_to_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    blocks: DevicePtr,
    out: DevicePtr,
    n: u32,
    k: u32,
    group: u32,
    stream: u64,
) -> Result<()> {
    let n_blocks = n * (k / group);
    let block_bytes = 2 + group / 4;
    KernelLaunch::new(gpu, kernel)
        .grid([n_blocks, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(blocks)
        .arg_ptr(out)
        .arg_u32(n_blocks)
        .arg_u32(group)
        .arg_u32(block_bytes)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use half::f16;

    /// 2026-09-25: Build one `block_q2_0` from `group` codes in {0,1,2,3} and an fp16 scale, in
    /// the layout the `q2_0_gemv` kernel reads: `[fp16 d][group/4 bytes, 4 codes per byte,
    /// low bits first]` (34 bytes at group 128, 18 at group 64).
    fn pack_block(d: f32, codes: &[u8]) -> Vec<u8> {
        let group = codes.len();
        let mut b = Vec::with_capacity(2 + group / 4);
        b.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for chunk in codes.chunks(4) {
            let mut byte = 0u8;
            for (t, &c) in chunk.iter().enumerate() {
                debug_assert!(c < 4);
                byte |= (c & 3) << (2 * t as u8);
            }
            b.push(byte);
        }
        b
    }

    /// 2026-09-25: Host model of the kernel's dot product,
    /// `out = sum_k a[k] * (code(k)-1) * d(k/group)`, walking one weight row as `k/group`
    /// contiguous blocks as `q2_0_gemv.cu` does.
    fn packed_dot(row_bytes: &[u8], a: &[f32], group: usize) -> f32 {
        let block_bytes = 2 + group / 4;
        let blocks = a.len() / group;
        let mut acc = 0.0f32;
        for b in 0..blocks {
            let blk = &row_bytes[b * block_bytes..(b + 1) * block_bytes];
            let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let qs = &blk[2..];
            for j in 0..group {
                let code = (qs[j >> 2] >> (2 * (j & 3))) & 3;
                acc += a[b * group + j] * ((code as i32 - 1) as f32) * d;
            }
        }
        acc
    }

    /// 2026-09-25: Oracle: dequantize each code to `(code-1)*d` first, then a plain dense dot.
    /// Comparing it with `packed_dot` checks the block layout and symbol mapping.
    fn dense_dot(codes: &[u8], d: &[f32], a: &[f32], group: usize) -> f32 {
        (0..a.len())
            .map(|k| a[k] * ((codes[k] as i32 - 1) as f32) * d[k / group])
            .sum()
    }

    #[test]
    fn q2_block_layout_reference_matches_dense_for_supported_groups() {
        for group in [64usize, 128] {
            let k = group * 2;
            let codes: Vec<u8> = (0..k).map(|i| ((i * 7 + 3) % 4) as u8).collect();
            let a: Vec<f32> = (0..k).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
            let d = [0.0123f32, -0.0456f32];

            let mut row = Vec::new();
            row.extend(pack_block(d[0], &codes[0..group]));
            row.extend(pack_block(d[1], &codes[group..]));
            assert_eq!(row.len(), 2 * (2 + group / 4), "group {group}");

            let got = packed_dot(&row, &a, group);
            let want = dense_dot(&codes, &d, &a, group);
            assert!(
                (got - want).abs() < 1e-3,
                "group {group}: packed {got} vs dense {want}"
            );
        }
    }

    #[test]
    fn ternary_symbols_are_code_minus_one() {
        // 2026-09-25: code {0,1,2,3} maps to {-1, 0, +1, +2}.
        let group = 4usize;
        let codes = [0u8, 1, 2, 3];
        let a = [1.0f32, 1.0, 1.0, 1.0];
        let d = [2.0f32];
        let row = pack_block(d[0], &codes);
        // 2026-09-25: sum a*(code-1)*d = (-1 + 0 + 1 + 2) * 2 = 4.
        assert!((packed_dot(&row, &a, group) - 4.0).abs() < 1e-4);
        // 2026-09-25: Low bits first: byte 0 packs codes[0..4] as 0|1<<2|2<<4|3<<6 = 0xE4.
        assert_eq!(row[2], 0xE4);
    }

    #[test]
    fn shipped_kernels_use_the_q2_layout_and_symbol_mapping() {
        let baseline = include_str!("../../../../../kernels/gb10/common/q2_0_gemv.cu");
        let vector = include_str!("../../../../../kernels/gb10/common/q2_0_gemv_vec.cu");
        let strip_comments = |source: &str| {
            source
                .lines()
                .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let baseline = strip_comments(baseline);
        let vector = strip_comments(vector);

        assert_eq!(
            baseline
                .matches("const unsigned int block_bytes = 2u + group / 4u;")
                .count(),
            2,
            "baseline single-row and batch kernels must use the inline-scale layout"
        );
        assert_eq!(
            baseline.matches("const float d = q2_rd_f16(blk);").count(),
            2
        );
        assert_eq!(
            baseline
                .matches("const unsigned char* qs = blk + 2;")
                .count(),
            2
        );
        assert!(baseline.contains("acc += a * (float)(code - 1) * d;"));
        assert!(baseline.contains("const float wv = (float)(code - 1) * d;"));

        assert_eq!(
            vector
                .matches("const unsigned int block_bytes = 2u + group / 4u;")
                .count(),
            2,
            "vector single-row and batch kernels must use the inline-scale layout"
        );
        assert_eq!(
            vector.matches("const float d = q2v_rd_f16(blk);").count(),
            2
        );
        assert_eq!(vector.matches("q2v_rd_u32(blk + 2 + jg * 4u)").count(), 2);
        assert!(vector.contains("acc += a[j] * (float)(code - 1) * d;"));
        assert!(vector.contains("wv[j] = (float)((int)((codes >> (2 * j)) & 3u) - 1) * d;"));
    }
}
