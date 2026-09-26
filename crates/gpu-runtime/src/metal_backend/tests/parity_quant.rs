// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Parity of the MLX 8-bit kernels `mlx_int8_dequant`, `mlx_int8_gemv`
//! and `mlx_int8_gemm`.
//!
//! Owner: gpu-runtime (metal tests).
//! Invariants: none beyond the types.

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{DevicePtr, GpuBackend, KernelArg};

/// 2026-09-25: `mlx_int8_dequant` against the CPU formula
/// `w[r,c] = byte * scales[r, c/group_size] + biases[r, c/group_size]`, over two
/// groups of 64 columns per row.
#[test]
fn metal_mlx_int8_dequant_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let out_features = 4u32;
    let in_features = 128u32;
    let group_size = 64u32;
    let groups_per_row = (in_features / group_size) as usize;
    let n_rows = out_features as usize;
    let n_cols = in_features as usize;

    let mut bytes_flat: Vec<u8> = Vec::with_capacity(n_rows * n_cols);
    for r in 0..n_rows {
        for c in 0..n_cols {
            bytes_flat.push(((r * 7 + c) % 256) as u8);
        }
    }
    let mut packed: Vec<u32> = Vec::with_capacity(n_rows * n_cols / 4);
    for r in 0..n_rows {
        for c in (0..n_cols).step_by(4) {
            let base = r * n_cols + c;
            let word = (bytes_flat[base] as u32)
                | ((bytes_flat[base + 1] as u32) << 8)
                | ((bytes_flat[base + 2] as u32) << 16)
                | ((bytes_flat[base + 3] as u32) << 24);
            packed.push(word);
        }
    }

    let mut scales: Vec<half::bf16> = Vec::with_capacity(n_rows * groups_per_row);
    let mut biases: Vec<half::bf16> = Vec::with_capacity(n_rows * groups_per_row);
    for r in 0..n_rows {
        for g in 0..groups_per_row {
            scales.push(half::bf16::from_f32(
                0.01 * (1.0 + r as f32) + 0.001 * g as f32,
            ));
            biases.push(half::bf16::from_f32(
                -0.5 + 0.1 * r as f32 + 0.05 * g as f32,
            ));
        }
    }

    let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n_rows * n_cols];
    for r in 0..n_rows {
        for c in 0..n_cols {
            let byte = bytes_flat[r * n_cols + c] as f32;
            let g = c / group_size as usize;
            let s = scales[r * groups_per_row + g].to_f32();
            let b = biases[r * groups_per_row + g].to_f32();
            expected[r * n_cols + c] = half::bf16::from_f32(byte * s + b);
        }
    }

    let packed_bytes_buf = u32_slice_to_bytes(&packed);
    let scales_bytes_buf = bf16_slice_to_bytes(&scales);
    let biases_bytes_buf = bf16_slice_to_bytes(&biases);

    let packed_ptr = backend.alloc(packed_bytes_buf.len()).expect("alloc packed");
    let scales_ptr = backend.alloc(scales_bytes_buf.len()).expect("alloc scales");
    let biases_ptr = backend.alloc(biases_bytes_buf.len()).expect("alloc biases");
    let out_ptr = backend.alloc(n_rows * n_cols * 2).expect("alloc out");

    backend
        .copy_h2d(&packed_bytes_buf, packed_ptr)
        .expect("h2d packed");
    backend
        .copy_h2d(&scales_bytes_buf, scales_ptr)
        .expect("h2d scales");
    backend
        .copy_h2d(&biases_bytes_buf, biases_ptr)
        .expect("h2d biases");

    let kernel = backend
        .kernel("mlx_int8_dequant", "mlx_int8_dequant")
        .expect("kernel lookup");

    let block_x = 16u32;
    let block_y = 1u32;
    let grid_x = in_features.div_ceil(block_x);
    let grid_y = out_features;
    backend
        .launch_typed(
            kernel,
            [grid_x, grid_y, 1],
            [block_x, block_y, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&out_features.to_le_bytes()),
                KernelArg::Bytes(&in_features.to_le_bytes()),
                KernelArg::Bytes(&group_size.to_le_bytes()),
                KernelArg::Buffer(packed_ptr),
                KernelArg::Buffer(scales_ptr),
                KernelArg::Buffer(biases_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch_typed dequant");
    backend
        .synchronize(backend.default_stream())
        .expect("synchronize");

    let mut out_raw = vec![0u8; n_rows * n_cols * 2];
    backend.copy_d2h(out_ptr, &mut out_raw).expect("d2h out");
    let actual = bytes_to_bf16_vec(&out_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..expected.len() {
        let e = expected[i].to_f32();
        let a = actual[i].to_f32();
        let d = (e - a).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    // 2026-09-25: Outputs reach about 5.92 (row 3, column 127: 148 * 0.041 - 0.15),
    // where one BF16 ulp is 0.03125; the 0.1 bound allows three ulps.
    assert!(
        max_abs_diff < 0.1,
        "mlx_int8_dequant: max |expected - actual| = {max_abs_diff}, expected < 0.1"
    );

    backend.free(packed_ptr).unwrap();
    backend.free(scales_ptr).unwrap();
    backend.free(biases_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}

/// 2026-09-25: `mlx_int8_gemv` (8 rows, K = 256) against an FP32-accumulated CPU
/// reference over the `build_mlx_fixture` weights.
#[test]
fn metal_mlx_int8_gemv_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let n: u32 = 8;
    let k: u32 = 256;
    let group_size: u32 = 64;

    let (packed_bytes, scales_bytes, biases_bytes, w_ref) =
        build_mlx_fixture(n as usize, k as usize, group_size as usize);

    let x_bf16: Vec<half::bf16> = (0..k)
        .map(|i| half::bf16::from_f32(0.5 + 0.001 * i as f32))
        .collect();
    let x_bytes = bf16_slice_to_bytes(&x_bf16);

    let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n as usize];
    for r in 0..n as usize {
        let mut acc: f32 = 0.0;
        for c in 0..k as usize {
            acc += w_ref[r * k as usize + c].to_f32() * x_bf16[c].to_f32();
        }
        expected[r] = half::bf16::from_f32(acc);
    }

    let packed_ptr = backend.alloc(packed_bytes.len()).unwrap();
    let scales_ptr = backend.alloc(scales_bytes.len()).unwrap();
    let biases_ptr = backend.alloc(biases_bytes.len()).unwrap();
    let x_ptr = backend.alloc(x_bytes.len()).unwrap();
    let y_ptr = backend.alloc(n as usize * 2).unwrap();
    backend.copy_h2d(&packed_bytes, packed_ptr).unwrap();
    backend.copy_h2d(&scales_bytes, scales_ptr).unwrap();
    backend.copy_h2d(&biases_bytes, biases_ptr).unwrap();
    backend.copy_h2d(&x_bytes, x_ptr).unwrap();

    let kernel = backend.kernel("mlx_int8_gemv", "mlx_int8_gemv").unwrap();
    const ROWS_PER_TG: u32 = 4;
    let threads_per_tg: u32 = 128;
    let row_groups = n.div_ceil(ROWS_PER_TG);
    backend
        .launch_typed(
            kernel,
            [row_groups, 1, 1],
            [threads_per_tg, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&k.to_le_bytes()),
                KernelArg::Bytes(&group_size.to_le_bytes()),
                KernelArg::Buffer(packed_ptr),
                KernelArg::Buffer(scales_ptr),
                KernelArg::Buffer(biases_ptr),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(y_ptr),
            ],
        )
        .expect("launch_typed gemv");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut y_raw = vec![0u8; n as usize * 2];
    backend.copy_d2h(y_ptr, &mut y_raw).unwrap();
    let actual = bytes_to_bf16_vec(&y_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..n as usize {
        let e = expected[i].to_f32();
        let a = actual[i].to_f32();
        let d = (e - a).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff < 0.05,
        "mlx_int8_gemv: max |expected - actual| = {max_abs_diff}; \
         expected/actual head: {:?} vs {:?}",
        &expected
            .iter()
            .take(4)
            .map(|v| v.to_f32())
            .collect::<Vec<_>>(),
        &actual
            .iter()
            .take(4)
            .map(|v| v.to_f32())
            .collect::<Vec<_>>()
    );

    backend.free(packed_ptr).unwrap();
    backend.free(scales_ptr).unwrap();
    backend.free(biases_ptr).unwrap();
    backend.free(x_ptr).unwrap();
    backend.free(y_ptr).unwrap();
}

/// 2026-09-25: `mlx_int8_gemm` (m = 2, n = 8, k = 128) against an FP32 CPU
/// reference over the `build_mlx_fixture` weights.
#[test]
fn metal_mlx_int8_gemm_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let m: u32 = 2;
    let n: u32 = 8;
    let k: u32 = 128;
    let group_size: u32 = 64;

    let (packed_bytes, scales_bytes, biases_bytes, w_ref) =
        build_mlx_fixture(n as usize, k as usize, group_size as usize);

    let x_bf16: Vec<half::bf16> = (0..(m * k))
        .map(|i| {
            let row = i / k;
            let col = i % k;
            half::bf16::from_f32(0.3 + 0.01 * row as f32 + 0.001 * col as f32)
        })
        .collect();
    let x_bytes = bf16_slice_to_bytes(&x_bf16);

    let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; (m * n) as usize];
    for mi in 0..m as usize {
        for ni in 0..n as usize {
            let mut acc: f32 = 0.0;
            for ki in 0..k as usize {
                acc += x_bf16[mi * k as usize + ki].to_f32() * w_ref[ni * k as usize + ki].to_f32();
            }
            expected[mi * n as usize + ni] = half::bf16::from_f32(acc);
        }
    }

    let packed_ptr = backend.alloc(packed_bytes.len()).unwrap();
    let scales_ptr = backend.alloc(scales_bytes.len()).unwrap();
    let biases_ptr = backend.alloc(biases_bytes.len()).unwrap();
    let x_ptr = backend.alloc(x_bytes.len()).unwrap();
    let y_ptr = backend.alloc((m * n) as usize * 2).unwrap();
    backend.copy_h2d(&packed_bytes, packed_ptr).unwrap();
    backend.copy_h2d(&scales_bytes, scales_ptr).unwrap();
    backend.copy_h2d(&biases_bytes, biases_ptr).unwrap();
    backend.copy_h2d(&x_bytes, x_ptr).unwrap();

    let kernel = backend.kernel("mlx_int8_gemm", "mlx_int8_gemm").unwrap();
    let block_x = 16u32;
    let block_y = 16u32;
    backend
        .launch_typed(
            kernel,
            [n.div_ceil(block_x), m.div_ceil(block_y), 1],
            [block_x, block_y, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&m.to_le_bytes()),
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&k.to_le_bytes()),
                KernelArg::Bytes(&group_size.to_le_bytes()),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(packed_ptr),
                KernelArg::Buffer(scales_ptr),
                KernelArg::Buffer(biases_ptr),
                KernelArg::Buffer(y_ptr),
            ],
        )
        .expect("launch_typed gemm");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut y_raw = vec![0u8; (m * n) as usize * 2];
    backend.copy_d2h(y_ptr, &mut y_raw).unwrap();
    let actual = bytes_to_bf16_vec(&y_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..(m * n) as usize {
        let e = expected[i].to_f32();
        let a = actual[i].to_f32();
        let d = (e - a).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff < 0.05,
        "mlx_int8_gemm: max |expected - actual| = {max_abs_diff}"
    );

    backend.free(packed_ptr).unwrap();
    backend.free(scales_ptr).unwrap();
    backend.free(biases_ptr).unwrap();
    backend.free(x_ptr).unwrap();
    backend.free(y_ptr).unwrap();
}
