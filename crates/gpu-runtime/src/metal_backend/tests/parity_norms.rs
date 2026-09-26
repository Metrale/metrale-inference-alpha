// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Parity of `rope_apply`, `silu_gate`, `rms_norm`, `embed_lookup` and
//! `argmax_bf16`.
//!
//! Owner: gpu-runtime (metal tests).
//! Invariants: none beyond the types.

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{DevicePtr, GpuBackend, KernelArg};

/// 2026-09-25: `rope_apply` over the whole head (`rotary_dim == head_dim`) against
/// an FP32 reference that rotates the pairs `(d, d + head_dim/2)`.
#[test]
fn metal_rope_apply_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let num_tokens: u32 = 4;
    let num_heads: u32 = 2;
    let head_dim: u32 = 16;
    let half_dim = head_dim / 2;

    let total = (num_tokens * num_heads * head_dim) as usize;
    let x: Vec<half::bf16> = (0..total)
        .map(|i| half::bf16::from_f32(0.1 + 0.001 * i as f32))
        .collect();

    let rope_theta: f32 = 10000.0;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let positions: Vec<u32> = (0..num_tokens).collect();

    let mut expected = x.clone();
    for tok in 0..num_tokens as usize {
        let pos = positions[tok] as f32;
        for h in 0..num_heads as usize {
            let base = (tok * num_heads as usize + h) * head_dim as usize;
            for d in 0..half_dim as usize {
                let theta = pos * inv_freq[d];
                let c = theta.cos();
                let s = theta.sin();
                let lo = x[base + d].to_f32();
                let hi = x[base + d + half_dim as usize].to_f32();
                expected[base + d] = half::bf16::from_f32(lo * c - hi * s);
                expected[base + d + half_dim as usize] = half::bf16::from_f32(lo * s + hi * c);
            }
        }
    }

    let x_bytes = bf16_slice_to_bytes(&x);
    let inv_freq_bytes: Vec<u8> = inv_freq.iter().flat_map(|f| f.to_le_bytes()).collect();
    let positions_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();

    let x_ptr = backend.alloc(x_bytes.len()).unwrap();
    let inv_freq_ptr = backend.alloc(inv_freq_bytes.len()).unwrap();
    let positions_ptr = backend.alloc(positions_bytes.len()).unwrap();
    backend.copy_h2d(&x_bytes, x_ptr).unwrap();
    backend.copy_h2d(&inv_freq_bytes, inv_freq_ptr).unwrap();
    backend.copy_h2d(&positions_bytes, positions_ptr).unwrap();

    let kernel = backend.kernel("rope_apply", "rope_apply").unwrap();
    let rotary_dim: u32 = head_dim;
    backend
        .launch_typed(
            kernel,
            [half_dim, num_heads, num_tokens],
            [1, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&num_tokens.to_le_bytes()),
                KernelArg::Bytes(&num_heads.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&rotary_dim.to_le_bytes()),
                KernelArg::Buffer(positions_ptr),
                KernelArg::Buffer(inv_freq_ptr),
                KernelArg::Buffer(x_ptr),
            ],
        )
        .expect("launch rope_apply");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut x_after = vec![0u8; x_bytes.len()];
    backend.copy_d2h(x_ptr, &mut x_after).unwrap();
    let actual = bytes_to_bf16_vec(&x_after);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..total {
        let e = expected[i].to_f32();
        let a = actual[i].to_f32();
        let d = (e - a).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        assert!(
            d < 0.05,
            "rope_apply mismatch at idx {i}: expected {e}, got {a}"
        );
    }
    assert!(max_abs_diff < 0.02);

    backend.free(x_ptr).unwrap();
    backend.free(inv_freq_ptr).unwrap();
    backend.free(positions_ptr).unwrap();
    let _ = x;
}

/// 2026-09-25: `silu_gate` (`gate * sigmoid(gate) * up`) against an FP32 reference,
/// for gates from -4 to 4.
#[test]
fn metal_silu_gate_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let n: u32 = 256;
    let gate: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(-4.0 + 8.0 * i as f32 / (n as f32 - 1.0)))
        .collect();
    let up: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(0.5 + 0.01 * i as f32))
        .collect();

    let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n as usize];
    for i in 0..n as usize {
        let g = gate[i].to_f32();
        let u = up[i].to_f32();
        let sig = 1.0 / (1.0 + (-g).exp());
        expected[i] = half::bf16::from_f32(g * sig * u);
    }

    let gate_bytes = bf16_slice_to_bytes(&gate);
    let up_bytes = bf16_slice_to_bytes(&up);
    let gate_ptr = backend.alloc(gate_bytes.len()).unwrap();
    let up_ptr = backend.alloc(up_bytes.len()).unwrap();
    let out_ptr = backend.alloc(n as usize * 2).unwrap();
    backend.copy_h2d(&gate_bytes, gate_ptr).unwrap();
    backend.copy_h2d(&up_bytes, up_ptr).unwrap();

    let kernel = backend.kernel("silu_gate", "silu_gate").unwrap();
    let block: u32 = 64;
    let grid = n.div_ceil(block);
    backend
        .launch_typed(
            kernel,
            [grid, 1, 1],
            [block, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Buffer(gate_ptr),
                KernelArg::Buffer(up_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch silu_gate");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; n as usize * 2];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

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
        max_abs_diff < 0.02,
        "silu_gate: max |expected - actual| = {max_abs_diff}"
    );

    backend.free(gate_ptr).unwrap();
    backend.free(up_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}

/// 2026-09-25: `rms_norm` against an FP32 reference, launched with 128 threads per
/// row so the kernel's sum of squares crosses 4 simdgroups.
#[test]
fn metal_rms_norm_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let num_tokens: u32 = 3;
    let hidden: u32 = 256;
    let eps: f32 = 1e-5;

    let x: Vec<half::bf16> = (0..(num_tokens * hidden))
        .map(|i| {
            let r = i / hidden;
            let c = i % hidden;
            half::bf16::from_f32(0.1 + 0.01 * r as f32 + 0.001 * c as f32)
        })
        .collect();
    let weight: Vec<half::bf16> = (0..hidden)
        .map(|i| half::bf16::from_f32(1.0 + 0.005 * i as f32))
        .collect();

    let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; (num_tokens * hidden) as usize];
    for r in 0..num_tokens as usize {
        let mut ssq: f32 = 0.0;
        for c in 0..hidden as usize {
            let v = x[r * hidden as usize + c].to_f32();
            ssq += v * v;
        }
        let inv_rms = (ssq / hidden as f32 + eps).powf(-0.5);
        for c in 0..hidden as usize {
            let v = x[r * hidden as usize + c].to_f32();
            let w = weight[c].to_f32();
            expected[r * hidden as usize + c] = half::bf16::from_f32(v * inv_rms * w);
        }
    }

    let x_bytes = bf16_slice_to_bytes(&x);
    let w_bytes = bf16_slice_to_bytes(&weight);
    let x_ptr = backend.alloc(x_bytes.len()).unwrap();
    let w_ptr = backend.alloc(w_bytes.len()).unwrap();
    let out_ptr = backend.alloc(x_bytes.len()).unwrap();
    backend.copy_h2d(&x_bytes, x_ptr).unwrap();
    backend.copy_h2d(&w_bytes, w_ptr).unwrap();

    let kernel = backend.kernel("rms_norm", "rms_norm").unwrap();
    backend
        .launch_typed(
            kernel,
            [num_tokens, 1, 1],
            [128, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&hidden.to_le_bytes()),
                KernelArg::Bytes(&eps.to_le_bytes()),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(w_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch rms_norm");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; x_bytes.len()];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..(num_tokens * hidden) as usize {
        let e = expected[i].to_f32();
        let a = actual[i].to_f32();
        let d = (e - a).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff < 0.02,
        "rms_norm: max |expected - actual| = {max_abs_diff}"
    );

    backend.free(x_ptr).unwrap();
    backend.free(w_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}

/// 2026-09-25: `embed_lookup` copies each token's row, and writes a zero row for
/// the out-of-range id 99.
#[test]
fn metal_embed_lookup_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let vocab: u32 = 16;
    let hidden: u32 = 8;
    let num_tokens: u32 = 4;
    let tokens: [u32; 4] = [3, 7, 99, 0];

    let table: Vec<half::bf16> = (0..(vocab * hidden))
        .map(|i| {
            let v = i / hidden;
            let h = i % hidden;
            half::bf16::from_f32(0.1 * v as f32 + 0.01 * h as f32)
        })
        .collect();

    let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; (num_tokens * hidden) as usize];
    for (ti, &v) in tokens.iter().enumerate() {
        for h in 0..hidden as usize {
            if v < vocab {
                expected[ti * hidden as usize + h] = table[v as usize * hidden as usize + h];
            }
        }
    }

    let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let table_bytes = bf16_slice_to_bytes(&table);
    let token_ptr = backend.alloc(token_bytes.len()).unwrap();
    let table_ptr = backend.alloc(table_bytes.len()).unwrap();
    let out_ptr = backend.alloc((num_tokens * hidden) as usize * 2).unwrap();
    backend.copy_h2d(&token_bytes, token_ptr).unwrap();
    backend.copy_h2d(&table_bytes, table_ptr).unwrap();

    let kernel = backend.kernel("embed_lookup", "embed_lookup").unwrap();
    backend
        .launch_typed(
            kernel,
            [hidden.div_ceil(8), num_tokens, 1],
            [8, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&num_tokens.to_le_bytes()),
                KernelArg::Bytes(&hidden.to_le_bytes()),
                KernelArg::Bytes(&vocab.to_le_bytes()),
                KernelArg::Buffer(token_ptr),
                KernelArg::Buffer(table_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch embed_lookup");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; (num_tokens * hidden) as usize * 2];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

    for i in 0..(num_tokens * hidden) as usize {
        let e = expected[i].to_f32();
        let a = actual[i].to_f32();
        assert!(
            (e - a).abs() < 1e-4,
            "embed_lookup mismatch at idx {i}: expected {e}, got {a}"
        );
    }

    backend.free(token_ptr).unwrap();
    backend.free(table_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}

/// 2026-09-25: `argmax_bf16` returns the index of a planted maximum (99.0 at index
/// 723 of 1024).
#[test]
fn metal_argmax_bf16_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let n: u32 = 1024;
    let mut values: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(0.001 * i as f32))
        .collect();
    let target_idx = 723usize;
    values[target_idx] = half::bf16::from_f32(99.0);

    let bytes = bf16_slice_to_bytes(&values);
    let logits_ptr = backend.alloc(bytes.len()).unwrap();
    let result_ptr = backend.alloc(4).unwrap();
    backend.copy_h2d(&bytes, logits_ptr).unwrap();

    let kernel = backend.kernel("argmax_bf16", "argmax_bf16").unwrap();
    backend
        .launch_typed(
            kernel,
            [1, 1, 1],
            [128, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Buffer(logits_ptr),
                KernelArg::Buffer(result_ptr),
            ],
        )
        .expect("launch_typed argmax");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut result_raw = [0u8; 4];
    backend.copy_d2h(result_ptr, &mut result_raw).unwrap();
    let actual_idx = u32::from_le_bytes(result_raw) as usize;
    assert_eq!(
        actual_idx, target_idx,
        "argmax: expected {target_idx}, got {actual_idx}"
    );

    backend.free(logits_ptr).unwrap();
    backend.free(result_ptr).unwrap();
}
