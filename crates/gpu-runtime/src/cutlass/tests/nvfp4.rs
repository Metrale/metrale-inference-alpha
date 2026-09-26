// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: NVFP4 GEMM comparator and packed-weight transpose tests. Both
//! need a GPU and are `#[ignore]`d.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

// 2026-09-25: For each projection shape the comparator computes three results
// from the same inputs:
//   - out_cutlass: `nvfp4_gemm_bf16_act_weight_t` on the packed weight.
//   - out_ref: a host W4A4 reference that decodes the same packed nibbles and
//     E4M3 scales the kernel reads, and quantizes the activation with the
//     wrapper's per-16-group max/6 rule and E2M1 thresholds, keeping each
//     scale in f32 where the wrapper rounds it to UE4M3.
//   - out_true: the unquantized GEMM.
// cos(cutlass, ref) near 1 means the kernel computes what its packed operands
// imply, and any gap to out_true is W4A4 loss; a low value points at the
// wrapper's layout or scales. It prints a verdict; it asserts no accuracy bound.

use super::super::*;
use super::*;

#[test]
#[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
fn cutlass_nvfp4_projection_numeric_comparator() {
    // 2026-09-25: The weight scale layout depends on N and K only, so a small M
    // keeps the host reference cheap.
    const M: usize = 128;
    let shapes = [
        ("ssm_qkvz", 12288usize, 2048usize),
        ("attn_q", 8192, 2048),
        ("attn_kv", 512, 2048),
        ("attn_o", 2048, 4096),
    ];

    for (name, n, k) in shapes {
        assert_eq!(k % 16, 0, "{name}: K must be a multiple of 16");

        let weight_bf16: Vec<u16> = (0..n * k)
            .map(|i| f32_to_bf16(gen_val(i as u64) * 0.2))
            .collect();
        let act_bf16: Vec<u16> = (0..M * k)
            .map(|i| f32_to_bf16(gen_val((i as u64) ^ 0xA5A5_0000_0000) * 2.0))
            .collect();

        let packed_len = (k / 2) * n;
        let scale_len = (k / 16) * n;

        let weight_dev;
        let act_dev;
        let packed_dev;
        let scale_dev;
        let out_dev;
        unsafe {
            weight_dev = device_alloc(weight_bf16.len() * 2);
            act_dev = device_alloc(act_bf16.len() * 2);
            packed_dev = device_alloc(packed_len);
            scale_dev = device_alloc(scale_len);
            out_dev = device_alloc(M * n * 2);
            copy_h2d(weight_dev, &weight_bf16);
            copy_h2d(act_dev, &act_bf16);
        }

        // 2026-09-25: CUTLASS `[N,K/2]` nibbles and E4M3 `[K/16,N]` scales.
        pack_bf16_weight_to_nvfp4_t(
            weight_dev as u64,
            packed_dev as u64,
            scale_dev as u64,
            n as u32,
            k as u32,
            0,
        )
        .unwrap();
        unsafe {
            cuda_check(cudaDeviceSynchronize(), "pack synchronize");
        }

        // 2026-09-25: `weight_scale_2` is 1.0: the pack has no second-level scale.
        nvfp4_gemm_bf16_act_weight_t(
            act_dev as u64,
            packed_dev as u64,
            scale_dev as u64,
            1.0,
            out_dev as u64,
            M as u32,
            n as u32,
            k as u32,
            0,
        )
        .unwrap();
        unsafe {
            cuda_check(cudaDeviceSynchronize(), "cutlass gemm synchronize");
        }

        let mut out_cutlass_bf16 = vec![0u16; M * n];
        let mut packed = vec![0u8; packed_len];
        let mut scale = vec![0u8; scale_len];
        unsafe {
            copy_d2h(&mut out_cutlass_bf16, out_dev);
            copy_d2h(&mut packed, packed_dev);
            copy_d2h(&mut scale, scale_dev);
            cuda_check(cudaFree(weight_dev), "free weight");
            cuda_check(cudaFree(act_dev), "free act");
            cuda_check(cudaFree(packed_dev), "free packed");
            cuda_check(cudaFree(scale_dev), "free scale");
            cuda_check(cudaFree(out_dev), "free out");
        }

        // 2026-09-25: Decode the packed weight the kernel reads: the byte for
        // (n = col, k) is col*(K/2) + k/2, low nibble for even k; scales are
        // [K/16,N].
        let mut w_q = vec![0f32; n * k];
        let mut w_true = vec![0f32; n * k];
        for col in 0..n {
            for kk in 0..k {
                let g = kk / 16;
                let byte = packed[col * (k / 2) + kk / 2];
                let nib = if kk % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let s = e4m3_to_f32(scale[g * n + col]);
                w_q[col * k + kk] = decode_e2m1(nib) * s;
                w_true[col * k + kk] = bf16_to_f32(weight_bf16[col * k + kk]);
            }
        }

        // 2026-09-25: The wrapper's activation quantizer (per 16-element group,
        // scale = max|x| / 6, then E2M1), with the scale left in f32.
        let mut a_q = vec![0f32; M * k];
        let mut a_true = vec![0f32; M * k];
        for m in 0..M {
            for g in 0..(k / 16) {
                let base = g * 16;
                let mut max_abs = 0.0f32;
                for i in 0..16 {
                    let v = bf16_to_f32(act_bf16[m * k + base + i]);
                    max_abs = max_abs.max(v.abs());
                }
                let s = if max_abs > 0.0 { max_abs / 6.0 } else { 1.0 };
                let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                for i in 0..16 {
                    let v = bf16_to_f32(act_bf16[m * k + base + i]);
                    let nib = f32_to_e2m1(v * inv);
                    a_q[m * k + base + i] = decode_e2m1(nib) * s;
                    a_true[m * k + base + i] = v;
                }
            }
        }

        let mut out_ref = vec![0f32; M * n];
        let mut out_true = vec![0f32; M * n];
        let mut out_cutlass = vec![0f32; M * n];
        for m in 0..M {
            for col in 0..n {
                let mut acc_ref = 0.0f32;
                let mut acc_true = 0.0f32;
                for kk in 0..k {
                    acc_ref += a_q[m * k + kk] * w_q[col * k + kk];
                    acc_true += a_true[m * k + kk] * w_true[col * k + kk];
                }
                out_ref[m * n + col] = acc_ref;
                out_true[m * n + col] = acc_true;
                out_cutlass[m * n + col] = bf16_to_f32(out_cutlass_bf16[m * n + col]);
            }
        }

        let cos_cr = cosine(&out_cutlass, &out_ref);
        let cos_ct = cosine(&out_cutlass, &out_true);
        let cos_rt = cosine(&out_ref, &out_true);
        let max_abs_cr = out_cutlass
            .iter()
            .zip(&out_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let ref_rms =
            (out_ref.iter().map(|x| (x * x) as f64).sum::<f64>() / out_ref.len() as f64).sqrt();

        let verdict = if cos_cr > 0.999 {
            "KERNEL OK (cutlass matches W4A4 ref) -> divergence from true is inherent W4A4 loss"
        } else if cos_cr > 0.95 {
            "SUSPECT (minor cutlass<->ref drift; check scale rounding)"
        } else {
            "BUG (cutlass does NOT match its own packed operands -> layout/scale wrong)"
        };

        eprintln!(
            "NVFP4_COMPARATOR {name} M={M} N={n} K={k} \
             cos(cutlass,ref)={cos_cr:.6} cos(cutlass,true)={cos_ct:.6} \
             cos(ref,true)={cos_rt:.6} max_abs(cutlass-ref)={max_abs_cr:.5} \
             ref_rms={ref_rms:.5} => {verdict}"
        );
    }
}

#[test]
#[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
fn cutlass_nvfp4_transpose_is_bit_exact() {
    // 2026-09-25: The `[K/2,N] -> [N,K/2]` transpose that `cutlass_nvfp4_proj`
    // (metrale-model-layers) applies through
    // `cutlass_nvfp4_weight_transposed_cached`. A golden `[N,K/2]` pack is
    // transposed to `[K/2,N]` on the host, and the device transpose must
    // restore it byte for byte.
    const N: usize = 512;
    const K: usize = 256;
    let half = K / 2;

    let weight_bf16: Vec<u16> = (0..N * K)
        .map(|i| f32_to_bf16(gen_val(i as u64) * 0.2))
        .collect();

    let weight_dev;
    let golden_dev;
    let scale_dev;
    unsafe {
        weight_dev = device_alloc(weight_bf16.len() * 2);
        golden_dev = device_alloc(N * half);
        scale_dev = device_alloc((K / 16) * N);
        copy_h2d(weight_dev, &weight_bf16);
    }
    pack_bf16_weight_to_nvfp4_t(
        weight_dev as u64,
        golden_dev as u64,
        scale_dev as u64,
        N as u32,
        K as u32,
        0,
    )
    .unwrap();
    unsafe {
        cuda_check(cudaDeviceSynchronize(), "pack synchronize");
    }
    let mut golden = vec![0u8; N * half];
    unsafe {
        copy_d2h(&mut golden, golden_dev);
    }

    let mut checkpoint = vec![0u8; half * N];
    for c in 0..N {
        for h in 0..half {
            checkpoint[h * N + c] = golden[c * half + h];
        }
    }

    let src_dev;
    let dst_dev;
    unsafe {
        src_dev = device_alloc(checkpoint.len());
        dst_dev = device_alloc(N * half);
        copy_h2d(src_dev, &checkpoint);
    }
    transpose_nvfp4_packed_kton(src_dev as u64, dst_dev as u64, N as u32, K as u32, 0).unwrap();
    unsafe {
        cuda_check(cudaDeviceSynchronize(), "transpose synchronize");
    }
    let mut got = vec![0u8; N * half];
    unsafe {
        copy_d2h(&mut got, dst_dev);
        cuda_check(cudaFree(weight_dev), "free weight");
        cuda_check(cudaFree(golden_dev), "free golden");
        cuda_check(cudaFree(scale_dev), "free scale");
        cuda_check(cudaFree(src_dev), "free src");
        cuda_check(cudaFree(dst_dev), "free dst");
    }

    assert_eq!(
        got, golden,
        "device transpose must reproduce the golden [N,K/2] pack"
    );
    eprintln!(
        "NVFP4_TRANSPOSE bit-exact over N={N} K={K} ({} bytes) OK",
        N * half
    );
}
