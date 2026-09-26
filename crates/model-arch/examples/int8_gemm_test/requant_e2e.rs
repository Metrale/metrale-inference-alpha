// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The end-to-end requant arm: NVFP4 weights and BF16 activations through the
//! `requant_*` kernels, then faith2, faith5 to faith10, mmqf and mmqf3, each against a
//! host GEMM on the dequantised weights.
//!
//! Owner: model-arch examples (int8 GEMM kernels).
//! Invariants: every buffer the arm allocates is freed before it returns `Ok`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::{Rng, bf16_bits_to_f32, up};

pub(super) fn requant_e2e(
    gpu: &dyn GpuBackend,
    stream: u64,
    hreqw: KernelHandle,
    hreqa: KernelHandle,
    hfaith2: KernelHandle,
    hreqa_il: KernelHandle,
    hfaith5: KernelHandle,
    hfaith6: KernelHandle,
    hfaith7: KernelHandle,
    hfaith8: KernelHandle,
    hfaith9: KernelHandle,
    hfaith10: KernelHandle,
    hmmqf: KernelHandle,
    hmmqf3: KernelHandle,
) -> Result<()> {
    let (m2, n2, k2) = (256usize, 256usize, 512usize);
    let nb2 = k2 / 32;
    let mut rng = Rng(0xBEEF);
    // 2026-09-25: NVFP4 weights: packed E2M1 nibbles [n2, k2/2], per-16 E4M3 scales
    // [n2, k2/16], and one scale2.
    let e2m1_lut = [
        0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let e4m3_dec = |b: u8| -> f32 {
        let s = (b >> 7) & 1;
        let e = (b >> 3) & 0xF;
        let mm = b & 0x7;
        let v = if e == 0 {
            (mm as f32) * 0.001953125
        } else {
            (1.0 + (mm as f32) / 8.0) * 2f32.powi(e as i32 - 7)
        };
        if s == 1 { -v } else { v }
    };
    let scale2 = 0.05f32;
    let nibbles: Vec<u8> = (0..n2 * k2).map(|_| (rng.next_u64() % 16) as u8).collect();
    let mut packed = vec![0u8; n2 * k2 / 2];
    for i in 0..n2 * k2 / 2 {
        packed[i] = nibbles[2 * i] | (nibbles[2 * i + 1] << 4);
    }
    // 2026-09-25: Scale bytes with exponent 1..=14, any mantissa and sign 0: positive,
    // normal and finite.
    let e4m3_bytes: Vec<u8> = (0..n2 * (k2 / 16))
        .map(|_| {
            let e = 1 + (rng.next_u64() % 14) as u8;
            let mm = (rng.next_u64() % 8) as u8;
            (e << 3) | mm
        })
        .collect();
    let mut w_real = vec![0f32; n2 * k2];
    for ni in 0..n2 {
        for ki in 0..k2 {
            let nib = nibbles[ni * k2 + ki] as usize;
            let s16 = e4m3_dec(e4m3_bytes[ni * (k2 / 16) + ki / 16]) * scale2;
            w_real[ni * k2 + ki] = e2m1_lut[nib] * s16;
        }
    }
    // 2026-09-25: BF16 activations truncated from f32; the reference uses the truncated
    // values.
    let a_f: Vec<f32> = (0..m2 * k2).map(|_| (rng.i8() as f32) * 0.02).collect();
    let a_bf16_bits: Vec<u16> = a_f.iter().map(|&x| (x.to_bits() >> 16) as u16).collect();
    let a_bf16_round: Vec<f32> = a_bf16_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
    let mut cref = vec![0f32; m2 * n2];
    for mi in 0..m2 {
        for ni in 0..n2 {
            let mut acc = 0f32;
            for ki in 0..k2 {
                acc += a_bf16_round[mi * k2 + ki] * w_real[ni * k2 + ki];
            }
            cref[mi * n2 + ni] = acc;
        }
    }
    let packed_p = up(gpu, &packed)?;
    let e4m3_p = up(gpu, &e4m3_bytes)?;
    let a_bf16_u8: Vec<u8> = a_bf16_bits.iter().flat_map(|&b| b.to_le_bytes()).collect();
    let abf_p = up(gpu, &a_bf16_u8)?;
    let wi8_p = gpu.alloc(n2 * k2)?;
    let wsc_p = gpu.alloc(n2 * nb2 * 4)?;
    let ai8_p = gpu.alloc(m2 * k2)?;
    let asc_p = gpu.alloc(m2 * nb2 * 4)?;
    let wblocks = (n2 * nb2) as u32;
    KernelLaunch::new(gpu, hreqw)
        .grid([wblocks.div_ceil(128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(packed_p)
        .arg_ptr(e4m3_p)
        .arg_f32(scale2)
        .arg_ptr(wi8_p)
        .arg_ptr(wsc_p)
        .arg_u32(n2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    let ablocks = (m2 * nb2) as u32;
    KernelLaunch::new(gpu, hreqa)
        .grid([ablocks.div_ceil(128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(abf_p)
        .arg_ptr(ai8_p)
        .arg_ptr(asc_p)
        .arg_u32(m2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    let ce = gpu.alloc(m2 * n2 * 2)?;
    KernelLaunch::new(gpu, hfaith2)
        .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(ai8_p)
        .arg_ptr(wi8_p)
        .arg_ptr(asc_p)
        .arg_ptr(wsc_p)
        .arg_ptr(ce)
        .arg_u32(m2 as u32)
        .arg_u32(n2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut re = vec![0u8; m2 * n2 * 2];
    gpu.copy_d2h(ce, &mut re)?;
    let cg: Vec<f32> = re
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
    for i in 0..m2 * n2 {
        let (x, y) = (cref[i] as f64, cg[i] as f64);
        d += x * y;
        nr += x * x;
        ng += y * y;
    }
    let cos = d / (nr.sqrt() * ng.sqrt());
    println!(
        "REQUANT e2e (NVFP4 w->int8 + bf16 a->int8 -> faith2) vs host dequant: cosine={cos:.6}  RESULT: {}",
        if cos > 0.999 { "PASS" } else { "FAIL" }
    );

    // 2026-09-25: faith5 reads the interleaved activation layout `requant_a_bf16_int8_il`
    // writes, and is scored against the same host reference.
    let ai8_il_p = gpu.alloc(m2 * k2)?;
    let asc5_p = gpu.alloc(m2 * nb2 * 4)?;
    KernelLaunch::new(gpu, hreqa_il)
        .grid([ablocks.div_ceil(128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(abf_p)
        .arg_ptr(ai8_il_p)
        .arg_ptr(asc5_p)
        .arg_u32(m2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    let ce5 = gpu.alloc(m2 * n2 * 2)?;
    KernelLaunch::new(gpu, hfaith5)
        .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(ai8_il_p)
        .arg_ptr(wi8_p)
        .arg_ptr(asc5_p)
        .arg_ptr(wsc_p)
        .arg_ptr(ce5)
        .arg_u32(m2 as u32)
        .arg_u32(n2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut re5 = vec![0u8; m2 * n2 * 2];
    gpu.copy_d2h(ce5, &mut re5)?;
    let cg5: Vec<f32> = re5
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut d5, mut nr5, mut ng5) = (0f64, 0f64, 0f64);
    for i in 0..m2 * n2 {
        let (x, y) = (cref[i] as f64, cg5[i] as f64);
        d5 += x * y;
        nr5 += x * x;
        ng5 += y * y;
    }
    let cos5 = d5 / (nr5.sqrt() * ng5.sqrt());
    println!(
        "REQUANT e2e faith5 (interleaved-q8_1 B-load) vs host dequant: cosine={cos5:.6}  RESULT: {}",
        if cos5 > 0.999 { "PASS" } else { "FAIL" }
    );

    // 2026-09-25: faith6 to faith10, mmqf and mmqf3 run on faith2's int8 inputs and are
    // scored against the same host reference by cosine; identity with faith2 is not checked.
    let ce6 = gpu.alloc(m2 * n2 * 2)?;
    KernelLaunch::new(gpu, hfaith6)
        .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(ai8_p)
        .arg_ptr(wi8_p)
        .arg_ptr(asc_p)
        .arg_ptr(wsc_p)
        .arg_ptr(ce6)
        .arg_u32(m2 as u32)
        .arg_u32(n2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut re6 = vec![0u8; m2 * n2 * 2];
    gpu.copy_d2h(ce6, &mut re6)?;
    let cg6: Vec<f32> = re6
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut d6, mut nr6, mut ng6) = (0f64, 0f64, 0f64);
    for i in 0..m2 * n2 {
        let (x, y) = (cref[i] as f64, cg6[i] as f64);
        d6 += x * y;
        nr6 += x * x;
        ng6 += y * y;
    }
    let cos6 = d6 / (nr6.sqrt() * ng6.sqrt());
    println!(
        "REQUANT e2e faith6 (WAR-break 3-phase schedule) vs host dequant: cosine={cos6:.6}  RESULT: {}",
        if cos6 > 0.999 { "PASS" } else { "FAIL" }
    );

    let ce7 = gpu.alloc(m2 * n2 * 2)?;
    KernelLaunch::new(gpu, hfaith7)
        .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(ai8_p)
        .arg_ptr(wi8_p)
        .arg_ptr(asc_p)
        .arg_ptr(wsc_p)
        .arg_ptr(ce7)
        .arg_u32(m2 as u32)
        .arg_u32(n2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut re7 = vec![0u8; m2 * n2 * 2];
    gpu.copy_d2h(ce7, &mut re7)?;
    let cg7: Vec<f32> = re7
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut d7, mut nr7, mut ng7) = (0f64, 0f64, 0f64);
    for i in 0..m2 * n2 {
        let (x, y) = (cref[i] as f64, cg7[i] as f64);
        d7 += x * y;
        nr7 += x * x;
        ng7 += y * y;
    }
    let cos7 = d7 / (nr7.sqrt() * ng7.sqrt());
    println!(
        "REQUANT e2e faith7 (64Nx32M 4:1-reuse reshape) vs host dequant: cosine={cos7:.6}  RESULT: {}",
        if cos7 > 0.999 { "PASS" } else { "FAIL" }
    );

    let ce8 = gpu.alloc(m2 * n2 * 2)?;
    KernelLaunch::new(gpu, hfaith8)
        .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(ai8_p)
        .arg_ptr(wi8_p)
        .arg_ptr(asc_p)
        .arg_ptr(wsc_p)
        .arg_ptr(ce8)
        .arg_u32(m2 as u32)
        .arg_u32(n2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut re8 = vec![0u8; m2 * n2 * 2];
    gpu.copy_d2h(ce8, &mut re8)?;
    let cg8: Vec<f32> = re8
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut d8, mut nr8, mut ng8) = (0f64, 0f64, 0f64);
    for i in 0..m2 * n2 {
        let (x, y) = (cref[i] as f64, cg8[i] as f64);
        d8 += x * y;
        nr8 += x * x;
        ng8 += y * y;
    }
    let cos8 = d8 / (nr8.sqrt() * ng8.sqrt());
    println!(
        "REQUANT e2e faith8 (2-stage cp.async pipeline) vs host dequant: cosine={cos8:.6}  RESULT: {}",
        if cos8 > 0.999 { "PASS" } else { "FAIL" }
    );

    let ce9 = gpu.alloc(m2 * n2 * 2)?;
    KernelLaunch::new(gpu, hfaith9)
        .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(ai8_p)
        .arg_ptr(wi8_p)
        .arg_ptr(asc_p)
        .arg_ptr(wsc_p)
        .arg_ptr(ce9)
        .arg_u32(m2 as u32)
        .arg_u32(n2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut re9 = vec![0u8; m2 * n2 * 2];
    gpu.copy_d2h(ce9, &mut re9)?;
    let cg9: Vec<f32> = re9
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut d9, mut nr9, mut ng9) = (0f64, 0f64, 0f64);
    for i in 0..m2 * n2 {
        let (x, y) = (cref[i] as f64, cg9[i] as f64);
        d9 += x * y;
        nr9 += x * x;
        ng9 += y * y;
    }
    let cos9 = d9 / (nr9.sqrt() * ng9.sqrt());
    println!(
        "REQUANT e2e faith9 (ITER_K=256) vs host dequant: cosine={cos9:.6}  RESULT: {}",
        if cos9 > 0.999 { "PASS" } else { "FAIL" }
    );

    let ce10 = gpu.alloc(m2 * n2 * 2)?;
    KernelLaunch::new(gpu, hfaith10)
        .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(ai8_p)
        .arg_ptr(wi8_p)
        .arg_ptr(asc_p)
        .arg_ptr(wsc_p)
        .arg_ptr(ce10)
        .arg_u32(m2 as u32)
        .arg_u32(n2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut re10 = vec![0u8; m2 * n2 * 2];
    gpu.copy_d2h(ce10, &mut re10)?;
    let cg10: Vec<f32> = re10
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut d10, mut nr10, mut ng10) = (0f64, 0f64, 0f64);
    for i in 0..m2 * n2 {
        let (x, y) = (cref[i] as f64, cg10[i] as f64);
        d10 += x * y;
        nr10 += x * x;
        ng10 += y * y;
    }
    let cos10 = d10 / (nr10.sqrt() * ng10.sqrt());
    println!(
        "REQUANT e2e faith10 (128-token weight reuse) vs host dequant: cosine={cos10:.6}  RESULT: {}",
        if cos10 > 0.999 { "PASS" } else { "FAIL" }
    );

    let cemf = gpu.alloc(m2 * n2 * 2)?;
    KernelLaunch::new(gpu, hmmqf)
        .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(ai8_p)
        .arg_ptr(wi8_p)
        .arg_ptr(asc_p)
        .arg_ptr(wsc_p)
        .arg_ptr(cemf)
        .arg_u32(m2 as u32)
        .arg_u32(n2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut remf = vec![0u8; m2 * n2 * 2];
    gpu.copy_d2h(cemf, &mut remf)?;
    let cgmf: Vec<f32> = remf
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut dmf, mut nrmf, mut ngmf) = (0f64, 0f64, 0f64);
    for i in 0..m2 * n2 {
        let (x, y) = (cref[i] as f64, cgmf[i] as f64);
        dmf += x * y;
        nrmf += x * x;
        ngmf += y * y;
    }
    let cosmf = dmf / (nrmf.sqrt() * ngmf.sqrt());
    println!(
        "REQUANT e2e mmqf (faithful llama-MMQ port) vs host dequant: cosine={cosmf:.6}  RESULT: {}",
        if cosmf > 0.999 { "PASS" } else { "FAIL" }
    );

    let cemf3 = gpu.alloc(m2 * n2 * 2)?;
    KernelLaunch::new(gpu, hmmqf3)
        .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(ai8_p)
        .arg_ptr(wi8_p)
        .arg_ptr(asc_p)
        .arg_ptr(wsc_p)
        .arg_ptr(cemf3)
        .arg_u32(m2 as u32)
        .arg_u32(n2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut remf3 = vec![0u8; m2 * n2 * 2];
    gpu.copy_d2h(cemf3, &mut remf3)?;
    let cgmf3: Vec<f32> = remf3
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut dmf3, mut nrmf3, mut ngmf3) = (0f64, 0f64, 0f64);
    for i in 0..m2 * n2 {
        let (x, y) = (cref[i] as f64, cgmf3[i] as f64);
        dmf3 += x * y;
        nrmf3 += x * x;
        ngmf3 += y * y;
    }
    let cosmf3 = dmf3 / (nrmf3.sqrt() * ngmf3.sqrt());
    println!(
        "REQUANT e2e mmqf3 (ILP-max MMA schedule) vs host dequant: cosine={cosmf3:.6}  RESULT: {}",
        if cosmf3 > 0.999 { "PASS" } else { "FAIL" }
    );

    for p in [
        packed_p, e4m3_p, abf_p, wi8_p, wsc_p, ai8_p, asc_p, ce, ai8_il_p, asc5_p, ce5, ce6, ce7,
        ce8, ce9, ce10, cemf, cemf3,
    ] {
        let _ = gpu.free(p);
    }
    Ok(())
}
