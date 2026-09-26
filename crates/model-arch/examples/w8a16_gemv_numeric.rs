// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Numeric check of the block-scaled FP8 kernels `w8a16_gemv` (decode)
//! and `w8a16_gemm_pipelined` / `w8a16_gemm` (prefill, M = 256) on inputs whose
//! expected output is known in closed form. Results are printed as PASS/FAIL
//! (relative error above 2%); the exit code does not reflect them.
//!
//! GEMV construction:
//!   A[k]         = 1.0        for all k
//!   B[n,k]       = 0x38       = 1.0 in E4M3 (sign 0, exp 0111 -> 2^0, mantissa 0)
//!   block_scale  = S          constant over every block
//! then  C[n] = sum_k 1.0 * 1.0 * S = K * S  for every n.
//!
//! A per-block vs per-row scale-layout mix-up still gives the right answer
//! under a constant scale, so the run repeats with a scale that varies per
//! block, `scale[nb, kb] = 1 + nb + 100*kb`, where the two layouts disagree;
//! then C[n] = sum_kb width(kb) * (1 + nb + 100*kb), with width 128 except for
//! a short last K block.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Usage: cargo run --release -p metrale-model-arch --example w8a16_gemv_numeric -- [N] [K]
//! Defaults N=18048, K=4096. K must be a multiple of 16.

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const FP8_BLOCK: usize = 128;

fn bf16_bytes(v: f32) -> [u8; 2] {
    // 2026-09-25: f32 to BF16, round to nearest even (no NaN handling).
    let bits = v.to_bits();
    let rounded = ((bits >> 16) & 1).wrapping_add(0x7fff).wrapping_add(bits);
    ((rounded >> 16) as u16).to_le_bytes()
}

fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits(((u16::from_le_bytes([lo, hi])) as u32) << 16)
}

fn run(gpu: &dyn GpuBackend, stream: u64, n: u32, k: u32, vary: bool) -> Result<()> {
    let nb = (n as usize).div_ceil(FP8_BLOCK);
    let kb = (k as usize).div_ceil(FP8_BLOCK);

    let a_host: Vec<u8> = (0..k).flat_map(|_| bf16_bytes(1.0)).collect();
    let a = gpu.alloc(a_host.len())?;
    gpu.copy_h2d(&a_host, a)?;

    let b_host = vec![0x38u8; (n as usize) * (k as usize)];
    let b = gpu.alloc(b_host.len())?;
    gpu.copy_h2d(&b_host, b)?;

    // 2026-09-25: block_scale `[ceil(N/128), ceil(K/128)]` FP32, row-major, the
    // layout `w8a16_gemv` reads.
    let mut s_host = Vec::with_capacity(nb * kb * 4);
    for i in 0..nb {
        for j in 0..kb {
            let s = if vary {
                1.0f32 + i as f32 + 100.0 * j as f32
            } else {
                0.00088065f32
            };
            s_host.extend_from_slice(&s.to_le_bytes());
        }
    }
    let s = gpu.alloc(s_host.len())?;
    gpu.copy_h2d(&s_host, s)?;

    let c = gpu.alloc((n as usize) * 2)?;
    let kern = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    KernelLaunch::new(gpu, kern)
        .grid([n.div_ceil(4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(s)
        .arg_ptr(c)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    let mut c_host = vec![0u8; (n as usize) * 2];
    gpu.copy_d2h(c, &mut c_host)?;

    // 2026-09-25: Expected, per row n: the sum over k of scale[n/128, k/128]; each
    // K block contributes min(128, K - kb*128) terms.
    let expect = |row: usize| -> f32 {
        let i = row / FP8_BLOCK;
        (0..kb)
            .map(|j| {
                let width = ((k as usize) - j * FP8_BLOCK).min(FP8_BLOCK) as f32;
                let sv = if vary {
                    1.0f32 + i as f32 + 100.0 * j as f32
                } else {
                    0.00088065f32
                };
                sv * width
            })
            .sum()
    };

    let mut worst = 0.0f32;
    let mut worst_row = 0usize;
    let mut first_bad: Option<(usize, f32, f32)> = None;
    for row in 0..(n as usize) {
        let got = bf16_to_f32(c_host[row * 2], c_host[row * 2 + 1]);
        let want = expect(row);
        let rel = if want.abs() > 0.0 {
            ((got - want) / want).abs()
        } else {
            got.abs()
        };
        if rel > worst {
            worst = rel;
            worst_row = row;
        }
        if rel > 0.02 && first_bad.is_none() {
            first_bad = Some((row, got, want));
        }
    }

    let label = if vary {
        "VARYING block scale"
    } else {
        "constant scale"
    };
    println!("  {label}: N={n} K={k} blocks=[{nb},{kb}]");
    println!(
        "    row0: got={:.6} want={:.6}",
        bf16_to_f32(c_host[0], c_host[1]),
        expect(0)
    );
    if let Some((row, got, want)) = first_bad {
        println!("    FAIL first bad row {row}: got={got:.6} want={want:.6}");
        println!("    worst rel err {:.4} at row {worst_row}", worst);
    } else {
        println!("    PASS (worst rel err {:.6} at row {worst_row})", worst);
    }
    Ok(())
}

/// 2026-09-25: The prefill GEMM check: every weight is 1.0, row m of A is
/// `a_row(m)` and N block nb's scale is `s_blk(nb)`, so
/// `C[m, n] = a_row(m) * s_blk(n / 128) * K`.
fn run_gemm(gpu: &dyn GpuBackend, stream: u64, which: &str, m: u32, n: u32, k: u32) -> Result<()> {
    let nb = (n as usize).div_ceil(FP8_BLOCK);
    let kb = (k as usize).div_ceil(FP8_BLOCK);
    const S: f32 = 0.00088065;

    // 2026-09-25: A varies by row and the scale by N block, so the expected value
    // depends on both m and n: with all-ones inputs every element would be equal,
    // and a transposed or wrong-stride result would look correct.
    let a_row = |mi: usize| -> f32 { ((mi % 7) + 1) as f32 };
    let s_blk = |nbi: usize| -> f32 { S * (1 + (nbi % 5)) as f32 };

    let a_host: Vec<u8> = (0..(m as usize))
        .flat_map(|mi| (0..(k as usize)).flat_map(move |_| bf16_bytes(a_row(mi))))
        .collect();
    let a = gpu.alloc(a_host.len())?;
    gpu.copy_h2d(&a_host, a)?;

    let b_host = vec![0x38u8; (n as usize) * (k as usize)];
    let b = gpu.alloc(b_host.len())?;
    gpu.copy_h2d(&b_host, b)?;

    let mut s_host = Vec::with_capacity(nb * kb * 4);
    for nbi in 0..nb {
        for _ in 0..kb {
            s_host.extend_from_slice(&s_blk(nbi).to_le_bytes());
        }
    }
    let s = gpu.alloc(s_host.len())?;
    gpu.copy_h2d(&s_host, s)?;

    let c = gpu.alloc((m as usize) * (n as usize) * 2)?;
    let kern = gpu.kernel(which, which)?;
    let launch = KernelLaunch::new(gpu, kern);
    let launch = if which == "w8a16_gemm_pipelined" {
        launch
            .grid([n.div_ceil(32), m.div_ceil(128), 1])
            .block([256, 1, 1])
    } else {
        launch
            .grid([n.div_ceil(64), m.div_ceil(64), 1])
            .block([128, 1, 1])
    };
    launch
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(s)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    let mut c_host = vec![0u8; (m as usize) * (n as usize) * 2];
    gpu.copy_d2h(c, &mut c_host)?;
    let want_at = |mi: usize, ni: usize| -> f32 { a_row(mi) * s_blk(ni / FP8_BLOCK) * k as f32 };
    let mut worst = 0.0f32;
    let mut bad = 0usize;
    let mut first_bad: Option<(usize, usize, f32, f32)> = None;
    for mi in 0..(m as usize) {
        for ni in 0..(n as usize) {
            let idx = mi * (n as usize) + ni;
            let got = bf16_to_f32(c_host[idx * 2], c_host[idx * 2 + 1]);
            let want = want_at(mi, ni);
            let rel = ((got - want) / want).abs();
            if rel > worst {
                worst = rel;
            }
            if rel > 0.02 {
                bad += 1;
                if first_bad.is_none() {
                    first_bad = Some((mi, ni, got, want));
                }
            }
        }
    }
    println!("  {which}: M={m} N={n} K={k}");
    println!(
        "    [0,0] got={:.4} want={:.4} | [1,0] got={:.4} want={:.4} | [0,{}] got={:.4} want={:.4}",
        bf16_to_f32(c_host[0], c_host[1]),
        want_at(0, 0),
        bf16_to_f32(c_host[(n as usize) * 2], c_host[(n as usize) * 2 + 1]),
        want_at(1, 0),
        FP8_BLOCK,
        bf16_to_f32(c_host[FP8_BLOCK * 2], c_host[FP8_BLOCK * 2 + 1]),
        want_at(0, FP8_BLOCK),
    );
    if let Some((mi, ni, got, want)) = first_bad {
        println!(
            "    FAIL first bad [{mi},{ni}] got={got:.4} want={want:.4}  bad={bad}/{}",
            (m as usize) * (n as usize)
        );
    } else {
        println!("    PASS (worst rel {worst:.5})");
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n: u32 = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(18048);
    let k: u32 = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(4096);
    if !k.is_multiple_of(16) {
        bail!("w8a16_gemv requires K%16==0");
    }
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    println!("w8a16_gemv numeric check (DECODE)");
    run(&gpu, stream, n, k, false)?;
    run(&gpu, stream, n, k, true)?;
    println!("w8a16 GEMM numeric check (PREFILL), M=256");
    run_gemm(&gpu, stream, "w8a16_gemm_pipelined", 256, n, k)?;
    run_gemm(&gpu, stream, "w8a16_gemm", 256, n, k)?;
    Ok(())
}
