// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Correctness gate for `attn_prefill_paged_fp8`, paged prefill
//! attention over an FP8 E4M3 KV cache, against an FP32 CPU reference.
//!
//! gb10's `attn_prefill_paged_fp8.cu` is the only includer of
//! `prefill_paged_compute.cuh` that defines `METRALE_ATTN_FP8_SMEM`, which keeps
//! K/V in shared memory as raw E4M3 bytes and dequantizes them in registers
//! before each MMA. `attn_prefill_microtest` launches the BF16 `attn_prefill_64`
//! and does not reach that path.
//!
//! Setup: one paged KV block (`block_table = [0]`), HDIM 256, random E4M3 K/V
//! bytes with per-tensor `k_scale` and `v_scale`. The reference dequantizes the
//! same bytes, so the cosine measures the attention, not input quantization.
//! Queries attend causally over themselves (`qlen = kvlen = seq`, `qoff = 0`)
//! unless `METRALE_QLEN`, `METRALE_KVLEN` or `METRALE_QOFF` is set.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Usage: cargo run --release -p metrale-model-arch \
//!          --example attn_prefill_fp8_microtest --features cuda,gpu-examples \
//!          -- [seq] [nq] [nkv] [seed]
//! Exits 1 when the cosine is below the gate. When `qlen * kvlen` exceeds
//! 4,000,000 the CPU reference is skipped and the run exits 0 without a cosine.

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

const HDIM: usize = 256; // 2026-09-25: the kernel's compile-time HDIM (prefill_paged_compute.cuh).
const COSINE_GATE: f64 = 0.99;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32)
    }
    /// 2026-09-25: A random E4M3 byte with a biased exponent in [4, 9]:
    /// magnitudes 0.125 to 7.5, never NaN or subnormal.
    fn e4m3_byte(&mut self) -> u8 {
        let s = (self.next_u64() & 1) as u8;
        let e = 4u8 + (self.next_u64() % 6) as u8;
        let m = (self.next_u64() % 8) as u8;
        (s << 7) | (e << 3) | m
    }
}

/// 2026-09-25: Decode an E4M3 byte (exponent bias 7; only S.1111.111 is NaN) to
/// f32. The kernel's `fp8_to_bf16` converts the same byte through
/// `__nv_cvt_fp8_to_halfraw` before scaling (gb10 `attn_prefill_paged_fp8.cu`).
fn e4m3_to_f32(b: u8) -> f32 {
    let s = (b >> 7) & 1;
    let e = (b >> 3) & 0xF;
    let m = b & 0x7;
    let sign = if s == 1 { -1.0 } else { 1.0 };
    if e == 0 {
        sign * (m as f32 / 8.0) * 2f32.powi(1 - 7)
    } else if e == 0xF && m == 0x7 {
        f32::NAN
    } else {
        sign * (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
    }
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    ((bits.wrapping_add(0x7FFF + ((bits >> 16) & 1))) >> 16) as u16
}
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn upload(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len())?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let seq: usize = a.get(1).map_or(64, |s| s.parse().unwrap());
    let nq: usize = a.get(2).map_or(2, |s| s.parse().unwrap());
    let nkv: usize = a.get(3).map_or(1, |s| s.parse().unwrap());
    let seed: u64 = a.get(4).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });
    let env_usize = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let qlen = env_usize("METRALE_QLEN", seq);
    let kvlen = env_usize("METRALE_KVLEN", seq);
    let qoff = env_usize("METRALE_QOFF", 0);
    let hd = HDIM;
    let inv_sqrt_d = 1.0f32 / (hd as f32).sqrt();
    let k_scale = 0.25f32;
    let v_scale = 0.25f32;
    println!(
        "=== attn_prefill_paged_fp8 (FP8-smem) microtest: \
         qlen={qlen} kvlen={kvlen} qoff={qoff} nq={nq} nkv={nkv} hd={hd} \
         k_scale={k_scale} v_scale={v_scale} seed=0x{seed:X} ==="
    );

    let mut rng = Rng(seed);
    let q: Vec<u16> = (0..qlen * nq * hd)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let k_fp8: Vec<u8> = (0..kvlen * nkv * hd).map(|_| rng.e4m3_byte()).collect();
    let v_fp8: Vec<u8> = (0..kvlen * nkv * hd).map(|_| rng.e4m3_byte()).collect();

    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let qp = upload(gpu, &u16s_to_le(&q))?;
    let kp = upload(gpu, &k_fp8)?;
    let vp = upload(gpu, &v_fp8)?;
    let op = gpu.alloc(qlen * nq * hd * 2)?;
    let btp = upload(gpu, &0i32.to_le_bytes())?;

    let cache_block_size = kvlen as u32;
    let cache_stride = (kvlen * nkv * hd) as u64;

    // 2026-09-25: The launch of `ops::prefill_attention_paged_fp8`: 32 query rows
    // per block, 128 threads, the same argument order.
    let br = 32u32;
    let handle = gpu.kernel("prefill_paged_fp8", "attn_prefill_paged_fp8")?;
    KernelLaunch::new(gpu, handle)
        .grid([nq as u32, div_ceil(qlen as u32, br), 1])
        .block([128, 1, 1])
        .arg_ptr(qp)
        .arg_ptr(kp)
        .arg_ptr(vp)
        .arg_ptr(op)
        .arg_ptr(btp)
        .arg_u32(qlen as u32)
        .arg_u32(kvlen as u32)
        .arg_u32(qoff as u32)
        .arg_u32(nq as u32)
        .arg_u32(nkv as u32)
        .arg_u32(hd as u32)
        .arg_u32(cache_block_size)
        .arg_u32(0)
        .arg_u32(1)
        .arg_f32(inv_sqrt_d)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u64(cache_stride)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    // 2026-09-25: With METRALE_BENCH_ITERS set (50 when it does not parse), run
    // 10 warm-up launches, then time that many launches between two synchronizes
    // and print the mean wall time per launch.
    if let Ok(iters_s) = std::env::var("METRALE_BENCH_ITERS") {
        let iters: usize = iters_s.parse().unwrap_or(50);
        let relaunch = || -> Result<()> {
            KernelLaunch::new(gpu, handle)
                .grid([nq as u32, div_ceil(qlen as u32, br), 1])
                .block([128, 1, 1])
                .arg_ptr(qp)
                .arg_ptr(kp)
                .arg_ptr(vp)
                .arg_ptr(op)
                .arg_ptr(btp)
                .arg_u32(qlen as u32)
                .arg_u32(kvlen as u32)
                .arg_u32(qoff as u32)
                .arg_u32(nq as u32)
                .arg_u32(nkv as u32)
                .arg_u32(hd as u32)
                .arg_u32(cache_block_size)
                .arg_u32(0)
                .arg_u32(1)
                .arg_f32(inv_sqrt_d)
                .arg_f32(k_scale)
                .arg_f32(v_scale)
                .arg_u64(cache_stride)
                .launch(stream)?;
            Ok(())
        };
        for _ in 0..10 {
            relaunch()?;
        }
        gpu.synchronize(stream)?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            relaunch()?;
        }
        gpu.synchronize(stream)?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "BENCH: {us:.2} us/launch  (iters={iters}, grid_y={})",
            div_ceil(qlen as u32, br)
        );
    }

    let mut raw = vec![0u8; qlen * nq * hd * 2];
    gpu.copy_d2h(op, &mut raw)?;
    let o_gpu: Vec<u16> = raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    if qlen * kvlen > 4_000_000 {
        println!(
            "cosine skipped (warm shape qlen*kvlen={} > 4M; timing-only)",
            qlen * kvlen
        );
        for p in [qp, kp, vp, op, btp] {
            gpu.free(p).ok();
        }
        return Ok(());
    }

    // 2026-09-25: FP32 causal GQA reference over the same E4M3 bytes: query `i`
    // attends to KV positions `0..=qoff + i`, capped at `kvlen - 1`.
    let gqa = nq / nkv;
    let mut o_cpu = vec![0u16; qlen * nq * hd];
    for h in 0..nq {
        let kvh = h / gqa;
        for i in 0..qlen {
            let nkeys = (qoff + i + 1).min(kvlen);
            let mut scores = vec![0f32; nkeys];
            let mut mx = -1e30f32;
            for (j, sj) in scores.iter_mut().enumerate() {
                let mut s = 0f32;
                for d in 0..hd {
                    let kdq = e4m3_to_f32(k_fp8[(j * nkv + kvh) * hd + d]) * k_scale;
                    s += bf16_bits_to_f32(q[(i * nq + h) * hd + d]) * kdq;
                }
                s *= inv_sqrt_d;
                *sj = s;
                if s > mx {
                    mx = s;
                }
            }
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for d in 0..hd {
                let mut acc = 0f32;
                for (j, sj) in scores.iter().enumerate() {
                    let vdq = e4m3_to_f32(v_fp8[(j * nkv + kvh) * hd + d]) * v_scale;
                    acc += *sj / sum * vdq;
                }
                o_cpu[(i * nq + h) * hd + d] = f32_to_bf16_bits(acc);
            }
        }
    }

    let cosall = cos(&o_gpu, &o_cpu, 0, qlen * nq * hd);
    println!("cosine(all)={cosall:.6}");
    for p in [qp, kp, vp, op, btp] {
        gpu.free(p).ok();
    }
    if cosall >= COSINE_GATE {
        println!("RESULT: PASS (cosine {cosall:.6} >= {COSINE_GATE})");
        Ok(())
    } else {
        println!("RESULT: FAIL (cosine {cosall:.6} < {COSINE_GATE})");
        std::process::exit(1);
    }
}

fn cos(a: &[u16], b: &[u16], off: usize, n: usize) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in off..off + n {
        let x = bf16_bits_to_f32(a[i]) as f64;
        let y = bf16_bits_to_f32(b[i]) as f64;
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return f64::NAN;
    }
    d / (na.sqrt() * nb.sqrt())
}
