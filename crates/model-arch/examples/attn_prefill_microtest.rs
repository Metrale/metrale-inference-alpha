// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Correctness oracle for `attn_prefill_64`, the contiguous BF16
//! causal GQA flash-attention prefill kernel, against an FP32 CPU reference.
//!
//! The launch is the one `ops::prefill_attention_64` uses (model-layers
//! `prefill_attn_main_a.rs`): grid `(nq, ceil(seq / br), 1)`, 256 threads, `br`
//! 32 under `metrale_scale` and 64 otherwise. For `seq > 32` the run also fails
//! on any all-zero output row at position 32 or later, the rows a mismatch
//! between `br` and the kernel's `BR64` leaves unwritten.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Usage: cargo run --release -p metrale-model-arch --example attn_prefill_microtest \
//!          --features cuda,gpu-examples -- [seq] [nq] [nkv] [seed]
//! Exits 1 on FAIL.

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

const HDIM: usize = 256; // 2026-09-25: the kernel's compile-time HDIM (attn_prefill.cu).
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
    let seq: usize = a.get(1).map_or(16, |s| s.parse().unwrap());
    let nq: usize = a.get(2).map_or(2, |s| s.parse().unwrap());
    let nkv: usize = a.get(3).map_or(1, |s| s.parse().unwrap());
    let seed: u64 = a.get(4).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });
    let hd = HDIM;
    let inv_sqrt_d = 1.0f32 / (hd as f32).sqrt();
    println!(
        "=== attn_prefill_64 microtest: seq={seq} nq={nq} nkv={nkv} hd={hd} seed=0x{seed:X} ==="
    );

    let mut rng = Rng(seed);
    let q: Vec<u16> = (0..seq * nq * hd)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let k: Vec<u16> = (0..seq * nkv * hd)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let v: Vec<u16> = (0..seq * nkv * hd)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();

    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let qp = upload(gpu, &u16s_to_le(&q))?;
    let kp = upload(gpu, &u16s_to_le(&k))?;
    let vp = upload(gpu, &u16s_to_le(&v))?;
    let op = gpu.alloc(seq * nq * hd * 2)?;

    // 2026-09-25: Rows per block must equal the kernel's `BR64`, which is 32 on
    // the `metrale_scale` targets and 64 otherwise, as in `ops::prefill_attention_64`.
    let br = if cfg!(metrale_scale) { 32u32 } else { 64u32 };
    let handle = gpu.kernel("attn_prefill", "attn_prefill_64")?;
    KernelLaunch::new(gpu, handle)
        .grid([nq as u32, div_ceil(seq as u32, br), 1])
        .block([256, 1, 1])
        .arg_ptr(qp)
        .arg_ptr(kp)
        .arg_ptr(vp)
        .arg_ptr(op)
        .arg_u32(seq as u32)
        .arg_u32(nq as u32)
        .arg_u32(nkv as u32)
        .arg_u32(hd as u32)
        .arg_f32(inv_sqrt_d)
        .arg_u32(1)
        .arg_u32(0)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    let mut raw = vec![0u8; seq * nq * hd * 2];
    gpu.copy_d2h(op, &mut raw)?;
    let o_gpu: Vec<u16> = raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let gqa = nq / nkv;
    let mut o_cpu = vec![0u16; seq * nq * hd];
    for h in 0..nq {
        let kvh = h / gqa;
        for i in 0..seq {
            let mut scores = vec![0f32; i + 1];
            let mut mx = -1e30f32;
            for j in 0..=i {
                let mut s = 0f32;
                for d in 0..hd {
                    s += bf16_bits_to_f32(q[(i * nq + h) * hd + d])
                        * bf16_bits_to_f32(k[(j * nkv + kvh) * hd + d]);
                }
                s *= inv_sqrt_d;
                scores[j] = s;
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
                for j in 0..=i {
                    acc += scores[j] / sum * bf16_bits_to_f32(v[(j * nkv + kvh) * hd + d]);
                }
                o_cpu[(i * nq + h) * hd + d] = f32_to_bf16_bits(acc);
            }
        }
    }

    let cosall = cos(&o_gpu, &o_cpu, 0, seq * nq * hd);
    let mut dot = 0f64;
    let mut zero_rows = 0usize;
    if seq > 32 {
        for h in 0..nq {
            for i in 32..seq {
                let off = (i * nq + h) * hd;
                let mag: f64 = (0..hd)
                    .map(|d| {
                        let g = bf16_bits_to_f32(o_gpu[off + d]) as f64;
                        g * g
                    })
                    .sum();
                if mag < 1e-9 {
                    zero_rows += 1;
                }
                let _ = &mut dot;
            }
        }
    }
    println!(
        "cosine(all)={cosall:.6}  zero_rows(i>=32)={zero_rows}/{}",
        if seq > 32 { (seq - 32) * nq } else { 0 }
    );
    for p in [qp, kp, vp, op] {
        gpu.free(p).ok();
    }
    if cosall >= COSINE_GATE && zero_rows == 0 {
        println!("RESULT: PASS (cosine {cosall:.6} >= {COSINE_GATE})");
        Ok(())
    } else {
        println!("RESULT: FAIL (cosine {cosall:.6} or {zero_rows} dropped rows)");
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
