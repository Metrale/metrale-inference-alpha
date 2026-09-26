// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Correctness test for the runtime LoRA delta, `ops::lora_delta::apply_lora_delta`,
//! stage by stage against a CPU reference that rounds to BF16 where the GPU stores BF16.
//!
//! Owner: model-arch examples (LoRA).
//! Invariants:
//! - The run fails unless every stage it compares reaches `COSINE_GATE`.
//!
//! The stages, so a divergence names the stage:
//!   shrink:  `xa[j] = sum_k x[k] * A[j, k]`      (A padded to `[max_rank, k_in]`)
//!   expand:  `delta[n] = sum_j xa[j] * B[n, j]`  (B padded to `[n_out, max_rank]`)
//!   fold:    `out[n] += scale * delta[n]`        (out starts at zero)
//!
//! Usage:
//!   cargo run --release -p metrale-model-arch --example lora_apply_microtest \
//!       -- `[k_in] [n_out] [r] [max_rank] [m] [seed]`
//! Defaults: k_in=1024 n_out=512 (the Holo-3.1-0.8B `k_proj` shape) r=8 max_rank=64 m=1.
//! `apply_lora_delta` runs `dense_gemv` per row for `m <= lora_gemv_max_m()` and a GEMM path
//! above that.

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::lora_delta::{LoraKernels, LoraPair, apply_lora_delta};
use metrale_model_layers::weight_map::DenseWeight;

const COSINE_GATE: f64 = 0.999;

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
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + u * (hi - lo)
    }
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le_to_u16s(v: &[u8]) -> Vec<u16> {
    v.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}
fn upload(gpu: &dyn GpuBackend, bits: &[u16]) -> Result<DevicePtr> {
    let bytes = u16s_to_le(bits);
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

/// 2026-09-25: Cosine and max/mean relative error of BF16 GPU values against an f32 reference;
/// passes at `COSINE_GATE`.
fn compare(label: &str, gpu: &[u16], reference: &[f32]) -> bool {
    let (mut dot, mut ng, mut nr, mut maxrel, mut sumrel) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g_bits, &r) in gpu.iter().zip(reference) {
        let g = bf16_bits_to_f32(*g_bits) as f64;
        let r = r as f64;
        dot += g * r;
        ng += g * g;
        nr += r * r;
        let denom = r.abs().max(1e-3);
        let rel = (g - r).abs() / denom;
        maxrel = maxrel.max(rel);
        sumrel += rel;
    }
    let cos = if ng > 0.0 && nr > 0.0 {
        dot / (ng.sqrt() * nr.sqrt())
    } else {
        0.0
    };
    let pass = cos >= COSINE_GATE;
    println!(
        "  {:6} {label:12} cosine={cos:.6} max_rel={maxrel:.4} mean_rel={:.4}  |gpu|₂={:.4} |ref|₂={:.4}",
        if pass { "PASS" } else { "FAIL ❌" },
        sumrel / gpu.len().max(1) as f64,
        ng.sqrt(),
        nr.sqrt(),
    );
    pass
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let k_in: usize = a.get(1).map_or(1024, |s| s.parse().unwrap());
    let n_out: usize = a.get(2).map_or(512, |s| s.parse().unwrap());
    let r: usize = a.get(3).map_or(8, |s| s.parse().unwrap());
    let max_rank: usize = a.get(4).map_or(64, |s| s.parse().unwrap());
    let m: usize = a.get(5).map_or(1, |s| s.parse().unwrap());
    let seed: u64 = a.get(6).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });
    let scale = 2.0f32;
    assert!(r <= max_rank);
    let path = if m == 1 {
        "decode dense_gemv"
    } else {
        "prefill dense_gemm"
    };
    println!(
        "=== lora_apply microtest: k_in={k_in} n_out={n_out} r={r} max_rank={max_rank} m={m} ({path}) scale={scale} seed=0x{seed:X} ==="
    );

    let mut rng = Rng(seed);
    let x: Vec<u16> = (0..m * k_in)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    // 2026-09-25: A is [r, k_in] and B is [n_out, r] before padding.
    let a_real: Vec<u16> = (0..r * k_in)
        .map(|_| f32_to_bf16_bits(rng.uniform(-0.05, 0.05)))
        .collect();
    let b_real: Vec<u16> = (0..n_out * r)
        .map(|_| f32_to_bf16_bits(rng.uniform(-0.05, 0.05)))
        .collect();

    // 2026-09-25: Pad to the pool layout: A [max_rank, k_in] with zero rows past r, and
    // B [n_out, max_rank] with zero columns past r.
    let mut a_pool = vec![0u16; max_rank * k_in];
    for j in 0..r {
        a_pool[j * k_in..(j + 1) * k_in].copy_from_slice(&a_real[j * k_in..(j + 1) * k_in]);
    }
    let mut b_pool = vec![0u16; n_out * max_rank];
    for n in 0..n_out {
        b_pool[n * max_rank..n * max_rank + r].copy_from_slice(&b_real[n * r..(n + 1) * r]);
    }

    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let kernels = LoraKernels::new(gpu)?;

    let x_ptr = upload(gpu, &x)?;
    let a_ptr = upload(gpu, &a_pool)?;
    let b_ptr = upload(gpu, &b_pool)?;
    let xa_ptr = gpu.alloc(m * max_rank * 2)?;
    let delta_ptr = gpu.alloc(m * n_out * 2)?;
    let out_ptr = gpu.alloc(m * n_out * 2)?;
    gpu.memset(out_ptr, 0, m * n_out * 2)?;

    let pair = LoraPair {
        a: DenseWeight { weight: a_ptr },
        b: DenseWeight { weight: b_ptr },
        rank: r as u32,
        k_in: k_in as u32,
        n_out: n_out as u32,
        scale,
        max_rank: max_rank as u32,
    };

    apply_lora_delta(
        gpu, &kernels, &pair, x_ptr, out_ptr, m as u32, xa_ptr, delta_ptr, stream,
    )?;
    gpu.synchronize(stream)?;

    let mut xa_raw = vec![0u8; m * max_rank * 2];
    let mut delta_raw = vec![0u8; m * n_out * 2];
    let mut out_raw = vec![0u8; m * n_out * 2];
    gpu.copy_d2h(xa_ptr, &mut xa_raw)?;
    gpu.copy_d2h(delta_ptr, &mut delta_raw)?;
    gpu.copy_d2h(out_ptr, &mut out_raw)?;
    let xa_gpu = le_to_u16s(&xa_raw);
    let delta_gpu = le_to_u16s(&delta_raw);
    let out_gpu = le_to_u16s(&out_raw);

    // 2026-09-25: CPU reference per row: f32 accumulation, rounded to BF16 after the shrink
    // and after the expand.
    let xf: Vec<f32> = x.iter().map(|&b| bf16_bits_to_f32(b)).collect();
    let af: Vec<f32> = a_real.iter().map(|&b| bf16_bits_to_f32(b)).collect();
    let bf: Vec<f32> = b_real.iter().map(|&b| bf16_bits_to_f32(b)).collect();
    let mut xa_ref = vec![0f32; m * max_rank];
    let mut delta_ref = vec![0f32; m * n_out];
    for row in 0..m {
        for j in 0..r {
            let mut acc = 0f32;
            for k in 0..k_in {
                acc += xf[row * k_in + k] * af[j * k_in + k];
            }
            xa_ref[row * max_rank + j] = bf16_bits_to_f32(f32_to_bf16_bits(acc));
        }
        for n in 0..n_out {
            let mut acc = 0f32;
            for j in 0..r {
                acc += xa_ref[row * max_rank + j] * bf[n * r + j];
            }
            delta_ref[row * n_out + n] = bf16_bits_to_f32(f32_to_bf16_bits(acc));
        }
    }
    let out_ref: Vec<f32> = delta_ref.iter().map(|&d| scale * d).collect();

    println!("stage bisection ({path}):");
    // 2026-09-25: Compare only the xa columns below r.
    let mut xa_gpu_v = Vec::new();
    let mut xa_ref_v = Vec::new();
    for row in 0..m {
        for j in 0..r {
            xa_gpu_v.push(xa_gpu[row * max_rank + j]);
            xa_ref_v.push(xa_ref[row * max_rank + j]);
        }
    }
    let p1 = compare("shrink xa", &xa_gpu_v, &xa_ref_v);
    let p2 = compare("expand delta", &delta_gpu, &delta_ref);
    let p3 = compare("fold out", &out_gpu, &out_ref);

    // 2026-09-25: Padded xa columns that are not zero are reported, not gated.
    let mut pad_nonzero = 0usize;
    for row in 0..m {
        for j in r..max_rank {
            if bf16_bits_to_f32(xa_gpu[row * max_rank + j]) != 0.0 {
                pad_nonzero += 1;
            }
        }
    }
    if pad_nonzero > 0 {
        println!("  NOTE: {pad_nonzero} padded xa cols are NONZERO across {m} rows (should be 0)");
    }

    if p1 && p2 && p3 {
        println!("RESULT: PASS ✅ — runtime apply matches reference");
        Ok(())
    } else {
        bail!("RESULT: FAIL — first diverging stage localizes the bug");
    }
}
