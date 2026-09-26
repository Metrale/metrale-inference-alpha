// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Accuracy gate for `w4a16_gemm_t_m128_bf16_v2`.
//!
//! Per shape, v2 must reach cosine >= `COSINE_GATE` against the base
//! `w4a16_gemm`, and at least 99.9999% of its output elements must be
//! bit-identical to `w4a16_gemm_t_m128_bf16` (v1). When the target has
//! `w4a16_v2::w4a16_gemm_t_m128_v2`, it must meet the same bit-identity bar
//! against `w4a16_gemm_t_m128`; otherwise those legs are skipped. The launches
//! are in `run_shape` (`common/w4a16_bf16_v2_shape.rs`).
//!
//! Layouts (those of `QuantizedWeight::transpose_for_gemm`):
//!   - base `w4a16_gemm`:              B_packed [N, K/2],   B_scale [N, K/16]
//!   - the `w4a16_gemm_t_m128*` kernels: B_packed [K/2, N],   B_scale [K/16, N]
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Needs a target whose `w4a16` module has these kernels (the gb10 qwen3.6-27b
//! kernels). Usage, with an optional hex seed:
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example w4a16_bf16_v2_microtest -- [seed]
//! Exit 0 = all shapes pass, 1 = any shape fails.

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

#[path = "common/w4a16_bf16_v2_shape.rs"]
pub(crate) mod w4a16_bf16_v2_shape;
use w4a16_bf16_v2_shape::*;

/// 2026-09-25: NVFP4 group size along K (`GROUP_SIZE` in w4a16_gemm.cu).
const GROUP_SIZE: usize = 16;

/// 2026-09-25: Minimum base-vs-v2 cosine for a shape to pass.
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
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
/// 2026-09-25: f32 to BF16 bits, round to nearest even; NaN stays a quiet NaN.
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

/// 2026-09-25: The E2M1 value table, written out here rather than taken from
/// w4a16_gemm.cu, so the check does not reuse the kernel's own table.
const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// 2026-09-25: An E4M3 group-scale byte with mantissa 0 and exponent field
/// 5..=9, i.e. one of the exact values {0.25, 0.5, 1, 2, 4}.
fn e4m3_scale_byte(sel: u32) -> u8 {
    let e = 5 + (sel % 5);
    ((e as u8) << 3) & 0x7F
}
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2f32.powi(-6)
    } else if exp == 0x0F && mant == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// 2026-09-25: One generated NVFP4 weight in both layouts, plus `scale2`.
struct Nvfp4Weight {
    packed_nt: Vec<u8>,
    scale_nt: Vec<u8>,
    packed_t: Vec<u8>,
    scale_t: Vec<u8>,
    scale2: f32,
}

/// 2026-09-25: Build a random NVFP4 weight `[N, K]` in packed form, then its
/// transposed layout by the byte transpose `transpose_for_gemm` performs.
fn gen_weight(rng: &mut Rng, n: usize, k: usize) -> Nvfp4Weight {
    assert!(
        k.is_multiple_of(GROUP_SIZE),
        "K must be a multiple of {GROUP_SIZE}"
    );
    let half_k = k / 2;
    let num_groups = k / GROUP_SIZE;
    let mut packed_nt = vec![0u8; n * half_k];
    let mut scale_nt = vec![0u8; n * num_groups];

    for i in 0..n {
        for g in 0..num_groups {
            scale_nt[i * num_groups + g] = e4m3_scale_byte(rng.next_u64() as u32);
        }
        for j in 0..half_k {
            // 2026-09-25: Low nibble is even k (2j), high nibble odd k (2j+1).
            let lo = (rng.next_u64() % 16) as u8;
            let hi = (rng.next_u64() % 16) as u8;
            packed_nt[i * half_k + j] = (hi << 4) | lo;
        }
    }

    let mut packed_t = vec![0u8; n * half_k];
    for i in 0..n {
        for j in 0..half_k {
            packed_t[j * n + i] = packed_nt[i * half_k + j];
        }
    }
    let mut scale_t = vec![0u8; n * num_groups];
    for i in 0..n {
        for g in 0..num_groups {
            scale_t[g * n + i] = scale_nt[i * num_groups + g];
        }
    }

    Nvfp4Weight {
        packed_nt,
        scale_nt,
        packed_t,
        scale_t,
        // 2026-09-25: Not 1.0, so every path's scale2 multiply is exercised.
        scale2: 0.5,
    }
}

struct Stats {
    cosine: f64,
    max_abs: f64,
    #[allow(dead_code)]
    max_rel: f64,
    frac_bit_identical: f64,
}

fn compare(a: &[u16], b: &[u16]) -> Stats {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let (mut max_abs, mut max_rel) = (0f64, 0f64);
    let mut bit_eq = 0usize;
    for i in 0..a.len() {
        if a[i] == b[i] {
            bit_eq += 1;
        }
        let x = bf16_bits_to_f32(a[i]) as f64;
        let y = bf16_bits_to_f32(b[i]) as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        let denom = x.abs().max(y.abs());
        if denom > 1e-6 {
            let r = d / denom;
            if r > max_rel {
                max_rel = r;
            }
        }
    }
    let cosine = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        1.0
    };
    Stats {
        cosine,
        max_abs,
        max_rel,
        frac_bit_identical: bit_eq as f64 / a.len() as f64,
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });

    debug_assert_eq!(E2M1_LUT[7], 6.0);
    debug_assert!((e4m3_to_f32(e4m3_scale_byte(2)) - 1.0).abs() < 1e-6);

    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let base_h = gpu.kernel("w4a16", "w4a16_gemm")?;
    let bf16_h = gpu.kernel("w4a16", "w4a16_gemm_t_m128_bf16")?;
    let v2_h = gpu.kernel("w4a16", "w4a16_gemm_t_m128_bf16_v2")?;
    // 2026-09-25: `w4a16_gemm_t_m128` is required; `w4a16_gemm_t_m128_v2` is
    // optional, and without it the crush legs are skipped.
    let crush1_h = gpu.kernel("w4a16", "w4a16_gemm_t_m128")?;
    let crush2_h = gpu
        .kernel("w4a16_v2", "w4a16_gemm_t_m128_v2")
        .unwrap_or(metrale_gpu_runtime::gpu::KernelHandle(0));
    if crush2_h.0 == 0 {
        println!("NOTE: w4a16_v2::w4a16_gemm_t_m128_v2 absent — crush v1/v2 legs SKIPPED\n");
    }

    // 2026-09-25: `(label, M, N, K)`: four full prefill shapes, M edges around
    // the 64-row tile, and a K that is a multiple of 16 but not of 32.
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("gate/up   ", 1024, 17408, 5120),
        ("down      ", 1024, 5120, 17408),
        ("gate/up4k ", 4096, 17408, 5120),
        ("down4k    ", 4096, 5120, 17408),
        ("M=33  edge", 33, 5120, 5120),
        ("M=128 edge", 128, 5120, 5120),
        ("M=1015edge", 1015, 5120, 5120),
        ("K-tail    ", 256, 4096, 5104),
    ];

    println!(
        "=== w4a16_bf16 v2 losslessness microtest seed=0x{seed:X} ===\n\
         base = w4a16_gemm, v1 = w4a16_gemm_t_m128_bf16, v2 = w4a16_gemm_t_m128_bf16_v2\n\
         GATE: base-vs-v2 cosine >= {COSINE_GATE}  AND  v1-vs-v2 bit_id == 100%% (schedule-only change)\n"
    );
    println!(
        "{:<12} {:>6} {:>6} {:>6} | {:>10} {:>10} {:>9} | {:>10} {:>9}  result",
        "shape", "M", "N", "K", "b/v2 cos", "b/v2 abs", "b/v2 bit%", "v1/v2 cos", "v1/v2 bit%"
    );
    println!("{}", "-".repeat(108));

    let mut all_pass = true;
    for &(label, m, n, k) in shapes {
        let r = run_shape(
            gpu, stream, base_h, bf16_h, v2_h, crush1_h, crush2_h, seed, m, n, k,
        )?;
        let bv = &r.base_vs_v2;
        let v12 = &r.v1_vs_v2;
        let crush_ok = match &r.crush_v1_vs_v2 {
            Some(c) => c.frac_bit_identical >= 0.999_999,
            None => true,
        };
        let pass = bv.cosine >= COSINE_GATE
            && bv.cosine.is_finite()
            && v12.frac_bit_identical >= 0.999_999
            && crush_ok;
        all_pass &= pass;
        let crush_col = match &r.crush_v1_vs_v2 {
            Some(c) => format!("{:>8.3}%", c.frac_bit_identical * 100.0),
            None => "   skip".to_string(),
        };
        println!(
            "{label:<12} {m:>6} {n:>6} {k:>6} | {:>10.6} {:>10.3e} {:>8.3}% | {:>10.6} {:>8.3}% | crush {} {}",
            bv.cosine,
            bv.max_abs,
            bv.frac_bit_identical * 100.0,
            v12.cosine,
            v12.frac_bit_identical * 100.0,
            crush_col,
            if pass { "PASS" } else { "FAIL" },
        );
    }

    println!("{}", "-".repeat(108));
    if all_pass {
        println!(
            "RESULT: PASS — v2 is bit-identical to v1 (100%%) and numerically equivalent to base (cosine >= {COSINE_GATE})"
        );
        Ok(())
    } else {
        println!(
            "RESULT: FAIL — v2 diverged from v1 (must be 100%% bit-identical) or from base (cosine < {COSINE_GATE})"
        );
        std::process::exit(1);
    }
}
