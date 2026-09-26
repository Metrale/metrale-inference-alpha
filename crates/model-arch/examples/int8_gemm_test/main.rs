// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Correctness and speed of the int8 GEMM family in the `w4a16` module
//! (`int8_gemm_*`, the `requant_*` kernels, split-K).
//!
//! Owner: model-arch examples (int8 GEMM kernels).
//! Invariants: none beyond the types. Each correctness arm prints a cosine against a host
//! reference and PASS above 0.999; a FAIL is printed, not fatal.
//!
//! The kernels are only in `kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu`, so the build's first
//! target, which `ptx_modules()` loads, must be the qwen3.6-27b one. The speed section times two
//! prefill shapes and prints TFLOP/s.

mod checks_faith;
mod checks_ldm;
mod requant_e2e;
mod speed;

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use checks_faith::{check_faith, check_faith2, check_faith3, check_faith4, check_mmq2};
use checks_ldm::{check_8w_ldm, check_8w_ldmab, check_8w_pipe, check_mmq, check_pada};
use requant_e2e::requant_e2e;
use speed::speed_shape;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn i8(&mut self) -> i8 {
        (self.next_u64() % 255) as i8 - 127
    }
    fn pos_scale(&mut self) -> f32 {
        0.001 + (self.next_u64() % 1000) as f32 * 0.0005
    }
}

fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn run(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    asc: DevicePtr,
    bsc: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(gpu, h)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(asc)
        .arg_ptr(bsc)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let h = gpu.kernel("w4a16", "int8_gemm_t_m128")?;
    let h64 = gpu.kernel("w4a16", "int8_gemm_t_m64")?;
    let hk64 = gpu.kernel("w4a16", "int8_gemm_t_m128_k64")?;
    let h8w = gpu.kernel("w4a16", "int8_gemm_8w")?;
    let h8w3 = gpu.kernel("w4a16", "int8_gemm_8w3")?;
    let h8wl = gpu.kernel("w4a16", "int8_gemm_8w_ldm")?;
    let h8wi = gpu.kernel("w4a16", "int8_gemm_8w_ilp")?;
    let h8wab = gpu.kernel("w4a16", "int8_gemm_8w_ldmab")?;
    let hpipe = gpu.kernel("w4a16", "int8_gemm_8w_pipe")?;
    let hpada = gpu.kernel("w4a16", "int8_gemm_padA")?;
    let hfaith = gpu.kernel("w4a16", "int8_gemm_faith")?;
    let hfaith2 = gpu.kernel("w4a16", "int8_gemm_faith2")?;
    let hfaith3 = gpu.kernel("w4a16", "int8_gemm_faith3")?;
    let hfaith4 = gpu.kernel("w4a16", "int8_gemm_faith4")?;
    let hfaith5 = gpu.kernel("w4a16", "int8_gemm_faith5")?;
    let hfaith6 = gpu.kernel("w4a16", "int8_gemm_faith6")?;
    let hfaith7 = gpu.kernel("w4a16", "int8_gemm_faith7")?;
    let hfaith8 = gpu.kernel("w4a16", "int8_gemm_faith8")?;
    let hfaith9 = gpu.kernel("w4a16", "int8_gemm_faith9")?;
    let hfaith10 = gpu.kernel("w4a16", "int8_gemm_faith10")?;
    let hmmqf = gpu.kernel("w4a16", "int8_gemm_mmqf")?;
    let hmmqf2 = gpu.kernel("w4a16", "int8_gemm_mmqf2")?;
    let hmmqf3 = gpu.kernel("w4a16", "int8_gemm_mmqf3")?;
    let hreqa_il = gpu.kernel("w4a16", "requant_a_bf16_int8_il")?;
    let hmmq = gpu.kernel("w4a16", "int8_gemm_mmq")?;
    let hmmq2 = gpu.kernel("w4a16", "int8_gemm_mmq2")?;
    let hreqw = gpu.kernel("w4a16", "requant_w_nvfp4_int8")?;
    let hreqa = gpu.kernel("w4a16", "requant_a_bf16_int8")?;
    let hsk = gpu.kernel("w4a16", "int8_gemm_splitk")?;
    let hred = gpu.kernel("w4a16", "int8_splitk_reduce")?;

    let (m, n, k) = (128usize, 256usize, 512usize);
    let nb = k / 32;
    let mut rng = Rng(0xC0FFEE);
    let a_i8: Vec<i8> = (0..m * k).map(|_| rng.i8()).collect();
    let b_i8: Vec<i8> = (0..n * k).map(|_| rng.i8()).collect();
    let a_sc: Vec<f32> = (0..m * nb).map(|_| rng.pos_scale()).collect();
    let b_sc: Vec<f32> = (0..n * nb).map(|_| rng.pos_scale()).collect();

    // 2026-09-25: Host reference: C[m,n] = sum over 32-wide blocks of
    // (integer sum of A[m,k] * B[n,k]) * As[m,blk] * Bs[n,blk].
    let mut c_ref = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for blk in 0..nb {
                let mut s = 0i32;
                for kk in 0..32 {
                    let ki = blk * 32 + kk;
                    s += a_i8[mi * k + ki] as i32 * b_i8[ni * k + ki] as i32;
                }
                acc += s as f32 * a_sc[mi * nb + blk] * b_sc[ni * nb + blk];
            }
            c_ref[mi * n + ni] = acc;
        }
    }

    let a_p = up(gpu, bytemuck_i8(&a_i8))?;
    let b_p = up(gpu, bytemuck_i8(&b_i8))?;
    let as_p = up(gpu, bytemuck_f32(&a_sc))?;
    let bs_p = up(gpu, bytemuck_f32(&b_sc))?;
    let c_p = gpu.alloc(m * n * 2)?;
    run(gpu, stream, h, a_p, b_p, as_p, bs_p, c_p, m, n, k)?;
    gpu.synchronize(stream)?;
    let mut raw = vec![0u8; m * n * 2];
    gpu.copy_d2h(c_p, &mut raw)?;
    let c_gpu: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    let (mut dot, mut na, mut nbb) = (0f64, 0f64, 0f64);
    let mut maxrel = 0f64;
    for i in 0..m * n {
        let (x, y) = (c_ref[i] as f64, c_gpu[i] as f64);
        dot += x * y;
        na += x * x;
        nbb += y * y;
        if x.abs() > 1.0 {
            maxrel = maxrel.max(((x - y).abs() / x.abs()).min(9.9));
        }
    }
    let cos = dot / (na.sqrt() * nbb.sqrt());
    println!("int8_gemm correctness {m}x{n}x{k}: cosine={cos:.6}  max_rel={maxrel:.4}");
    println!("  ref[0..4]={:?}  gpu[0..4]={:?}", &c_ref[..4], &c_gpu[..4]);
    let pass = cos > 0.999;
    println!("  RESULT: {}", if pass { "PASS" } else { "FAIL" });

    check_8w_ldm(gpu, stream, h8wl, a_p, b_p, as_p, bs_p, &c_ref, m, n, k)?;
    check_mmq(gpu, stream, hmmq, a_p, b_p, as_p, bs_p, &c_ref, m, n, k)?;
    check_8w_ldmab(gpu, stream, h8wab, a_p, b_p, as_p, bs_p, &c_ref, m, n, k)?;
    check_8w_pipe(gpu, stream, hpipe, a_p, b_p, as_p, bs_p, &c_ref, m, n, k)?;
    check_pada(gpu, stream, hpada, a_p, b_p, as_p, bs_p, &c_ref, m, n, k)?;
    check_faith(gpu, stream, hfaith, a_p, b_p, as_p, bs_p, &c_ref, m, n, k)?;
    check_faith2(gpu, stream, hfaith2, a_p, b_p, as_p, bs_p, &c_ref, m, n, k)?;
    check_faith3(gpu, stream, hfaith3, a_p, b_p, as_p, bs_p, &c_ref, m, n, k)?;
    check_faith4(gpu, stream, hfaith4, a_p, b_p, as_p, bs_p, &c_ref, m, n, k)?;
    check_mmq2(gpu, stream, hmmq2, a_p, b_p, as_p, bs_p, &c_ref, m, n, k)?;

    // 2026-09-25: End to end: NVFP4 weights through `requant_w_nvfp4_int8` (per-16 E4M3 scales
    // re-blocked to per-32 int8 scales) and BF16 activations through `requant_a_bf16_int8`,
    // then the int8 GEMMs, against a host GEMM on the dequantised weights.
    requant_e2e(
        gpu, stream, hreqw, hreqa, hfaith2, hreqa_il, hfaith5, hfaith6, hfaith7, hfaith8, hfaith9,
        hfaith10, hmmqf, hmmqf3,
    )?;

    println!("\n=== int8 speed (TFLOP/s) ===");
    for &(label, m, n, k) in &[
        ("gate/up M=4096", 4096usize, 17408usize, 5120usize),
        ("down    M=4096", 4096, 5120, 17408),
    ] {
        speed_shape(
            gpu, stream, h, h64, hk64, h8w, h8w3, h8wl, h8wi, hmmq, h8wab, hpipe, hpada, hfaith,
            hfaith2, hfaith3, hfaith4, hfaith5, hfaith6, hfaith7, hfaith8, hfaith9, hfaith10,
            hmmqf, hmmqf2, hmmqf3, hmmq2, hsk, hred, label, m, n, k,
        )?;
    }
    Ok(())
}

fn bytemuck_i8(v: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}
fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
