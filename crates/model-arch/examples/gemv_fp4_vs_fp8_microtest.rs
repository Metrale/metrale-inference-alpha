// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Timing of the M=1 GEMVs `w8a16_gemv` (FP8 weight, 1 byte per element) and
//! `w4a16_gemv` (NVFP4 weight, half a byte) at one [N, K] shape.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types. Buffers are allocated at the right sizes but not
//! initialised, and no output is checked.
//!
//! Each kernel runs 10 warm-up and 200 timed launches between CUDA events on the launch
//! stream. GB/s counts each kernel's weight bytes.
//!
//! Usage: cargo run --release -p metrale-model-arch --example gemv_fp4_vs_fp8_microtest -- [N] [K]
//! N and K default to 12288 and 2048.

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::kernel_args::KernelLaunch;

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// 2026-09-25: Time `iters` back-to-back launches of `f` on `stream` between CUDA events,
/// after 10 warm-up launches, with no host sync between launches. Returns mean ms per
/// launch.
fn time_kernel(
    gpu: &dyn GpuBackend,
    stream: u64,
    iters: u32,
    mut f: impl FnMut() -> Result<()>,
) -> Result<f64> {
    for _ in 0..10 {
        f()?;
    }
    gpu.synchronize(stream)?;
    let (mut ev0, mut ev1): (u64, u64) = (0, 0);
    unsafe {
        if cuEventCreate(&mut ev0, 0) != 0 || cuEventCreate(&mut ev1, 0) != 0 {
            bail!("cuEventCreate failed");
        }
        if cuEventRecord(ev0, stream) != 0 {
            bail!("cuEventRecord(start) failed");
        }
    }
    for _ in 0..iters {
        f()?;
    }
    let mut ms: f32 = 0.0;
    unsafe {
        if cuEventRecord(ev1, stream) != 0 {
            bail!("cuEventRecord(end) failed");
        }
        if cuEventSynchronize(ev1) != 0 {
            bail!("cuEventSynchronize failed");
        }
        if cuEventElapsedTime(&mut ms, ev0, ev1) != 0 {
            bail!("cuEventElapsedTime failed");
        }
        cuEventDestroy_v2(ev0);
        cuEventDestroy_v2(ev1);
    }
    Ok(ms as f64 / iters as f64)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n: u32 = args.get(1).map_or(12288, |s| s.parse().unwrap());
    let k: u32 = args.get(2).map_or(2048, |s| s.parse().unwrap());
    let iters = 200u32;

    println!("=== decode GEMV A/B: N={n} K={k} (M=1)  iters={iters} ===");

    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let input = gpu.alloc((k as usize) * 2)?;
    let output = gpu.alloc((n as usize) * 2)?;

    // 2026-09-25: Each launch reads the next buffer of a ring of about 768 MB of distinct
    // weight buffers per kernel, so the weight read is not served from L2.
    let fp8_w_bytes = (n as usize) * (k as usize);
    let fp4_w_bytes = (n as usize) * (k as usize) / 2;
    let ring_fp8 = (768 * 1024 * 1024 / fp8_w_bytes).max(2);
    let ring_fp4 = (768 * 1024 * 1024 / fp4_w_bytes).max(2);
    let w_fp8: Vec<_> = (0..ring_fp8)
        .map(|_| gpu.alloc(fp8_w_bytes))
        .collect::<Result<_>>()?;
    let s_fp8 = gpu.alloc((n as usize) * (k as usize / 128 + 1) * 4)?;
    let w_fp4: Vec<_> = (0..ring_fp4)
        .map(|_| gpu.alloc(fp4_w_bytes))
        .collect::<Result<_>>()?;
    let s_fp4 = gpu.alloc((n as usize) * (k as usize / 16 + 1))?;
    println!(
        "DRAM-cold: FP8 ring={ring_fp8} bufs ({} MB), FP4 ring={ring_fp4} bufs ({} MB)",
        ring_fp8 * fp8_w_bytes / (1 << 20),
        ring_fp4 * fp4_w_bytes / (1 << 20)
    );

    let k_w8 = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    let k_w4 = gpu.kernel("w4a16_gemv", "w4a16_gemv")?;

    let mut i8 = 0usize;
    let fp8_ms = time_kernel(gpu, stream, iters, || {
        let w = w_fp8[i8 % ring_fp8];
        i8 += 1;
        KernelLaunch::new(gpu, k_w8)
            .grid([div_ceil(n, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(w)
            .arg_ptr(s_fp8)
            .arg_ptr(output)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)
    })?;

    let mut i4 = 0usize;
    let fp4_ms = time_kernel(gpu, stream, iters, || {
        let w = w_fp4[i4 % ring_fp4];
        i4 += 1;
        KernelLaunch::new(gpu, k_w4)
            .grid([div_ceil(n, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(w)
            .arg_ptr(s_fp4)
            .arg_f32(1.0)
            .arg_ptr(output)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)
    })?;

    let nk = n as f64 * k as f64;
    let fp8_gbs = (nk) / (fp8_ms / 1e3) / 1e9;
    let fp4_gbs = (nk / 2.0) / (fp4_ms / 1e3) / 1e9;
    println!(
        "FP8 w8a16_gemv : {:.4} ms/iter   {:.0} GB/s (weight {:.1} MB)",
        fp8_ms,
        fp8_gbs,
        nk / 1e6
    );
    println!(
        "FP4 w4a16_gemv : {:.4} ms/iter   {:.0} GB/s (weight {:.1} MB)",
        fp4_ms,
        fp4_gbs,
        nk / 2.0 / 1e6
    );
    println!(
        "SPEEDUP (FP8/FP4): {:.2}x  {}",
        fp8_ms / fp4_ms,
        if fp4_ms < fp8_ms {
            "→ FP4 decode is FASTER on this stack"
        } else {
            "→ FP8 decode still wins"
        }
    );

    for p in [input, output, s_fp8, s_fp4] {
        gpu.free(p).ok();
    }
    for p in w_fp8.into_iter().chain(w_fp4) {
        gpu.free(p).ok();
    }
    Ok(())
}
