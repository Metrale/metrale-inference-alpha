// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Wall-time benchmark of prefill GEMM kernels on four `(M, N, K)`
//! shapes: the `w4a16` module's m128/m64/k64 NVFP4 and FP8 kernels, the
//! `w4a16_v2` module's `w4a16_gemm_t_m128_v2` when present, and the dense BF16
//! `dense_gemm_bf16_pipelined` and `dense_gemm_tc`.
//!
//! Each kernel runs 3 warm-up launches, then `iters` back-to-back launches
//! between two stream syncs; the time is their mean. Weights and activations
//! are random bytes: this measures speed, not correctness.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Needs a target whose `w4a16` module has all of these kernels (the gb10
//! qwen3.6-27b kernels). Usage:
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example w4a16_bf16_v2_bench

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

const GROUP_SIZE: usize = 16;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

#[allow(clippy::too_many_arguments)]
fn time_kernel(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    a: DevicePtr,
    packed: DevicePtr,
    scale: DevicePtr,
    scale2: f32,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    // 2026-09-25: The ninth `ldb` argument, for kernels that declare it
    // (`w4a16_gemm_t_m128_bf16_v2`); it must match the kernel's signature.
    ldb: Option<u32>,
    block_x: u32,
    iters: usize,
) -> Result<f64> {
    let launch_once = |c: DevicePtr| -> Result<()> {
        let mut l = KernelLaunch::new(gpu, h)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([block_x, 1, 1])
            .arg_ptr(a)
            .arg_ptr(packed)
            .arg_ptr(scale)
            .arg_f32(scale2)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32);
        if let Some(ldb) = ldb {
            l = l.arg_u32(ldb);
        }
        l.launch(stream)
    };
    for _ in 0..3 {
        launch_once(c)?;
    }
    gpu.synchronize(stream)?;

    let t0 = Instant::now();
    for _ in 0..iters {
        launch_once(c)?;
    }
    gpu.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

// 2026-09-25: Time a kernel with the argument list `(A, B_fp8, C, M, N, K)`:
// `fp8_gemm_t_m128` and `fp8_fp8_gemm_t_m128`.
fn time_kernel_fp8(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    a: DevicePtr,
    b_fp8: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
) -> Result<f64> {
    let launch = |h: KernelHandle| -> Result<()> {
        KernelLaunch::new(gpu, h)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b_fp8)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launch(h)?;
    }
    gpu.synchronize(stream)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        launch(h)?;
    }
    gpu.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

// 2026-09-25: `time_kernel` for the 64-row-tile kernels: grid.y is ceil(M/64).
fn time_kernel_m64(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    a: DevicePtr,
    packed: DevicePtr,
    scale: DevicePtr,
    scale2: f32,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
) -> Result<f64> {
    let launch = || -> Result<()> {
        KernelLaunch::new(gpu, h)
            .grid([n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(a)
            .arg_ptr(packed)
            .arg_ptr(scale)
            .arg_f32(scale2)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launch()?;
    }
    gpu.synchronize(stream)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        launch()?;
    }
    gpu.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let v1 = gpu.kernel("w4a16", "w4a16_gemm_t_m128_bf16")?;
    let v2 = gpu.kernel("w4a16", "w4a16_gemm_t_m128_bf16_v2")?;
    let fp8 = gpu.kernel("w4a16", "fp8_gemm_t_m128")?;
    // 2026-09-25: `w4a16_gemm_t_m128_v2` is optional; it runs with 256 threads.
    let crush1 = gpu.kernel("w4a16", "w4a16_gemm_t_m128")?;
    let crush2 = gpu
        .kernel("w4a16_v2", "w4a16_gemm_t_m128_v2")
        .unwrap_or(KernelHandle(0));
    let fp8fp8 = gpu.kernel("w4a16", "fp8_fp8_gemm_t_m128")?;
    let gt_k32 = gpu.kernel("w4a16", "w4a16_gemm_t_m64_bf16")?;
    let gt_k64 = gpu.kernel("w4a16", "w4a16_gemm_t_k64")?;
    let dbf16 = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;
    let dtc = gpu.kernel("gemm_tc", "dense_gemm_tc")?;

    let shapes: &[(&str, usize, usize, usize)] = &[
        ("gate/up M=1024", 1024, 17408, 5120),
        ("down    M=1024", 1024, 5120, 17408),
        ("gate/up M=4096", 4096, 17408, 5120),
        ("down    M=4096", 4096, 5120, 17408),
    ];

    println!("=== w4a16 BF16 prefill GEMM bench (v1 vs v2) ===\n");
    println!(
        "{:<16} {:>6} {:>6} {:>6} | {:>10} {:>9} | {:>10} {:>9} | {:>7}",
        "shape", "M", "N", "K", "v1 ms", "v1 TFLOP", "v2 ms", "v2 TFLOP", "speedup"
    );
    println!("{}", "-".repeat(96));

    let mut rng = Rng(0x1234);
    for &(label, m, n, k) in shapes {
        let half_k = k / 2;
        let num_groups = k / GROUP_SIZE;
        // 2026-09-25: Transposed layout: packed `[K/2, N]`, scale `[K/16, N]`.
        let mut packed = vec![0u8; half_k * n];
        let mut scale = vec![0u8; num_groups * n];
        for b in packed.iter_mut() {
            *b = rng.next_u64() as u8;
        }
        for s in scale.iter_mut() {
            // 2026-09-25: E4M3 with exponent field 5..=9 and mantissa 0: 0.25 to 4.
            *s = (((5 + (rng.next_u64() % 5)) as u8) << 3) & 0x7F;
        }
        let a: Vec<u8> = (0..m * k * 2).map(|_| rng.next_u64() as u8).collect();

        let a_ptr = upload(gpu, &a)?;
        let p_ptr = upload(gpu, &packed)?;
        let s_ptr = upload(gpu, &scale)?;
        let c1 = gpu.alloc(m * n * 2)?;
        let c2 = gpu.alloc(m * n * 2)?;

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let iters = if m >= 4096 { 30 } else { 60 };

        let b_fp8: Vec<u8> = (0..n * k).map(|_| rng.next_u64() as u8).collect();
        let bf8_ptr = upload(gpu, &b_fp8)?;
        let c3 = gpu.alloc(m * n * 2)?;

        let t1 = time_kernel(
            gpu, stream, v1, a_ptr, p_ptr, s_ptr, 0.5, c1, m, n, k, None, 128, iters,
        )?;
        let t2 = time_kernel(
            gpu,
            stream,
            v2,
            a_ptr,
            p_ptr,
            s_ptr,
            0.5,
            c2,
            m,
            n,
            k,
            Some(n as u32),
            128,
            iters,
        )?;
        let t3 = time_kernel_fp8(gpu, stream, fp8, a_ptr, bf8_ptr, c3, m, n, k, iters)?;
        let tc1 = time_kernel(
            gpu, stream, crush1, a_ptr, p_ptr, s_ptr, 0.5, c1, m, n, k, None, 128, iters,
        )?;
        let tc2 = if crush2.0 != 0 {
            Some(time_kernel(
                gpu, stream, crush2, a_ptr, p_ptr, s_ptr, 0.5, c2, m, n, k, None, 256, iters,
            )?)
        } else {
            None
        };
        match tc2 {
            Some(tc2) => println!(
                "{label:<16} CRUSH v1 {:>7.3} ms {:>6.1} TF | v2 {:>7.3} ms {:>6.1} TF | {:>5.3}x",
                tc1 * 1e3,
                flops / tc1 / 1e12,
                tc2 * 1e3,
                flops / tc2 / 1e12,
                tc1 / tc2,
            ),
            None => println!(
                "{label:<16} CRUSH v1 {:>7.3} ms {:>6.1} TF | v2 ABSENT",
                tc1 * 1e3,
                flops / tc1 / 1e12,
            ),
        }
        let a_fp8: Vec<u8> = (0..m * k).map(|_| rng.next_u64() as u8).collect();
        let af8_ptr = upload(gpu, &a_fp8)?;
        let c4 = gpu.alloc(m * n * 2)?;
        let t4 = time_kernel_fp8(gpu, stream, fp8fp8, af8_ptr, bf8_ptr, c4, m, n, k, iters)?;

        let c5 = gpu.alloc(m * n * 2)?;
        let c6 = gpu.alloc(m * n * 2)?;
        let t5 = time_kernel_m64(
            gpu, stream, gt_k32, a_ptr, p_ptr, s_ptr, 0.5, c5, m, n, k, iters,
        )?;
        let t6 = time_kernel_m64(
            gpu, stream, gt_k64, a_ptr, p_ptr, s_ptr, 0.5, c6, m, n, k, iters,
        )?;

        let b_bf16: Vec<u8> = (0..n * k * 2).map(|_| rng.next_u64() as u8).collect();
        let bbf_ptr = upload(gpu, &b_bf16)?;
        let c7 = gpu.alloc(m * n * 2)?;
        let launch_dense = |it: usize| -> Result<f64> {
            let go = || -> Result<()> {
                KernelLaunch::new(gpu, dbf16)
                    .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(a_ptr)
                    .arg_ptr(bbf_ptr)
                    .arg_ptr(c7)
                    .arg_u32(m as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .launch(stream)
            };
            for _ in 0..3 {
                go()?;
            }
            gpu.synchronize(stream)?;
            let t0 = Instant::now();
            for _ in 0..it {
                go()?;
            }
            gpu.synchronize(stream)?;
            Ok(t0.elapsed().as_secs_f64() / it as f64)
        };
        let t7 = launch_dense(iters)?;
        let c8 = gpu.alloc(m * n * 2)?;
        let launch_tc = |it: usize| -> Result<f64> {
            let go = || -> Result<()> {
                KernelLaunch::new(gpu, dtc)
                    .grid([n.div_ceil(64) as u32, m.div_ceil(16) as u32, 1])
                    .block([128, 1, 1])
                    .arg_ptr(a_ptr)
                    .arg_ptr(bbf_ptr)
                    .arg_ptr(c8)
                    .arg_u32(m as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .launch(stream)
            };
            for _ in 0..3 {
                go()?;
            }
            gpu.synchronize(stream)?;
            let t0 = Instant::now();
            for _ in 0..it {
                go()?;
            }
            gpu.synchronize(stream)?;
            Ok(t0.elapsed().as_secs_f64() / it as f64)
        };
        let t8 = launch_tc(iters)?;

        let tf2 = flops / t2 / 1e12;
        let tf7 = flops / t7 / 1e12;
        let tf8 = flops / t8 / 1e12;
        println!(
            "{label:<16} {m:>6} {n:>6} {k:>6} | bf16v2(deq) {:>6.2} | denseBF16 {:>6.2} | denseTC(ldmatrix) {:>6.2} | {:>5.3}x v2",
            tf2,
            tf7,
            tf8,
            t2 / t8,
        );
        let _ = (t1, t3, t4, t5, t6);
        for ptr in [c5, c6, bbf_ptr, c7, c8] {
            let _ = gpu.free(ptr);
        }

        for ptr in [a_ptr, p_ptr, s_ptr, c1, c2, bf8_ptr, c3, af8_ptr, c4] {
            let _ = gpu.free(ptr);
        }
    }
    println!("{}", "-".repeat(96));
    Ok(())
}
