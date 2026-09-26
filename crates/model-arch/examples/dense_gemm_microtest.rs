// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Correctness and kernel-only throughput microtest for the dense
//! BF16 GEMMs `dense_gemm_bf16`, `dense_gemm_bf16_pipelined` and `dense_gemm_tc`:
//! C[m, n] = Σ_k A[m, k] · B[n, k] with FP32 accumulation and a BF16 result,
//! against a CPU recompute.
//!
//! The GPU kernels accumulate in another order than the CPU loop, so the gate
//! is cosine similarity (`COSINE_GATE`), not byte equality.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Usage:
//!   cargo run --release -p metrale-model-arch --example dense_gemm_microtest \
//!       --features cuda,gpu-examples -- [kernel_name] [M] [N] [K] [seed]
//! Defaults: dense_gemm_bf16 1024 2048 4096 0x51A7
//!
//! Exits 1 when the cosine is below the gate or not finite.

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

// 2026-09-25: CUDA driver event API for kernel-only timing, declared as in
// `w8a16_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-09-25: Minimum cosine between the GPU and CPU outputs.
const COSINE_GATE: f64 = 0.999;

// 2026-09-25: splitmix64, seeded from the command line.
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

/// 2026-09-25: f32 to BF16 bits, round to nearest even; a NaN becomes a quiet NaN.
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

fn upload_bytes(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 2026-09-25: CPU reference: C[m, n] = bf16(Σ_k A[m, k] · B[n, k]) with FP32
/// accumulation in ascending k. A is [M, K] and B is [N, K], both row-major.
fn cpu_reference(a_bf16: &[u16], b_bf16: &[u16], m: usize, n: usize, k: usize) -> Vec<u16> {
    // 2026-09-25: Row blocks split across `available_parallelism` threads (8 when
    // unknown).
    let nthreads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(8);
    let mut out = vec![0u16; m * n];
    let rows_per = m.div_ceil(nthreads);
    std::thread::scope(|sc| {
        for (t, chunk) in out.chunks_mut(rows_per * n).enumerate() {
            let row0 = t * rows_per;
            sc.spawn(move || {
                let rows = chunk.len() / n;
                for rr in 0..rows {
                    let row = row0 + rr;
                    for col in 0..n {
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            let a = bf16_bits_to_f32(a_bf16[row * k + kk]);
                            let b = bf16_bits_to_f32(b_bf16[col * k + kk]);
                            acc += a * b;
                        }
                        chunk[rr * n + col] = f32_to_bf16_bits(acc);
                    }
                }
            });
        }
    });
    out
}

/// 2026-09-25: Launch geometry per kernel name; an unknown name is refused.
fn grid_block(name: &str, m: u32, n: u32) -> Result<([u32; 3], [u32; 3])> {
    Ok(match name {
        // 2026-09-25: 16 × 16 outputs per 16 × 16-thread block, as `ops::dense_gemm`
        // launches it.
        "dense_gemm_bf16" => ([n.div_ceil(16), m.div_ceil(16), 1], [16u32, 16, 1]),
        // 2026-09-25: 128 rows by `DM_N_TILE_SWEEP` columns (default 128) per 256-thread
        // block; set the variable to match a kernel built with `-DDM_N_TILE=`.
        "dense_gemm_bf16_pipelined" => {
            let n_tile: u32 = std::env::var("DM_N_TILE_SWEEP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(128);
            ([n.div_ceil(n_tile), m.div_ceil(128), 1], [256u32, 1, 1])
        }
        // 2026-09-25: 16 rows by 64 columns per 128-thread block, as `ops::dense_gemm_tc`
        // launches it.
        "dense_gemm_tc" => ([n.div_ceil(64), m.div_ceil(16), 1], [128u32, 1, 1]),
        other => bail!("no launch geometry registered for kernel '{other}' — add an arm"),
    })
}

/// 2026-09-25: PTX module the kernel is looked up in: `gemm` (dense_gemm_bf16.cu)
/// for the `dense_gemm_bf16` kernels, and `dense_gemm_tc` for `dense_gemm_tc`,
/// although every KERNEL.toml that builds dense_gemm_tc.cu names its module
/// `gemm_tc`.
fn module_for(name: &str) -> &'static str {
    match name {
        "dense_gemm_tc" => "dense_gemm_tc",
        _ => "gemm",
    }
}

fn launch(
    gpu: &dyn GpuBackend,
    name: &str,
    ptrs: [DevicePtr; 3],
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    sync: bool,
) -> Result<()> {
    let [a, b, c] = ptrs;
    let handle = gpu.kernel(module_for(name), name)?;
    let (grid, block) = grid_block(name, m, n)?;
    KernelLaunch::new(gpu, handle)
        .grid(grid)
        .block(block)
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    if sync {
        gpu.synchronize(stream)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let kernel = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "dense_gemm_bf16".to_string());
    let m: usize = args.get(2).map_or(1024, |s| s.parse().unwrap());
    let n: usize = args.get(3).map_or(2048, |s| s.parse().unwrap());
    let k: usize = args.get(4).map_or(4096, |s| s.parse().unwrap());
    let seed: u64 = args.get(5).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });

    println!("=== dense_gemm microtest: kernel='{kernel}' M={m} N={n} K={k} seed=0x{seed:X} ===");

    let mut rng = Rng(seed);
    let a_bf16: Vec<u16> = (0..m * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let b_bf16: Vec<u16> = (0..n * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();

    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
    let b_ptr = upload_bytes(gpu, &u16s_to_le(&b_bf16))?;
    let c_ptr = gpu.alloc(m * n * 2)?;
    let ptrs = [a_ptr, b_ptr, c_ptr];

    launch(
        gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream, true,
    )?;
    let mut c_raw = vec![0u8; m * n * 2];
    gpu.copy_d2h(c_ptr, &mut c_raw)?;
    let c_gpu: Vec<u16> = c_raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let c_cpu = cpu_reference(&a_bf16, &b_bf16, m, n, k);

    let (mut dot, mut ng, mut nc, mut max_rel, mut sum_rel) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..m * n {
        let g = bf16_bits_to_f32(c_gpu[i]) as f64;
        let c = bf16_bits_to_f32(c_cpu[i]) as f64;
        dot += g * c;
        ng += g * g;
        nc += c * c;
        let rel = (g - c).abs() / c.abs().max(1e-3);
        max_rel = max_rel.max(rel);
        sum_rel += rel;
    }
    let cosine = dot / (ng.sqrt() * nc.sqrt());
    let mean_rel = sum_rel / (m * n) as f64;

    // 2026-09-25: Wall-clock timing, with a synchronize after every launch.
    let iters = 50;
    for _ in 0..5 {
        launch(
            gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream, true,
        )?;
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        launch(
            gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream, true,
        )?;
    }
    let per_iter_s = t0.elapsed().as_secs_f64() / iters as f64;
    let tflops = (2.0 * m as f64 * n as f64 * k as f64) / per_iter_s / 1e12;

    // 2026-09-25: Kernel-only timing: CUDA events around 50 launches without a
    // synchronize between them.
    let (mut ev_start, mut ev_end): (u64, u64) = (0, 0);
    let rc = unsafe { cuEventCreate(&mut ev_start, 0) };
    if rc != 0 {
        bail!("cuEventCreate(start) failed: status {rc}");
    }
    let rc = unsafe { cuEventCreate(&mut ev_end, 0) };
    if rc != 0 {
        bail!("cuEventCreate(end) failed: status {rc}");
    }
    let rc = unsafe { cuEventRecord(ev_start, stream) };
    if rc != 0 {
        bail!("cuEventRecord(start) failed: status {rc}");
    }
    for _ in 0..iters {
        launch(
            gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream, false,
        )?;
    }
    let rc = unsafe { cuEventRecord(ev_end, stream) };
    if rc != 0 {
        bail!("cuEventRecord(end) failed: status {rc}");
    }
    let rc = unsafe { cuEventSynchronize(ev_end) };
    if rc != 0 {
        bail!("cuEventSynchronize(end) failed: status {rc}");
    }
    let mut elapsed_ms: f32 = 0.0;
    let rc = unsafe { cuEventElapsedTime(&mut elapsed_ms, ev_start, ev_end) };
    if rc != 0 {
        bail!("cuEventElapsedTime failed: status {rc}");
    }
    unsafe {
        cuEventDestroy_v2(ev_start);
        cuEventDestroy_v2(ev_end);
    }
    let kernel_s = (elapsed_ms as f64 / 1e3) / iters as f64;
    let kernel_tflops = (2.0 * m as f64 * n as f64 * k as f64) / kernel_s / 1e12;

    for p in ptrs {
        gpu.free(p).ok();
    }

    println!("cosine={cosine:.6}  mean_rel={mean_rel:.2e}  max_rel={max_rel:.2e}");
    println!(
        "perf: {:.3} ms/iter  ~{tflops:.2} TFLOP/s (wall-clock incl. launch)",
        per_iter_s * 1e3
    );
    println!(
        "kernel-only: {:.4} ms/iter  ~{kernel_tflops:.2} TFLOP/s (CUDA events)",
        kernel_s * 1e3
    );

    if cosine >= COSINE_GATE && cosine.is_finite() {
        println!("RESULT: PASS (cosine {cosine:.6} >= {COSINE_GATE})");
        Ok(())
    } else {
        eprintln!(
            "RESULT: FAIL (cosine {cosine:.6} < {COSINE_GATE}) — layout/accumulation mismatch"
        );
        std::process::exit(1);
    }
}
