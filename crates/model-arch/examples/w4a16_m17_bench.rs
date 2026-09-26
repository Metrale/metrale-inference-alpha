// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Wall-time bench of the W4A16 (NVFP4) tile GEMMs and batched GEMVs
//! over `SHAPES` x `M_SWEEP`, reported against a weight-bandwidth floor:
//! `(K*N/2 packed + K*N/16 scale) bytes / peak`.
//!
//! Each point is the mean of `ITERS` back-to-back launches on stream 0 after
//! `WARMUP` launches, timed between two syncs. Buffers are filled with 0x5A, so
//! this measures speed only. A kernel the target lacks is skipped.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//!   cargo run -p metrale-model-arch --release --example w4a16_m17_bench \
//!       --features cuda,gpu-examples
//!
//! Env: METRALE_PEAK_GBPS (default 273) is the peak used for the floor and the
//! %-of-peak column.

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::time::Instant;

const WARMUP: usize = 20;
const ITERS: usize = 100;

/// 2026-09-25: `(label, N, K)` with `C[M, N] = A[M, K] · W`.
const SHAPES: &[(&str, u32, u32)] = &[
    ("ffn_gate/up  N=17408 K=5120 ", 17408, 5120),
    ("ffn_down     N=5120  K=17408", 5120, 17408),
    ("ssm_qkvz     N=16384 K=5120 ", 16384, 5120),
    ("ssm_out_proj N=5120  K=6144 ", 5120, 6144),
    ("attn_qkv     N=8192  K=5120 ", 8192, 5120),
    ("attn_o_proj  N=5120  K=6144 ", 5120, 6144),
    ("attn_k       N=1024  K=5120 ", 1024, 5120),
    ("attn_v       N=1024  K=5120 ", 1024, 5120),
    ("attn_qkv_FUSED N=14336 K=5120", 14336, 5120),
    ("lm_head      N=248320 K=5120", 248320, 5120),
];

const M_SWEEP: &[u32] = &[1, 4, 8, 16, 32, 64];

/// 2026-09-25: Launch geometry: N and M tile (grid), and block 256 for the two
/// 256-thread cases, 128 otherwise.
#[derive(Clone, Copy)]
enum Geom {
    N64M64,
    N128M64,
    N128M128,
    N128M128W256,
    /// 2026-09-25: `w4a16_gemv_batch*`, which take the same arguments as the
    /// GEMMs (`A, B_packed, B_scale, scale2, C, M, N, K`).
    GemvN4,
}

fn grid_for(g: Geom, m: u32, n: u32) -> [u32; 3] {
    match g {
        Geom::N64M64 => [div_ceil(n, 64), div_ceil(m, 64), 1],
        Geom::N128M64 => [div_ceil(n, 128), div_ceil(m, 64), 1],
        Geom::N128M128 => [div_ceil(n, 128), div_ceil(m, 128), 1],
        Geom::N128M128W256 => [div_ceil(n, 128), div_ceil(m, 128), 1],
        Geom::GemvN4 => [div_ceil(n, 4), 1, 1],
    }
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    k_h: KernelHandle,
    geom: Geom,
    a: DevicePtr,
    b: DevicePtr,
    b_scale: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    KernelLaunch::new(g, k_h)
        .grid(grid_for(geom, m, n))
        .block([
            if matches!(geom, Geom::N128M128W256 | Geom::GemvN4) {
                256
            } else {
                128
            },
            1,
            1,
        ])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(b_scale)
        .arg_f32(1.0)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        // 2026-09-25: `ldb`, the transposed-B row stride (`n` when packed), for the
        // kernels that declare a ninth argument; the others do not read it.
        .arg_u32(n)
        .launch(0)
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;

    let peak_gbps: f64 = std::env::var("METRALE_PEAK_GBPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(273.0);

    // 2026-09-25: `(display name, function, geometry)`; the module follows from
    // the function name.
    let kernels: Vec<(&str, KernelHandle, Geom)> = [
        ("w4a16_gemm       (N64,M64) ", "w4a16_gemm", Geom::N64M64),
        ("w4a16_gemm_t     (N128,M64)", "w4a16_gemm_t", Geom::N128M64),
        (
            "w4a16_gemm_t_k64 (N128,M64)",
            "w4a16_gemm_t_k64",
            Geom::N128M64,
        ),
        (
            "w4a16_gemm_t_m128(N128,M128)",
            "w4a16_gemm_t_m128",
            Geom::N128M128,
        ),
        (
            "w4a16_gemm_t_m128_v2(W256)",
            "w4a16_gemm_t_m128_v2",
            Geom::N128M128W256,
        ),
        (
            "w4a16_gemm_t_m64_bf16 (N128,M64)",
            "w4a16_gemm_t_m64_bf16",
            Geom::N128M64,
        ),
        (
            "w4a16_gemm_t_m128_bf16_v2",
            "w4a16_gemm_t_m128_bf16_v2",
            Geom::N128M128,
        ),
        ("w4a16_gemv_batch4", "w4a16_gemv_batch4", Geom::GemvN4),
        ("w4a16_gemv_batch8", "w4a16_gemv_batch8", Geom::GemvN4),
        ("w4a16_gemv_batch16", "w4a16_gemv_batch16", Geom::GemvN4),
    ]
    .into_iter()
    .filter_map(|(name, func, geom)| {
        match g.kernel(
            if func == "w4a16_gemm_t_m128_v2" {
                "w4a16_v2"
            } else if func.starts_with("w4a16_gemv") {
                "w4a16_gemv"
            } else {
                "w4a16"
            },
            func,
        ) {
            Ok(h) => Some((name, h, geom)),
            Err(_) => {
                eprintln!("SKIP {name}: kernel w4a16::{func} not in this target's module set");
                None
            }
        }
    })
    .collect();

    let m_max = *M_SWEEP.iter().max().unwrap() as usize;

    eprintln!(
        "W4A16 M=17 speed-of-light bench  (peak {peak_gbps:.0} GB/s, \
         {ITERS} iters, floor = packed(K*N/2) + scales(K*N/16))\n"
    );

    for &(label, n, k) in SHAPES {
        let (n_us, k_us) = (n as usize, k as usize);
        let a = g.alloc(m_max * k_us * 2)?;
        let b = g.alloc(k_us * n_us / 2)?;
        let b_scale = g.alloc(k_us * n_us / 16)?;
        let c = g.alloc(m_max * n_us * 2)?;
        for (p, bytes) in [
            (a, m_max * k_us * 2),
            (b, k_us * n_us / 2),
            (b_scale, k_us * n_us / 16),
        ] {
            g.memset(p, 0x5A, bytes)?;
        }

        let weight_bytes = (k_us * n_us / 2 + k_us * n_us / 16) as f64;
        let floor_us = weight_bytes / (peak_gbps * 1e9) * 1e6;
        eprintln!(
            "── {label}  weights {:.1} MB, floor {floor_us:.0} us @ {peak_gbps:.0} GB/s ──",
            weight_bytes / 1e6
        );

        for &(kname, kh, geom) in &kernels {
            for &m in M_SWEEP {
                for _ in 0..WARMUP {
                    launch(g, kh, geom, a, b, b_scale, c, m, n, k)?;
                }
                g.synchronize(0)?;
                let t0 = Instant::now();
                for _ in 0..ITERS {
                    launch(g, kh, geom, a, b, b_scale, c, m, n, k)?;
                }
                g.synchronize(0)?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
                let gbps = weight_bytes / (us * 1e-6) / 1e9;
                eprintln!(
                    "  {kname}  M={m:>3}  {us:>8.1} us  {gbps:>6.1} GB/s  \
                     {:>5.1}% of peak  {:>4.2}x floor",
                    100.0 * gbps / peak_gbps,
                    us / floor_us,
                );
            }
            eprintln!();
        }

        for p in [a, b, b_scale, c] {
            let _ = g.free(p);
        }
    }

    eprintln!(
        "context: verify has 64 layers x (gate,up @ N=17408) + 64 x (down @ N=5120K17408).\n\
         per-step FFN cost = 128 x t(gate/up shape, M=17) + 64 x t(down shape, M=17).\n\
         profile said ~100ms via w4a16_gemm_t_m128 — compare against that here."
    );
    Ok(())
}
