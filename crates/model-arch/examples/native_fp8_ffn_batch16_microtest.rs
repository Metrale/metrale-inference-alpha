// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU oracle for the 5..=32-row `w8a16_gemv_batch16` dense-FFN
//! decode tier (opt-in, `METRALE_FFN_BATCH16=1`).
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if, for both shapes at every M in {5, 8, 16, 32},
//!   the batch16 route's first M output rows equal the per-row scalar
//!   `w8a16_gemv` bytes (`unequal_bf16=0 max_abs=0`) and every other byte of
//!   its output buffer, guard bands included, still holds the sentinel.
//!
//! Shapes: gate/up `[17408, 5120]` and down `[5120, 17408]`, from `hidden_dim`
//! and `intermediate_size` in `kernels/hopper/qwen3.8-27b/MODEL.toml`.
//! `batch16_route` launches as `DenseFfnLayer::w8a16_batch16_proj` does.
//!
//! The tier is also timed against `w8a16_gemm_pipelined` at the same M.
//! GB/s counts the FP8 weight bytes once per kernel pass, so the 17..=32 rung,
//! which makes two passes, counts them twice. Times are `synchronize` plus a
//! host `Instant` over `REPS` after `WARMUP` runs: `GpuBackend` can record,
//! wait on and query events but has no elapsed-time query.
//!
//! Run:
//!     cargo run --release --example native_fp8_ffn_batch16_microtest \
//!       --features cuda,gpu-examples
use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use std::time::Instant;

/// 2026-09-25: Qwen3.8-27B dense FFN, `hidden_dim` 5120 and
/// `intermediate_size` 17408 (`kernels/hopper/qwen3.8-27b/MODEL.toml`).
const H: usize = 5120;
const INTER: usize = 17408;
const MAX_M: usize = 32;
const GUARD: usize = 64;
const REPS: u32 = 20;
const WARMUP: u32 = 3;

struct Shape {
    name: &'static str,
    n: usize,
    k: usize,
}

/// 2026-09-25: The two orientations the FFN runs. Gate and up share a shape,
/// so one entry covers both; down has the deep K.
const SHAPES: [Shape; 2] = [
    Shape {
        name: "gate/up",
        n: INTER,
        k: H,
    },
    Shape {
        name: "down",
        n: H,
        k: INTER,
    },
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// 2026-09-25: The batch16 route as `DenseFfnLayer::w8a16_batch16_proj` runs
/// it: one launch at m <= 16, two on contiguous row halves at 17..=32 (the
/// first half `m.div_ceil(2)` rows). Returns the number of launches.
#[allow(clippy::too_many_arguments)]
fn batch16_route(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    scale: DevicePtr,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<usize> {
    let launch = |rows: usize, first: usize| {
        ops::w8a16_gemv_batch16(
            gpu,
            kernel,
            input.offset(first * k * 2),
            weight,
            scale,
            out.offset(first * n * 2),
            rows as u32,
            n as u32,
            k as u32,
            0,
        )
    };
    if m <= 16 {
        launch(m, 0)?;
        Ok(1)
    } else {
        let first = m.div_ceil(2);
        launch(first, 0)?;
        launch(m - first, first)?;
        Ok(2)
    }
}

/// 2026-09-25: Synchronised wall clock over `REPS`, after `WARMUP` untimed
/// runs. Returns milliseconds per rep.
fn time_ms(gpu: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    gpu.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..REPS {
        run()?;
    }
    gpu.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e3 / f64::from(REPS))
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let scalar = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    let batch16 = gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16")?;
    let pipelined = gpu.kernel("w8a16_gemm_pipelined", "w8a16_gemm_pipelined")?;
    let mut rng = Rng(0x927_8a16_2026);
    let mut failures = 0_usize;

    for shape in &SHAPES {
        let (n, k) = (shape.n, shape.k);
        // 2026-09-25: E4M3 byte draws skip 0x7F/0xFF (NaN).
        let weights: Vec<u8> = (0..n * k)
            .map(|_| {
                let x = rng.next();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect();
        let acts: Vec<u8> = (0..MAX_M * k)
            .flat_map(|_| {
                bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let scales: Vec<u8> = (0..(n / 128) * (k / 128))
            .flat_map(|_| (((rng.next() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect();
        let weight = upload(&gpu, &weights)?;
        let scale = upload(&gpu, &scales)?;
        let input = upload(&gpu, &acts)?;
        let out_bytes = MAX_M * n * 2;
        let sentinel = vec![0x5a_u8; out_bytes + 2 * GUARD];
        let scalar_base = upload(&gpu, &sentinel)?;
        let batch_base = upload(&gpu, &sentinel)?;
        let tile_base = upload(&gpu, &sentinel)?;
        let scalar_out = scalar_base.offset(GUARD);
        let batch_out = batch_base.offset(GUARD);
        let tile_out = tile_base.offset(GUARD);
        let weight_gb = (n * k) as f64 / 1e9;

        for m in [5_usize, 8, 16, 32] {
            gpu.copy_h2d(&sentinel, scalar_base)?;
            gpu.copy_h2d(&sentinel, batch_base)?;
            for row in 0..m {
                ops::w8a16_gemv(
                    &gpu,
                    scalar,
                    input.offset(row * k * 2),
                    weight,
                    scale,
                    scalar_out.offset(row * n * 2),
                    n as u32,
                    k as u32,
                    0,
                )?;
            }
            let passes = batch16_route(&gpu, batch16, input, weight, scale, batch_out, m, n, k)?;
            gpu.synchronize(0)?;

            let mut baseline = vec![0_u8; sentinel.len()];
            let mut observed = vec![0_u8; sentinel.len()];
            gpu.copy_d2h(scalar_base, &mut baseline)?;
            gpu.copy_d2h(batch_base, &mut observed)?;
            let bytes = m * n * 2;
            let expected = &baseline[GUARD..GUARD + bytes];
            let actual = &observed[GUARD..GUARD + bytes];
            let unequal = actual
                .chunks_exact(2)
                .zip(expected.chunks_exact(2))
                .filter(|(a, b)| a != b)
                .count();
            let max_abs = actual
                .chunks_exact(2)
                .zip(expected.chunks_exact(2))
                .map(|(a, b)| {
                    let f = |x: &[u8]| {
                        bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32() as f64
                    };
                    (f(a) - f(b)).abs()
                })
                .fold(0.0_f64, f64::max);
            let guards_intact = observed[..GUARD] == sentinel[..GUARD]
                && observed[GUARD + bytes..] == sentinel[GUARD + bytes..];

            let batch_ms = time_ms(&gpu, || {
                batch16_route(&gpu, batch16, input, weight, scale, batch_out, m, n, k)?;
                Ok(())
            })?;
            let tile_ms = time_ms(&gpu, || {
                ops::w8a16_gemm_pipelined(
                    &gpu, pipelined, input, weight, scale, tile_out, m as u32, n as u32, k as u32,
                    0,
                )
            })?;
            let batch_gbs = weight_gb * passes as f64 / (batch_ms / 1e3);
            let tile_gbs = weight_gb / (tile_ms / 1e3);
            println!(
                "{name:<8} M={m:<3} N={n} K={k} passes={passes} \
                 unequal_bf16={unequal} max_abs={max_abs:.9} guards={guards} | \
                 batch16 {batch_ms:.3}ms ({batch_gbs:.1} GB/s) \
                 vs pipelined {tile_ms:.3}ms ({tile_gbs:.1} GB/s) \
                 = {speedup:.2}x",
                name = shape.name,
                guards = if guards_intact { "ok" } else { "CLOBBERED" },
                speedup = tile_ms / batch_ms,
            );
            if unequal != 0 || max_abs != 0.0 || !guards_intact {
                failures += 1;
            }
        }

        // 2026-09-25: A one-bit flip of the baseline's first element,
        // compared against an unflipped copy with `!=`.
        let mut bad = vec![0_u8; sentinel.len()];
        gpu.copy_d2h(scalar_base, &mut bad)?;
        bad[GUARD] ^= 1;
        let mut good = vec![0_u8; sentinel.len()];
        gpu.copy_d2h(scalar_base, &mut good)?;
        let caught = bad[GUARD..GUARD + 2] != good[GUARD..GUARD + 2];
        println!("KNOWN_BAD {} output-bit: refused={caught}", shape.name);
        ensure!(caught, "comparison oracle admitted a one-bit mutation");
    }

    ensure!(
        failures == 0,
        "{failures} shape/M cases differed from the scalar w8a16_gemv bits"
    );
    println!(
        "ALL PASS: real Qwen3.8-27B FFN shapes, batch16 M5/M8/M16 and 2x-halves M32 \
         exactly equal to scalar w8a16_gemv, guards intact"
    );
    Ok(())
}
