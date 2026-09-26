// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU oracle for the tensor-core 5..=32-row native-FP8 dense-FFN
//! decode tier, `w8a16_gemm_m16` and `w8a16_gemm_m16_n64` (the
//! `ffn_m16_tc` target default, overridden by `METRALE_FFN_M16_TC` or, when
//! that is unset, `METRALE_M16_TC`).
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if, for both shapes at every M in
//!   {1, 5, 8, 13, 16, 32}, both instantiations pass `case_passed`
//!   (`examples/common/m16_tc_compare.rs`) against the per-row scalar
//!   `w8a16_gemv`, and `assert_oracle_bites` refused its known-bad blocks.
//!
//! Shapes: gate/up `[17408, 5120]` and down `[5120, 17408]` (`SHAPES`).
//!
//! The pass condition is a tolerance, not byte equality: an m16n8k16 MMA
//! reduces K-products in the tensor core's own order. Each element must pass
//! `layers::dense_ffn::m16_tc::oracle::within_m16_tc_budget` (within
//! `M16_TC_MAX_ULP` ordinal BF16 ULP, or an absolute error within
//! `m16_tc_acc_floor(k, row_rms)`), and each block's rel_rms must be at most
//! `REL_RMS_GATE`. The host simulations evaluate the same predicate.
//!
//! Also checked: no write outside `[M, N]` on either instantiation, and the
//! `w8a16_gemm_m16_strided` leg, which takes the same two-halves route at
//! padded pitches, leaves each row's pitch gap and every row past M at the
//! sentinel. The tier is timed against `w8a16_gemv_batch16` and against
//! `w8a16_gemm_n128_m128` on a host-transposed weight. GB/s counts the FP8
//! weight bytes once per kernel pass, so the 17..=32 rung, which makes two
//! passes, counts them twice.
//!
//! Timing: `synchronize` plus a host `Instant` over `REPS` after `WARMUP`
//! runs.
//!
//! Run:
//!     cargo run --release --example native_fp8_ffn_m16_tc_microtest \
//!       --features cuda,gpu-examples
use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::dense_ffn::m16_tc::oracle::{
    M16_TC_MAX_ULP, compare_m16_tc_block,
};
use metrale_model_layers::layers::ops;
use std::time::Instant;

#[path = "common/m16_tc_compare.rs"]
pub(crate) mod m16_tc_compare;
use m16_tc_compare::{
    A_PAD, C_PAD, Case, GUARD, MAX_M, REL_RMS_GATE, Rng, SHAPES, Timings, assert_oracle_bites,
    check_strided, draw_inputs, guards_intact, report_case, transpose,
};

const REPS: u32 = 20;
const WARMUP: u32 = 3;

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// 2026-09-25: The TC route as `DenseFfnLayer::w8a16_m16_tc_proj` runs it:
/// one launch at m <= 16, two on contiguous row halves at 17..=32 (the first
/// `m.div_ceil(2)` rows). `gemm` is the instantiation under test,
/// `w8a16_gemm_m16` or `w8a16_gemm_m16_n64`, which take the same arguments.
/// Returns the number of launches.
#[allow(clippy::too_many_arguments)]
fn tc_route(
    gpu: &dyn GpuBackend,
    gemm: ops::ContiguousM16Gemm,
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
        gemm(
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

/// 2026-09-25: The same route through `w8a16_gemm_m16_strided`, at
/// caller-supplied A and C row pitches. The halves are a pointer offset on a
/// padded pitch just as on a packed one, so at M=32 rows 16..31 are measured
/// too.
#[allow(clippy::too_many_arguments)]
fn tc_route_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    scale: DevicePtr,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    a_pitch: usize,
    c_pitch: usize,
) -> Result<()> {
    let launch = |rows: usize, first: usize| {
        ops::w8a16_gemm_m16_strided(
            gpu,
            kernel,
            input.offset(first * a_pitch * 2),
            weight,
            scale,
            out.offset(first * c_pitch * 2),
            rows as u32,
            n as u32,
            k as u32,
            a_pitch as u32,
            c_pitch as u32,
            0,
        )
    };
    if m <= 16 {
        launch(m, 0)
    } else {
        let first = m.div_ceil(2);
        launch(first, 0)?;
        launch(m - first, first)
    }
}

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
) -> Result<()> {
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
        launch(m, 0)
    } else {
        let first = m.div_ceil(2);
        launch(first, 0)?;
        launch(m - first, first)
    }
}

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
    let tc = gpu.kernel("w8a16_gemm_m16", "w8a16_gemm_m16")?;
    let tc_n64 = gpu.kernel("w8a16_gemm_m16", "w8a16_gemm_m16_n64")?;
    let tc_strided = gpu.kernel("w8a16_gemm_m16", "w8a16_gemm_m16_strided")?;
    let tile_t = gpu.kernel("w8a16_gemm_t_m128", "w8a16_gemm_t_m128")?;
    let mut rng = Rng(0x927_16_7C_2026);
    let mut failures = 0_usize;

    for shape in &SHAPES {
        let (n, k) = (shape.n, shape.k);
        let inp = draw_inputs(&mut rng, n, k);
        let a_pitch = k + A_PAD;
        let (wt, st) = transpose(&inp.weights, &inp.scales, n, k);

        let weight = upload(&gpu, &inp.weights)?;
        let scale = upload(&gpu, &inp.scales)?;
        let input = upload(&gpu, &inp.acts)?;
        let input_s = upload(&gpu, &inp.acts_strided)?;
        let weight_t = upload(&gpu, &wt)?;
        let scale_t = upload(&gpu, &st)?;
        let out_bytes = MAX_M * n * 2;
        let sentinel = vec![0x5a_u8; out_bytes + 2 * GUARD];
        let c_pitch = n + C_PAD;
        let sentinel_s = vec![0x5a_u8; MAX_M * c_pitch * 2 + 2 * GUARD];
        let scalar_base = upload(&gpu, &sentinel)?;
        let tc_base = upload(&gpu, &sentinel)?;
        let n64_base = upload(&gpu, &sentinel)?;
        let b16_base = upload(&gpu, &sentinel)?;
        let tile_base = upload(&gpu, &sentinel)?;
        let strided_base = upload(&gpu, &sentinel_s)?;
        let (scalar_out, tc_out, n64_out, b16_out, tile_out, strided_out) = (
            scalar_base.offset(GUARD),
            tc_base.offset(GUARD),
            n64_base.offset(GUARD),
            b16_base.offset(GUARD),
            tile_base.offset(GUARD),
            strided_base.offset(GUARD),
        );
        let weight_gb = (n * k) as f64 / 1e9;

        for m in [1_usize, 5, 8, 13, 16, 32] {
            gpu.copy_h2d(&sentinel, scalar_base)?;
            gpu.copy_h2d(&sentinel, tc_base)?;
            gpu.copy_h2d(&sentinel, n64_base)?;
            gpu.copy_h2d(&sentinel_s, strided_base)?;
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
            let passes = tc_route(
                &gpu,
                ops::w8a16_gemm_m16,
                tc,
                input,
                weight,
                scale,
                tc_out,
                m,
                n,
                k,
            )?;
            tc_route(
                &gpu,
                ops::w8a16_gemm_m16_n64,
                tc_n64,
                input,
                weight,
                scale,
                n64_out,
                m,
                n,
                k,
            )?;
            // 2026-09-25: Strided leg: the same route, padded pitches on both
            // sides.
            tc_route_strided(
                &gpu,
                tc_strided,
                input_s,
                weight,
                scale,
                strided_out,
                m,
                n,
                k,
                a_pitch,
                c_pitch,
            )?;
            gpu.synchronize(0)?;

            let mut baseline = vec![0_u8; sentinel.len()];
            let mut observed = vec![0_u8; sentinel.len()];
            let mut observed_n64 = vec![0_u8; sentinel.len()];
            let mut strided = vec![0_u8; sentinel_s.len()];
            gpu.copy_d2h(scalar_base, &mut baseline)?;
            gpu.copy_d2h(tc_base, &mut observed)?;
            gpu.copy_d2h(n64_base, &mut observed_n64)?;
            gpu.copy_d2h(strided_base, &mut strided)?;
            let bytes = m * n * 2;
            let d = compare_m16_tc_block(
                &observed[GUARD..GUARD + bytes],
                &baseline[GUARD..GUARD + bytes],
                n,
                k,
            );
            let d64 = compare_m16_tc_block(
                &observed_n64[GUARD..GUARD + bytes],
                &baseline[GUARD..GUARD + bytes],
                n,
                k,
            );

            // 2026-09-25: Guards: nothing outside [M, N] on either
            // instantiation, and the strided leg's row gaps, rows past M and
            // used extent (`m16_tc_compare.rs`).
            let guards = guards_intact(&observed, &observed_n64, &sentinel, bytes);
            let strided_check = check_strided(&strided, &sentinel_s, &baseline, m, n, k, c_pitch);

            let tc_ms = time_ms(&gpu, || {
                tc_route(
                    &gpu,
                    ops::w8a16_gemm_m16,
                    tc,
                    input,
                    weight,
                    scale,
                    tc_out,
                    m,
                    n,
                    k,
                )?;
                Ok(())
            })?;
            let n64_ms = time_ms(&gpu, || {
                tc_route(
                    &gpu,
                    ops::w8a16_gemm_m16_n64,
                    tc_n64,
                    input,
                    weight,
                    scale,
                    n64_out,
                    m,
                    n,
                    k,
                )?;
                Ok(())
            })?;
            let b16_ms = time_ms(&gpu, || {
                batch16_route(&gpu, batch16, input, weight, scale, b16_out, m, n, k)
            })?;
            let tile_ms = time_ms(&gpu, || {
                ops::w8a16_gemm_n128_m128(
                    &gpu, tile_t, input, weight_t, scale_t, tile_out, m as u32, n as u32, k as u32,
                    0,
                )
            })?;
            let timings = Timings {
                tc_ms,
                n64_ms,
                b16_ms,
                tile_ms,
            };
            let case = Case {
                name: shape.name,
                m,
                n,
                k,
                passes,
                d: &d,
                d64: &d64,
                strided: &strided_check,
                guards_intact: guards,
                timings: &timings,
                weight_gb,
            };
            if !report_case(&case) {
                failures += 1;
            }
        }

        // 2026-09-25: The comparison the loop runs must refuse known-bad
        // blocks, so a green report cannot come from a vacuous comparison.
        let mut good = vec![0_u8; sentinel.len()];
        gpu.copy_d2h(scalar_base, &mut good)?;
        assert_oracle_bites(&good, shape.name, n, k)?;
    }

    ensure!(
        failures == 0,
        "{failures} shape/M cases exceeded the {M16_TC_MAX_ULP}-ULP / accumulation-floor \
         criterion or the {REL_RMS_GATE} rel_rms budget, or broke a guard"
    );
    println!(
        "ALL PASS: real Qwen3.8-27B FFN shapes, w8a16_gemm_m16 AND w8a16_gemm_m16_n64 at \
         M 1/5/8/13/16/32 within {M16_TC_MAX_ULP} BF16 ULP (or the accumulation floor) and \
         {REL_RMS_GATE} rel_rms of the scalar w8a16_gemv, strided halves, gaps and [M,N] \
         bounds intact"
    );
    Ok(())
}
