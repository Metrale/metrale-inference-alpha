// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The Qwen3.8-27B attention decode projections on four W8A16
//! routes over identical inputs: a per-row scalar `w8a16_gemv` loop (the
//! oracle), `w8a16_gemv_batch{4,16}[_strided]` and
//! `w8a16_gemv_batch16_ncol{2,4}[_strided]`.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if, at every M in `ROWS`, each non-scalar route's
//!   Q/K/V and o_proj output equals the scalar loop's BF16 bytes, is finite,
//!   and leaves every byte outside rows 0..M (the rows past M and the guard
//!   bands) at `SENTINEL`.
//!
//! The tolerance is zero: any unequal element fails. The run also prints
//! `unequal` and `max_abs` per route and per projection, and the time of
//! `REPS` synchronised runs of the q + k + v + o_proj bundle, as ms/layer and
//! GB/s of weight bytes.
//!
//! Shapes from `kernels/hopper/qwen3.8-27b/MODEL.toml`: hidden 5120, head_dim
//! 256, q_heads 24, kv_heads 4, attn_output_gate = true. So K = 5120,
//! q_dim = 6144, q_proj_dim = 12288 (doubled by the output gate),
//! kv_dim = 1024, one sequence's [Q|K|V] block = 14336 BF16 elements (the
//! `PER_SEQ_QKV` stride), and o_proj is N = 5120 over K = q_dim = 6144.
//!
//! Run on a `kernels/hopper` build (the only tree with
//! `w8a16_gemv_ncol.cu`):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example native_fp8_attn_decode_batch_microtest
use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use std::time::Instant;

const K: usize = 5120;
const Q_DIM: usize = 6144;
const Q_PROJ_DIM: usize = 12288;
const KV_DIM: usize = 1024;
const PER_SEQ_QKV: usize = Q_PROJ_DIM + 2 * KV_DIM;
const O_N: usize = K;
const O_K: usize = Q_DIM;
const MAX_M: usize = 16;
const ROWS: [usize; 4] = [2, 4, 8, 16];
const REPS: usize = 20;
const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;

/// 2026-09-25: The four routes under comparison. `Scalar` is the oracle.
/// `Prod` is `w8a16_gemv_batch4` up to 4 rows and `w8a16_gemv_batch16` above,
/// the GEMV arms the layer takes when neither `attn_m16_tc` nor
/// `attn_ncol_gemv` is on. `Ncol2`/`Ncol4` are the N-column-blocked kernels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    Scalar,
    Prod,
    Ncol2,
    Ncol4,
}

impl Route {
    fn label(self, m: usize) -> &'static str {
        match self {
            Route::Scalar => "scalar-per-row",
            Route::Prod if m <= 4 => "batch4",
            Route::Prod => "batch16",
            Route::Ncol2 => "ncol2",
            Route::Ncol4 => "ncol4",
        }
    }
}

/// 2026-09-25: One strided projection's slot inside a sequence's [Q|K|V]
/// block.
struct Proj {
    /// 2026-09-25: Printed on the per-projection line, so an unequal count is
    /// attributed to a projection.
    name: &'static str,
    weight: DevicePtr,
    scale: DevicePtr,
    offset: usize,
    n: usize,
}

/// 2026-09-25: Every kernel handle the routes need, resolved once.
struct Kernels {
    scalar: KernelHandle,
    batch4_s: KernelHandle,
    batch16_s: KernelHandle,
    ncol2_s: KernelHandle,
    ncol4_s: KernelHandle,
    batch4: KernelHandle,
    batch16: KernelHandle,
    ncol2: KernelHandle,
    ncol4: KernelHandle,
}

type StridedBatchGemv = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn values(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32() as f64)
        .collect()
}

/// 2026-09-25: The pass/fail oracle. Every byte outside `live` must still be
/// the sentinel in both buffers; inside `live` the values must be finite and
/// the bytes equal. `main` first feeds it four corrupted copies of the
/// baseline and requires it to refuse each.
fn check(observed: &[u8], baseline: &[u8], sentinel: &[u8], live: &[(usize, usize)]) -> Result<()> {
    ensure!(
        observed.len() == sentinel.len() && baseline.len() == sentinel.len(),
        "output extent mismatch"
    );
    let mut mask = vec![false; sentinel.len()];
    for &(start, len) in live {
        mask[GUARD + start..GUARD + start + len].fill(true);
    }
    for i in 0..sentinel.len() {
        if !mask[i] {
            ensure!(
                observed[i] == sentinel[i],
                "route wrote outside its extent at byte {i}"
            );
            ensure!(
                baseline[i] == sentinel[i],
                "scalar oracle wrote outside its extent at byte {i}"
            );
        }
    }
    for &(start, len) in live {
        let a = &observed[GUARD + start..GUARD + start + len];
        let b = &baseline[GUARD + start..GUARD + start + len];
        ensure!(
            values(a)
                .iter()
                .chain(values(b).iter())
                .all(|x| x.is_finite()),
            "nonfinite projection output"
        );
        ensure!(a == b, "output differs from the scalar loop's BF16 bits");
    }
    Ok(())
}

/// 2026-09-25: Unequal BF16 elements and max absolute delta over the live
/// extents.
fn compare(observed: &[u8], baseline: &[u8], live: &[(usize, usize)]) -> (usize, f64) {
    let mut mismatches = 0;
    let mut max_abs = 0.0_f64;
    for &(start, len) in live {
        let a = &observed[GUARD + start..GUARD + start + len];
        let b = &baseline[GUARD + start..GUARD + start + len];
        mismatches += a
            .chunks_exact(2)
            .zip(b.chunks_exact(2))
            .filter(|(x, y)| x != y)
            .count();
        max_abs = values(a)
            .iter()
            .zip(values(b).iter())
            .map(|(x, y)| (x - y).abs())
            .fold(max_abs, f64::max);
    }
    (mismatches, max_abs)
}

/// 2026-09-25: One strided Q/K/V projection, on the given route.
fn run_qkv(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    route: Route,
    input: DevicePtr,
    out: DevicePtr,
    p: &Proj,
    m: usize,
) -> Result<()> {
    if route == Route::Scalar {
        for row in 0..m {
            ops::w8a16_gemv(
                gpu,
                k.scalar,
                input.offset(row * K * 2),
                p.weight,
                p.scale,
                out.offset((row * PER_SEQ_QKV + p.offset) * 2),
                p.n as u32,
                K as u32,
                0,
            )?;
        }
        return Ok(());
    }
    let (launch, kernel): (StridedBatchGemv, KernelHandle) = match route {
        Route::Prod if m <= 4 => (ops::w8a16_gemv_batch4_strided, k.batch4_s),
        Route::Prod => (ops::w8a16_gemv_batch16_strided, k.batch16_s),
        Route::Ncol2 => (ops::w8a16_gemv_batch16_ncol2_strided, k.ncol2_s),
        Route::Ncol4 => (ops::w8a16_gemv_batch16_ncol4_strided, k.ncol4_s),
        Route::Scalar => unreachable!("handled above"),
    };
    launch(
        gpu,
        kernel,
        input,
        p.weight,
        p.scale,
        out.offset(p.offset * 2),
        m as u32,
        p.n as u32,
        K as u32,
        K as u32,
        PER_SEQ_QKV as u32,
        0,
    )
}

/// 2026-09-25: The contiguous o_proj, on the given route.
/// `[m, O_K] -> [m, O_N]`.
#[allow(clippy::too_many_arguments)]
fn run_oproj(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    route: Route,
    input: DevicePtr,
    out: DevicePtr,
    weight: DevicePtr,
    scale: DevicePtr,
    m: usize,
) -> Result<()> {
    if route == Route::Scalar {
        for row in 0..m {
            ops::w8a16_gemv(
                gpu,
                k.scalar,
                input.offset(row * O_K * 2),
                weight,
                scale,
                out.offset(row * O_N * 2),
                O_N as u32,
                O_K as u32,
                0,
            )?;
        }
        return Ok(());
    }
    let (launch, kernel): (ops::ContiguousBatchGemv, KernelHandle) = match route {
        Route::Prod if m <= 4 => (ops::w8a16_gemv_batch4, k.batch4),
        Route::Prod => (ops::w8a16_gemv_batch16, k.batch16),
        Route::Ncol2 => (ops::w8a16_gemv_batch16_ncol2, k.ncol2),
        Route::Ncol4 => (ops::w8a16_gemv_batch16_ncol4, k.ncol4),
        Route::Scalar => unreachable!("handled above"),
    };
    launch(
        gpu, kernel, input, weight, scale, out, m as u32, O_N as u32, O_K as u32, 0,
    )
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let k = Kernels {
        scalar: gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
        batch4_s: gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch4_strided")?,
        batch16_s: gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16_strided")?,
        ncol2_s: gpu.kernel("w8a16_gemv_ncol", "w8a16_gemv_batch16_ncol2_strided")?,
        ncol4_s: gpu.kernel("w8a16_gemv_ncol", "w8a16_gemv_batch16_ncol4_strided")?,
        batch4: gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch4")?,
        batch16: gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16")?,
        ncol2: gpu.kernel("w8a16_gemv_ncol", "w8a16_gemv_batch16_ncol2")?,
        ncol4: gpu.kernel("w8a16_gemv_ncol", "w8a16_gemv_batch16_ncol4")?,
    };

    let mut state = 0x0927_a77f_2026_u64;
    let mut random = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 32) as u32
    };
    let mut fp8_weight = |n: usize, depth: usize| -> Vec<u8> {
        (0..n * depth)
            .map(|_| {
                let x = random();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect()
    };
    let q_bytes = fp8_weight(Q_PROJ_DIM, K);
    let k_bytes = fp8_weight(KV_DIM, K);
    let v_bytes = fp8_weight(KV_DIM, K);
    let o_bytes = fp8_weight(O_N, O_K);
    let mut scales = |n: usize, depth: usize| -> Vec<u8> {
        (0..(n / 128) * (depth / 128))
            .flat_map(|_| (((random() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect()
    };
    let q_scale = scales(Q_PROJ_DIM, K);
    let k_scale = scales(KV_DIM, K);
    let v_scale = scales(KV_DIM, K);
    let o_scale = scales(O_N, O_K);
    let mut acts = |elems: usize| -> Vec<u8> {
        (0..elems)
            .flat_map(|_| {
                bf16::from_f32(((random() % 2049) as f32 - 1024.0) / 1024.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect()
    };
    // 2026-09-25: `normed` [MAX_M, K] for q/k/v, `attn_out` [MAX_M, O_K] for
    // o_proj.
    let normed_bytes = acts(MAX_M * K);
    let attn_out_bytes = acts(MAX_M * O_K);

    let normed = upload(&gpu, &normed_bytes)?;
    let attn_out = upload(&gpu, &attn_out_bytes)?;
    let projections = [
        Proj {
            name: "q_proj",
            weight: upload(&gpu, &q_bytes)?,
            scale: upload(&gpu, &q_scale)?,
            offset: 0,
            n: Q_PROJ_DIM,
        },
        Proj {
            name: "k_proj",
            weight: upload(&gpu, &k_bytes)?,
            scale: upload(&gpu, &k_scale)?,
            offset: Q_PROJ_DIM,
            n: KV_DIM,
        },
        Proj {
            name: "v_proj",
            weight: upload(&gpu, &v_bytes)?,
            scale: upload(&gpu, &v_scale)?,
            offset: Q_PROJ_DIM + KV_DIM,
            n: KV_DIM,
        },
    ];
    let o_weight = upload(&gpu, &o_bytes)?;
    let o_scale_ptr = upload(&gpu, &o_scale)?;

    let qkv_bytes = MAX_M * PER_SEQ_QKV * 2;
    let o_out_bytes = MAX_M * O_N * 2;
    let qkv_sentinel = vec![SENTINEL; qkv_bytes + 2 * GUARD];
    let o_sentinel = vec![SENTINEL; o_out_bytes + 2 * GUARD];
    let qkv_base = upload(&gpu, &qkv_sentinel)?;
    let o_base = upload(&gpu, &o_sentinel)?;
    let qkv_out = qkv_base.offset(GUARD);
    let o_out = o_base.offset(GUARD);

    // 2026-09-25: The bundle every route is checked and timed on: q, k and v
    // strided, then o_proj.
    let bundle = |route: Route, m: usize| -> Result<()> {
        for p in &projections {
            run_qkv(&gpu, &k, route, normed, qkv_out, p, m)?;
        }
        run_oproj(&gpu, &k, route, attn_out, o_out, o_weight, o_scale_ptr, m)
    };
    let capture = |route: Route, m: usize| -> Result<(Vec<u8>, Vec<u8>)> {
        gpu.copy_h2d(&qkv_sentinel, qkv_base)?;
        gpu.copy_h2d(&o_sentinel, o_base)?;
        bundle(route, m)?;
        gpu.synchronize(0)?;
        let mut qkv = vec![0_u8; qkv_sentinel.len()];
        let mut o = vec![0_u8; o_sentinel.len()];
        gpu.copy_d2h(qkv_base, &mut qkv)?;
        gpu.copy_d2h(o_base, &mut o)?;
        Ok((qkv, o))
    };

    let mut failures = 0usize;
    let mut first_oracle = true;
    for m in ROWS {
        let qkv_live: Vec<(usize, usize)> = (0..m)
            .map(|r| (r * PER_SEQ_QKV * 2, PER_SEQ_QKV * 2))
            .collect();
        let o_live: Vec<(usize, usize)> = (0..m).map(|r| (r * O_N * 2, O_N * 2)).collect();
        let (qkv_ref, o_ref) = capture(Route::Scalar, m)?;

        if first_oracle {
            // 2026-09-25: A green run has to be able to go red: corrupt the
            // baseline four ways and require the oracle to refuse each.
            for mutation in ["output-bit", "gap", "guard", "nonfinite"] {
                let mut bad = qkv_ref.clone();
                match mutation {
                    "output-bit" => bad[GUARD] ^= 1,
                    // 2026-09-25: The buffer's last row is past M here
                    // (M = ROWS[0]), so a route that writes too many rows
                    // changes it.
                    "gap" => bad[GUARD + MAX_M * PER_SEQ_QKV * 2 - 2] ^= 1,
                    "guard" => bad[0] ^= 1,
                    _ => bad[GUARD..GUARD + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes()),
                }
                let err = check(&bad, &qkv_ref, &qkv_sentinel, &qkv_live)
                    .expect_err("known-bad output was admitted by the real oracle");
                println!("KNOWN_BAD {mutation}: refused: {err}");
            }
            first_oracle = false;
        }

        for route in [Route::Prod, Route::Ncol2, Route::Ncol4] {
            let (qkv, o) = capture(route, m)?;
            let (qkv_bad, qkv_max) = compare(&qkv, &qkv_ref, &qkv_live);
            let (o_bad, o_max) = compare(&o, &o_ref, &o_live);
            println!(
                "M={m:>2} route={:<14} qkv unequal={qkv_bad} max_abs={qkv_max:.9} | \
                 o_proj unequal={o_bad} max_abs={o_max:.9}",
                route.label(m)
            );
            // 2026-09-25: Per projection, so a mismatch names the projection
            // and its N rather than an offset in the packed `PER_SEQ_QKV` row.
            for p in &projections {
                let live_p: Vec<(usize, usize)> = (0..m)
                    .map(|r| ((r * PER_SEQ_QKV + p.offset) * 2, p.n * 2))
                    .collect();
                let (bad, mx) = compare(&qkv, &qkv_ref, &live_p);
                println!(
                    "          {:<7} N={:<5} unequal={bad} max_abs={mx:.9}",
                    p.name, p.n
                );
            }
            for (what, observed, baseline, sentinel, live) in [
                ("qkv", &qkv, &qkv_ref, &qkv_sentinel, &qkv_live),
                ("o_proj", &o, &o_ref, &o_sentinel, &o_live),
            ] {
                if let Err(e) = check(observed, baseline, sentinel, live) {
                    println!("FAIL {what} M={m} route={:?}: {e}", route);
                    failures += 1;
                }
            }
        }

        // 2026-09-25: Timing: the bundle, synchronised, after one warmup run.
        println!("  --- timing, {REPS} reps of the per-layer q+k+v+o bundle ---");
        for route in [Route::Scalar, Route::Prod, Route::Ncol2, Route::Ncol4] {
            bundle(route, m)?;
            gpu.synchronize(0)?;
            let t0 = Instant::now();
            for _ in 0..REPS {
                bundle(route, m)?;
            }
            gpu.synchronize(0)?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
            // 2026-09-25: GB/s counts the FP8 weight bytes (73.4 MB q/k/v +
            // 31.5 MB o_proj) once per bundle for a batched route and m times
            // for the scalar loop; scales and activations are not counted.
            let passes = if route == Route::Scalar { m } else { 1 };
            let gb = (Q_PROJ_DIM + 2 * KV_DIM) as f64 * K as f64 + (O_N * O_K) as f64;
            let gbs = gb * passes as f64 / (ms / 1000.0) / 1e9;
            println!(
                "  M={m:>2} route={:<14} {ms:.3} ms/layer  {gbs:.0} GB/s",
                route.label(m)
            );
        }
    }

    ensure!(failures == 0, "{failures} case(s) failed");
    println!(
        "ALL PASS: Qwen3.8-27B attention projections, scalar == batch4/batch16 == ncol2 == ncol4 \
         on every row count, gaps and guards intact"
    );
    Ok(())
}
