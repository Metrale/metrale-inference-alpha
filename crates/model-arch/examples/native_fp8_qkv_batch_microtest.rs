// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The Qwen3.8-27B Q/K/V decode projections: a per-row scalar
//! `w8a16_gemv` loop against one strided batched launch per projection
//! (`w8a16_gemv_batch4_strided` up to 4 rows, `w8a16_gemv_batch16_strided` up
//! to 16, `w8a16_gemm_pipelined_m32_strided` above).
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if, at every M in {2, 3, 4, 5, 8, 16, 17, 24, 32},
//!   for each projection alone and for all three into one buffer, every byte
//!   outside the live extents (rows past M, the other projections' slots in a
//!   single-projection run, the guard bands) still holds the sentinel; and the
//!   live extents are byte-equal to the scalar loop for M <= 16, or within the
//!   tensor-core budget of `common/fp8_qkv_m32_leg.rs` above 16. Every
//!   known-bad control must be refused.
//!
//! Shapes from `kernels/gb10/qwen3.8-27b/MODEL.toml`: hidden_dim 5120,
//! head_dim 256, q_heads 24, kv_heads 4, attn_output_gate = true. So
//! K = 5120, q_dim = 6144, q_proj_dim = 2 * q_dim = 12288, kv_dim = 1024, and
//! one sequence's [Q|K|V] block is 14336 BF16 elements, the `PER_SEQ_QKV` row
//! stride the batched kernels write at.
//!
//! Run:
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example native_fp8_qkv_batch_microtest
use anyhow::{Result, ensure};
use half::bf16;
use metrale_model_layers::layers::ops;

#[path = "common/fp8_qkv_m32_leg.rs"]
mod m32_leg;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const K: usize = 5120;
const Q_PROJ_DIM: usize = 12288;
const KV_DIM: usize = 1024;
const PER_SEQ_QKV: usize = Q_PROJ_DIM + 2 * KV_DIM;
const MAX_M: usize = 32;
const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;

/// 2026-09-25: One projection's slot inside a sequence's [Q|K|V] block.
struct Proj {
    name: &'static str,
    weight: DevicePtr,
    scale: DevicePtr,
    /// 2026-09-25: Element offset of this projection inside the row.
    offset: usize,
    /// 2026-09-25: Output width (N).
    n: usize,
}

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

/// 2026-09-25: Every byte outside the live extents must still be sentinel, on
/// both runs. Shared by the exact legs and the M32 leg.
fn check_untouched(
    observed: &[u8],
    baseline: &[u8],
    sentinel: &[u8],
    live: &[(usize, usize)],
) -> Result<()> {
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
                "batched projection wrote outside its extent at byte {i}"
            );
            ensure!(
                baseline[i] == sentinel[i],
                "scalar projection wrote outside its extent at byte {i}"
            );
        }
    }
    Ok(())
}

/// 2026-09-25: The comparison oracle for the GEMV legs: `check_untouched`,
/// then finite and byte-equal live extents. `main` first feeds it four
/// corrupted copies of a baseline and requires it to refuse each.
fn check(observed: &[u8], baseline: &[u8], sentinel: &[u8], live: &[(usize, usize)]) -> Result<()> {
    check_untouched(observed, baseline, sentinel, live)?;
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
        ensure!(
            a == b,
            "batch output differs from production scalar BF16 bits"
        );
    }
    Ok(())
}

/// 2026-09-25: The signature the three strided wrappers `run_batched` picks
/// between share.
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

/// 2026-09-25: The tier's arm chain as `ms_qkv_batchm_fp8_gemv` walks it when
/// neither `attn_m16_tc` nor `attn_ncol_gemv` is on: batch4 to 4 rows,
/// batch16 to 16, the 32-row M-tile twin past 16.
fn run_batched(
    gpu: &dyn GpuBackend,
    batch4: KernelHandle,
    batch16: KernelHandle,
    m32: KernelHandle,
    input: DevicePtr,
    out: DevicePtr,
    p: &Proj,
    m: usize,
) -> Result<()> {
    let (launch, kernel): (StridedBatchGemv, KernelHandle) = if m <= 4 {
        (ops::w8a16_gemv_batch4_strided, batch4)
    } else if m <= 16 {
        (ops::w8a16_gemv_batch16_strided, batch16)
    } else {
        (ops::w8a16_gemm_pipelined_m32_strided, m32)
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

fn run_scalar(
    gpu: &dyn GpuBackend,
    scalar: KernelHandle,
    input: DevicePtr,
    out: DevicePtr,
    p: &Proj,
    m: usize,
) -> Result<()> {
    for row in 0..m {
        ops::w8a16_gemv(
            gpu,
            scalar,
            input.offset(row * K * 2),
            p.weight,
            p.scale,
            out.offset((row * PER_SEQ_QKV + p.offset) * 2),
            p.n as u32,
            K as u32,
            0,
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let scalar = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    let batch4 = gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch4_strided")?;
    let batch16 = gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16_strided")?;
    let m32 = gpu.kernel("w8a16_gemm_pipelined_m32", "w8a16_gemm_pipelined_m32")?;

    let mut state = 0x0132_8a16_2026_u64;
    let mut random = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 32) as u32
    };
    let mut fp8_weight = |n: usize| -> Vec<u8> {
        (0..n * K)
            .map(|_| {
                let x = random();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect()
    };
    let q_bytes = fp8_weight(Q_PROJ_DIM);
    let k_bytes = fp8_weight(KV_DIM);
    let v_bytes = fp8_weight(KV_DIM);
    let mut scales = |n: usize| -> Vec<u8> {
        (0..(n / 128) * (K / 128))
            .flat_map(|_| (((random() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect()
    };
    let q_scale = scales(Q_PROJ_DIM);
    let k_scale = scales(KV_DIM);
    let v_scale = scales(KV_DIM);
    // 2026-09-25: Activations: contiguous [MAX_M, K].
    let acts: Vec<u8> = (0..MAX_M * K)
        .flat_map(|_| {
            bf16::from_f32(((random() % 2049) as f32 - 1024.0) / 1024.0)
                .to_bits()
                .to_le_bytes()
        })
        .collect();

    let input_base = upload(
        &gpu,
        &[vec![SENTINEL; GUARD], acts, vec![SENTINEL; GUARD]].concat(),
    )?;
    let input = input_base.offset(GUARD);
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

    let output_bytes = MAX_M * PER_SEQ_QKV * 2;
    let sentinel = vec![SENTINEL; output_bytes + 2 * GUARD];
    let scalar_base = upload(&gpu, &sentinel)?;
    let batch_base = upload(&gpu, &sentinel)?;
    let scalar_out = scalar_base.offset(GUARD);
    let batch_out = batch_base.offset(GUARD);
    let mut first_oracle = true;
    let mut first_m32_oracle = true;
    let mut failures = 0usize;

    for m in [2_usize, 3, 4, 5, 8, 16, 17, 24, 32] {
        // 2026-09-25: Pass 1: each projection alone. The other two
        // projections' slots in the row are a gap that must stay sentinel, so
        // this tests the C row stride directly.
        for p in &projections {
            gpu.copy_h2d(&sentinel, scalar_base)?;
            gpu.copy_h2d(&sentinel, batch_base)?;
            run_scalar(&gpu, scalar, input, scalar_out, p, m)?;
            run_batched(&gpu, batch4, batch16, m32, input, batch_out, p, m)?;
            gpu.synchronize(0)?;
            let mut baseline = vec![0_u8; sentinel.len()];
            let mut observed = vec![0_u8; sentinel.len()];
            gpu.copy_d2h(scalar_base, &mut baseline)?;
            gpu.copy_d2h(batch_base, &mut observed)?;
            let live: Vec<(usize, usize)> = (0..m)
                .map(|r| ((r * PER_SEQ_QKV + p.offset) * 2, p.n * 2))
                .collect();

            if m > 16 && first_m32_oracle {
                // 2026-09-25: The first case above 16 rows is q_proj, so the
                // K slot is a gap here.
                m32_leg::known_bad_controls(&baseline, &sentinel, &live, (Q_PROJ_DIM + 1) * 2, K)?;
                first_m32_oracle = false;
            }
            if first_oracle {
                for mutation in ["output-bit", "gap", "guard", "nonfinite"] {
                    let mut bad = baseline.clone();
                    match mutation {
                        "output-bit" => bad[GUARD] ^= 1,
                        "gap" => bad[GUARD + (Q_PROJ_DIM + 1) * 2] ^= 1,
                        "guard" => bad[0] ^= 1,
                        _ => bad[GUARD..GUARD + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes()),
                    }
                    let err = check(&bad, &baseline, &sentinel, &live)
                        .expect_err("known-bad output was admitted by the real oracle");
                    println!("KNOWN_BAD {mutation}: refused: {err}");
                }
                first_oracle = false;
            }

            let (mismatches, max_abs) = compare(&observed, &baseline, &live);
            let kernel = kernel_name(m);
            println!(
                "{} M={m} N={} K={K} stride={PER_SEQ_QKV} kernel={kernel} \
                 unequal_bf16={mismatches} max_abs={max_abs:.9}",
                p.name, p.n
            );
            if let Err(e) = grade(m, &observed, &baseline, &sentinel, &live) {
                println!("FAIL {} M={m}: {e}", p.name);
                failures += 1;
            }
        }

        // 2026-09-25: Pass 2: all three projections into one strided buffer,
        // one launch each, as `ms_qkv_batchm_fp8_gemv` issues them.
        gpu.copy_h2d(&sentinel, scalar_base)?;
        gpu.copy_h2d(&sentinel, batch_base)?;
        for p in &projections {
            run_scalar(&gpu, scalar, input, scalar_out, p, m)?;
            run_batched(&gpu, batch4, batch16, m32, input, batch_out, p, m)?;
        }
        gpu.synchronize(0)?;
        let mut baseline = vec![0_u8; sentinel.len()];
        let mut observed = vec![0_u8; sentinel.len()];
        gpu.copy_d2h(scalar_base, &mut baseline)?;
        gpu.copy_d2h(batch_base, &mut observed)?;
        // 2026-09-25: Rows m..MAX_M are the gap here: a launch that wrote too
        // many rows would change them.
        let live: Vec<(usize, usize)> = (0..m)
            .map(|r| (r * PER_SEQ_QKV * 2, PER_SEQ_QKV * 2))
            .collect();
        let (mismatches, max_abs) = compare(&observed, &baseline, &live);
        println!(
            "full-qkv M={m} row_elems={PER_SEQ_QKV} K={K} kernel={} \
             unequal_bf16={mismatches} max_abs={max_abs:.9}",
            kernel_name(m)
        );
        if let Err(e) = grade(m, &observed, &baseline, &sentinel, &live) {
            println!("FAIL full-qkv M={m}: {e}");
            failures += 1;
        }
    }

    ensure!(failures == 0, "{failures} case(s) failed");
    println!(
        "ALL PASS: Qwen3.8-27B q/k/v shapes — strided batch4 M2/M3/M4 and batch16 \
         M5/M8/M16 byte-exact vs scalar; M32 tile M17/M24/M32 within the tensor-core \
         budget vs scalar; gaps and guards intact on every leg"
    );
    Ok(())
}

/// 2026-09-25: Which arm of the tier a row count lands on (`run_batched`'s
/// chain).
fn kernel_name(m: usize) -> &'static str {
    if m <= 4 {
        "batch4"
    } else if m <= 16 {
        "batch16"
    } else {
        "m32_tile"
    }
}

/// 2026-09-25: The GEMV legs (M <= 16) are graded byte-exact; the M32 leg on
/// the tensor-core budget, with its worst-case numbers printed.
fn grade(
    m: usize,
    observed: &[u8],
    baseline: &[u8],
    sentinel: &[u8],
    live: &[(usize, usize)],
) -> Result<()> {
    if m <= 16 {
        return check(observed, baseline, sentinel, live);
    }
    let v = m32_leg::check_m32(observed, baseline, sentinel, live, K)?;
    println!(
        "  m32_tile M={m}: max_ulp={} over_ulp_only(floor-admitted)={} rel_rms={:.3e} \
         max_abs={:.3e}",
        v.max_ulp, v.over_ulp_only, v.rel_rms, v.max_abs
    );
    Ok(())
}

/// 2026-09-25: Unequal BF16 elements and max absolute delta over the live
/// extents only.
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
