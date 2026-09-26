// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU oracle for the tensor-core 5..=16-row BF16 LM-head kernels
//! `dense_gemm_m16_bf16` and `dense_gemm_m16_bf16_n64` (lever `METRALE_LM_HEAD_M16_TC`),
//! against one scalar `dense_gemv_bf16` per row, on a `[N = 248,077, K = 5120]` BF16 weight
//! (2.54 GB) at M in {5, 8, 13, 16}.
//!
//! Owner: model-arch examples (hopper LM-head kernels).
//! Invariants:
//! - The run fails if any M case has an element outside the per-element budget, a block
//!   `rel_rms` above `REL_RMS_GATE` or an overwritten guard, or if any of the three KNOWN_BAD
//!   controls passes. The speed target is printed, not asserted.
//!
//! The pass condition is a tolerance, not bit equality: the m16n8k16 MMA sums 16 K-products in
//! its own order. Per element it is `within_m16_tc_budget` (within `M16_TC_MAX_ULP` ordinal
//! BF16 ULP, or an absolute error under `m16_tc_acc_floor(K, row_rms)` =
//! 8 * u32 * sqrt(K) * row RMS), through `compare_m16_tc_block`, plus `rel_rms <= 1e-3` over
//! the block. The lever is declared per target (`lm_head_m16_tc` in each HARDWARE.toml).
//!
//! N is odd, so the last CTA is partial at both tile widths; the output guards and the
//! partial-tail control cover it. The run also times both kernels against
//! `dense_gemv_bf16_batchm`, with `synchronize` and host `Instant` over `REPS` reps.
//!
//! The weight takes 2.54 GB of host and of device memory, generated from 1.27e9 LCG draws.
//! `dense_gemm_m16_bf16` exists only in `kernels/hopper/common/dense_gemm_m16_bf16.cu`.
//!
//! Run on an H100:
//!     cargo run --release -p metrale-model-arch --example native_bf16_lm_head_m16_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::dense_ffn::m16_tc::oracle::{
    M16_TC_MAX_ULP, M16TcDiff, compare_m16_tc_block, m16_tc_acc_floor,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;
use std::time::Instant;

/// 2026-09-25: Qwen3.8-27B's hidden size, the head's reduction depth.
const K: usize = 5120;
/// 2026-09-25: Odd, so the last CTA at either tile width is partial.
const N: usize = 248_077;
const MAX_M: usize = 16;
/// 2026-09-25: The two CTA widths, from the launcher: 7,753 and 3,877 CTAs at this N, both
/// ending on the same 13-column tail because 248,064 is a multiple of both.
const N_TILE: usize = ops::DENSE_GEMM_M16_BF16_N_TILE as usize;
const N_TILE_WIDE: usize = ops::DENSE_GEMM_M16_BF16_N_TILE_WIDE as usize;
const GUARD: usize = 64;
const REPS: u32 = 20;
const WARMUP: u32 = 3;
/// 2026-09-25: Block-level relative-RMS gate; the per-element budget is in
/// `dense_ffn_m16_tc_oracle`.
const REL_RMS_GATE: f64 = 1e-3;
/// 2026-09-25: The baseline the speed line compares with: `dense_gemv_bf16_batchm` at M = 16 on
/// this head, 3.571 ms, measured with nsys on one H100 on 2026-09-11.
const NSYS_BATCHM_MS: f64 = 3.571;
/// 2026-09-25: The speed target at M = 16, in ms and GB/s; printed only.
const TARGET_MS: f64 = 1.3;
const TARGET_GBS: f64 = 1_950.0;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// 2026-09-25: A BF16 value in [-1, 1] on a 1/1024 grid.
    fn bf16_bits(&mut self) -> u16 {
        bf16::from_f32(((self.next() % 2049) as f32 - 1024.0) / 1024.0).to_bits()
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// 2026-09-25: Fill `dst` with BF16 draws in place, rather than collecting 1.3e9 draws into a
/// growing vector.
fn fill_bf16(rng: &mut Rng, dst: &mut [u8]) {
    for slot in dst.chunks_exact_mut(2) {
        slot.copy_from_slice(&rng.bf16_bits().to_le_bytes());
    }
}

/// 2026-09-25: Print up to 16 rejected elements with their coordinates, `|reference|` relative
/// to the row's RMS, the floor they missed, and their CTA with a tail flag: the size relative to
/// the row separates a cancellation tail from a defect, and the last, partial CTA is the defect
/// class the statistics alone cannot exclude.
fn report_outliers(label: &str, d: &M16TcDiff, n_tile: usize) {
    for o in d.over_budget.iter().take(16) {
        let relative = if o.row_rms > 0.0 {
            f64::from(o.reference).abs() / o.row_rms
        } else {
            f64::NAN
        };
        println!(
            "  OVER_BUDGET {label} (m={}, n={}) reference={:+.9e} actual={:+.9e} \
             ulp={} |ref|/row_rms={relative:.3e} row_rms={:.4} floor={:.6e} \
             cta={} of {} (tail={}) budget={M16_TC_MAX_ULP} ULP or the floor",
            o.row,
            o.col,
            o.reference,
            o.actual,
            o.ulp,
            o.row_rms,
            m16_tc_acc_floor(K, o.row_rms),
            o.col / n_tile,
            N.div_ceil(n_tile),
            o.col >= (N / n_tile) * n_tile,
        );
    }
    if d.over_budget.len() > 16 {
        println!("  … and {} more", d.over_budget.len() - 16);
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

/// 2026-09-25: One arm of the comparison: a name, its launcher and its handle. Both kernels
/// take the same arguments, so the arm is data, not a branch.
struct Arm {
    label: &'static str,
    gemm: ops::DenseM16Bf16Gemm,
    kernel: KernelHandle,
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let scalar = gpu.kernel("gemv", "dense_gemv_bf16")?;
    let batchm = gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?;
    let tc = gpu.kernel("dense_gemm_m16_bf16", "dense_gemm_m16_bf16")?;
    let tc_n64 = gpu.kernel("dense_gemm_m16_bf16", "dense_gemm_m16_bf16_n64")?;

    let mut rng = Rng(0x0927_167C_2026);
    println!(
        "generating [{N}, {K}] BF16 weights ({:.2} GB) — this takes a moment",
        (N * K * 2) as f64 / 1e9
    );
    let mut weights = vec![0_u8; N * K * 2];
    fill_bf16(&mut rng, &mut weights);
    let mut acts = vec![0_u8; MAX_M * K * 2];
    fill_bf16(&mut rng, &mut acts);

    let weight = DenseWeight {
        weight: upload(&gpu, &weights)?,
    };
    let input = upload(&gpu, &acts)?;
    drop(weights);

    let out_bytes = MAX_M * N * 2;
    let sentinel = vec![0x5a_u8; out_bytes + 2 * GUARD];
    let scalar_base = upload(&gpu, &sentinel)?;
    let tc_base = upload(&gpu, &sentinel)?;
    let n64_base = upload(&gpu, &sentinel)?;
    let batchm_base = upload(&gpu, &sentinel)?;
    let (scalar_out, tc_out, n64_out, batchm_out) = (
        scalar_base.offset(GUARD),
        tc_base.offset(GUARD),
        n64_base.offset(GUARD),
        batchm_base.offset(GUARD),
    );
    let weight_gb = (N * K * 2) as f64 / 1e9;

    let arms = [
        Arm {
            label: "m16_tc",
            gemm: ops::dense_gemm_m16_bf16,
            kernel: tc,
        },
        Arm {
            label: "n64",
            gemm: ops::dense_gemm_m16_bf16_n64,
            kernel: tc_n64,
        },
    ];
    let mut failures = 0_usize;

    for m in [5_usize, 8, 13, 16] {
        gpu.copy_h2d(&sentinel, scalar_base)?;
        gpu.copy_h2d(&sentinel, tc_base)?;
        gpu.copy_h2d(&sentinel, n64_base)?;
        // 2026-09-25: The reference: one scalar `dense_gemv_bf16` per row.
        for row in 0..m {
            ops::dense_gemv(
                &gpu,
                scalar,
                input.offset(row * K * 2),
                &weight,
                scalar_out.offset(row * N * 2),
                N as u32,
                K as u32,
                0,
            )?;
        }
        let run = |arm: &Arm, out: DevicePtr| {
            (arm.gemm)(
                &gpu, arm.kernel, input, &weight, out, m as u32, N as u32, K as u32, K as u32,
                N as u32, 0,
            )
        };
        run(&arms[0], tc_out)?;
        run(&arms[1], n64_out)?;
        gpu.synchronize(0)?;

        let mut baseline = vec![0_u8; sentinel.len()];
        let mut observed = vec![0_u8; sentinel.len()];
        let mut observed_n64 = vec![0_u8; sentinel.len()];
        gpu.copy_d2h(scalar_base, &mut baseline)?;
        gpu.copy_d2h(tc_base, &mut observed)?;
        gpu.copy_d2h(n64_base, &mut observed_n64)?;
        let bytes = m * N * 2;
        let d = compare_m16_tc_block(
            &observed[GUARD..GUARD + bytes],
            &baseline[GUARD..GUARD + bytes],
            N,
            K,
        );
        let d64 = compare_m16_tc_block(
            &observed_n64[GUARD..GUARD + bytes],
            &baseline[GUARD..GUARD + bytes],
            N,
            K,
        );

        // 2026-09-25: Nothing outside [M, N] may be written: the leading sentinel and everything
        // after row M must be untouched, for both kernels. The last CTA is always partial at this
        // N, so the trailing bytes are the column mask's only witness.
        let guards_intact = observed[..GUARD] == sentinel[..GUARD]
            && observed[GUARD + bytes..] == sentinel[GUARD + bytes..]
            && observed_n64[..GUARD] == sentinel[..GUARD]
            && observed_n64[GUARD + bytes..] == sentinel[GUARD + bytes..];

        let tc_ms = time_ms(&gpu, || run(&arms[0], tc_out))?;
        let n64_ms = time_ms(&gpu, || run(&arms[1], n64_out))?;
        let batchm_ms = time_ms(&gpu, || {
            ops::dense_gemv_batchm(
                &gpu, batchm, input, &weight, batchm_out, m as u32, N as u32, K as u32, N as u32, 0,
            )
        })?;
        let gbs = |ms: f64| weight_gb / (ms / 1e3);
        let ok = d.over_budget.is_empty()
            && d64.over_budget.is_empty()
            && d.rel_rms <= REL_RMS_GATE
            && d64.rel_rms <= REL_RMS_GATE
            && guards_intact;
        println!(
            "lm_head M={m:<3} N={N} K={K} rms={rms:.3} max_ulp={ulp} over_budget={ob} \
             over_ulp_only={ou} sign_flips={sf} max_abs={ma:.9} rel_rms={rr:.3e} \
             n64_over_budget={ob64} n64_max_ulp={ulp64} guards={g} | \
             m16_tc {tc_ms:.3}ms ({tcg:.1} GB/s) \
             vs n64 {n64_ms:.3}ms ({n64g:.1} GB/s) = {sp0:.2}x \
             vs batchm {batchm_ms:.3}ms ({bmg:.1} GB/s) = {sp1:.2}x  {verdict}",
            rms = d.rms,
            ulp = d.max_ulp,
            ob = d.over_budget.len(),
            ou = d.over_ulp_only,
            sf = d.sign_flips,
            ma = d.max_abs,
            rr = d.rel_rms,
            ob64 = d64.over_budget.len(),
            ulp64 = d64.max_ulp,
            g = if guards_intact { "ok" } else { "CLOBBERED" },
            tcg = gbs(tc_ms),
            n64g = gbs(n64_ms),
            bmg = gbs(batchm_ms),
            sp0 = n64_ms / tc_ms,
            sp1 = batchm_ms / tc_ms,
            verdict = if ok { "PASS" } else { "FAIL" },
        );
        report_outliers(arms[0].label, &d, N_TILE);
        report_outliers(arms[1].label, &d64, N_TILE_WIDE);
        if m == MAX_M {
            // 2026-09-25: The speed line is printed, not asserted: the numerics are this file's
            // pass/fail, and timing on a busy box would make it flaky.
            let best = tc_ms.min(n64_ms);
            println!(
                "TARGET M=16: {TARGET_MS} ms / {TARGET_GBS} GB/s — best arm {best:.3} ms \
                 ({:.1} GB/s), {:.2}x the round-7 nsys baseline of {NSYS_BATCHM_MS} ms: {}",
                gbs(best),
                NSYS_BATCHM_MS / best,
                if best <= TARGET_MS { "MET" } else { "MISSED" },
            );
        }
        if !ok {
            failures += 1;
        }
    }

    // 2026-09-25: Self-checks: the comparison must refuse known-bad blocks, so a pass cannot
    // come from a vacuous comparison. One adds three ULP to a value above 1 (which the absolute
    // floor must not rescue); one copies row 8 over row 9 (a row or pitch defect).
    let mut good = vec![0_u8; sentinel.len()];
    gpu.copy_d2h(scalar_base, &mut good)?;
    let rows = &good[GUARD..GUARD + MAX_M * N * 2];
    let mut bad = good.clone();
    let idx = (GUARD..GUARD + N * 2)
        .step_by(2)
        .find(|i| {
            bf16::from_bits(u16::from_le_bytes([good[*i], good[*i + 1]]))
                .to_f32()
                .abs()
                > 1.0
        })
        .expect("baseline has a value above 1.0");
    let bits = u16::from_le_bytes([good[idx], good[idx + 1]]);
    bad[idx..idx + 2].copy_from_slice(&bits.wrapping_add(3).to_le_bytes());
    let caught = !compare_m16_tc_block(&bad[GUARD..GUARD + MAX_M * N * 2], rows, N, K)
        .over_budget
        .is_empty();
    println!("KNOWN_BAD three-ULP mutation on a |value| > 1: refused={caught}");
    ensure!(
        caught,
        "comparison oracle admitted a three-ULP mutation above the accumulation floor"
    );
    let mut shifted = good.clone();
    let (src, dst) = (GUARD + 8 * N * 2, GUARD + 9 * N * 2);
    let row8 = good[src..src + N * 2].to_vec();
    shifted[dst..dst + N * 2].copy_from_slice(&row8);
    let caught_row = !compare_m16_tc_block(&shifted[GUARD..GUARD + MAX_M * N * 2], rows, N, K)
        .over_budget
        .is_empty();
    println!("KNOWN_BAD misplaced output row (row 9 <- row 8): refused={caught_row}");
    ensure!(
        caught_row,
        "comparison oracle admitted a misplaced output row — the absolute floor is too wide"
    );
    // 2026-09-25: The partial-tail control. Both kernels end on the same 13-column tail, so their
    // agreement says nothing about the last CTA; the comparison is served the tail columns
    // copied from 13 columns to their left, and must refuse them.
    let tail = N - (N / N_TILE) * N_TILE;
    let mut wrapped = good.clone();
    let (src, dst) = (GUARD + (N - 2 * tail) * 2, GUARD + (N - tail) * 2);
    let moved = good[src..src + tail * 2].to_vec();
    wrapped[dst..dst + tail * 2].copy_from_slice(&moved);
    let caught_tail = !compare_m16_tc_block(&wrapped[GUARD..GUARD + MAX_M * N * 2], rows, N, K)
        .over_budget
        .is_empty();
    println!(
        "KNOWN_BAD partial-tail store (last {tail} columns of the 7,753rd CTA shifted): \
         refused={caught_tail}"
    );
    ensure!(
        caught_tail,
        "comparison oracle admitted a shifted partial-CTA tail — the round-9 tail \
         hypothesis would have been unfalsifiable"
    );

    ensure!(
        failures == 0,
        "{failures} M cases exceeded the {M16_TC_MAX_ULP}-ULP / accumulation-floor criterion \
         or the {REL_RMS_GATE} rel_rms budget, or broke a guard"
    );
    println!(
        "ALL PASS: real Qwen3.8-27B BF16 lm_head [{N}, {K}], dense_gemm_m16_bf16 AND \
         dense_gemm_m16_bf16_n64 at M 5/8/13/16 within {M16_TC_MAX_ULP} BF16 ULP (or the \
         accumulation floor) and {REL_RMS_GATE} rel_rms of the scalar dense_gemv_bf16, \
         [M,N] bounds intact"
    );
    Ok(())
}
