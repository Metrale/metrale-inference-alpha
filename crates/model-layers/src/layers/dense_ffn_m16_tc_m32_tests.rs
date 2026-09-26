// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host simulation of the `w8a16_gemm_m16` two-halves rung at M=32, graded by the oracle.
//!
//! Owner: model-layers (dense FFN).
//! Invariants: none beyond the types.
//!
//! Two things are pinned. The byte offsets [`halves_route`] applies (`first * k`
//! on the input, `first * n` on the output, as in
//! `DenseFfnLayer::w8a16_m16_tc_proj`) give every row bit for bit what a
//! launch of its own gives, and a wrong input or output offset is shown to
//! break that. And [`within_m16_tc_budget`] admits outputs that cancelled to
//! near zero while still rejecting an error of structural size.
//!
//! [`gemv_reference`] models `w8a16_gemv`'s summation order (64 lanes striding
//! by 64 over 16-wide K chunks, a shfl butterfly per warp, then the two-warp
//! add); [`mma_row`] models `w8a16_gemm_m16`'s (eight 16-wide MMA sub-steps per
//! 128-wide scale block into an unscaled FP32 accumulator, folded once per
//! block onto the outer one). The MMA's internal 16-product order is
//! unspecified hardware, so the sequential model is one possible order.
//!
//! K is the real 5120 (40 whole scale blocks); N is a 128-column sample.

use super::{M16TcPlan, bf16_ord, m16_tc_acc_floor, m16_tc_plan, within_m16_tc_budget};
use half::bf16;

/// 2026-09-25: The reduction depth, 40 whole 128-wide FP8 scale blocks.
const K: usize = 5120;
/// 2026-09-25: Sampled output width, one whole scale block.
const N: usize = 128;
const M: usize = 32;
/// 2026-09-25: The microtest's seed (`0x927_16_7C_2026` in
/// `native_fp8_ffn_m16_tc_microtest.rs`), written with equal-sized digit
/// groups for clippy.
const SEED: u64 = 0x0927_167C_2026;
const FP8_BLOCK: usize = 128;

/// 2026-09-25: The microtest's LCG (`examples/common/m16_tc_compare.rs`).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
}

/// 2026-09-25: E4M3 -> f32, with the two NaN bytes (0x7F/0xFF) decoded to
/// +-0; the generator never draws them.
fn e4m3(byte: u8) -> f32 {
    if byte & 0x7F == 0x7F {
        return 0.0;
    }
    let sign = if byte & 0x80 != 0 { -1.0_f32 } else { 1.0 };
    let exp = i32::from((byte >> 3) & 0xF);
    let mant = f32::from(byte & 0x7) / 8.0;
    if exp == 0 {
        sign * mant * 2.0_f32.powi(-6)
    } else {
        sign * (1.0 + mant) * 2.0_f32.powi(exp - 7)
    }
}

struct Fixture {
    /// 2026-09-25: `[N, K]` FP8 E4M3, dequantized to f32 (exact).
    weight: Vec<f32>,
    /// 2026-09-25: `[M, K]` activations, rounded to BF16.
    acts: Vec<f32>,
    /// 2026-09-25: `[N/128, K/128]` FP32 block scales.
    scales: Vec<f32>,
}

impl Fixture {
    /// 2026-09-25: Draws in the microtest's order (weights, then activations,
    /// then scales) with the same value formulas as its `draw_inputs`.
    fn new() -> Self {
        let mut rng = Rng(SEED);
        let weight = (0..N * K)
            .map(|_| {
                let x = rng.next();
                e4m3(((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128))
            })
            .collect();
        let acts = (0..M * K)
            .map(|_| bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0).to_f32())
            .collect();
        let scales = (0..(N / FP8_BLOCK) * (K / FP8_BLOCK))
            .map(|_| ((rng.next() % 16 + 1) as f32) / 1024.0)
            .collect();
        Self {
            weight,
            acts,
            scales,
        }
    }

    fn scale(&self, col: usize, k_block: usize) -> f32 {
        self.scales[(col / FP8_BLOCK) * (K / FP8_BLOCK) + k_block]
    }
}

/// 2026-09-25: `w8a16_gemv`'s summation order: 64 lanes per output, each
/// walking 16-wide K chunks with a stride of 64, then a 32-lane butterfly per
/// warp and one add across the two warps. The block scale multiplies each
/// weight, per element.
fn gemv_reference(f: &Fixture, row: &[f32], col: usize) -> u16 {
    let chunks = K / 16;
    let mut lanes = [0.0_f32; 64];
    for (lane, acc) in lanes.iter_mut().enumerate() {
        let mut chunk = lane;
        while chunk < chunks {
            for i in 0..16 {
                let k = chunk * 16 + i;
                *acc += row[k] * (f.weight[col * K + k] * f.scale(col, k / FP8_BLOCK));
            }
            chunk += 64;
        }
    }
    let mut warps = [0.0_f32; 2];
    for (w, out) in warps.iter_mut().enumerate() {
        let mut v = [0.0_f32; 32];
        v.copy_from_slice(&lanes[w * 32..(w + 1) * 32]);
        let mut off = 16;
        while off > 0 {
            for l in 0..off {
                v[l] += v[l + off];
            }
            off >>= 1;
        }
        *out = v[0];
    }
    bf16::from_f32(warps[0] + warps[1]).to_bits()
}

/// 2026-09-25: `w8a16_gemm_m16`'s summation order: eight 16-wide sub-steps per
/// 128-K scale block into an unscaled FP32 inner accumulator, folded onto the
/// outer one once per block with that block's scale.
fn mma_row(f: &Fixture, row: &[f32], col: usize) -> u16 {
    let mut outer = 0.0_f32;
    for block in 0..K / FP8_BLOCK {
        let mut inner = 0.0_f32;
        for sub in 0..FP8_BLOCK / 16 {
            let base = block * FP8_BLOCK + sub * 16;
            for i in 0..16 {
                inner += row[base + i] * f.weight[col * K + base + i];
            }
        }
        outer += inner * f.scale(col, block);
    }
    bf16::from_f32(outer).to_bits()
}

/// 2026-09-25: One `w8a16_gemm_m16` launch: activation rows
/// `[a_off, a_off + rows)` into output rows `[c_off, c_off + rows)`, in rows.
fn launch(f: &Fixture, out: &mut [u16], rows: usize, a_off: usize, c_off: usize) {
    assert!(rows <= 16, "the kernel's M tile is 16 rows");
    for r in 0..rows {
        let row = &f.acts[(a_off + r) * K..(a_off + r + 1) * K];
        for col in 0..N {
            out[(c_off + r) * N + col] = mma_row(f, row, col);
        }
    }
}

/// 2026-09-25: The launches `m16_tc_plan` gives for `m` rows, with the offsets
/// `w8a16_m16_tc_proj` uses. `a_skew` / `c_skew` shift the second half's
/// input / output offset for the negative controls; 0 is the real route.
fn halves_route(f: &Fixture, m: usize, a_skew: isize, c_skew: isize) -> Vec<u16> {
    let mut out = vec![0_u16; M * N];
    match m16_tc_plan(m as u32, K as u32, true, true).expect("5..=32 is this tier's band") {
        M16TcPlan::Single => launch(f, &mut out, m, 0, 0),
        M16TcPlan::Halves { first } => {
            let first = first as usize;
            launch(f, &mut out, first, 0, 0);
            launch(
                f,
                &mut out,
                m - first,
                (first as isize + a_skew) as usize,
                (first as isize + c_skew) as usize,
            );
        }
    }
    out
}

/// 2026-09-25: RMS of a BF16 row, the scale the oracle's floor takes.
fn rms(block: &[u16]) -> f64 {
    let sum: f64 = block
        .iter()
        .map(|b| {
            let v = f64::from(bf16::from_bits(*b).to_f32());
            v * v
        })
        .sum();
    (sum / block.len() as f64).sqrt()
}

/// 2026-09-25: At M=32 the rung launches the 16-row kernel twice on
/// contiguous halves; every row must equal, bit for bit, a launch of its own.
#[test]
fn the_two_halves_rung_reproduces_every_row_of_a_per_row_launch() {
    let f = Fixture::new();
    let split = halves_route(&f, M, 0, 0);
    let mut direct = vec![0_u16; M * N];
    for r in 0..M {
        launch(&f, &mut direct, 1, r, r);
    }
    for r in 0..M {
        assert_eq!(
            &split[r * N..(r + 1) * N],
            &direct[r * N..(r + 1) * N],
            "row {r} differs between the two-halves route and a per-row launch"
        );
    }
}

/// 2026-09-25: Negative control for the test above: a second half with a
/// wrong input offset or a wrong output offset changes the result.
#[test]
fn a_wrong_second_half_offset_is_caught() {
    let f = Fixture::new();
    let good = halves_route(&f, M, 0, 0);
    for (label, a_skew, c_skew) in [
        ("input offset short by one row", -1_isize, 0_isize),
        ("input offset dropped entirely", -16, 0),
        ("output offset short by one row", 0, -1),
    ] {
        assert_ne!(
            good,
            halves_route(&f, M, a_skew, c_skew),
            "{label}: the offset pin would not have caught this"
        );
    }
}

/// 2026-09-25: Which activation row each launch reads into which output row
/// for `m` rows, without the MACs.
fn row_map(m: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let mut record = |rows: usize, a_off: usize, c_off: usize| {
        assert!(
            rows <= 16,
            "m={m}: a launch exceeded the kernel's 16-row M tile"
        );
        for r in 0..rows {
            pairs.push((a_off + r, c_off + r));
        }
    };
    match m16_tc_plan(m as u32, K as u32, true, true).expect("5..=32 is this tier's band") {
        M16TcPlan::Single => record(m, 0, 0),
        M16TcPlan::Halves { first } => {
            let first = first as usize;
            record(first, 0, 0);
            record(m - first, first, first);
        }
    }
    pairs
}

/// 2026-09-25: Every `m` in 5..=32 maps activation row `r` onto output row
/// `r`, once each, for every `r < m`.
#[test]
fn every_row_count_in_the_band_maps_each_row_to_itself_exactly_once() {
    for m in 5..=M {
        let pairs = row_map(m);
        assert_eq!(pairs.len(), m, "m={m}: {} rows covered", pairs.len());
        for (i, (a, c)) in pairs.iter().enumerate() {
            assert_eq!(
                (*a, *c),
                (i, i),
                "m={m}: launch slot {i} reads/writes the wrong row"
            );
        }
    }
}

/// 2026-09-25: Outputs that cancelled to |ref| 5.7e-6..1.6e-4 at a reference
/// RMS of 39.13 are outside the ordinal budget but inside the accumulation
/// floor at K=5120.
#[test]
fn the_m32_red_cell_is_a_cancelled_output_not_a_row_offset() {
    const RMS: f64 = 39.13;
    let reference = bf16::from_f32(-5.722e-6).to_bits();
    let actual = bf16::from_f32(-9.537e-6).to_bits();
    assert!(
        (bf16_ord(actual) - bf16_ord(reference)).abs() > 2,
        "the round-6 pair must be the one the ordinal budget rejects"
    );
    assert!(
        within_m16_tc_budget(actual, reference, K, RMS),
        "an output cancelled to 1.5e-7 of the matrix RMS has no relative accuracy to grade"
    );
    for (r, a) in [(-1.163_48e-4, -1.058_58e-4), (-1.564_03e-4, -1.506_81e-4)] {
        let (rb, ab) = (bf16::from_f32(r).to_bits(), bf16::from_f32(a).to_bits());
        assert!(
            within_m16_tc_budget(ab, rb, K, RMS),
            "ref={r} actual={a} is inside the accumulation floor"
        );
    }
}

/// 2026-09-25: The floor still rejects errors of structural size at the same
/// RMS, and an error of one BF16 quantum at magnitude 96 passes on the ordinal
/// budget alone.
#[test]
fn the_accumulation_floor_still_rejects_a_structural_error() {
    const RMS: f64 = 39.13;
    for (label, r, a) in [
        ("a misplaced output", 64.0_f32, 71.5_f32),
        ("three BF16 quanta at the top of the range", 96.0, 97.5),
        (
            "a cancelled output moved by a matrix-scale error",
            1.0e-5,
            0.5,
        ),
    ] {
        let (rb, ab) = (bf16::from_f32(r).to_bits(), bf16::from_f32(a).to_bits());
        assert!(
            !within_m16_tc_budget(ab, rb, K, RMS),
            "{label}: ref={r} actual={a} must still fail the budget"
        );
    }
    let (r, a) = (
        bf16::from_f32(96.0).to_bits(),
        bf16::from_f32(96.5).to_bits(),
    );
    assert_eq!(
        (bf16_ord(a) - bf16_ord(r)).abs(),
        1,
        "max_abs 0.500 is 1 ULP"
    );
    assert!(within_m16_tc_budget(a, r, K, RMS));
    assert!(
        (m16_tc_acc_floor(K, RMS) - 1.335_103e-3).abs() < 1e-8,
        "the absolute floor at the round-6 RMS and K=5120 is ~1.34e-3"
    );
}

/// 2026-09-25: End to end on the sampled geometry: every row of both halves
/// is within the oracle's budget of the scalar `w8a16_gemv` model.
#[test]
fn the_two_halves_rung_meets_the_oracle_budget_against_the_scalar_gemv() {
    let f = Fixture::new();
    let actual = halves_route(&f, M, 0, 0);
    let mut reference = vec![0_u16; M * N];
    for r in 0..M {
        let row = &f.acts[r * K..(r + 1) * K];
        for col in 0..N {
            reference[r * N + col] = gemv_reference(&f, row, col);
        }
    }
    assert!(
        rms(&reference) > 1.0,
        "the fixture must produce a real matrix scale"
    );
    let mut worst = (0_i32, 0_usize, 0_usize);
    for r in 0..M {
        let scale = rms(&reference[r * N..(r + 1) * N]);
        for col in 0..N {
            let (a, b) = (actual[r * N + col], reference[r * N + col]);
            let ulp = (bf16_ord(a) - bf16_ord(b)).abs();
            if ulp > worst.0 {
                worst = (ulp, r, col);
            }
            assert!(
                within_m16_tc_budget(a, b, K, scale),
                "row {r} col {col}: reference {} actual {} ({ulp} ordinal ULP) is outside \
                 both the 2-ULP budget and the accumulation floor",
                bf16::from_bits(b),
                bf16::from_bits(a),
            );
        }
    }
    // 2026-09-25: The worst element depends on the draw, so it is printed, not
    // asserted.
    println!(
        "worst ordinal ULP {} at row {} col {} (that row's reference RMS {:.3})",
        worst.0,
        worst.1,
        worst.2,
        rms(&reference[worst.1 * N..(worst.1 + 1) * N])
    );
}
