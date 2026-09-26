// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host half of `native_fp8_ffn_m16_tc_microtest`: the shape table
//! and input draws, the host transpose, the guard and gap audits, the per-case
//! verdict and report line, and the known-bad controls. The example keeps the
//! kernel handles, uploads, launches and timing.
//!
//! The per-element predicate (`compare_m16_tc_block` with `M16_TC_MAX_ULP` and
//! `m16_tc_acc_floor`) lives in `layers::dense_ffn::m16_tc::oracle`, which
//! `dense_ffn_m16_tc_m32_tests.rs` also calls; this module only calls it. Sign
//! flips below `M16_TC_SIGN_FLIP_BAND` (0.05) are counted separately, not graded.
//! Every rejected element is printed with its row and column.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::{Result, ensure};
use half::bf16;
use metrale_model_layers::layers::dense_ffn::m16_tc::oracle::{
    M16_TC_MAX_ULP, M16TcDiff, compare_m16_tc_block, m16_tc_acc_floor,
};

/// 2026-09-25: Qwen3.8-27B hidden size (`hidden_dim` in
/// `kernels/hopper/qwen3.8-27b/MODEL.toml`).
pub(crate) const H: usize = 5120;
/// 2026-09-25: Qwen3.8-27B FFN intermediate size (`intermediate_size`, same file).
pub(crate) const INTER: usize = 17408;
pub(crate) const MAX_M: usize = 32;
pub(crate) const GUARD: usize = 64;
/// 2026-09-25: Elements added to each activation row's pitch in the strided leg.
pub(crate) const A_PAD: usize = 8;
pub(crate) const C_PAD: usize = 64;
/// 2026-09-25: Block-level relative-RMS gate. The per-element budget is the
/// library's (`M16_TC_MAX_ULP` and `m16_tc_acc_floor`).
pub(crate) const REL_RMS_GATE: f64 = 1e-3;

pub(crate) struct Shape {
    pub(crate) name: &'static str,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

pub(crate) const SHAPES: [Shape; 2] = [
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

pub(crate) struct Rng(pub(crate) u64);
impl Rng {
    pub(crate) fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
}

/// 2026-09-25: One shape's host-drawn inputs: the FP8 weight, the BF16
/// activations for all [`MAX_M`] rows, the per-128x128 block scales, and the
/// same activations re-laid at the padded pitch the strided leg reads.
pub(crate) struct Inputs {
    pub(crate) weights: Vec<u8>,
    pub(crate) acts: Vec<u8>,
    pub(crate) scales: Vec<u8>,
    pub(crate) acts_strided: Vec<u8>,
}

pub(crate) fn draw_inputs(rng: &mut Rng, n: usize, k: usize) -> Inputs {
    // 2026-09-25: Weight bytes are 0x00..=0x7E with a random sign bit, so never
    // the E4M3 NaN codes 0x7F and 0xFF.
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
    // 2026-09-25: Strided A: the same rows, re-laid at a `k + A_PAD` pitch.
    let a_pitch = k + A_PAD;
    let mut acts_strided = vec![0x3c_u8; MAX_M * a_pitch * 2];
    for row in 0..MAX_M {
        let src = row * k * 2;
        let dst = row * a_pitch * 2;
        acts_strided[dst..dst + k * 2].copy_from_slice(&acts[src..src + k * 2]);
    }
    Inputs {
        weights,
        acts,
        scales,
        acts_strided,
    }
}

/// 2026-09-25: Transpose `[N, K]` FP8 weights to `[K, N]` and `[N/128, K/128]`
/// FP32 scales to `[K/128, N/128]`.
pub(crate) fn transpose(weights: &[u8], scales: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let mut wt = vec![0_u8; n * k];
    for row in 0..n {
        for col in 0..k {
            wt[col * n + row] = weights[row * k + col];
        }
    }
    let (nb, kb) = (n / 128, k / 128);
    let mut st = vec![0_u8; nb * kb * 4];
    for bn in 0..nb {
        for bk in 0..kb {
            let src = (bn * kb + bk) * 4;
            let dst = (bk * nb + bn) * 4;
            st[dst..dst + 4].copy_from_slice(&scales[src..src + 4]);
        }
    }
    (wt, st)
}

/// 2026-09-25: Print every element the predicate rejected, with its coordinates,
/// values, ULP distance, magnitude relative to its row's RMS, and its row's floor.
pub(crate) fn report_outliers(label: &str, d: &M16TcDiff, k: usize) {
    for o in &d.over_budget {
        let relative = if o.row_rms > 0.0 {
            f64::from(o.reference).abs() / o.row_rms
        } else {
            f64::NAN
        };
        println!(
            "  OVER_BUDGET {label} (m={}, n={}) reference={:+.9e} actual={:+.9e} \
             ulp={} |ref|/row_rms={relative:.3e} row_rms={:.4} floor={:.6e} \
             budget={M16_TC_MAX_ULP} ULP or the floor",
            o.row,
            o.col,
            o.reference,
            o.actual,
            o.ulp,
            o.row_rms,
            m16_tc_acc_floor(k, o.row_rms)
        );
    }
}

/// 2026-09-25: The bytes before `GUARD` and from `GUARD + bytes` on must still
/// be the sentinel, in both `observed` and `observed_n64`.
pub(crate) fn guards_intact(
    observed: &[u8],
    observed_n64: &[u8],
    sentinel: &[u8],
    bytes: usize,
) -> bool {
    observed[..GUARD] == sentinel[..GUARD]
        && observed[GUARD + bytes..] == sentinel[GUARD + bytes..]
        && observed_n64[..GUARD] == sentinel[..GUARD]
        && observed_n64[GUARD + bytes..] == sentinel[GUARD + bytes..]
}

pub(crate) struct StridedCheck {
    pub(crate) gaps_intact: bool,
    pub(crate) max_ulp: i32,
}

/// 2026-09-25: The strided leg's audit: the leading guard, each row's
/// `[n, c_pitch)` gap and every row from `m` on must still hold the sentinel,
/// and each row below `m` must pass `compare_m16_tc_block` against the scalar
/// baseline.
pub(crate) fn check_strided(
    strided: &[u8],
    sentinel_s: &[u8],
    baseline: &[u8],
    m: usize,
    n: usize,
    k: usize,
    c_pitch: usize,
) -> StridedCheck {
    let mut gaps_intact = strided[..GUARD] == sentinel_s[..GUARD];
    let mut max_ulp = 0_i32;
    for row in 0..MAX_M {
        let base = GUARD + row * c_pitch * 2;
        let gap = &strided[base + n * 2..base + c_pitch * 2];
        gaps_intact &= gap.iter().all(|b| *b == 0x5a);
        if row < m {
            let sd = compare_m16_tc_block(
                &strided[base..base + n * 2],
                &baseline[GUARD + row * n * 2..GUARD + (row + 1) * n * 2],
                n,
                k,
            );
            max_ulp = max_ulp.max(sd.max_ulp);
            if !sd.over_budget.is_empty() {
                gaps_intact = false;
            }
        } else {
            gaps_intact &= strided[base..base + n * 2].iter().all(|b| *b == 0x5a);
        }
    }
    StridedCheck {
        gaps_intact,
        max_ulp,
    }
}

/// 2026-09-25: The four arms' wall times for one shape/M case, in ms per rep.
pub(crate) struct Timings {
    pub(crate) tc_ms: f64,
    pub(crate) n64_ms: f64,
    pub(crate) b16_ms: f64,
    pub(crate) tile_ms: f64,
}

/// 2026-09-25: Everything one shape/M case's verdict and report line read.
pub(crate) struct Case<'a> {
    pub(crate) name: &'a str,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) passes: usize,
    pub(crate) d: &'a M16TcDiff,
    pub(crate) d64: &'a M16TcDiff,
    pub(crate) strided: &'a StridedCheck,
    pub(crate) guards_intact: bool,
    pub(crate) timings: &'a Timings,
    pub(crate) weight_gb: f64,
}

/// 2026-09-25: The case-level verdict: both instantiations inside the
/// per-element budget and the block rel-RMS gate, with every guard and gap intact.
pub(crate) fn case_passed(c: &Case) -> bool {
    c.d.over_budget.is_empty()
        && c.d64.over_budget.is_empty()
        && c.d.rel_rms <= REL_RMS_GATE
        && c.d64.rel_rms <= REL_RMS_GATE
        && c.guards_intact
        && c.strided.gaps_intact
}

/// 2026-09-25: Print the case's line, then every rejected element with its
/// coordinates, and return the verdict.
pub(crate) fn report_case(c: &Case) -> bool {
    let ok = case_passed(c);
    let (d, d64, t) = (c.d, c.d64, c.timings);
    let gbs = |ms: f64| c.weight_gb * c.passes as f64 / (ms / 1e3);
    println!(
        "{name:<8} M={m:<3} N={n} K={k} passes={passes} rms={rms:.3} \
         max_ulp={ulp} over_budget={ob} over_ulp_only={ou} sign_flips={sf} \
         max_abs={ma:.9} rel_rms={rr:.3e} strided_max_ulp={su} \
         n64_over_budget={ob64} n64_max_ulp={ulp64} guards={g} gaps={gp} | \
         m16_tc {tc_ms:.3}ms ({tcg:.1} GB/s) \
         vs n64 {n64_ms:.3}ms ({n64g:.1} GB/s) = {sp0:.2}x \
         vs batch16 {b16_ms:.3}ms ({b16g:.1} GB/s) = {sp1:.2}x \
         vs t_m128 {tile_ms:.3}ms ({tileg:.1} GB/s) = {sp2:.2}x  {verdict}",
        name = c.name,
        m = c.m,
        n = c.n,
        k = c.k,
        passes = c.passes,
        rms = d.rms,
        ulp = d.max_ulp,
        ob = d.over_budget.len(),
        ou = d.over_ulp_only,
        sf = d.sign_flips,
        ma = d.max_abs,
        rr = d.rel_rms,
        su = c.strided.max_ulp,
        ob64 = d64.over_budget.len(),
        ulp64 = d64.max_ulp,
        g = if c.guards_intact { "ok" } else { "CLOBBERED" },
        gp = if c.strided.gaps_intact {
            "ok"
        } else {
            "CLOBBERED"
        },
        tc_ms = t.tc_ms,
        n64_ms = t.n64_ms,
        b16_ms = t.b16_ms,
        tile_ms = t.tile_ms,
        tcg = gbs(t.tc_ms),
        n64g = gbs(t.n64_ms),
        b16g = gbs(t.b16_ms),
        tileg = c.weight_gb / (t.tile_ms / 1e3),
        sp0 = t.n64_ms / t.tc_ms,
        sp1 = t.b16_ms / t.tc_ms,
        sp2 = t.tile_ms / t.tc_ms,
        verdict = if ok { "PASS" } else { "FAIL" },
    );
    report_outliers("m16_tc", d, c.k);
    report_outliers("n64", d64, c.k);
    ok
}

/// 2026-09-25: Known-bad controls, once per shape, each of which
/// `compare_m16_tc_block` must refuse: a three-ULP change to the first row-0
/// element with |value| > 1, row 17 overwritten with row 16, and the tail shift
/// of [`assert_tail_control`].
///
/// `good` is the scalar baseline buffer, sentinel guards included.
pub(crate) fn assert_oracle_bites(good: &[u8], name: &str, n: usize, k: usize) -> Result<()> {
    let rows = &good[GUARD..GUARD + MAX_M * n * 2];
    let mut bad = good.to_vec();
    // 2026-09-25: The first row-0 element with |value| > 1, outside the sign-flip
    // band, moved by three ULP.
    let idx = (GUARD..GUARD + n * 2)
        .step_by(2)
        .find(|i| {
            bf16::from_bits(u16::from_le_bytes([good[*i], good[*i + 1]]))
                .to_f32()
                .abs()
                > 1.0
        })
        .expect("baseline has a value above 1.0");
    let bits = u16::from_le_bytes([good[idx], good[idx + 1]]);
    bad[idx..idx + 2].copy_from_slice(&(bits.wrapping_add(3)).to_le_bytes());
    let caught = !compare_m16_tc_block(&bad[GUARD..GUARD + MAX_M * n * 2], rows, n, k)
        .over_budget
        .is_empty();
    println!("KNOWN_BAD {name} three-ULP mutation on a |value| > 1: refused={caught}");
    ensure!(
        caught,
        "comparison oracle admitted a three-ULP mutation above the accumulation floor"
    );
    let mut shifted = good.to_vec();
    let (src, dst) = (GUARD + 16 * n * 2, GUARD + 17 * n * 2);
    let row16 = good[src..src + n * 2].to_vec();
    shifted[dst..dst + n * 2].copy_from_slice(&row16);
    let caught_row = !compare_m16_tc_block(&shifted[GUARD..GUARD + MAX_M * n * 2], rows, n, k)
        .over_budget
        .is_empty();
    println!("KNOWN_BAD {name} second-half row offset (row 17 <- row 16): refused={caught_row}");
    ensure!(
        caught_row,
        "comparison oracle admitted a misplaced output row — the absolute floor is too wide"
    );
    assert_tail_control(good, rows, name, n, k)
}

/// 2026-09-25: Tail control: the last `min(n, 16)` columns of row 0 are replaced
/// by the `min(n, 16)` columns to their left, and `compare_m16_tc_block` must
/// refuse it.
fn assert_tail_control(good: &[u8], rows: &[u8], name: &str, n: usize, k: usize) -> Result<()> {
    let tail = n.min(16);
    let mut wrapped = good.to_vec();
    let src = GUARD + (n - 2 * tail) * 2;
    let dst = GUARD + (n - tail) * 2;
    let moved = good[src..src + tail * 2].to_vec();
    wrapped[dst..dst + tail * 2].copy_from_slice(&moved);
    let caught = !compare_m16_tc_block(&wrapped[GUARD..GUARD + MAX_M * n * 2], rows, n, k)
        .over_budget
        .is_empty();
    println!("KNOWN_BAD {name} partial-tail store (last {tail} columns shifted): refused={caught}");
    ensure!(
        caught,
        "comparison oracle admitted a shifted partial-CTA tail — the round-9 tail \
         hypothesis would have been unfalsifiable"
    );
    Ok(())
}
