// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host simulation of the arithmetic of the two Hopper GDN prefill
//! twins, which build for sm_90a only. Each algorithm is reproduced in the
//! precision its kernel uses; the twin and the parent both run against an f64
//! reference on the same fixture, and the twin must land within 1.25x of the
//! parent's error.
//!
//! Modelled exactly: bf16 operand limbs, f32 accumulation, the order of the
//! limb products, the blocked solve's block order and the explicit 16x16
//! diagonal inverse. The reduction inside one `mma.sync.m16n8k16` is
//! hardware-defined and is modelled as an ascending f32 sum, so these numbers
//! are the algorithm's error budget, not a substitute for
//! `native_gdn_prefill_remnants_microtest`.
//!
//! The fixture: a fixed LCG, gates in [0.80, 0.999], beta in [0, 1], keys and
//! values in [-0.5, 0.5] stored as bf16. Serving L2-normalises GDN keys, which
//! bounds the Gram by 1; here `<k_l, k_i>` has an rms near 0.94, so `(I + L)`
//! is worse conditioned than in serving.
//!
//! Owner: model-layers ops (GDN).
//! Invariants: none beyond the types.

use half::bf16;

const KD: usize = 128;
const VD: usize = 128;
const C: usize = 64;
const BLK: usize = 16;
const NB: usize = C / BLK;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn r(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
}

fn bf(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
/// 2026-09-25: hi = bf16(x), lo = bf16(x - hi), as `gdnh_split` in
/// `gdn_prefill_hopper.cuh`.
fn limbs(x: f32) -> (f32, f32) {
    let h = bf(x);
    (h, bf(x - h))
}

fn rel_rms(got: &[f32], want: &[f64]) -> f64 {
    let (mut se, mut sr) = (0.0f64, 0.0f64);
    for (a, b) in got.iter().zip(want.iter()) {
        se += (*a as f64 - b) * (*a as f64 - b);
        sr += b * b;
    }
    if sr > 0.0 { (se / sr).sqrt() } else { 0.0 }
}

/// 2026-09-25: One `mma.sync` pass: `acc[m][n] += SUM_k a[m][k] * b[k][n]`,
/// f32 accumulate, each operand taken from the chosen bf16 limb.
fn mma_pass(
    acc: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    al: bool,
    bl: bool,
) {
    for i in 0..m {
        for j in 0..n {
            let mut s = acc[i * n + j];
            for t in 0..k {
                let (a_hi, a_lo) = limbs(a[i * k + t]);
                let (b_hi, b_lo) = limbs(b[t * n + j]);
                s += if al { a_lo } else { a_hi } * if bl { b_lo } else { b_hi };
            }
            acc[i * n + j] = s;
        }
    }
}

/// 2026-09-25: `wu`'s three-product limb scheme: Ah.Bh + Ah.Bl + Al.Bh, in
/// that order.
fn mma_x2(acc: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    mma_pass(acc, a, b, m, k, n, false, false);
    mma_pass(acc, a, b, m, k, n, false, true);
    mma_pass(acc, a, b, m, k, n, true, false);
}

// 2026-09-25: The fixture: one chunk's L, and the two right-hand sides.

struct Chunk {
    l: Vec<f32>,   // 2026-09-25: [C][C], strict lower; the L both kernels build.
    rhs: Vec<f32>, // 2026-09-25: [C][VD], beta_i * V.
    uc: Vec<f32>,  // 2026-09-25: [C][VD], bf16-valued.
    kq: Vec<f32>,  // 2026-09-25: [C][C], exp(gc_i - gc_l) * <q_i, k_l> for l <= i.
}

fn fixture(seed: u64) -> Chunk {
    let mut r = Lcg(seed);
    let key: Vec<f32> = (0..C * KD).map(|_| bf(r.r(-0.5, 0.5) as f32)).collect();
    let query: Vec<f32> = (0..C * KD).map(|_| bf(r.r(-0.5, 0.5) as f32)).collect();
    let val: Vec<f32> = (0..C * VD).map(|_| bf(r.r(-0.5, 0.5) as f32)).collect();
    let beta: Vec<f32> = (0..C).map(|_| r.r(0.0, 1.0) as f32).collect();
    let mut gc = vec![0.0f32; C];
    let mut a = 0.0f32;
    for g in gc.iter_mut() {
        a += r.r(0.80, 0.999).ln() as f32;
        *g = a;
    }
    // 2026-09-25: Gram in f32, ascending k.
    let gram = |x: &[f32], y: &[f32], i: usize, l: usize| -> f32 {
        let mut s = 0.0f32;
        for t in 0..KD {
            s += x[i * KD + t] * y[l * KD + t];
        }
        s
    };
    let mut l = vec![0.0f32; C * C];
    let mut kq = vec![0.0f32; C * C];
    for i in 0..C {
        for j in 0..C {
            if j < i {
                l[i * C + j] = beta[i] * (gc[i] - gc[j]).exp() * gram(&key, &key, i, j);
            }
            if j <= i {
                kq[i * C + j] = (gc[i] - gc[j]).exp() * gram(&query, &key, i, j);
            }
        }
    }
    let mut rhs = vec![0.0f32; C * VD];
    for i in 0..C {
        for v in 0..VD {
            rhs[i * VD + v] = beta[i] * val[i * VD + v];
        }
    }
    Chunk {
        l,
        rhs,
        uc: val,
        kq,
    }
}

// 2026-09-25: The three solvers.

/// 2026-09-25: The oracle: `(I + L) x = b` by plain forward substitution in
/// f64.
fn reference_solve(ch: &Chunk) -> Vec<f64> {
    let mut x = vec![0.0f64; C * VD];
    for v in 0..VD {
        for i in 0..C {
            let mut s = ch.rhs[i * VD + v] as f64;
            for j in 0..i {
                s -= ch.l[i * C + j] as f64 * x[j * VD + v];
            }
            x[i * VD + v] = s;
        }
    }
    x
}

/// 2026-09-25: The parent: right-looking blocked forward substitution,
/// RL_BLK = 16, one thread per column, f32 throughout, bf16 on store, as in
/// `gated_delta_rule_recompute_wu`'s two solve passes.
fn parent_solve(ch: &Chunk) -> Vec<f32> {
    let mut out = vec![0.0f32; C * VD];
    for v in 0..VD {
        let mut acc: Vec<f32> = (0..C).map(|i| ch.rhs[i * VD + v]).collect();
        for jb in (0..C).step_by(BLK) {
            let mut xb = [0.0f32; BLK];
            for r in 0..BLK {
                let mut x = acc[jb + r];
                for q in 0..r {
                    x -= ch.l[(jb + r) * C + jb + q] * xb[q];
                }
                xb[r] = x;
                acc[jb + r] = x;
                out[(jb + r) * VD + v] = bf(x);
            }
            for i in jb + BLK..C {
                let mut a = acc[i];
                for (q, xq) in xb.iter().enumerate() {
                    a -= ch.l[i * C + jb + q] * xq;
                }
                acc[i] = a;
            }
        }
    }
    out
}

/// 2026-09-25: The twin: blocked solve with an explicit f32 16x16 diagonal
/// inverse applied by MMA, and two bf16 limbs on every MMA operand, as in
/// `gdn_recompute_wu_hopper.cu` steps (2) and (3).
fn twin_solve(ch: &Chunk) -> Vec<f32> {
    // 2026-09-25: (2) T_jj = (I + L_jj)^-1, f32 forward substitution per
    // column.
    let mut t = vec![0.0f32; NB * BLK * BLK];
    for j in 0..NB {
        for c in 0..BLK {
            t[j * BLK * BLK + c * BLK + c] = 1.0;
            for r in c + 1..BLK {
                let mut s = 0.0f32;
                for m in c..r {
                    s -= ch.l[(j * BLK + r) * C + j * BLK + m] * t[j * BLK * BLK + m * BLK + c];
                }
                t[j * BLK * BLK + r * BLK + c] = s;
            }
        }
    }
    // 2026-09-25: -L for the off-diagonal updates, because an MMA only
    // accumulates.
    let neg: Vec<f32> = ch.l.iter().map(|x| -x).collect();
    // 2026-09-25: (3) The solve. Every warp holds the same [64][n] panel
    // shape and the warps partition the columns, so one panel of VD columns
    // is the whole computation.
    let mut x = ch.rhs.clone();
    for j in 0..NB {
        let bj: Vec<f32> = x[j * BLK * VD..(j + 1) * BLK * VD].to_vec();
        let mut xj = vec![0.0f32; BLK * VD];
        mma_x2(
            &mut xj,
            &t[j * BLK * BLK..(j + 1) * BLK * BLK],
            &bj,
            BLK,
            BLK,
            VD,
        );
        x[j * BLK * VD..(j + 1) * BLK * VD].copy_from_slice(&xj);
        for i in j + 1..NB {
            let mut blk = vec![0.0f32; BLK * BLK];
            for r in 0..BLK {
                blk[r * BLK..(r + 1) * BLK]
                    .copy_from_slice(&neg[(i * BLK + r) * C + j * BLK..][..BLK]);
            }
            let mut acc: Vec<f32> = x[i * BLK * VD..(i + 1) * BLK * VD].to_vec();
            mma_x2(&mut acc, &blk, &xj, BLK, BLK, VD);
            x[i * BLK * VD..(i + 1) * BLK * VD].copy_from_slice(&acc);
        }
    }
    x.iter().map(|v| bf(*v)).collect()
}

// 2026-09-25: 1. The blocked solve lands on the parent's own bf16 floor.

#[test]
fn the_blocked_solve_matches_the_parent_within_the_bf16_storage_floor() {
    for seed in [0x0928_A11A_u64, 0x0928_B22B, 0x0928_C33C] {
        let ch = fixture(seed);
        let want = reference_solve(&ch);
        let par = rel_rms(&parent_solve(&ch), &want);
        let twin = rel_rms(&twin_solve(&ch), &want);
        // 2026-09-25: The parent accumulates in f32 and only its store rounds,
        // so its deviation is the bf16 storage floor of this output. The twin
        // must land within 1.25x of it, the bound
        // `native_gdn_prefill_remnants_microtest` applies on the device.
        assert!(
            twin <= 1.25 * par,
            "seed {seed:#x}: blocked solve rel_rms {twin:e} against the parent's \
             {par:e}; two bf16 limbs must keep the solve on the storage floor"
        );
        // 2026-09-25: The floor itself must be a bf16 floor, not a larger error
        // both arms share.
        assert!(
            par < 5e-3,
            "seed {seed:#x}: parent rel_rms {par:e} is not a bf16 floor"
        );
    }
}

/// 2026-09-25: Negative control: the test above is evidence for the limb
/// scheme only if one limb fails it. Same solve, `Ah.Bh` only.
#[test]
fn a_single_bf16_limb_does_not_meet_the_contract() {
    let ch = fixture(0x0928_A11A);
    let want = reference_solve(&ch);
    let par = rel_rms(&parent_solve(&ch), &want);
    let mut t = vec![0.0f32; NB * BLK * BLK];
    for j in 0..NB {
        for c in 0..BLK {
            t[j * BLK * BLK + c * BLK + c] = 1.0;
            for r in c + 1..BLK {
                let mut s = 0.0f32;
                for m in c..r {
                    s -= ch.l[(j * BLK + r) * C + j * BLK + m] * t[j * BLK * BLK + m * BLK + c];
                }
                t[j * BLK * BLK + r * BLK + c] = s;
            }
        }
    }
    let neg: Vec<f32> = ch.l.iter().map(|x| -x).collect();
    let mut x = ch.rhs.clone();
    for j in 0..NB {
        let bj: Vec<f32> = x[j * BLK * VD..(j + 1) * BLK * VD].to_vec();
        let mut xj = vec![0.0f32; BLK * VD];
        mma_pass(
            &mut xj,
            &t[j * BLK * BLK..(j + 1) * BLK * BLK],
            &bj,
            BLK,
            BLK,
            VD,
            false,
            false,
        );
        x[j * BLK * VD..(j + 1) * BLK * VD].copy_from_slice(&xj);
        for i in j + 1..NB {
            let mut blk = vec![0.0f32; BLK * BLK];
            for r in 0..BLK {
                blk[r * BLK..(r + 1) * BLK]
                    .copy_from_slice(&neg[(i * BLK + r) * C + j * BLK..][..BLK]);
            }
            let mut acc: Vec<f32> = x[i * BLK * VD..(i + 1) * BLK * VD].to_vec();
            mma_pass(&mut acc, &blk, &xj, BLK, BLK, VD, false, false);
            x[i * BLK * VD..(i + 1) * BLK * VD].copy_from_slice(&acc);
        }
    }
    let one: Vec<f32> = x.iter().map(|v| bf(*v)).collect();
    let got = rel_rms(&one, &want);
    assert!(
        got > 1.25 * par,
        "one bf16 limb measured {got:e} against the parent's {par:e}; if a \
         single limb were enough, the kernel's second and third products are \
         MMA issue spent on nothing and the header's claim is wrong"
    );
}

// 2026-09-25: 2. The masked triangular product.

/// 2026-09-25: `fwd_o`'s triangular product three ways: the parent's f32 chain
/// over `l <= i`, the twin's masked square on two bf16 limbs of `kq~`, and f64.
#[test]
fn the_masked_triangular_product_matches_the_parent() {
    let ch = fixture(0x0928_F0F0);
    let mut want = vec![0.0f64; C * VD];
    let mut par = vec![0.0f32; C * VD];
    for i in 0..C {
        for v in 0..VD {
            let mut sw = 0.0f64;
            let mut sp = 0.0f32;
            for l in 0..=i {
                sw += ch.kq[i * C + l] as f64 * ch.uc[l * VD + v] as f64;
                sp += ch.kq[i * C + l] * ch.uc[l * VD + v];
            }
            want[i * VD + v] = sw;
            // 2026-09-25: Both kernels store bf16, so both are rounded.
            par[i * VD + v] = bf(sp);
        }
    }
    // 2026-09-25: The twin: `kq~` is zero above `i`, so the MMA over the full
    // 64-wide square is the same sum. Two limbs on `kq~`; `uc` is already bf16,
    // so one limb is exact.
    let mut acc = vec![0.0f32; C * VD];
    mma_pass(&mut acc, &ch.kq, &ch.uc, C, C, VD, false, false);
    mma_pass(&mut acc, &ch.kq, &ch.uc, C, C, VD, true, false);
    let twin: Vec<f32> = acc.iter().map(|v| bf(*v)).collect();

    let rp = rel_rms(&par, &want);
    let rt = rel_rms(&twin, &want);
    assert!(
        rt <= 1.25 * rp,
        "masked square rel_rms {rt:e} against the parent's {rp:e}"
    );
    assert!(rp < 5e-3, "parent rel_rms {rp:e} is not a bf16 floor");
}

/// 2026-09-25: Negative control for the mask: nonzero values above the
/// diagonal must move the result.
#[test]
fn an_unmasked_gram_breaks_the_triangular_product() {
    let ch = fixture(0x0928_F0F0);
    let mut want = vec![0.0f64; C * VD];
    for i in 0..C {
        for v in 0..VD {
            let mut s = 0.0f64;
            for l in 0..=i {
                s += ch.kq[i * C + l] as f64 * ch.uc[l * VD + v] as f64;
            }
            want[i * VD + v] = s;
        }
    }
    // 2026-09-25: Fill the strictly-upper half, as a fold without the `l <= i`
    // test would.
    let mut bad = ch.kq.clone();
    let mut r = Lcg(0xBAD_0928);
    for i in 0..C {
        for l in i + 1..C {
            bad[i * C + l] = r.r(-1.0, 1.0) as f32;
        }
    }
    let mut acc = vec![0.0f32; C * VD];
    mma_pass(&mut acc, &bad, &ch.uc, C, C, VD, false, false);
    mma_pass(&mut acc, &bad, &ch.uc, C, C, VD, true, false);
    let got: Vec<f32> = acc.iter().map(|v| bf(*v)).collect();
    assert!(
        rel_rms(&got, &want) > 0.1,
        "an unmasked upper triangle must move the result"
    );
}
