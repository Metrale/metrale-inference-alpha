// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Fixture, guarded allocation, f64 references and self-checks for
//! `native_gdn_prefill_remnants_microtest`, the only file that includes it.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

pub const KD: usize = 128;
pub const VD: usize = 128;
pub const NK: usize = 16;
pub const NV: usize = 48;
pub const C: usize = 64;
/// 2026-09-25: Heads, out of `NV`, that the f64 references recompute.
pub const REF_HEADS: usize = 2;
/// 2026-09-25: Sentinel tail after each [`alloc_guarded`] buffer, in bytes.
pub const GUARD: usize = 512;
pub const GUARD_BYTE: u8 = 0xA5;

pub struct Lcg(pub u64);
impl Lcg {
    pub fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    pub fn r(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
}

pub fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
pub fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
/// 2026-09-25: An output buffer followed by a `GUARD`-byte sentinel tail, which
/// [`guard_intact`] checks.
pub fn alloc_guarded(g: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    let p = g.alloc(bytes + GUARD)?;
    g.copy_h2d(&vec![GUARD_BYTE; bytes + GUARD], p)?;
    Ok(p)
}
pub fn guard_intact(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<bool> {
    let mut tail = vec![0u8; GUARD];
    g.copy_d2h(DevicePtr(p.0 + bytes as u64), &mut tail)?;
    Ok(tail.iter().all(|b| *b == GUARD_BYTE))
}
pub fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
pub fn dn_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
/// 2026-09-25: `(max_abs, rel_rms)` with rel_rms = ||a - r|| / ||r|| (0 when
/// ||r|| is 0), computed in f64.
pub fn metrics(a: &[f32], r: &[f64]) -> (f64, f64) {
    let (mut mx, mut se, mut sr) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(r.iter()) {
        let d = (*x as f64 - y).abs();
        mx = mx.max(d);
        se += d * d;
        sr += y * y;
    }
    (mx, if sr > 0.0 { (se / sr).sqrt() } else { 0.0 })
}

pub struct Case {
    pub t: usize,
    pub nt: usize,
    pub query: Vec<bf16>,
    pub key: Vec<bf16>,
    pub val: Vec<bf16>,
    pub gate: Vec<f32>,
    pub beta: Vec<f32>,
    pub h0: Vec<f32>,
}

/// 2026-09-25: Fixed-LCG fixture for `t` tokens: q, k and v in [-0.5, 0.5)
/// rounded to bf16, gates in [0.80, 0.999), beta in [0, 1), h0 in [-0.1, 0.1).
pub fn gen_case(t: usize) -> Case {
    let mut r = Lcg(0x9D8E_2026 ^ (t as u64));
    let bf = |r: &mut Lcg| bf16::from_f64(r.r(-0.5, 0.5));
    Case {
        t,
        nt: t.div_ceil(C),
        query: (0..t * NK * KD).map(|_| bf(&mut r)).collect(),
        key: (0..t * NK * KD).map(|_| bf(&mut r)).collect(),
        val: (0..t * NV * VD).map(|_| bf(&mut r)).collect(),
        gate: (0..t * NV).map(|_| r.r(0.80, 0.999) as f32).collect(),
        beta: (0..t * NV).map(|_| r.r(0.0, 1.0) as f32).collect(),
        h0: (0..NV * KD * VD).map(|_| r.r(-0.1, 0.1) as f32).collect(),
    }
}

/// 2026-09-25: f64 reference for the WY pass on heads [0, REF_HEADS): the
/// log-gate scan `gc`, the gated Gram matrix L, then forward substitution for U
/// and W. Laid out `[chunk][REF_HEADS][row][col]`, the order [`take`] gathers
/// the kernels' outputs into.
pub fn ref_wu(c: &Case) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let hr = NV / NK;
    let (mut w, mut u, mut gc) = (
        vec![0.0; c.nt * REF_HEADS * C * KD],
        vec![0.0; c.nt * REF_HEADS * C * VD],
        vec![0.0; c.nt * REF_HEADS * C],
    );
    for vh in 0..REF_HEADS {
        let kh = vh / hr;
        for ch in 0..c.nt {
            let (cs, rb) = (ch * C, ch * REF_HEADS + vh);
            let ce = (c.t - cs).min(C);
            let mut g = vec![0.0f64; C];
            let mut a = 0.0f64;
            for (i, gi) in g.iter_mut().enumerate().take(ce) {
                a += (c.gate[(cs + i) * NV + vh] as f64).max(1e-30).ln();
                *gi = a;
                gc[rb * C + i] = a;
            }
            let kv = |i: usize, d: usize| c.key[(cs + i) * NK * KD + kh * KD + d].to_f64();
            let mut l = vec![0.0f64; C * C];
            for i in 0..ce {
                for j in 0..i {
                    let gram: f64 = (0..KD).map(|d| kv(i, d) * kv(j, d)).sum();
                    l[i * C + j] = c.beta[(cs + i) * NV + vh] as f64 * (g[i] - g[j]).exp() * gram;
                }
            }
            for (n, (out, cols)) in [(0usize, VD), (1, KD)].iter().enumerate() {
                let _ = out;
                for col in 0..*cols {
                    for i in 0..ce {
                        let b = c.beta[(cs + i) * NV + vh] as f64;
                        let mut s = if n == 0 {
                            b * c.val[(cs + i) * NV * VD + vh * VD + col].to_f64()
                        } else {
                            b * g[i].exp() * kv(i, col)
                        };
                        for j in 0..i {
                            s -= l[i * C + j]
                                * if n == 0 {
                                    u[rb * C * VD + j * VD + col]
                                } else {
                                    w[rb * C * KD + j * KD + col]
                                };
                        }
                        if n == 0 {
                            u[rb * C * VD + i * VD + col] = s;
                        } else {
                            w[rb * C * KD + i * KD + col] = s;
                        }
                    }
                }
            }
        }
    }
    (w, u, gc)
}

/// 2026-09-25: f64 reference for the output pass on heads [0, REF_HEADS),
/// taking the kernels' own full-`NV` `sc`, `uc` and `gc` as inputs.
pub fn ref_fwd_o(c: &Case, sc: &[f32], uc: &[f32], gc: &[f32]) -> Vec<f64> {
    let hr = NV / NK;
    let inv = 1.0 / (KD as f64).sqrt();
    let mut o = vec![0.0f64; c.t * REF_HEADS * VD];
    for vh in 0..REF_HEADS {
        let kh = vh / hr;
        for ch in 0..c.nt {
            let (cs, base) = (ch * C, ch * NV + vh);
            let ce = (c.t - cs).min(C);
            let g = |i: usize| gc[base * C + i] as f64;
            let qk = |i: usize, l: usize| -> f64 {
                (0..KD)
                    .map(|d| {
                        c.query[(cs + i) * NK * KD + kh * KD + d].to_f64()
                            * c.key[(cs + l) * NK * KD + kh * KD + d].to_f64()
                    })
                    .sum()
            };
            for i in 0..ce {
                let kqi: Vec<f64> = (0..=i).map(|l| (g(i) - g(l)).exp() * qk(i, l)).collect();
                for v in 0..VD {
                    let mut s: f64 = (0..KD)
                        .map(|d| {
                            c.query[(cs + i) * NK * KD + kh * KD + d].to_f64()
                                * sc[base * KD * VD + d * VD + v] as f64
                        })
                        .sum();
                    s *= g(i).exp();
                    for (l, kq) in kqi.iter().enumerate() {
                        s += kq * uc[base * C * VD + l * VD + v] as f64;
                    }
                    o[((cs + i) * REF_HEADS + vh) * VD + v] = s * inv;
                }
            }
        }
    }
    o
}

/// 2026-09-25: Gather heads [0, REF_HEADS) out of a full-NV output laid out
/// `[rows][NV][per]`.
///
/// `rows` is the outer dimension only: the chunk count for `W`, `U` and `gc`,
/// the token count for `O`. The function applies `NV` itself, and its length
/// assert refuses a `rows` already multiplied by `NV` with a message that says so.
pub fn take(full: &[f32], rows: usize, per: usize) -> Vec<f32> {
    assert_eq!(
        full.len(),
        rows * NV * per,
        "take(rows={rows}, per={per}) gathers {REF_HEADS} of NV={NV} heads out \
         of a [rows][NV][per] buffer and therefore wants {} elements, but was \
         handed {}. `rows` is the OUTER dimension only — pass `c.nt` (or `t`), \
         never `c.nt * NV`: NV is applied here",
        rows * NV * per,
        full.len(),
    );
    let mut out = Vec::with_capacity(rows * REF_HEADS * per);
    for r in 0..rows {
        for vh in 0..REF_HEADS {
            let b = (r * NV + vh) * per;
            out.extend_from_slice(&full[b..b + per]);
        }
    }
    out
}

/// 2026-09-25: Every `take` the example performs, at the example's geometry.
/// `main` runs it before creating the device backend: the example requires the
/// non-default `gpu-examples` feature, so `cargo test -p metrale-model-arch`
/// does not build it and the unit tests below do not run there.
///
/// At T=256 each gathered element is checked against the `(row, head, offset)`
/// it came from. At all three T the example walks, each `(rows, per)` must
/// describe the buffer length `main` allocates.
pub fn selfcheck_take() {
    // 2026-09-25: `(what, rows, per)` as the example calls `take`: W, U and gc
    // per chunk, O per token.
    let shapes = |t: usize| {
        let nt = t.div_ceil(C);
        [
            ("W", nt, C * KD),
            ("U", nt, C * VD),
            ("gc", nt, C),
            ("O", t, VD),
        ]
    };

    for (what, rows, per) in shapes(256) {
        let full: Vec<f32> = (0..rows * NV * per).map(|i| i as f32).collect();
        let got = take(&full, rows, per);
        assert_eq!(got.len(), rows * REF_HEADS * per, "{what}: gathered length");
        for r in 0..rows {
            for vh in 0..REF_HEADS {
                for (e, x) in got[(r * REF_HEADS + vh) * per..][..per].iter().enumerate() {
                    assert_eq!(
                        *x,
                        ((r * NV + vh) * per + e) as f32,
                        "{what}: row {r} head {vh} element {e} came from the wrong stride",
                    );
                }
            }
        }
    }

    for &t in &[256usize, 1193, 4593] {
        let nt = t.div_ceil(C);
        // 2026-09-25: The four buffer lengths `main` allocates, restated here.
        let (wb, ub, gcb, outs) = (nt * NV * C * KD, nt * NV * C * VD, nt * NV * C, t * NV * VD);
        for ((what, rows, per), len) in shapes(t).into_iter().zip([wb, ub, gcb, outs]) {
            assert_eq!(
                rows * NV * per,
                len,
                "T={t} {what}: take's (rows, per) must describe the buffer main \
                 allocates",
            );
        }
    }
}

/// 2026-09-25: The known-bad control for one arm: perturb one reference element
/// and return `(perturbed, clean)` `max_abs`. The example fails the run unless
/// `perturbed > clean`.
///
/// The perturbation moves the element that holds the clean extreme by
/// `3 * max_abs_clean + 0.1 * rms(reference)`, away from the arm's value, so the
/// deviation there becomes `|d_j|` plus that amount and exceeds `max_abs_clean`
/// whenever the amount is positive. The `rms` term keeps it positive for a
/// bit-exact arm, whose clean extreme is 0.
pub fn known_bad_probe(actual: &[f32], reference: &[f64]) -> (f64, f64) {
    assert_eq!(
        actual.len(),
        reference.len(),
        "the KNOWN_BAD control scores an arm against ITS OWN reference; unequal \
         lengths mean `metrics` silently truncated one of the two and the \
         control would be measuring a prefix",
    );
    assert!(
        !reference.is_empty(),
        "an empty reference cannot be perturbed"
    );
    let (clean, _) = metrics(actual, reference);
    // 2026-09-25: The element that holds the clean extreme, and the signed
    // deviation there.
    let (mut j, mut dj) = (0usize, 0.0f64);
    for (i, (x, y)) in actual.iter().zip(reference.iter()).enumerate() {
        let d = *x as f64 - y;
        if d.abs() > dj.abs() {
            (j, dj) = (i, d);
        }
    }
    let rms = (reference.iter().map(|x| x * x).sum::<f64>() / reference.len() as f64).sqrt();
    let mag = 3.0 * clean + 0.1 * rms;
    assert!(
        mag > 0.0,
        "an all-zero reference against a bit-exact arm leaves nothing to \
         perturb, and a control that cannot trip is not a control",
    );
    let mut bad = reference.to_vec();
    // 2026-09-25: Away from the arm's value, so the deviation at j adds rather
    // than cancels.
    bad[j] -= mag * if dj < 0.0 { -1.0 } else { 1.0 };
    let (perturbed, _) = metrics(actual, &bad);
    (perturbed, clean)
}

/// 2026-09-25: Synthetic `(reference, actual)` at the example's `O` geometry
/// whose clean `max_abs` is `clean_max`, held by one element that is not
/// element 0.
///
/// The reference is uniform on [-0.35, 0.35), so `0.1 * rms` is about 2.0e-2,
/// between the T=1193 and T=4593 extremes of [`selfcheck_known_bad`]; its
/// element-0 assertion depends on that.
fn known_bad_fixture(t: usize, clean_max: f64) -> (Vec<f64>, Vec<f32>) {
    let n = t * REF_HEADS * VD;
    let mut r = Lcg(0x4B4E_4F57 ^ (t as u64));
    let reference: Vec<f64> = (0..n).map(|_| r.r(-0.35, 0.35)).collect();
    // 2026-09-25: The f64 -> f32 rounding moves each value by at most 1.5e-8,
    // far under every planted extreme.
    let mut actual: Vec<f32> = reference.iter().map(|x| *x as f32).collect();
    let ex = n / 2 + 7;
    actual[ex] = (reference[ex] + clean_max) as f32;
    (reference, actual)
}

/// 2026-09-25: [`known_bad_probe`] on synthetic data at the three T the example
/// walks, with the clean extremes listed below. `main` runs it before creating
/// the device backend, for the reason given on [`selfcheck_take`].
///
/// The probe must trip at every T. The element-0 rule,
/// `reference[0] += 0.1 * rms`, must trip at T=256 and T=1193 and must not trip
/// at T=4593.
pub fn selfcheck_known_bad() {
    for (t, clean_max) in [(256usize, 1.895e-3f64), (1193, 7.668e-3), (4593, 2.774e-2)] {
        let (reference, actual) = known_bad_fixture(t, clean_max);
        let (perturbed, clean) = known_bad_probe(&actual, &reference);
        assert!(
            (clean / clean_max - 1.0).abs() < 1e-4,
            "T={t}: the fixture must hold round 14's clean extreme {clean_max:.3e}, \
             got {clean:.3e}",
        );
        assert!(
            perturbed > clean,
            "T={t}: the KNOWN_BAD control must trip against a clean extreme of \
             {clean:.3e}, got {perturbed:.3e} — round 14's failure, at T=4593",
        );
        // 2026-09-25: The element-0 rule on the same fixture trips only while
        // `0.1 * rms` exceeds the extreme already present.
        let rms = (reference.iter().map(|x| x * x).sum::<f64>() / reference.len() as f64).sqrt();
        let mut old = reference.clone();
        old[0] += 0.1 * rms;
        let (old_perturbed, _) = metrics(&actual, &old);
        assert_eq!(
            old_perturbed > clean,
            t != 4593,
            "T={t}: round 14's `0.1 * rms` injection ({:.3e}) against a clean \
             extreme of {clean:.3e} — the scaling defect this replaces",
            0.1 * rms,
        );
    }
}

pub fn report(tag: &str, a: &[f32], r: &[f64]) -> f64 {
    let (mx, rel) = metrics(a, r);
    println!("    {tag:<22} max_abs={mx:.6e}  rel_rms={rel:.4e}");
    rel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_known_bad_control_trips_at_every_t() {
        selfcheck_known_bad();
    }

    #[test]
    fn take_gathers_the_example_geometry() {
        selfcheck_take();
    }

    /// 2026-09-25: A `rows` already multiplied by `NV` is refused by the length
    /// assert's message, not by an out-of-range slice.
    #[test]
    #[should_panic(expected = "`rows` is the OUTER dimension only")]
    fn a_pre_multiplied_rows_is_refused() {
        let (t, per) = (256usize, C * KD);
        let nt = t.div_ceil(C);
        let full = vec![0.0f32; nt * NV * per];
        let _ = take(&full, nt * NV, per);
    }

    /// 2026-09-25: At T=256, `W` holds nt*NV*C*KD = 4*48*64*128 = 1,572,864
    /// elements; with `rows = nt * NV`, the first slice past it, at r = nt and
    /// vh = 0, ends at (4*48)*8192 + 8192 = 1,581,056.
    #[test]
    fn the_round_thirteen_panic_indices_are_this_geometry() {
        let (t, per) = (256usize, C * KD);
        let nt = t.div_ceil(C);
        assert_eq!(nt * NV * per, 1_572_864);
        assert_eq!((nt * NV) * per + per, 1_581_056);
    }
}
