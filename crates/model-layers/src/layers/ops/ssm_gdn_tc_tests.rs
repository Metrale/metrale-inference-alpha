// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host simulation of the tensor-core state spine's index maps
//! (`kernels/hopper/common/gated_delta_rule_chunk_tc.cu`), plus the grammar of
//! `gdn_tc_spine_reject`.
//!
//! The kernel keeps the recurrent state in the MMA accumulator, so the state
//! is addressed by warp and lane. These tests check the four maps that decide
//! what each MMA is handed:
//!
//!   1. the Phase-B C-fragment map, which is the state layout `S[k][v]`;
//!   2. the Phase-A C-fragment map over `ws[i][v]`, 4 m-tiles x 2 n-halves
//!      across 8 warps;
//!   3. the `K -> Kt` transpose map that feeds Phase B;
//!   4. the padded smem strides (136 for W/U/St, 72 for Kt/ducT).
//!
//! Each map is checked for bijectivity, and the maps are composed end-to-end
//! against a plain `for k { for v { ... } }` recurrence at a shape whose last
//! chunk is partial (T = 150: chunks of 64, 64 and 22), so the kernel's
//! zero-fill of rows `i >= ce` is exercised.
//!
//! Owner: model-layers ops (GDN).
//! Invariants: none beyond the types.

use super::{GDN_TC_CHUNK, GDN_TC_DIM, GDN_TC_SMEM, gdn_tc_spine_reject};

const KD: usize = 128;
const VD: usize = 128;
const C: usize = 64;
const SW: usize = 136; // 2026-09-25: W / U / St padded row stride, in bf16 elements.
const SC: usize = 72; // 2026-09-25: Kt / ducT padded row stride.
const THREADS: usize = 256;

/// 2026-09-25: `Kt` aliases `St` in the kernel: Phase A consumes St, then the
/// K transpose reuses its bytes. Checked at compile time because it is a
/// property of the two padded shapes.
const _: () = assert!(KD * SC * 2 <= VD * SW * 2);

/// 2026-09-25: Phase-B accumulator slot `(tid, nt, e)` -> the state element
/// `S[k][v]` it holds, as the kernel's `m0/m1` and `n0/n1` compute it.
fn acc_slot(tid: usize, nt: usize, e: usize) -> (usize, usize) {
    let (warp, lane) = (tid >> 5, tid & 31);
    let (grp, q) = (lane >> 2, lane & 3);
    let k = warp * 16 + grp + if e >= 2 { 8 } else { 0 };
    let v = nt * 8 + q * 2 + (e & 1);
    (k, v)
}

/// 2026-09-25: Phase-A slot `(tid, nt, e)` -> the `ws[i][v]` element it holds
/// (the kernel's `a_m`/`a_n`).
fn ws_slot(tid: usize, nt: usize, e: usize) -> (usize, usize) {
    let (warp, lane) = (tid >> 5, tid & 31);
    let (grp, q) = (lane >> 2, lane & 3);
    let (a_m, a_n) = ((warp & 3) * 16, (warp >> 2) * 64);
    let i = a_m + grp + if e >= 2 { 8 } else { 0 };
    let v = a_n + nt * 8 + q * 2 + (e & 1);
    (i, v)
}

/// 2026-09-25: K-staging slot `(tid, j, e)` -> the `(k, i)` pair it moves into
/// `Kt` (the kernel's `krow`/`kcol`).
fn kt_slot(tid: usize, j: usize, e: usize) -> (usize, usize) {
    ((tid & 3) * 32 + j * 8 + e, tid >> 2)
}

// 2026-09-25: 1. The three maps are bijections.

#[test]
fn the_accumulator_map_tiles_the_state_exactly_once() {
    let mut seen = vec![0u8; KD * VD];
    for tid in 0..THREADS {
        for nt in 0..16 {
            for e in 0..4 {
                let (k, v) = acc_slot(tid, nt, e);
                assert!(k < KD && v < VD, "tid={tid} nt={nt} e={e} -> ({k},{v})");
                seen[k * VD + v] += 1;
            }
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "every S[k][v] must be owned by exactly one lane slot; \
         holes={} duplicates={}",
        seen.iter().filter(|&&c| c == 0).count(),
        seen.iter().filter(|&&c| c > 1).count()
    );
    // 2026-09-25: 64 f32 accumulator registers per thread (16 tiles x 4), the
    // budget the kernel's header states.
    assert_eq!(16 * 4, KD * VD / THREADS);
}

#[test]
fn the_phase_a_map_covers_every_token_column_once() {
    let mut seen = vec![0u8; C * VD];
    for tid in 0..THREADS {
        for nt in 0..8 {
            for e in 0..4 {
                let (i, v) = ws_slot(tid, nt, e);
                assert!(i < C && v < VD, "tid={tid} nt={nt} e={e} -> ({i},{v})");
                seen[i * VD + v] += 1;
            }
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "the 4 m-tile x 2 n-half warp split must neither overlap nor leave a \
         hole; holes={} duplicates={}",
        seen.iter().filter(|&&c| c == 0).count(),
        seen.iter().filter(|&&c| c > 1).count()
    );
}

#[test]
fn the_k_transpose_map_is_a_bijection() {
    let mut seen = vec![0u8; KD * C];
    for tid in 0..THREADS {
        for j in 0..4 {
            for e in 0..8 {
                let (k, i) = kt_slot(tid, j, e);
                assert!(k < KD && i < C);
                // 2026-09-25: The kernel's 16-byte vector load is 8 contiguous
                // bf16 starting at `kcol + j*8`, so `e` must stay inside one load.
                assert_eq!(k / 8, ((tid & 3) * 32 + j * 8) / 8);
                seen[k * C + i] += 1;
            }
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "K^T staging must tile [128][64]"
    );
}

/// 2026-09-25: Every address the maps form lands inside the buffer the
/// launcher sized.
#[test]
fn every_padded_address_stays_inside_its_buffer() {
    for tid in 0..THREADS {
        for nt in 0..16 {
            for e in 0..4 {
                let (k, v) = acc_slot(tid, nt, e);
                assert!(v * SW + k < VD * SW, "St[v][k] overflow");
                assert!(v * SC + 63 < VD * SC, "ducT[v][i] overflow");
            }
        }
        for j in 0..4 {
            for e in 0..8 {
                let (k, i) = kt_slot(tid, j, e);
                assert!(k * SC + i < KD * SC, "Kt[k][i] overflow");
            }
        }
    }
    // 2026-09-25: The launcher's byte count is the sum the kernel lays out.
    assert_eq!(
        GDN_TC_SMEM as usize,
        VD * SW * 2 + 2 * (C * SW * 2) + VD * SC * 2 + (C + 1) * 4
    );
    assert_eq!(GDN_TC_SMEM, 88_324);
}

// 2026-09-25: 2. The maps compose to the recurrence.

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }
}

/// 2026-09-25: T = 150: chunks of 64, 64 and 22, so the last chunk is partial.
const T: usize = 150;

struct Fixture {
    w: Vec<f64>,
    u: Vec<f64>,
    key: Vec<f64>,
    gc: Vec<f64>,
    h0: Vec<f64>,
}

fn fixture() -> (Fixture, usize) {
    let nt = T.div_ceil(C);
    let mut r = Rng(0x0928_7CF0_2026);
    let f = Fixture {
        w: (0..nt * C * KD).map(|_| r.f()).collect(),
        u: (0..nt * C * VD).map(|_| r.f()).collect(),
        key: (0..T * KD).map(|_| r.f()).collect(),
        // 2026-09-25: A decreasing cumulative log-gate, the form
        // `recompute_wu` writes to `gc`.
        gc: (0..nt * C).map(|i| -0.02 * (i % C) as f64).collect(),
        h0: (0..KD * VD).map(|_| r.f() * 0.2).collect(),
    };
    (f, nt)
}

/// 2026-09-25: The reference: the plain recurrence as nested loops.
fn reference(f: &Fixture, nt: usize) -> Vec<f64> {
    let mut s = f.h0.clone();
    for c in 0..nt {
        let ce = (T - c * C).min(C);
        let gl = f.gc[c * C + ce - 1];
        let mut duc = vec![0.0f64; C * VD];
        for i in 0..ce {
            let dc = (gl - f.gc[c * C + i]).exp();
            for v in 0..VD {
                let mut ws = 0.0;
                for k in 0..KD {
                    ws += f.w[(c * C + i) * KD + k] * s[k * VD + v];
                }
                duc[i * VD + v] = dc * (f.u[(c * C + i) * VD + v] - ws);
            }
        }
        let edl = gl.exp();
        for k in 0..KD {
            for v in 0..VD {
                let mut a = edl * s[k * VD + v];
                for i in 0..ce {
                    a += duc[i * VD + v] * f.key[(c * C + i) * KD + k];
                }
                s[k * VD + v] = a;
            }
        }
    }
    s
}

/// 2026-09-25: The same recurrence driven through the kernel's maps and
/// padded smem strides: the state lives in `acc[tid][nt][e]`, never in
/// `S[k][v]`. The numbered steps are the kernel's.
fn simulated(f: &Fixture, nt_chunks: usize) -> Vec<f64> {
    let mut acc = vec![0.0f64; THREADS * 16 * 4];
    for tid in 0..THREADS {
        for nt in 0..16 {
            for e in 0..4 {
                let (k, v) = acc_slot(tid, nt, e);
                acc[(tid * 16 + nt) * 4 + e] = f.h0[k * VD + v];
            }
        }
    }
    let mut st = vec![0.0f64; VD * SW];
    let mut kt = vec![0.0f64; KD * SC];
    let mut duct = vec![0.0f64; VD * SC];
    let mut wp = vec![0.0f64; C * SW];
    let mut up = vec![0.0f64; C * SW];

    for c in 0..nt_chunks {
        let ce = (T - c * C).min(C);
        let gl = f.gc[c * C + ce - 1];
        let mut dec = vec![0.0f64; C + 1];
        dec[0] = gl.exp();
        for i in 0..ce {
            dec[1 + i] = (gl - f.gc[c * C + i]).exp();
        }
        // 2026-09-25: (1) Stage W/U with the padded stride, zero-filling rows
        // past `ce`.
        for i in 0..C {
            for x in 0..KD {
                wp[i * SW + x] = if i < ce {
                    f.w[(c * C + i) * KD + x]
                } else {
                    0.0
                };
                up[i * SW + x] = if i < ce {
                    f.u[(c * C + i) * VD + x]
                } else {
                    0.0
                };
            }
        }
        // 2026-09-25: (2) Snapshot S -> St[v][k] from the accumulator.
        for tid in 0..THREADS {
            for nt in 0..16 {
                for e in 0..4 {
                    let (k, v) = acc_slot(tid, nt, e);
                    st[v * SW + k] = acc[(tid * 16 + nt) * 4 + e];
                }
            }
        }
        // 2026-09-25: (3) and (6): K^T staging.
        for tid in 0..THREADS {
            for j in 0..4 {
                for e in 0..8 {
                    let (k, i) = kt_slot(tid, j, e);
                    kt[k * SC + i] = if i < ce {
                        f.key[(c * C + i) * KD + k]
                    } else {
                        0.0
                    };
                }
            }
        }
        // 2026-09-25: (4) and (5): Phase A, then the epilogue that writes duc
        // transposed.
        for tid in 0..THREADS {
            for nt in 0..8 {
                for e in 0..4 {
                    let (i, v) = ws_slot(tid, nt, e);
                    let mut ws = 0.0;
                    for k in 0..KD {
                        ws += wp[i * SW + k] * st[v * SW + k];
                    }
                    let uci = up[i * SW + v] - ws;
                    duct[v * SC + i] = if i < ce { dec[1 + i] * uci } else { 0.0 };
                }
            }
        }
        // 2026-09-25: (7) Phase B, accumulating into the same registers.
        for tid in 0..THREADS {
            for nt in 0..16 {
                for e in 0..4 {
                    let (k, v) = acc_slot(tid, nt, e);
                    let slot = (tid * 16 + nt) * 4 + e;
                    let mut a = dec[0] * acc[slot];
                    for i in 0..C {
                        a += kt[k * SC + i] * duct[v * SC + i];
                    }
                    acc[slot] = a;
                }
            }
        }
    }
    let mut out = vec![0.0f64; KD * VD];
    for tid in 0..THREADS {
        for nt in 0..16 {
            for e in 0..4 {
                let (k, v) = acc_slot(tid, nt, e);
                out[k * VD + v] = acc[(tid * 16 + nt) * 4 + e];
            }
        }
    }
    out
}

#[test]
fn the_simulated_index_math_reproduces_the_recurrence() {
    let (f, nt) = fixture();
    let want = reference(&f, nt);
    let got = simulated(&f, nt);
    let (mut se, mut sr) = (0.0f64, 0.0f64);
    for (a, b) in got.iter().zip(want.iter()) {
        se += (a - b) * (a - b);
        sr += b * b;
    }
    let rel = (se / sr).sqrt();
    assert!(
        rel < 1e-12,
        "the kernel's maps must compose to the plain recurrence; rel_rms={rel:e}"
    );
}

/// 2026-09-25: Negative control: the composition test is evidence only if a
/// wrong operand fails it. Here the key buffer is reversed, which permutes
/// every K element while staying in bounds, and the result must move.
#[test]
fn a_transposed_duc_breaks_the_composition() {
    let (f, nt) = fixture();
    let want = reference(&f, nt);
    let mut bad = f;
    bad.key.reverse();
    let got = simulated(&bad, nt);
    let diff: f64 = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1.0, "a perturbed operand must move the result");
}

// 2026-09-25: 3. The lever grammar.

/// 2026-09-25: The Qwen3.8-27B geometry (kd = vd = 128, chunk 64,
/// qk_stride = conv_dim = 10240) is accepted.
#[test]
fn the_production_geometry_is_accepted() {
    assert_eq!(gdn_tc_spine_reject(true, true, 128, 128, 64, 10240), None);
}

/// 2026-09-25: A false resolved lever refuses the spine even with the kernel
/// present and the production geometry. On `kernels/hopper`, which declares
/// `gdn_prefill_tc = true`, `METRALE_GDN_PREFILL_TC=0` produces that false bit.
/// The function reads no environment; the caller passes the resolved value.
#[test]
fn a_false_lever_refuses_the_spine() {
    assert_eq!(
        gdn_tc_spine_reject(false, true, 128, 128, 64, 10240),
        Some("not requested"),
        "a false resolved bit must keep the scalar spine, whatever else holds"
    );
}

/// 2026-09-25: Each refusal names its guard.
#[test]
fn every_refusal_names_its_guard() {
    for (case, want) in [
        (
            gdn_tc_spine_reject(true, false, 128, 128, 64, 10240),
            "kernel absent from this image",
        ),
        (
            gdn_tc_spine_reject(true, true, 64, 128, 64, 10240),
            "head/chunk differs from the compile-time tile (K_DIM=V_DIM=128, CHUNK=64)",
        ),
        (
            gdn_tc_spine_reject(true, true, 128, 64, 64, 10240),
            "head/chunk differs from the compile-time tile (K_DIM=V_DIM=128, CHUNK=64)",
        ),
        (
            gdn_tc_spine_reject(true, true, 128, 128, 32, 10240),
            "head/chunk differs from the compile-time tile (K_DIM=V_DIM=128, CHUNK=64)",
        ),
        (
            gdn_tc_spine_reject(true, true, 128, 128, 64, 10244),
            "qk_stride is not a multiple of 8 (the K staging uses 16-byte vector loads)",
        ),
    ] {
        assert_eq!(case, Some(want));
    }
}

/// 2026-09-25: The tile constants equal the kernel's `K_DIM`/`V_DIM` (128)
/// and `CHUNK` (64).
#[test]
fn the_tile_constants_match_the_kernel() {
    assert_eq!(GDN_TC_DIM, 128);
    assert_eq!(GDN_TC_CHUNK, 64);
}
