// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host simulation of `dense_gemm_m16_bf16`'s index math (the
//! cp.async staging map, the per-lane m16n8k16 fragment gather and the
//! accumulator-to-`[M, N]` store map), checked against a reference in
//! `dense_gemv_bf16`'s reduction order.
//!
//! The kernel is `kernels/hopper/common/dense_gemm_m16_bf16.cu`; this needs no
//! GPU, and it says where a mismatch is. It follows the three maps rather than
//! computing row times column:
//!
//! 1. **Staging.** Each of the 128 threads runs the kernel's chunk map (A:
//!    `row = tid>>3`, `col = (tid&7)*8`; B: the same, `B_CHUNKS` times at
//!    16-row strides) into a padded `[STAGES][rows][72]` shared tile, with the
//!    zero-fill for `row >= M` and `gn >= N`.
//! 2. **Fragments.** The 16x16 A tile and 8x16 B tile each MMA consumes are
//!    rebuilt from the per-lane `a0..a3` / `b0,b1` shared-memory reads, so a
//!    wrong `group_id`/`quad`/`kc` term gives a wrong tile.
//! 3. **Store.** The four accumulator registers stay per lane and are written
//!    through the kernel's `(group_id, group_id+8) x (quad*2, quad*2+1)` map,
//!    masked to `[M, N]`.
//!
//! The simulation adds each MMA's 16 products in sequence, which the hardware
//! need not do, so the check is the tolerance [`within_m16_tc_budget`], not
//! bit equality.
//!
//! N = 70 is a multiple of neither CTA width (32, 64), so the last CTA is
//! partial on both and the `gn >= N` zero-fill and `col < N` store mask are
//! exercised. K = 384 is six 64-wide steps, so the 4-stage ring wraps around.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::{DENSE_GEMM_M16_BF16_N_TILE, DENSE_GEMM_M16_BF16_N_TILE_WIDE};
use crate::layers::dense_ffn::m16_tc::oracle::compare_m16_tc_block;
use crate::layers::dense_ffn::m16_tc::within_m16_tc_budget;
use half::bf16;

const M_TILE: usize = 16;
const K_STEP: usize = 64;
const K_SUB: usize = 16;
const WARPS: usize = 4;
const THREADS: usize = WARPS * 32;
const N_PER_MMA: usize = 8;
const STAGES: usize = 4;
const ROW_STRIDE: usize = 72;
const ELEMS_PER_CHUNK: usize = 8;
const ROWS_PER_PASS: usize = THREADS / (K_STEP / ELEMS_PER_CHUNK);

/// 2026-09-25: Sampled shape; the module header says why N is 70 and K is 384.
const N: usize = 70;
const K: usize = 384;
/// 2026-09-25: The seed the GPU oracle `native_bf16_lm_head_m16_microtest` uses.
const SEED: u64 = 0x0927_167C_2026;
/// 2026-09-25: Untouched-output sentinel, so "never written" differs from
/// "written as zero".
const SENTINEL: u16 = 0x5a5a;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// 2026-09-25: A BF16-representable value in [-1, 1], drawn as the GPU oracle
    /// draws them.
    fn bf16(&mut self) -> f32 {
        bf16::from_f32(((self.next() % 2049) as f32 - 1024.0) / 1024.0).to_f32()
    }
}

/// 2026-09-25: `[M_TILE, K]` activations and `[N, K]` weights, as the BF16 values
/// the kernel sees.
struct Fixture {
    acts: Vec<f32>,
    weight: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        let mut rng = Rng(SEED);
        let acts = (0..M_TILE * K).map(|_| rng.bf16()).collect();
        let weight = (0..N * K).map(|_| rng.bf16()).collect();
        Self { acts, weight }
    }
}

/// 2026-09-25: Which index term to corrupt. `None` is the real kernel; every
/// other variant is a negative control that must change the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mutation {
    None,
    /// 2026-09-25: `kc1 = kc0 + 4` instead of `+ 8`: the wrong half of the MMA's
    /// K pair.
    FragmentKPair,
    /// 2026-09-25: The B fragment row picked by `quad` instead of `group_id`.
    FragmentBRow,
    /// 2026-09-25: The store column stepping by `quad` instead of `quad * 2`.
    StoreColumnStride,
    /// 2026-09-25: The A staging chunk map reading row `tid >> 2`.
    StageARowMap,
    /// 2026-09-25: The second A fragment row at `group_id + 4` instead of `+ 8`.
    FragmentARowPair,
    /// 2026-09-25: The store wraps columns past N onto `col - N` instead of
    /// dropping them, so the last, partial CTA overwrites the first columns.
    TailStoreWrap,
}

/// 2026-09-25: One CTA's shared tiles for one pipeline stage.
struct Stage {
    a: Vec<f32>,
    b: Vec<f32>,
}

/// 2026-09-25: The kernel's `prefetch`, thread for thread: one chunk of 8 BF16
/// per thread into A, `B_CHUNKS` per thread into B, zeros outside `[M, N]`.
fn prefetch(
    f: &Fixture,
    stage: &mut Stage,
    m: usize,
    cta_n: usize,
    n_tile: usize,
    k_base: usize,
    mu: Mutation,
) {
    let b_chunks = n_tile / ROWS_PER_PASS;
    for tid in 0..THREADS {
        let col = (tid & 7) * ELEMS_PER_CHUNK;
        let a_row = if mu == Mutation::StageARowMap {
            (tid >> 2) % M_TILE
        } else {
            tid >> 3
        };
        for e in 0..ELEMS_PER_CHUNK {
            stage.a[a_row * ROW_STRIDE + col + e] = if a_row < m {
                f.acts[a_row * K + k_base + col + e]
            } else {
                0.0
            };
        }
        for c in 0..b_chunks {
            let row = (tid >> 3) + c * ROWS_PER_PASS;
            let gn = cta_n + row;
            for e in 0..ELEMS_PER_CHUNK {
                stage.b[row * ROW_STRIDE + col + e] = if gn < N {
                    f.weight[gn * K + k_base + col + e]
                } else {
                    0.0
                };
            }
        }
    }
}

/// 2026-09-25: Rebuild the 16x16 A tile and 8x16 B tile one m16n8k16 consumes,
/// from the per-lane `a0..a3` / `b0,b1` shared-memory reads.
#[allow(clippy::needless_range_loop)]
fn fragments(
    stage: &Stage,
    warp_n: usize,
    j: usize,
    s: usize,
    mu: Mutation,
) -> ([[f32; K_SUB]; M_TILE], [[f32; N_PER_MMA]; K_SUB]) {
    let mut a_tile = [[0.0_f32; K_SUB]; M_TILE];
    let mut b_tile = [[0.0_f32; N_PER_MMA]; K_SUB];
    for lane in 0..32 {
        let group = lane >> 2;
        let quad = lane & 3;
        let kc0 = s * K_SUB + quad * 2;
        let kc1 = kc0 + if mu == Mutation::FragmentKPair { 4 } else { 8 };
        let row_hi = group
            + if mu == Mutation::FragmentARowPair {
                4
            } else {
                8
            };
        for (pair, kc) in [(0_usize, kc0), (1, kc1)] {
            let kl = quad * 2 + pair * 8;
            for e in 0..2 {
                a_tile[group][kl + e] = stage.a[group * ROW_STRIDE + kc + e];
                a_tile[row_hi][kl + e] = stage.a[row_hi * ROW_STRIDE + kc + e];
                let b_row = warp_n
                    + j * N_PER_MMA
                    + if mu == Mutation::FragmentBRow {
                        quad
                    } else {
                        group
                    };
                b_tile[kl + e][group] = stage.b[b_row * ROW_STRIDE + kc + e];
            }
        }
    }
    (a_tile, b_tile)
}

/// 2026-09-25: The whole kernel for one `n_tile` and `m`: every CTA, K step,
/// warp and lane, then the masked store into a sentinel-filled `[M_TILE, N]`
/// output.
fn simulate(f: &Fixture, m: usize, n_tile: usize, mu: Mutation) -> Vec<u16> {
    let n_subs = n_tile / WARPS / N_PER_MMA;
    let mut out = vec![SENTINEL; M_TILE * N];
    for cta in 0..N.div_ceil(n_tile) {
        let cta_n = cta * n_tile;
        // 2026-09-25: `[warp][lane][4 * n_subs]`: the accumulators stay per
        // lane, so the store map is exercised.
        let mut acc = vec![vec![vec![0.0_f32; 4 * n_subs]; 32]; WARPS];
        let mut stages: Vec<Stage> = (0..STAGES)
            .map(|_| Stage {
                a: vec![0.0; M_TILE * ROW_STRIDE],
                b: vec![0.0; n_tile * ROW_STRIDE],
            })
            .collect();
        for step in 0..K / K_STEP {
            let cur = step % STAGES;
            prefetch(f, &mut stages[cur], m, cta_n, n_tile, step * K_STEP, mu);
            for (warp, acc_w) in acc.iter_mut().enumerate() {
                let warp_n = warp * (n_tile / WARPS);
                for s in 0..K_STEP / K_SUB {
                    for j in 0..n_subs {
                        let (a_tile, b_tile) = fragments(&stages[cur], warp_n, j, s, mu);
                        for (lane, regs) in acc_w.iter_mut().enumerate() {
                            let (group, quad) = (lane >> 2, lane & 3);
                            for (r, (row, col)) in [
                                (group, quad * 2),
                                (group, quad * 2 + 1),
                                (group + 8, quad * 2),
                                (group + 8, quad * 2 + 1),
                            ]
                            .into_iter()
                            .enumerate()
                            {
                                let mut sum = 0.0_f32;
                                for kl in 0..K_SUB {
                                    sum += a_tile[row][kl] * b_tile[kl][col];
                                }
                                regs[j * 4 + r] += sum;
                            }
                        }
                    }
                }
            }
        }
        for (warp, acc_w) in acc.iter().enumerate() {
            let warp_n = warp * (n_tile / WARPS);
            for (lane, regs) in acc_w.iter().enumerate() {
                let (group, quad) = (lane >> 2, lane & 3);
                let step = if mu == Mutation::StoreColumnStride {
                    1
                } else {
                    2
                };
                for j in 0..n_subs {
                    let col0 = cta_n + warp_n + j * N_PER_MMA + quad * step;
                    for (r, (row, col)) in [
                        (group, col0),
                        (group, col0 + 1),
                        (group + 8, col0),
                        (group + 8, col0 + 1),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let col = if mu == Mutation::TailStoreWrap && col >= N {
                            col - N
                        } else {
                            col
                        };
                        if row < m && col < N {
                            out[row * N + col] = bf16::from_f32(regs[j * 4 + r]).to_bits();
                        }
                    }
                }
            }
        }
    }
    out
}

/// 2026-09-25: `dense_gemv_bf16`'s reduction order: 64 lanes each walking 8-wide
/// chunks at a stride of 64, lo then hi inside each 32-bit pair, then a 32-lane
/// shuffle reduction per warp and one add across the two warps.
fn gemv_reference(f: &Fixture, row: usize, col: usize) -> u16 {
    let k_vec = K / 8;
    let mut lanes = [0.0_f32; 64];
    for (lane, acc) in lanes.iter_mut().enumerate() {
        let mut kv = lane;
        while kv < k_vec {
            for i in 0..8 {
                let k = kv * 8 + i;
                *acc += f.acts[row * K + k] * f.weight[col * K + k];
            }
            kv += 64;
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

fn reference(f: &Fixture, m: usize) -> Vec<u16> {
    let mut out = vec![SENTINEL; M_TILE * N];
    for row in 0..m {
        for col in 0..N {
            out[row * N + col] = gemv_reference(f, row, col);
        }
    }
    out
}

/// 2026-09-25: RMS of one reference row, the scale of
/// [`within_m16_tc_budget`]'s absolute floor.
fn row_rms(block: &[u16], row: usize) -> f64 {
    let r = &block[row * N..(row + 1) * N];
    let sum: f64 = r
        .iter()
        .map(|b| {
            let v = f64::from(bf16::from_bits(*b).to_f32());
            v * v
        })
        .sum();
    (sum / r.len() as f64).sqrt()
}

/// 2026-09-25: Staging map, fragment gather and store map together reproduce
/// the scalar GEMV within the tier's tolerance, on both CTA widths and at
/// M = 5, 8, 13 and 16.
#[test]
fn the_fragment_and_index_math_reproduce_the_scalar_gemv() {
    let f = Fixture::new();
    for m in [5_usize, 8, 13, 16] {
        let want = reference(&f, m);
        for n_tile in [
            DENSE_GEMM_M16_BF16_N_TILE as usize,
            DENSE_GEMM_M16_BF16_N_TILE_WIDE as usize,
        ] {
            let got = simulate(&f, m, n_tile, Mutation::None);
            for row in 0..m {
                let scale = row_rms(&want, row);
                for col in 0..N {
                    let i = row * N + col;
                    assert!(
                        within_m16_tc_budget(got[i], want[i], K, scale),
                        "m={m} n_tile={n_tile} row={row} col={col}: \
                         got {:+e} want {:+e}",
                        bf16::from_bits(got[i]).to_f32(),
                        bf16::from_bits(want[i]).to_f32()
                    );
                }
            }
        }
    }
}

/// 2026-09-25: Every [`Mutation`] other than `None` changes the result.
#[test]
fn every_corrupted_index_term_is_caught() {
    let f = Fixture::new();
    let good = simulate(&f, 16, DENSE_GEMM_M16_BF16_N_TILE as usize, Mutation::None);
    for mu in [
        Mutation::FragmentKPair,
        Mutation::FragmentBRow,
        Mutation::StoreColumnStride,
        Mutation::StageARowMap,
        Mutation::FragmentARowPair,
        Mutation::TailStoreWrap,
    ] {
        assert_ne!(
            good,
            simulate(&f, 16, DENSE_GEMM_M16_BF16_N_TILE as usize, mu),
            "{mu:?}: the index-math pin would not have caught this"
        );
    }
}

/// 2026-09-25: On both widths, rows past `m` keep the sentinel and every element
/// of the first `m` rows is written.
#[test]
fn the_kernel_writes_nothing_outside_the_used_extent() {
    let f = Fixture::new();
    for m in [5_usize, 16] {
        for n_tile in [
            DENSE_GEMM_M16_BF16_N_TILE as usize,
            DENSE_GEMM_M16_BF16_N_TILE_WIDE as usize,
        ] {
            let got = simulate(&f, m, n_tile, Mutation::None);
            for row in m..M_TILE {
                assert!(
                    got[row * N..(row + 1) * N].iter().all(|b| *b == SENTINEL),
                    "m={m} n_tile={n_tile}: row {row} past the batch was written"
                );
            }
            assert!(
                got[..m * N].iter().all(|b| *b != SENTINEL),
                "m={m} n_tile={n_tile}: an in-extent output was left unwritten"
            );
        }
    }
}

/// 2026-09-25: [`compare_m16_tc_block`] refuses a wrapped partial-CTA store on
/// both widths, with errors at the scale of the matrix.
#[test]
fn a_partial_tail_defect_is_refused_by_the_metric_on_both_widths() {
    let f = Fixture::new();
    let want = reference(&f, M_TILE);
    let want_bytes: Vec<u8> = want.iter().flat_map(|b| b.to_le_bytes()).collect();
    for n_tile in [
        DENSE_GEMM_M16_BF16_N_TILE as usize,
        DENSE_GEMM_M16_BF16_N_TILE_WIDE as usize,
    ] {
        let got = simulate(&f, M_TILE, n_tile, Mutation::TailStoreWrap);
        let got_bytes: Vec<u8> = got.iter().flat_map(|b| b.to_le_bytes()).collect();
        let d = compare_m16_tc_block(&got_bytes, &want_bytes, N, K);
        assert!(
            !d.over_budget.is_empty(),
            "n_tile={n_tile}: the metric admitted a wrapped partial-CTA store — \
             the round-9 tail hypothesis would have been unfalsifiable"
        );
        // 2026-09-25: The errors are of matrix scale, so the refusal does not
        // depend on the floor being narrow.
        let worst = d
            .over_budget
            .iter()
            .map(|o| f64::from((o.actual - o.reference).abs()))
            .fold(0.0_f64, f64::max);
        assert!(
            worst > 0.1 * d.rms,
            "n_tile={n_tile}: a tail defect must land errors of matrix scale, got {worst:.3e} \
             against rms {:.3e}",
            d.rms
        );
    }
}
