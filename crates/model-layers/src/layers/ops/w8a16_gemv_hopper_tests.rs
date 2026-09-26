// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host simulation of the Hopper `w8a16_gemv` override's loop
//! against the gb10 kernel's reduction order.
//!
//! `kernels/hopper/common/w8a16_gemv.cu` replaces the gb10 kernel with the same
//! entry point and a `ceil(N/4)` x 256 launch. It changes two things:
//!
//!  * the E4M3 decode instruction (the gb10 shared-memory `E4M3_LUT` lookup
//!    becomes `cvt.rn.f16x2.e4m3x2`). That is a hardware question; the
//!    `native_fp8_gemv_hopper_microtest` example runs the target's kernel on a
//!    device against a host model of the gb10 reduction and asserts
//!    `unequal=0`. Here [`decode`] stands in for both, so a
//!    difference this file reports can only be an ordering difference;
//!  * the loop shape: [`UNROLL`] chunk loads are issued before the first is
//!    consumed (`w8a16_gemv_hopper.cuh`, DIAGNOSIS 2). Reordering loads leaves
//!    the result alone; reordering accumulation would not, and that is what
//!    this file checks.
//!
//! [`hopper_chunks`] transcribes the override's loop (the `UNROLL`-wide
//! prefetch body plus the one-at-a-time tail) and [`reference_chunks`] the
//! gb10 loop. They are asserted to emit the same chunk sequence and the same
//! per-lane FP32 accumulator at every count in [`CHUNK_COUNTS`].
//! [`rotated_chunks`], the same loop with each unroll group consumed
//! last-to-first, is required to differ, so the assertions can fail.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use half::bf16;

/// 2026-09-25: `w8a16_gemv.cu`'s `threads_per_out` (256 / `N_PER_BLOCK`).
const LANES: usize = 64;
/// 2026-09-25: `w8a16_gemv_hopper.cuh`'s `HOPPER_GEMV_UNROLL`.
const UNROLL: usize = 4;
/// 2026-09-25: K values one lane consumes per chunk (`K_PER_CHUNK`).
const K_PER_CHUNK: usize = 16;
/// 2026-09-25: Chunks per 128-wide FP8 scale block (`CHUNKS_PER_SCALE`).
const CHUNKS_PER_SCALE: usize = 8;

/// 2026-09-25: Stand-in for the E4M3 decode, the same on both sides (see the
/// module note).
fn decode(byte: u8) -> f32 {
    // 2026-09-25: Sign-symmetric and exactly representable, with magnitudes
    // from 2^-9 to about 2, so an accumulation-order difference shows up as
    // unequal bits rather than cancelling.
    let mag = f32::from(byte & 0x7F) * 0.015_625 + 0.001_953_125;
    if byte & 0x80 != 0 { -mag } else { mag }
}

/// 2026-09-25: One lane's chunk sequence under the gb10 loop:
/// `k16 = lane; k16 < n; k16 += 64`.
fn reference_chunks(lane: usize, chunk_count: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut k16 = lane;
    while k16 < chunk_count {
        out.push(k16);
        k16 += LANES;
    }
    out
}

/// 2026-09-25: The override's chunk sequence: an `UNROLL`-wide prefetch body
/// whose group is consumed in issue order, then the one-at-a-time tail.
fn hopper_chunks(lane: usize, chunk_count: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut k16 = lane;
    while k16 + (UNROLL - 1) * LANES < chunk_count {
        for u in 0..UNROLL {
            out.push(k16 + u * LANES);
        }
        k16 += UNROLL * LANES;
    }
    while k16 < chunk_count {
        out.push(k16);
        k16 += LANES;
    }
    out
}

/// 2026-09-25: The same body with each unroll group consumed last-to-first: an
/// accumulation reordering, which the assertions must detect.
fn rotated_chunks(lane: usize, chunk_count: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut k16 = lane;
    while k16 + (UNROLL - 1) * LANES < chunk_count {
        for u in (0..UNROLL).rev() {
            out.push(k16 + u * LANES);
        }
        k16 += UNROLL * LANES;
    }
    while k16 < chunk_count {
        out.push(k16);
        k16 += LANES;
    }
    out
}

/// 2026-09-25: One lane's FP32 accumulator over a chunk sequence, in the
/// kernels' per-chunk operand order: `w = decode(byte) * scale`, then
/// `acc += a * w`, for the chunk's 16 K values in order.
///
/// `kernels/gb10/common/KERNEL.toml` compiles with `--fmad=false`, so the
/// multiply and the add round separately; plain `+`/`*` models that, and
/// `mul_add` would not.
fn lane_acc(chunks: &[usize], weights: &[u8], act: &[f32], scales: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for &k16 in chunks {
        let scale = scales[k16 / CHUNKS_PER_SCALE];
        for i in 0..K_PER_CHUNK {
            let w = decode(weights[k16 * K_PER_CHUNK + i]) * scale;
            acc += act[k16 * K_PER_CHUNK + i] * w;
        }
    }
    acc
}

/// 2026-09-25: The kernels' two-stage reduction: a 5-step `shfl.down` tree
/// inside each 32-lane warp, then the two warps' lane-0 partials added through
/// shared memory, then one round to BF16.
fn reduce(partials: &[f32; LANES]) -> bf16 {
    let warp = |base: usize| {
        let mut v = [0.0_f32; 32];
        v.copy_from_slice(&partials[base..base + 32]);
        let mut off = 16;
        while off > 0 {
            for i in 0..32 - off {
                v[i] += v[i + off];
            }
            off >>= 1;
        }
        v[0]
    };
    bf16::from_f32(warp(0) + warp(32))
}

/// 2026-09-25: The 64 lane accumulators of one output row, for a
/// chunk-sequence rule.
///
/// The comparisons use these FP32 partials, not the reduced BF16: the final
/// round to BF16 is coarse enough to hide a reassociation.
fn partials(
    seq: fn(usize, usize) -> Vec<usize>,
    chunk_count: usize,
    weights: &[u8],
    act: &[f32],
    scales: &[f32],
) -> [f32; LANES] {
    let mut out = [0.0_f32; LANES];
    for (lane, p) in out.iter_mut().enumerate() {
        *p = lane_acc(&seq(lane, chunk_count), weights, act, scales);
    }
    out
}

/// 2026-09-25: How many of the 64 lanes disagree, bit for bit.
fn lanes_differing(a: &[f32; LANES], b: &[f32; LANES]) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// 2026-09-25: Chunk counts (K / 16) for the loop's residues. 16,384, 5,120 and
/// 256 are whole unroll groups on every lane (64, 20 and 1 groups); 328 is one
/// group per lane plus a one- or two-chunk tail.
const CHUNK_COUNTS: [usize; 4] = [16_384, 5_120, 328, 256];

fn fixture(chunk_count: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let n = chunk_count * K_PER_CHUNK;
    let mut state = 0x0928_2026_5a5a_0011_u64;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 32) as u32
    };
    let weights: Vec<u8> = (0..n).map(|_| (next() & 0xFF) as u8).collect();
    let act: Vec<f32> = (0..n)
        .map(|_| bf16::from_f32((next() % 2049) as f32 / 1024.0 - 1.0).to_f32())
        .collect();
    let scales: Vec<f32> = (0..chunk_count.div_ceil(CHUNKS_PER_SCALE))
        .map(|_| (next() % 16 + 1) as f32 / 1024.0)
        .collect();
    (weights, act, scales)
}

#[test]
fn the_unrolled_loop_visits_the_same_chunks_in_the_same_order() {
    for count in CHUNK_COUNTS {
        for lane in [0, 1, 31, 32, 63] {
            assert_eq!(
                hopper_chunks(lane, count),
                reference_chunks(lane, count),
                "lane {lane}, {count} chunks"
            );
        }
    }
}

#[test]
fn every_chunk_is_visited_exactly_once_across_the_64_lanes() {
    for count in CHUNK_COUNTS {
        let mut seen: Vec<usize> = (0..LANES).flat_map(|l| hopper_chunks(l, count)).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..count).collect::<Vec<_>>(), "{count} chunks");
    }
}

#[test]
fn the_scale_index_is_the_gb10_expression() {
    // 2026-09-25: gb10 computes `(k16 * 16) / FP8_BLOCK`; the override computes
    // `k16 / CHUNKS_PER_SCALE`.
    for k16 in 0..4096_usize {
        assert_eq!(k16 / CHUNKS_PER_SCALE, (k16 * K_PER_CHUNK) / 128);
    }
}

#[test]
fn the_unrolled_accumulator_is_bit_identical_to_the_gb10_order() {
    for count in CHUNK_COUNTS {
        let (w, a, s) = fixture(count);
        let (h, r) = (
            partials(hopper_chunks, count, &w, &a, &s),
            partials(reference_chunks, count, &w, &a, &s),
        );
        assert_eq!(
            lanes_differing(&h, &r),
            0,
            "{count} chunks (K={}): lane accumulators diverged",
            count * K_PER_CHUNK
        );
        assert_eq!(
            reduce(&h).to_bits(),
            reduce(&r).to_bits(),
            "{count} chunks (K={})",
            count * K_PER_CHUNK
        );
    }
}

/// 2026-09-25: The comparison can fail: consuming each unroll group in reverse
/// changes lane 0's chunk sequence and at least a quarter of the lanes' FP32
/// accumulators, at every count in `CHUNK_COUNTS`.
#[test]
fn reordering_the_unroll_group_is_detected() {
    for count in CHUNK_COUNTS {
        assert_ne!(
            rotated_chunks(0, count),
            reference_chunks(0, count),
            "{count} chunks: the mutation did not even change the sequence"
        );
        let (w, a, s) = fixture(count);
        let differing = lanes_differing(
            &partials(rotated_chunks, count, &w, &a, &s),
            &partials(reference_chunks, count, &w, &a, &s),
        );
        // 2026-09-25: Not all 64 lanes need differ: a reordered group can round
        // to the same bits.
        assert!(
            differing >= LANES / 4,
            "{count} chunks: an accumulation reordering moved only {differing} \
             of {LANES} lane chains"
        );
    }
}
