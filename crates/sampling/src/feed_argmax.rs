// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host reference for one row of `argmax_bf16_batch_feed`
//! (`kernels/gb10/common/argmax_feed.cu`), computed from the same BF16 bytes.
//!
//! Owner: metrale-sampling.
//! Invariants: none beyond the types. NaN input is outside the contract.
//!
//! The kernel runs two reductions over a row and combines them:
//!
//! * `p`: the plain argmax, the same scan and tree reduction as
//!   `argmax_bf16_batch`. Each of the `BLOCK` threads keeps the first strict
//!   maximum of its strided slice above the `-1e30` floor. At each merge of
//!   threads `t` and `t + s`, `t` keeps its value unless `t + s` holds a
//!   strictly larger one. Ties therefore follow the tree order, not the
//!   lowest index. [`plain_kernel_argmax`] simulates this, floor included.
//! * `q`: the row with the two masked ids set to `-inf` (as
//!   `PostCloseThinkMask` does) and the last index holding the maximum, the
//!   tie rule of the temperature-0 host sampler ([`masked_last_wins`]).
//! * The answer is `q` when `p` is a masked id, else `p` ([`feed_argmax`]).
//!   The synchronous decode step does the same: it takes the device argmax
//!   unless a `think_ended` row's argmax landed on a masked id, and then
//!   samples on the host.
//!
//! `feed_argmax_tests.rs` checks `q` against the host sampler, `p` against
//! `argmax_bf16` for rows with a unique maximum, and pins the tie cases.

/// 2026-09-25: "No id" in a row mask.
pub const NO_ID: u32 = u32::MAX;
/// 2026-09-25: The floor the plain kernel starts its running maximum at.
pub const PLAIN_FLOOR: f32 = -1e30;
/// 2026-09-25: Threads per row block of the kernel; the tie order depends on it.
pub const BLOCK: usize = 1024;

/// 2026-09-25: BF16 bits to f32, exact (what `__bfloat162float` computes).
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// 2026-09-25: `p`: the plain kernel argmax of `row`, tie order and floor included.
pub fn plain_kernel_argmax(row: &[f32]) -> u32 {
    let mut val = [PLAIN_FLOOR; BLOCK];
    let mut idx = [0u32; BLOCK];
    for (i, &v) in row.iter().enumerate() {
        let t = i % BLOCK;
        if v > val[t] {
            val[t] = v;
            idx[t] = i as u32;
        }
    }
    let mut s = BLOCK / 2;
    while s > 0 {
        for t in 0..s {
            if val[t + s] > val[t] {
                val[t] = val[t + s];
                idx[t] = idx[t + s];
            }
        }
        s /= 2;
    }
    idx[0]
}

/// 2026-09-25: The row with the masked ids at `-inf`; out-of-range ids are ignored.
pub fn masked_row(row: &[f32], mask: [u32; 2]) -> Vec<f32> {
    let mut out = row.to_vec();
    for m in mask {
        if let Some(v) = out.get_mut(m as usize) {
            *v = f32::NEG_INFINITY;
        }
    }
    out
}

/// 2026-09-25: `q`: the last index holding the maximum of [`masked_row`], with
/// the maximum starting at `-inf`; an empty row gives 0.
pub fn masked_last_wins(row: &[f32], mask: [u32; 2]) -> u32 {
    let mut best = f32::NEG_INFINITY;
    let mut idx = 0u32;
    for (i, &v) in masked_row(row, mask).iter().enumerate() {
        if v > best || (v == best && i as u32 > idx) {
            best = v;
            idx = i as u32;
        }
    }
    idx
}

/// 2026-09-25: One row of `argmax_bf16_batch_feed`, from the BF16 bits.
pub fn feed_argmax(row_bf16: &[u16], mask: [u32; 2]) -> u32 {
    let row: Vec<f32> = row_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
    feed_argmax_f32(&row, mask)
}

/// 2026-09-25: [`feed_argmax`] over an f32 row.
pub fn feed_argmax_f32(row: &[f32], mask: [u32; 2]) -> u32 {
    let p = plain_kernel_argmax(row);
    if p == mask[0] || p == mask[1] {
        masked_last_wins(row, mask)
    } else {
        p
    }
}
