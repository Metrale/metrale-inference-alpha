// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The n-gram id core: the EOS-aware right shift and the
//! polynomial rolling hash, on token ids only (no device, no weights).
//!
//! Owner: model-layers (n-gram embedding).
//! Invariants: none beyond the types.

use super::NgramDims;

/// 2026-09-25: `out[t] = ctx[t - n]` when `t - n` lies in the same segment
/// as `t` (a segment ends at an EOS token, inclusive), else 0. A segment of
/// n tokens or fewer is all zeros.
fn shift_right_ignore_eos(ctx: &[u32], n: usize, eos: u32) -> Vec<u32> {
    let len = ctx.len();
    let mut out = vec![0u32; len];
    let mut prev = 0usize;
    for (pos, &tok) in ctx.iter().enumerate() {
        if tok == eos {
            let end = pos + 1;
            if end - prev > n {
                out[prev + n..end].copy_from_slice(&ctx[prev..end - n]);
            }
            prev = end;
        }
    }
    if prev < len && len - prev > n {
        out[prev + n..len].copy_from_slice(&ctx[prev..len - n]);
    }
    out
}

/// 2026-09-25: Row ids of every table over `ctx`. Returns `num_tables()`
/// vectors of `ctx.len()` ids each, in table index order
/// (`(ngram-2)*K + split`).
pub fn ngram_ids(dims: &NgramDims, ctx: &[u32]) -> Vec<Vec<u64>> {
    let mut out = Vec::with_capacity(dims.num_tables());
    // 2026-09-25: One shift per d in 1..N, shared by every table.
    let shifts: Vec<Vec<u32>> = (1..dims.neighbor_num)
        .map(|d| shift_right_ignore_eos(ctx, d, dims.eos_token_id))
        .collect();
    for ngram in 2..=dims.neighbor_num {
        for split in 0..dims.split_num {
            let index = (ngram - 2) * dims.split_num + split;
            let t = dims.table_rows(index);
            let mods = dims.vocab_mods(ngram, split);
            let ids = ctx
                .iter()
                .enumerate()
                .map(|(pos, &x)| {
                    let mut acc = x as u64;
                    for (d, &m) in mods.iter().enumerate() {
                        acc += shifts[d][pos] as u64 * m;
                    }
                    acc % t
                })
                .collect();
            out.push(ids);
        }
    }
    out
}
