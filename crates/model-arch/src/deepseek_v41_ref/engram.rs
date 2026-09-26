// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: DeepSeek-V4.1 engram, CPU reference, checked stage by stage
//! against the golden:
//!
//! 1. Hash (`NgramHashState`): compressed token map, n-gram window with pad
//!    and a cache that carries look-back across the prefill/decode split,
//!    per-lookback multiply, XOR-rolled, then `% prime` per (n-gram size,
//!    head) plus that slot's offset. Integer-exact.
//! 2. Table rows (`regen_table`, `embed_rows`): the fp8 e4m3 table,
//!    dequantised per `scale_block` values by a power-of-two scale, gathered
//!    by hash id. An e4m3 value times a power of two is exact in bf16, so
//!    this stage is exact too.
//! 3. Projection (`wkv`): one bf16 GEMM, `[n_hash_cols * head_dim] -> [dim *
//!    (hc_mult + 1)]`, split into one key per hyper-connection copy plus one
//!    shared value.
//! 4. Gate (`gate_and_add`): normalised dot product of stream against key,
//!    signed-sqrt sigmoid, `h + gate * value`, back to bf16.
//!
//! The hash tables (`token_map`, `multipliers`, `primes`, `offsets`) are
//! read from the golden, never recomputed, as the loader reads them from the
//! GGUF's `deepseek41.engram.*` metadata.
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants: none beyond the types.

use super::{Golden, fixed_value, to_bf16_rne};

/// 2026-09-25: The engram hash tables, geometry and table scales, from the
/// golden's `fixture.engram`.
pub struct EngramTables {
    pub layer_ids: Vec<usize>,
    pub max_ngram: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub num_embeddings: Vec<usize>,
    // 2026-09-25: `pad_id` is the pad in the compressed id space. Layouts:
    // `multipliers[layer][lookback]`, `primes[layer][ngram_size - 2][head]`,
    // `offsets[layer][(ngram_size - 2) * n_heads + head]`, and
    // `scales[layer][row * (head_dim / scale_block) + block]`, powers of two
    // with `scale_block` values each along `head_dim`.
    pub pad_id: i64,
    pub token_map: Vec<i64>,
    pub multipliers: Vec<Vec<i64>>,
    pub primes: Vec<Vec<Vec<i64>>>,
    pub offsets: Vec<Vec<i64>>,
    pub scale_block: usize,
    pub scales: Vec<Vec<f32>>,
}

impl EngramTables {
    pub fn n_hash_cols(&self) -> usize {
        (self.max_ngram - 1) * self.n_heads
    }

    pub fn from_golden(g: &Golden) -> Self {
        let e = &g.0["fixture"]["engram"];
        let ints = |v: &serde_json::Value| -> Vec<i64> {
            v.as_array()
                .expect("array")
                .iter()
                .map(|x| x.as_i64().expect("int"))
                .collect()
        };
        EngramTables {
            layer_ids: ints(&e["layer_ids"])
                .into_iter()
                .map(|x| x as usize)
                .collect(),
            max_ngram: e["max_ngram_size"].as_u64().unwrap() as usize,
            n_heads: e["n_heads"].as_u64().unwrap() as usize,
            head_dim: e["head_dim"].as_u64().unwrap() as usize,
            num_embeddings: ints(&e["num_embeddings"])
                .into_iter()
                .map(|x| x as usize)
                .collect(),
            pad_id: e["pad_id_compressed"].as_i64().unwrap(),
            token_map: ints(&e["token_map"]),
            multipliers: e["multipliers"]
                .as_array()
                .unwrap()
                .iter()
                .map(ints)
                .collect(),
            primes: e["primes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| l.as_array().unwrap().iter().map(ints).collect())
                .collect(),
            offsets: e["offsets"].as_array().unwrap().iter().map(ints).collect(),
            scale_block: e["scale_block"].as_u64().unwrap() as usize,
            scales: e["scales"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| {
                    l.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap() as f32)
                        .collect()
                })
                .collect(),
        }
    }
}

/// 2026-09-25: The n-gram hash state (the reference's `NgramHashState`). The cache
/// holds compressed ids for every position seen, so a decode step's look-back
/// reaches into the prefill.
pub struct NgramHashState<'a> {
    pub t: &'a EngramTables,
    pub batch: usize,
    pub max_seq: usize,
    cache: Vec<i64>,
}

impl<'a> NgramHashState<'a> {
    pub fn new(t: &'a EngramTables, batch: usize, max_seq: usize) -> Self {
        NgramHashState {
            t,
            batch,
            max_seq,
            cache: vec![0; batch * max_seq],
        }
    }

    /// 2026-09-25: `input_ids`: `[batch, seqlen]` row-major. Returns `[batch,
    /// seqlen, n_layers, n_hash_cols]`.
    pub fn forward(&mut self, input_ids: &[i64], seqlen: usize, start_pos: usize) -> Vec<i64> {
        let t = self.t;
        let (b_n, ng, nl, cols) = (self.batch, t.max_ngram, t.layer_ids.len(), t.n_hash_cols());
        assert_eq!(input_ids.len(), b_n * seqlen);
        assert!(start_pos + seqlen <= self.max_seq);
        for b in 0..b_n {
            for s in 0..seqlen {
                self.cache[b * self.max_seq + start_pos + s] =
                    t.token_map[input_ids[b * seqlen + s] as usize];
            }
        }
        let mut out = vec![0i64; b_n * seqlen * nl * cols];
        let mut tokens = vec![0i64; ng];
        for b in 0..b_n {
            for s in 0..seqlen {
                let pos = start_pos + s;
                // 2026-09-25: `blocked` stays set for every larger shift once a
                // shift reaches before position 0 or a -1 compressed id.
                let mut blocked = false;
                for (shift, tok) in tokens.iter_mut().enumerate() {
                    let src = self.cache[b * self.max_seq + pos.saturating_sub(shift)];
                    blocked = blocked || pos < shift || src == -1;
                    *tok = if blocked { t.pad_id } else { src };
                }
                for (li, mults) in t.multipliers.iter().enumerate() {
                    let mut rolling = tokens[0].wrapping_mul(mults[0]);
                    for i in 1..ng {
                        rolling ^= tokens[i].wrapping_mul(mults[i]);
                        for h in 0..t.n_heads {
                            let col = (i - 1) * t.n_heads + h;
                            out[((b * seqlen + s) * nl + li) * cols + col] =
                                rolling.rem_euclid(t.primes[li][i - 1][h]) + t.offsets[li][col];
                        }
                    }
                }
            }
        }
        out
    }
}

/// 2026-09-25: Encode to fp8 e4m3fn with round-to-nearest-even. NaN -> 0x7F;
/// above 448 saturates to 448 (0x7E).
pub fn f32_to_e4m3_rne(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    let bits = x.abs().to_bits();
    let exp_field = ((bits >> 23) & 0xFF) as i32;
    if exp_field == 0 {
        return sign;
    }
    let a = x.abs() as f64;
    if a > 448.0 {
        return sign | 0x7E;
    }
    let e = exp_field - 127;
    if e < -6 {
        // 2026-09-25: e4m3 subnormal: value = m/8 * 2^-6, m in 0..=8; m == 8 is
        // the first normal, whose code (exp_field 1, mantissa 0) is 0x08.
        let r = (a * 512.0).round_ties_even() as u8;
        return sign | r;
    }
    let mant = ((a * 2f64.powi(-e) - 1.0) * 8.0).round_ties_even() as u8;
    let (e, mant) = if mant == 8 { (e + 1, 0u8) } else { (e, mant) };
    sign | (((e + 7) as u8) << 3) | mant
}

pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp_field = ((b >> 3) & 0x0F) as i32;
    let mant = (b & 0x07) as f32;
    if exp_field == 15 && mant == 7.0 {
        return f32::NAN;
    }
    if exp_field == 0 {
        return sign * mant / 8.0 * 2f32.powi(-6);
    }
    sign * (1.0 + mant / 8.0) * 2f32.powi(exp_field - 7)
}

/// 2026-09-25: f32 -> e8m0 as a power of two: nearest exponent, ties (exactly
/// 1.5 * 2^e) to the even exponent. Pinned against the golden's scales by
/// `e8m0_rounding_rule_matches_torch`.
pub fn f32_to_e8m0_rne(x: f32) -> f32 {
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xFF) as i32;
    let m = bits & 0x7F_FFFF;
    if m > 0x40_0000 || (m == 0x40_0000 && (e & 1) == 1) {
        e += 1;
    }
    f32::from_bits((e as u32) << 23)
}

/// 2026-09-25: The fixture's fp8 table for layer index `layer_idx`, regenerated
/// and dequantised with the golden's per-block scales (value times scale,
/// then bf16). `[rows * head_dim]` f32; exact, since e4m3 times a power of
/// two is bf16-exact.
pub fn regen_table(t: &EngramTables, layer_idx: usize) -> Vec<f32> {
    let lid = t.layer_ids[layer_idx];
    let (rows, hd, blk) = (t.num_embeddings[layer_idx], t.head_dim, t.scale_block);
    let name = format!("layers.{lid}.engram.embed.weight");
    let scales = &t.scales[layer_idx];
    (0..(rows * hd) as u64)
        .map(|i| {
            let v = e4m3_to_f32(f32_to_e4m3_rne(fixed_value(&name, i, 0.5, 0.0)));
            let (r, d) = ((i as usize) / hd, (i as usize) % hd);
            to_bf16_rne(v * scales[r * (hd / blk) + d / blk])
        })
        .collect()
}

/// 2026-09-25: Gather rows of a dequantised table by hash id. Output `[n_ids, head_dim]`.
pub fn embed_rows(table: &[f32], head_dim: usize, hash_ids: &[i64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(hash_ids.len() * head_dim);
    for &id in hash_ids {
        let r = id as usize * head_dim;
        out.extend_from_slice(&table[r..r + head_dim]);
    }
    out
}

/// 2026-09-25: A bf16-stored `[out, in]` weight regenerated by the fixture rule
/// `scale = in^-0.5`, offset 0.
pub fn regen_bf16_matrix(name: &str, out_f: usize, in_f: usize) -> Vec<f32> {
    let scale = (in_f as f64).powf(-0.5);
    (0..(out_f * in_f) as u64)
        .map(|i| to_bf16_rne(fixed_value(name, i, scale, 0.0)))
        .collect()
}

/// 2026-09-25: A bf16-stored engram q/k weight of `n` values regenerated by the
/// fixture rule (offset 1, scale 0.05).
pub fn regen_bf16_qk(name: &str, n: usize) -> Vec<f32> {
    (0..n as u64)
        .map(|i| to_bf16_rne(fixed_value(name, i, 0.05, 1.0)))
        .collect()
}

/// 2026-09-25: bf16 linear: f32 accumulate, bf16 output. `x`: `[rows, in]`, `w`:
/// `[out, in]`.
pub fn linear_bf16(x: &[f32], rows: usize, in_f: usize, w: &[f32], out_f: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * out_f];
    for r in 0..rows {
        let xr = &x[r * in_f..(r + 1) * in_f];
        for o in 0..out_f {
            let wr = &w[o * in_f..(o + 1) * in_f];
            let acc: f32 = xr.iter().zip(wr).map(|(a, b)| a * b).sum();
            y[r * out_f + o] = to_bf16_rne(acc);
        }
    }
    y
}

/// 2026-09-25: The engram after the projection: `x` `[tokens, hc, dim]` (bf16
/// values), `kv` `[tokens, dim*(hc+1)]`, q/k weights `[hc, dim]`. Returns
/// bf16-rounded `[tokens, hc, dim]`.
pub fn gate_and_add(
    x: &[f32],
    kv: &[f32],
    q_w: &[f32],
    k_w: &[f32],
    tokens: usize,
    hc: usize,
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    let kv_w = dim * (hc + 1);
    let inv_sqrt_dim = (dim as f32).powf(-0.5);
    let mut out = vec![0f32; tokens * hc * dim];
    for t in 0..tokens {
        let value = &kv[t * kv_w + hc * dim..(t + 1) * kv_w];
        for c in 0..hc {
            let h = &x[(t * hc + c) * dim..(t * hc + c + 1) * dim];
            let key = &kv[t * kv_w + c * dim..t * kv_w + (c + 1) * dim];
            let (mut hh, mut kk, mut dot) = (0f32, 0f32, 0f32);
            for d in 0..dim {
                hh += h[d] * h[d];
                kk += key[d] * key[d];
                // 2026-09-25: The q and k weights enter only as their product.
                dot += h[d] * (q_w[c * dim + d] * k_w[c * dim + d]) * key[d];
            }
            let rstd =
                (hh / dim as f32 + eps).sqrt().recip() * (kk / dim as f32 + eps).sqrt().recip();
            let dot = dot * rstd * inv_sqrt_dim;
            // 2026-09-25: Signed sqrt (magnitude floored at 1e-6) before the sigmoid.
            let g = dot.abs().max(1e-6).sqrt().copysign(dot);
            let gate = 1.0 / (1.0 + (-g).exp());
            for d in 0..dim {
                out[(t * hc + c) * dim + d] = to_bf16_rne(h[d] + gate * value[d]);
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "engram_tests.rs"]
mod tests;
