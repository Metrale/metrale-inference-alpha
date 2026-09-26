// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The NoPE MLA half of the DSA reference: `expand_kv` and `mla_masked_attention`.
//!
//! Owner: model-arch (GLM-5.3-Flash DSA reference).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: NoPE MLA over a per-query selected key set.
///
/// `q`: `[q_rows, heads, qk_head_dim]`, `k`: `[kv_len, heads, qk_head_dim]`,
/// `v`: `[kv_len, heads, v_head_dim]`, `mask`: `[q_rows, kv_len]`.
///
/// The scale is `qk_head_dim^-0.5` over the full head dim; it equals `qk_nope_head_dim^-0.5` only
/// while the rope part is zero-width.
pub fn mla_masked_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[u8],
    dims: DsaDims,
    q_rows: usize,
    kv_len: usize,
) -> Vec<f32> {
    let (h, qd, vd) = (dims.heads, dims.qk_head_dim(), dims.v_head_dim);
    let scale = (qd as f32).powf(-0.5);
    let mut out = vec![0.0f32; q_rows * h * vd];
    for r in 0..q_rows {
        for hh in 0..h {
            // 2026-09-25: Online softmax: one pass, no `[kv_len]` score buffer.
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            let mut acc = vec![0.0f32; vd];
            for kk in 0..kv_len {
                if mask[r * kv_len + kk] == 0 {
                    continue;
                }
                let mut dot = 0.0f32;
                for dd in 0..qd {
                    dot += q[(r * h + hh) * qd + dd] * k[(kk * h + hh) * qd + dd];
                }
                let s = dot * scale;
                let m_new = m.max(s);
                let corr = if m == f32::NEG_INFINITY {
                    0.0
                } else {
                    (m - m_new).exp()
                };
                let p = (s - m_new).exp();
                l = l * corr + p;
                for dd in 0..vd {
                    acc[dd] = acc[dd] * corr + p * v[(kk * h + hh) * vd + dd];
                }
                m = m_new;
            }
            let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
            for dd in 0..vd {
                out[(r * h + hh) * vd + dd] = acc[dd] * inv;
            }
        }
    }
    out
}

/// 2026-09-25: Expand the compressed latent into per-head K and V. NoPE only: it asserts
/// `qk_rope_head_dim == 0`, so K is exactly `qk_nope_head_dim` wide.
pub fn expand_kv(kv_c: &[f32], w_kv_b: &[f32], dims: DsaDims, seq: usize) -> (Vec<f32>, Vec<f32>) {
    let (h, nope, vd, r) = (
        dims.heads,
        dims.qk_nope_head_dim,
        dims.v_head_dim,
        dims.kv_lora_rank,
    );
    assert_eq!(dims.qk_rope_head_dim, 0, "expand_kv is the NoPE path");
    let wide = linear(kv_c, seq, r, w_kv_b, h * (nope + vd));
    let mut k = vec![0.0f32; seq * h * nope];
    let mut v = vec![0.0f32; seq * h * vd];
    for t in 0..seq {
        for hh in 0..h {
            let src = t * h * (nope + vd) + hh * (nope + vd);
            k[(t * h + hh) * nope..(t * h + hh) * nope + nope]
                .copy_from_slice(&wide[src..src + nope]);
            v[(t * h + hh) * vd..(t * h + hh) * vd + vd]
                .copy_from_slice(&wide[src + nope..src + nope + vd]);
        }
    }
    (k, v)
}
