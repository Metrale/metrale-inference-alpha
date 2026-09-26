// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host tests of the load-time transforms in `build.rs`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;

fn cfg(local_heads: usize) -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 64,
        index_heads: 4,
        index_head_dim: 8,
        index_kpool: 4,
        index_topk: 16,
        always_select_tail: true,
        local_heads,
        q_lora_rank: 6,
        kv_lora_rank: 512,
        qk_nope_head_dim: 3,
        qk_rope_head_dim: 0,
        v_head_dim: 5,
        max_context: 16_384,
    }
}

/// 2026-09-25: `absorb_q` computes
/// `q_absorb[h*kvl + c][k] = Σ_r kv_b[h*(nope+vd) + r][c] · q_b[h*nope + r][k]`, checked
/// against a scalar recomputation.
#[test]
fn absorb_q_matches_the_contraction() {
    let mut c = cfg(2);
    c.kv_lora_rank = 4;
    let (heads, nope, vd, kvl, ql) = (2usize, 3usize, 5usize, 4usize, 6usize);
    let q_b: Vec<f32> = (0..heads * nope * ql)
        .map(|i| (i as f32) * 0.25 - 1.0)
        .collect();
    let kv_b: Vec<f32> = (0..heads * (nope + vd) * kvl)
        .map(|i| 1.0 - (i as f32) * 0.125)
        .collect();

    let got = absorb_q(&c, &q_b, &kv_b, heads).unwrap();
    assert_eq!(got.len(), heads * kvl * ql);
    for h in 0..heads {
        for cc in 0..kvl {
            for k in 0..ql {
                let want: f32 = (0..nope)
                    .map(|r| kv_b[(h * (nope + vd) + r) * kvl + cc] * q_b[(h * nope + r) * ql + k])
                    .sum();
                assert!(
                    (got[(h * kvl + cc) * ql + k] - want).abs() < 1e-5,
                    "h{h} c{cc} k{k}"
                );
            }
        }
    }
}

/// 2026-09-25: `q_b_proj` and `kv_b_proj` have different per-head widths (256 and 512 on
/// GLM-5.3); `absorb_q` refuses either at a wrong element count.
#[test]
fn absorb_q_refuses_mismatched_tensors() {
    let mut c = cfg(2);
    c.kv_lora_rank = 4;
    let ok_q = vec![0f32; 2 * 3 * 6];
    let ok_kv = vec![0f32; 2 * (3 + 5) * 4];
    assert!(absorb_q(&c, &ok_q, &ok_kv, 2).is_ok());
    assert!(absorb_q(&c, &[0f32; 2 * 8 * 6], &ok_kv, 2).is_err());
    assert!(absorb_q(&c, &ok_q, &[0f32; 2 * 3 * 4], 2).is_err());
}

/// 2026-09-25: `absorb_q` refuses a nonzero `qk_rope_head_dim`.
#[test]
fn absorb_q_refuses_a_rope_section() {
    let mut c = cfg(2);
    c.kv_lora_rank = 4;
    c.qk_rope_head_dim = 2;
    assert!(absorb_q(&c, &vec![0f32; 2 * 5 * 6], &vec![0f32; 2 * 8 * 4], 2).is_err());
}

/// 2026-09-25: Rank 1's rows of the full absorption equal absorbing head 1 alone, so taking
/// a rank's rows after absorbing keeps each `q_b` head paired with its own `kv_b` head.
#[test]
fn absorbing_before_sharding_keeps_heads_paired() {
    let mut c = cfg(1);
    c.kv_lora_rank = 4;
    let (heads, nope, vd, kvl, ql) = (2usize, 3usize, 5usize, 4usize, 6usize);
    let q_b: Vec<f32> = (0..heads * nope * ql).map(|i| i as f32).collect();
    let kv_b: Vec<f32> = (0..heads * (nope + vd) * kvl)
        .map(|i| (i as f32) * 0.5)
        .collect();
    let full = absorb_q(&c, &q_b, &kv_b, heads).unwrap();

    let rank1 = row_slice(&full, ql, kvl, 2 * kvl);
    let mut c1 = c;
    c1.local_heads = 1;
    let solo = absorb_q(&c1, &q_b[nope * ql..], &kv_b[(nope + vd) * kvl..], 1).unwrap();
    assert_eq!(rank1, solo, "head pairing must survive the shard");
}

/// 2026-09-25: The `index_heads^-0.5` factor `build_dsa_weights` folds into `weights_proj`
/// is 1/2 at 4 heads. This test does not call `build_dsa_weights`.
#[test]
fn the_head_scale_is_folded_into_weights_proj() {
    let c = cfg(64);
    let scale = (c.index_heads as f32).powf(-0.5);
    assert!((scale - 0.5).abs() < 1e-6, "4 heads => 1/2");
    let raw = [2.0f32, -4.0, 0.0];
    let folded: Vec<f32> = raw.iter().map(|x| x * scale).collect();
    assert_eq!(folded, vec![1.0, -2.0, 0.0]);
}

/// 2026-09-25: `col_slice`, the `o_proj` shard, takes a column range of every row;
/// `row_slice` of the same buffer gives different elements.
#[test]
fn o_proj_is_sliced_on_columns_not_rows() {
    let full: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let got = col_slice(&full, 4, 2, 4);
    assert_eq!(got, vec![2.0, 3.0, 6.0, 7.0, 10.0, 11.0]);
    let wrong = row_slice(&full, 4, 0, 1);
    assert_eq!(wrong, vec![0.0, 1.0, 2.0, 3.0]);
    let wrong_half = row_slice(&full, 4, 0, 2);
    assert_ne!(
        wrong_half.len(),
        got.len(),
        "the two axes give different shapes here"
    );
    let same_len: Vec<f32> = full[..6].to_vec();
    assert_eq!(same_len.len(), got.len());
    assert_ne!(
        same_len, got,
        "same length, wrong values — the real failure mode"
    );
}

/// 2026-09-25: Applying `o_absorb` to the latent equals expanding the latent through
/// `kv_b_proj`'s V half and applying the raw `o_proj`, within 1e-4.
///
/// The expansion is `glm5next_dsa_ref::expand_kv`, the CPU reference, so the test also
/// checks where the V half starts.
///
/// Absorbed decode leaves `attn_out[h] = Σ_t p[h][t] · latent[t]` in latent space. Since
/// `v_t[h] = KV_B_V[h] · latent[t]` is linear, `Σ_t p[h][t] · v_t[h] = KV_B_V[h] · attn_out[h]`,
/// so applying the absorbed weight to `attn_out` is exact, not an approximation.
#[test]
fn absorb_o_equals_expand_then_project() {
    use crate::glm5next_dsa_ref::{DsaDims, expand_kv};

    let mut c = cfg(2);
    c.hidden = 7;
    c.kv_lora_rank = 4;
    let (heads, nope, vd, kvl, hidden) = (
        2usize,
        c.qk_nope_head_dim,
        c.v_head_dim,
        c.kv_lora_rank,
        c.hidden,
    );

    let f = |i: usize, salt: usize| ((i * 37 + salt * 11) % 23) as f32 * 0.031 - 0.29;
    let o_proj: Vec<f32> = (0..hidden * heads * vd).map(|i| f(i, 1)).collect();
    let kv_b: Vec<f32> = (0..heads * (nope + vd) * kvl).map(|i| f(i, 5)).collect();
    let latent: Vec<f32> = (0..heads * kvl).map(|i| f(i, 9)).collect();

    let o_absorb = absorb_o(&c, &o_proj, &kv_b, heads).unwrap();
    assert_eq!(o_absorb.len(), hidden * heads * kvl);

    let dims = DsaDims {
        hidden,
        index_heads: c.index_heads,
        index_head_dim: c.index_head_dim,
        index_kpool: c.index_kpool,
        index_topk: c.index_topk,
        always_select_tail: c.always_select_tail,
        heads,
        q_lora_rank: c.q_lora_rank,
        kv_lora_rank: kvl,
        qk_nope_head_dim: nope,
        qk_rope_head_dim: 0,
        v_head_dim: vd,
    };

    // 2026-09-25: Reference: expand each head's latent row through kv_b, keep that head's
    // V slice.
    let mut v = vec![0f32; heads * vd];
    for h in 0..heads {
        let (_, v_all) = expand_kv(&latent[h * kvl..(h + 1) * kvl], &kv_b, dims, 1);
        v[h * vd..(h + 1) * vd].copy_from_slice(&v_all[h * vd..(h + 1) * vd]);
    }

    for i in 0..hidden {
        let reference: f32 = (0..heads * vd)
            .map(|j| o_proj[i * heads * vd + j] * v[j])
            .sum();
        let absorbed: f32 = (0..heads * kvl)
            .map(|j| o_absorb[i * heads * kvl + j] * latent[j])
            .sum();
        assert!(
            (reference - absorbed).abs() < 1e-4,
            "row {i}: expand-then-project {reference} != absorbed {absorbed}"
        );
    }
}

/// 2026-09-25: `o_absorb` is `hidden × local_heads·kv_lora_rank`, the K of the output GEMM
/// in `Glm5NextDsaLayer::decode_k`, not `local_heads·v_head_dim`.
#[test]
fn absorb_o_width_matches_the_decode_contraction() {
    let mut c = cfg(2);
    c.hidden = 7;
    c.kv_lora_rank = 4;
    let (heads, nope, vd, kvl) = (2usize, c.qk_nope_head_dim, c.v_head_dim, c.kv_lora_rank);
    assert_ne!(kvl, vd, "the widths must differ or this proves nothing");
    let o = absorb_o(
        &c,
        &vec![0.5f32; c.hidden * heads * vd],
        &vec![0.25f32; heads * (nope + vd) * kvl],
        heads,
    )
    .unwrap();
    assert_eq!(o.len(), c.hidden * heads * kvl);
}

/// 2026-09-25: A plain loop nest over the same contraction: the oracle for
/// `absorb_q_bit_exact`, kept separate from `absorb_q_rows` so the test does not compare
/// the code with itself.
fn absorb_q_reference(
    kv_b: &[f32],
    q_b: &[f32],
    full_heads: usize,
    kvl: usize,
    ql: usize,
    nope: usize,
    vd: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; full_heads * kvl * ql];
    for h in 0..full_heads {
        let kv_base = h * (nope + vd);
        let qb_base = h * nope;
        for c in 0..kvl {
            for k in 0..ql {
                let mut acc = 0f32;
                for r in 0..nope {
                    acc += kv_b[(kv_base + r) * kvl + c] * q_b[(qb_base + r) * ql + k];
                }
                out[(h * kvl + c) * ql + k] = acc;
            }
        }
    }
    out
}

/// 2026-09-25: Threaded `absorb_q_rows` matches the reference bit for bit. Both sum over `r`
/// in ascending order; an epsilon would also accept a reassociated sum.
#[test]
fn absorb_q_bit_exact() {
    // 2026-09-25: 35 rows over 4 threads: chunks of 9, 9, 9 and 8 rows.
    let (full_heads, kvl, ql, nope, vd) = (5usize, 7usize, 11usize, 6usize, 3usize);
    // 2026-09-25: Non-round values of both signs: with exactly representable sums an
    // ordering change would not change the bits.
    let mk = |n: usize, seed: u32| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                ((x % 20_011) as f32 / 20_011.0 - 0.5) * 3.7
            })
            .collect()
    };
    let kv_b = mk(full_heads * (nope + vd) * kvl, 12345);
    let q_b = mk(full_heads * nope * ql, 67890);

    let want = absorb_q_reference(&kv_b, &q_b, full_heads, kvl, ql, nope, vd);

    let rows = full_heads * kvl;
    let mut got = vec![0f32; rows * ql];
    // 2026-09-25: The split is done here rather than through `absorb_threads`, which reads
    // the process environment.
    let threads = 4usize;
    let rows_per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk_idx, chunk) in got.chunks_mut(rows_per * ql).enumerate() {
            let row0 = chunk_idx * rows_per;
            let kv_b = &kv_b;
            let q_b = &q_b;
            scope.spawn(move || {
                super::absorb_q_rows(chunk, row0, kv_b, q_b, kvl, ql, nope, vd);
            });
        }
    });

    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "element {i}: threaded {g:?} != reference {w:?}"
        );
    }
}

/// 2026-09-25: One `absorb_q_rows` call over all rows matches the reference bit for bit.
#[test]
fn absorb_q_single_thread_matches_reference() {
    let (full_heads, kvl, ql, nope, vd) = (3usize, 5usize, 9usize, 4usize, 2usize);
    let mk = |n: usize, seed: u32| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(40_503).wrapping_add(seed);
                ((x % 7_919) as f32 / 7_919.0 - 0.5) * 2.3
            })
            .collect()
    };
    let kv_b = mk(full_heads * (nope + vd) * kvl, 11);
    let q_b = mk(full_heads * nope * ql, 22);
    let want = absorb_q_reference(&kv_b, &q_b, full_heads, kvl, ql, nope, vd);
    let mut got = vec![0f32; full_heads * kvl * ql];
    super::absorb_q_rows(&mut got, 0, &kv_b, &q_b, kvl, ql, nope, vd);
    for (g, w) in got.iter().zip(want.iter()) {
        assert_eq!(g.to_bits(), w.to_bits());
    }
}
