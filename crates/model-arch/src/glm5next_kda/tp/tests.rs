// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of the KDA TP shard plan: row arithmetic only, no GPU and no checkpoint.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use super::*;

// 2026-09-25: The `glm-5.3-flash` KDA geometry (`kernels/gb10/glm-5.3-flash/MODEL.toml`); the
// gate rank is `f_a_proj`'s row count.
const H: usize = 4096;
const HEADS: usize = 64;
const HD: usize = 128;
const CONV_K: usize = 4;
const GATE_RANK: usize = 128;
/// 2026-09-25: One KDA layer's bytes at this geometry: the sum of the per-tensor sizes below.
const KDA_LAYER_BYTES: usize = 275_481_088;
const KDA_LAYERS: usize = 34;

fn plan(tp_rank: usize, tp_size: usize) -> KdaTpPlan {
    KdaTpPlan::new(tp_rank, tp_size, H, HD, HEADS, CONV_K, GATE_RANK).expect("valid geometry")
}

/// 2026-09-25: Every tensor's full size at this geometry, and the block total over `KDA_LAYERS`.
#[test]
fn full_shapes_match_the_reference_checkpoint() {
    let p = plan(0, 1);
    let expect: &[(&str, usize)] = &[
        ("q_proj", 67_108_864),
        ("k_proj", 67_108_864),
        ("v_proj", 67_108_864),
        ("q_conv1d", 65_536),
        ("k_conv1d", 65_536),
        ("v_conv1d", 65_536),
        ("f_a_proj", 1_048_576),
        ("f_b_proj", 2_097_152),
        ("g_a_proj", 1_048_576),
        ("g_b_proj", 2_097_152),
        ("b_proj", 524_288),
        ("A_log", 256),
        ("dt_bias", 32_768),
        ("o_norm", 256),
        ("o_proj", 67_108_864),
    ];
    for (name, bytes) in expect {
        let t = p
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"));
        assert_eq!(t.full_bytes(), *bytes, "{name} full size");
    }
    assert_eq!(p.tensors.len(), expect.len(), "tensor count drifted");
    assert_eq!(p.full_bytes(), KDA_LAYER_BYTES, "per-layer KDA bytes");
    assert_eq!(p.full_bytes() * KDA_LAYERS, 9_366_356_992);
}

/// 2026-09-25: TP=1 keeps full tensors at zero offsets, with no reduce.
#[test]
fn tp1_is_inert() {
    let p = plan(0, 1);
    assert!(!p.needs_output_all_reduce());
    assert_eq!(p.local_heads, HEADS);
    for t in &p.tensors {
        assert_eq!(t.local_rows, t.full_rows, "{} rows", t.name);
        assert_eq!(t.local_row_elems, t.full_row_elems, "{} row elems", t.name);
        assert_eq!(t.src_row_offset, 0, "{} row offset", t.name);
        assert_eq!(t.src_col_offset, 0, "{} col offset", t.name);
    }
    assert_eq!(p.local_bytes(), p.full_bytes());
}

/// 2026-09-25: At TP=2 every sharded tensor is partitioned exactly: disjoint slices, complete
/// cover.
#[test]
fn tp2_partitions_every_sharded_tensor_exactly() {
    let (r0, r1) = (plan(0, 2), plan(1, 2));
    assert!(r0.needs_output_all_reduce());
    assert_eq!(r0.local_heads, 32);
    assert_eq!(r1.local_heads, 32);

    for (a, b) in r0.tensors.iter().zip(r1.tensors.iter()) {
        assert_eq!(a.name, b.name, "tensor order must match across ranks");
        match a.kind {
            KdaShard::Replicated => {
                assert_eq!(a.local_bytes(), a.full_bytes(), "{} must replicate", a.name);
                assert_eq!(b.local_bytes(), b.full_bytes(), "{} must replicate", b.name);
            }
            KdaShard::HeadRows | KdaShard::ChannelRows => {
                assert_eq!(a.src_row_offset, 0, "{} rank0 starts at 0", a.name);
                assert_eq!(
                    b.src_row_offset, a.local_rows,
                    "{} rank1 must start where rank0 ends",
                    a.name
                );
                assert_eq!(
                    a.local_rows + b.local_rows,
                    a.full_rows,
                    "{} rows must partition exactly",
                    a.name
                );
                assert_eq!(a.local_bytes() + b.local_bytes(), a.full_bytes());
            }
            KdaShard::ChannelCols => {
                // 2026-09-25: Row-parallel: all rows, half the input columns each.
                assert_eq!(a.local_rows, a.full_rows, "{} keeps every row", a.name);
                assert_eq!(a.src_col_offset, 0);
                assert_eq!(b.src_col_offset, a.local_row_elems);
                assert_eq!(a.local_row_elems + b.local_row_elems, a.full_row_elems);
                assert_eq!(a.local_bytes() + b.local_bytes(), a.full_bytes());
            }
        }
    }
}

/// 2026-09-25: `A_log` is per head and `dt_bias` per channel, so they shard in different units.
#[test]
fn a_log_and_dt_bias_shard_at_different_granularity() {
    let p = plan(1, 2);
    let a = p.get("A_log").unwrap();
    let dt = p.get("dt_bias").unwrap();

    assert_eq!(a.full_rows, HEADS, "A_log is one entry per head");
    assert_eq!(dt.full_rows, HEADS * HD, "dt_bias is one entry per channel");
    assert_eq!(a.local_rows, 32);
    assert_eq!(dt.local_rows, 4096);
    assert_eq!(a.src_row_offset, 32);
    assert_eq!(dt.src_row_offset, 4096);
    assert_ne!(
        a.local_rows, dt.local_rows,
        "if these ever match, one of them is being sharded in the wrong unit"
    );
}

/// 2026-09-25: `o_norm` is `[head_dim]` and replicated.
#[test]
fn o_norm_replicates() {
    let p = plan(1, 2);
    let n = p.get("o_norm").unwrap();
    assert_eq!(n.kind, KdaShard::Replicated);
    assert_eq!(n.full_rows, HD, "o_norm is per head_dim, not per head");
    assert_eq!(n.local_bytes(), 256);
}

/// 2026-09-25: The low-rank gate down-projections replicate; the up-projections shard.
#[test]
fn gate_down_projections_replicate_up_projections_shard() {
    let p = plan(1, 2);
    for down in ["f_a_proj", "g_a_proj"] {
        let t = p.get(down).unwrap();
        assert_eq!(t.kind, KdaShard::Replicated, "{down} must replicate");
        assert_eq!(t.full_rows, GATE_RANK);
        assert_eq!(t.local_bytes(), t.full_bytes());
    }
    for up in ["f_b_proj", "g_b_proj"] {
        let t = p.get(up).unwrap();
        assert_eq!(t.kind, KdaShard::ChannelRows, "{up} must shard");
        assert_eq!(t.local_rows, HEADS * HD / 2);
        assert_eq!(t.full_row_elems, GATE_RANK, "{up} input dim is the rank");
    }
}

/// 2026-09-25: `o_proj` is row-parallel on its input dim, so the output needs the reduce.
#[test]
fn o_proj_is_row_parallel() {
    let p = plan(1, 2);
    let o = p.get("o_proj").unwrap();
    assert_eq!(o.kind, KdaShard::ChannelCols);
    assert_eq!(o.full_rows, H, "output dim is hidden and is never sharded");
    assert_eq!(o.full_row_elems, HEADS * HD);
    assert_eq!(o.local_row_elems, HEADS * HD / 2);
    assert_eq!(o.src_col_offset, HEADS * HD / 2);
    assert!(p.needs_output_all_reduce());
}

/// 2026-09-25: At TP=2 each rank stores the replicated tensors plus half of the rest.
#[test]
fn tp2_local_bytes_are_exact() {
    let p = plan(0, 2);
    // 2026-09-25: The replicated tensors: f_a + g_a + o_norm.
    let replicated = 1_048_576 + 1_048_576 + 256;
    let expect = replicated + (KDA_LAYER_BYTES - replicated) / 2;
    assert_eq!(p.local_bytes(), expect);
    assert_eq!(plan(1, 2).local_bytes(), expect);
    assert_eq!(
        plan(0, 2).local_bytes() + plan(1, 2).local_bytes(),
        KDA_LAYER_BYTES + replicated
    );
}

/// 2026-09-25: A head count that `tp_size` does not divide is refused.
#[test]
fn indivisible_head_count_is_rejected() {
    let e = KdaTpPlan::new(0, 3, H, HD, HEADS, CONV_K, GATE_RANK).unwrap_err();
    assert!(e.to_string().contains("divisible"), "unexpected error: {e}");
}

/// 2026-09-25: The 256-channel rule is checked on the local q|k width.
#[test]
fn local_qk_channel_contract_is_enforced() {
    assert!(KdaTpPlan::new(0, 2, H, HD, HEADS, CONV_K, GATE_RANK).is_ok());
    let e = KdaTpPlan::new(0, 2, H, 1, 2, CONV_K, GATE_RANK).unwrap_err();
    assert!(e.to_string().contains("256"), "unexpected error: {e}");
}

#[test]
fn tp_rank_must_be_in_range() {
    assert!(KdaTpPlan::new(2, 2, H, HD, HEADS, CONV_K, GATE_RANK).is_err());
}
