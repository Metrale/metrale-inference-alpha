// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU-free tests of the MoE expert LoRA host helpers: the table
//! packer `pack_expert_tables`, the gather-fold grids and row-to-token map, and
//! the grouped fold's window grid and window partition.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::{
    ExpertTables, gather_bgmv_grids, gather_row_token, grouped_down_wc, grouped_down_windows,
    pack_expert_tables,
};

#[test]
fn empty_entries_none() {
    assert!(pack_expert_tables(&[]).is_none());
}

#[test]
fn single_expert_zero_padded_prefix() {
    // 2026-09-25: Expert 3 adapted; ids 0..3 get the 0 / 0.0 sentinels.
    let t = pack_expert_tables(&[(3u16, 0xAAAA, 0xBBBB, 2.0)]).unwrap();
    assert_eq!(
        t,
        ExpertTables {
            a: vec![0, 0, 0, 0xAAAA],
            b: vec![0, 0, 0, 0xBBBB],
            scale: vec![0.0, 0.0, 0.0, 2.0],
            n_experts: 4,
        }
    );
}

#[test]
fn sparse_experts_dense_table() {
    // 2026-09-25: Experts 0 and 2 adapted; the gap at 1 gets the sentinels.
    let t = pack_expert_tables(&[(0u16, 0x10, 0x20, 0.5), (2u16, 0x30, 0x40, 0.25)]).unwrap();
    assert_eq!(t.a, vec![0x10, 0, 0x30]);
    assert_eq!(t.b, vec![0x20, 0, 0x40]);
    assert_eq!(t.scale, vec![0.5, 0.0, 0.25]);
    assert_eq!(t.n_experts, 3);
}

#[test]
fn entry_order_independent() {
    let t = pack_expert_tables(&[(2u16, 0x30, 0x40, 0.25), (0u16, 0x10, 0x20, 0.5)]).unwrap();
    assert_eq!(t.a, vec![0x10, 0, 0x30]);
    assert_eq!(t.scale, vec![0.5, 0.0, 0.25]);
}

#[test]
fn table_length_is_max_id_plus_one() {
    let t = pack_expert_tables(&[(7u16, 1, 2, 1.0)]).unwrap();
    assert_eq!(t.n_experts, 8);
    assert_eq!(t.a.len(), 8);
    assert_eq!(t.a[7], 1);
    assert_eq!(t.a[..7], [0u64; 7]);
}

#[test]
fn gather_grids_single_token_decode() {
    // 2026-09-25: One token with top_k 8 gives 8 rows; the shrink covers
    // max_rank 16, the expand an n_out of 4096.
    let (shrink, expand) = gather_bgmv_grids(16, 4096, 8);
    assert_eq!(shrink, [4, 8, 1]);
    assert_eq!(expand, [1024, 8, 1]);
}

#[test]
fn gather_grids_verify_flat_rows() {
    // 2026-09-25: grid.y follows the row count; grid.x depends only on the
    // output width.
    let (shrink, expand) = gather_bgmv_grids(32, 4096, 24);
    assert_eq!(shrink, [8, 24, 1]);
    assert_eq!(expand, [1024, 24, 1]);
}

#[test]
fn gather_grids_rank_not_multiple_of_four_rounds_up() {
    let (shrink, _) = gather_bgmv_grids(6, 512, 8);
    assert_eq!(shrink[0], 2);
}

#[test]
fn short_prefill_slot_rows_map_to_owning_token() {
    // 2026-09-25: Every slot s of token t maps back to t, so a gate/up gather
    // (`x_gather == 1`) never reads another token's input.
    for top_k in [1u32, 2, 8] {
        for t in 0..64u32 {
            for s in 0..top_k {
                assert_eq!(
                    gather_row_token(t * top_k + s, top_k),
                    t,
                    "t={t} s={s} k={top_k}"
                );
            }
        }
    }
}

#[test]
fn router_fold_top_k_one_row_is_token() {
    // 2026-09-25: With top_k 1, as the router fold uses, each row is its token.
    for row in [0u32, 1, 3, 4, 255, 4095] {
        assert_eq!(gather_row_token(row, 1), row, "row={row}");
    }
}

#[test]
fn router_fold_grid_covers_all_experts_qwen36() {
    // 2026-09-25: Router fold shape: n_out is the expert count (256), and there is
    // one row per token (4 tokens).
    let (shrink, expand) = gather_bgmv_grids(16, 256, 4);
    assert_eq!(shrink, [4, 4, 1]);
    assert_eq!(expand, [64, 4, 1]);
}

#[test]
fn router_fold_grid_single_token_decode() {
    let (shrink, expand) = gather_bgmv_grids(32, 256, 1);
    assert_eq!(shrink, [8, 1, 1]);
    assert_eq!(expand, [64, 1, 1]);
}

/// 2026-09-25: Independent oracle for `grouped_down_wc`: `ceil(rows / 64)`, at
/// least 1, computed with `u32::div_ceil` on a row count instead of the
/// launcher's saturating window.
fn div_ceil64(n: u32) -> u32 {
    n.div_ceil(64).max(1)
}

#[test]
fn full_window_equals_prechunk_wc() {
    for te in [0u32, 1, 63, 64, 65, 4096, 4097, 131072] {
        assert_eq!(grouped_down_wc(0, te), div_ceil64(te), "te={te}");
    }
}

#[test]
fn window_grid_covers_only_the_slice() {
    // 2026-09-25: grid.y depends on the window length, not on where the window
    // starts.
    assert_eq!(grouped_down_wc(0, 4096), 64);
    assert_eq!(grouped_down_wc(4096, 8192), 64);
    assert_eq!(grouped_down_wc(128000, 131072), 48);
}

#[test]
fn empty_window_still_launches_one_tile() {
    // 2026-09-25: An empty window still launches one tile, and `row_end <
    // row_offset` saturates to an empty window.
    assert_eq!(grouped_down_wc(100, 100), 1);
    assert_eq!(grouped_down_wc(200, 100), 1);
}

#[test]
fn chunks_tile_range_without_gap_or_overlap() {
    // 2026-09-25: The windows partition [0, te) with no gap or overlap, and each
    // window's grid matches its length.
    let cap = 4096u32;
    for te in [1u32, 4096, 4097, 10000, 131072] {
        let mut covered = 0u32;
        let mut prev_end = 0u32;
        for (off, end) in grouped_down_windows(te, cap) {
            assert_eq!(off, prev_end, "gap/overlap at off={off} (te={te})");
            let window = end - off;
            assert!(window <= cap, "window {window} exceeds cap {cap}");
            assert_eq!(grouped_down_wc(off, end), div_ceil64(window), "te={te}");
            covered += window;
            prev_end = end;
        }
        assert_eq!(covered, te, "windows must cover exactly [0,te) for te={te}");
    }
}
