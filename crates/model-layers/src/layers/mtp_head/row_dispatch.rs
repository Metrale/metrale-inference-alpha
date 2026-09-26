// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Kernel selection for the n-row BF16 projections of the
//! batched drafter propose (`gemm_rows`), as a pure function of the width,
//! the shape and two env levers.
//!
//! The batched GEMV (`dense_gemv_bf16_batchm`) streams each weight once for
//! all rows. It serves m in 2..=[`DENSE_GEMV_BATCHM_DECODE_MAX_M`], a policy
//! band below the kernel's compile-time `MAX_M` (16). Moving the upper edge
//! changes which kernel, and so which accumulation order, a width runs.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - `RowKernel::Batchm` is chosen only for m in the band and K a multiple
//!   of 8.
//! - Every other case follows the fallback rule: the tile GEMM for
//!   N >= `TILE_N_MIN`, or for smaller N at m >= 8 without
//!   `METRALE_MTP_KV_GEMV`, when K is a multiple of 8; else the per-row loop.

use crate::layers::ops::DENSE_GEMV_BATCHM_DECODE_MAX_M;

/// 2026-09-25: N at or above which the fallback rule takes the pipelined
/// tile GEMM rather than the per-row GEMV loop.
const TILE_N_MIN: u32 = 4096;

/// 2026-09-25: Which kernel one `gemm_rows` projection dispatches to after
/// the tensor-core GEMV declines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowKernel {
    /// 2026-09-25: `dense_gemv_bf16_batchm`: one weight pass for all rows.
    Batchm,
    /// 2026-09-25: `dense_gemm_bf16_pipelined`: tensor-core tile GEMM.
    Pipelined,
    /// 2026-09-25: `dense_gemv_bf16`, once per row.
    GemvLoop,
}

/// 2026-09-25: Select the kernel for an `[m, k] x [n, k]^T` drafter
/// projection. Every input is an argument, both env levers included, so the
/// tests below cover the whole table.
///
/// * `batchm_ready`: the `dense_gemv_bf16_batchm` handle resolved
///   (`try_kernel` returns handle 0 on a miss).
/// * `kv_gemv_pinned`: `METRALE_MTP_KV_GEMV` is set; projections with
///   N < `TILE_N_MIN` stay on the per-row GEMV loop at every width.
/// * `small_m_tier_off`: `METRALE_NO_DRAFTER_SMALL_M_TIER=1`; the batched
///   GEMV is never chosen.
pub(crate) fn drafter_row_kernel(
    m: usize,
    n: u32,
    k: u32,
    batchm_ready: bool,
    kv_gemv_pinned: bool,
    small_m_tier_off: bool,
) -> RowKernel {
    let small_n = n < TILE_N_MIN;
    // 2026-09-25: `dense_gemv_bf16_batchm` reads the rows `A + t*K` and
    // `B + n*K` as `uint4` (16-byte loads), which are aligned only when K is
    // a multiple of 8. An unaligned K follows the fallback rule.
    let k_vec8 = (k & 7) == 0;
    if batchm_ready
        && !small_m_tier_off
        && k_vec8
        && (2..=DENSE_GEMV_BATCHM_DECODE_MAX_M as usize).contains(&m)
        && !(small_n && kv_gemv_pinned)
    {
        return RowKernel::Batchm;
    }

    // 2026-09-25: The fallback rule, for every case the band does not take.
    let small_n_tile = m >= 8 && k_vec8 && !kv_gemv_pinned;
    if (!small_n || small_n_tile) && k_vec8 {
        RowKernel::Pipelined
    } else {
        RowKernel::GemvLoop
    }
}

/// 2026-09-25: `METRALE_NO_DRAFTER_SMALL_M_TIER=1` (exactly `1`) turns the
/// batched-GEMV tier off. Read once per process.
pub(crate) fn small_m_tier_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var("METRALE_NO_DRAFTER_SMALL_M_TIER")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// 2026-09-25: `METRALE_MTP_KV_GEMV` set to any value keeps the small-N
/// projections on the per-row GEMV loop. Read once per process.
pub(crate) fn kv_gemv_pinned() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_MTP_KV_GEMV").is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: `(label, N, K)` of the eight weight-bearing projections of
    /// one draft position, in forward order, for hidden 5120, 24 query heads,
    /// 4 KV heads, head_dim 256 and intermediate 17408.
    const DRAFTER_SHAPES: &[(&str, u32, u32)] = &[
        ("fc", 5120, 10240),
        ("q_proj", 12288, 5120),
        ("k_proj", 1024, 5120),
        ("v_proj", 1024, 5120),
        ("o_proj", 5120, 6144),
        ("ffn_gate", 17408, 5120),
        ("ffn_up", 17408, 5120),
        ("ffn_down", 5120, 17408),
    ];

    /// 2026-09-25: The fallback rule, written out independently of
    /// `drafter_row_kernel` so the tests compare against a separate statement
    /// of it.
    fn legacy(m: usize, n: u32, k: u32, kv_gemv_pinned: bool) -> RowKernel {
        let small_n_tile = m >= 8 && (k & 7) == 0 && !kv_gemv_pinned;
        if (n >= 4096 || small_n_tile) && (k & 7) == 0 {
            RowKernel::Pipelined
        } else {
            RowKernel::GemvLoop
        }
    }

    /// 2026-09-25: Across the band every projection, the N=1024 K/V ones
    /// included, takes the batched GEMV.
    #[test]
    fn every_projection_batches_across_the_covered_widths() {
        for m in 2..=DENSE_GEMV_BATCHM_DECODE_MAX_M as usize {
            for &(label, n, k) in DRAFTER_SHAPES {
                assert_eq!(
                    drafter_row_kernel(m, n, k, true, false, false),
                    RowKernel::Batchm,
                    "{label} at m={m} must take the batched GEMV"
                );
            }
        }
    }

    /// 2026-09-25: Outside the band the choice is the fallback rule.
    #[test]
    fn widths_outside_the_tier_are_untouched() {
        let outside = [1usize, 9, 10, 12, 16, 17, 24, 32, 64];
        for m in outside {
            for &(label, n, k) in DRAFTER_SHAPES {
                assert_eq!(
                    drafter_row_kernel(m, n, k, true, false, false),
                    legacy(m, n, k, false),
                    "{label} at m={m} must keep the pre-tier dispatch"
                );
            }
        }
    }

    /// 2026-09-25: With `small_m_tier_off` the choice is the fallback rule
    /// for every width, shape and `kv_gemv_pinned`.
    #[test]
    fn kill_switch_restores_the_pre_tier_dispatch_exactly() {
        for m in 1..=64usize {
            for &(label, n, k) in DRAFTER_SHAPES {
                for kv_pin in [false, true] {
                    assert_eq!(
                        drafter_row_kernel(m, n, k, true, kv_pin, true),
                        legacy(m, n, k, kv_pin),
                        "{label} m={m} kv_pin={kv_pin} under the kill switch"
                    );
                }
            }
        }
    }

    /// 2026-09-25: Without the batchm handle the choice is the fallback rule.
    #[test]
    fn missing_kernel_handle_falls_back_like_the_kill_switch() {
        for m in 1..=16usize {
            for &(_, n, k) in DRAFTER_SHAPES {
                assert_eq!(
                    drafter_row_kernel(m, n, k, false, false, false),
                    legacy(m, n, k, false),
                );
            }
        }
    }

    /// 2026-09-25: `kv_gemv_pinned` keeps small-N projections on the per-row
    /// loop at every width of the band; large-N projections still batch.
    #[test]
    fn kv_gemv_lever_still_pins_small_n_at_every_width() {
        for m in 2..=DENSE_GEMV_BATCHM_DECODE_MAX_M as usize {
            assert_eq!(
                drafter_row_kernel(m, 1024, 5120, true, true, false),
                RowKernel::GemvLoop,
                "K/V at m={m} under METRALE_MTP_KV_GEMV"
            );
            assert_eq!(
                drafter_row_kernel(m, 17408, 5120, true, true, false),
                RowKernel::Batchm,
                "ffn_gate at m={m} is unaffected by METRALE_MTP_KV_GEMV"
            );
        }
    }

    #[test]
    fn never_selects_batchm_above_the_kernel_row_cap() {
        for m in 1..=128usize {
            for &(_, n, k) in DRAFTER_SHAPES {
                for (batchm, kv_pin, off) in [
                    (true, false, false),
                    (true, true, false),
                    (true, false, true),
                    (false, false, false),
                ] {
                    if drafter_row_kernel(m, n, k, batchm, kv_pin, off) == RowKernel::Batchm {
                        assert!(
                            (2..=DENSE_GEMV_BATCHM_DECODE_MAX_M as usize).contains(&m),
                            "batchm selected at m={m}, outside 2..={DENSE_GEMV_BATCHM_DECODE_MAX_M}"
                        );
                    }
                }
            }
        }
    }

    /// 2026-09-25: An unaligned K takes the per-row loop, as the fallback
    /// rule does: the batched GEMV and the pipelined GEMM both load rows 16
    /// bytes at a time.
    #[test]
    fn unaligned_k_stays_on_the_per_row_loop() {
        for m in [1usize, 2, 4, 8, 16] {
            for &(_, n, _) in DRAFTER_SHAPES {
                assert_eq!(
                    drafter_row_kernel(m, n, 5123, true, false, false),
                    RowKernel::GemvLoop,
                    "m={m} n={n} with an unaligned K"
                );
                assert_eq!(
                    drafter_row_kernel(m, n, 5123, true, false, false),
                    legacy(m, n, 5123, false),
                    "m={m} n={n}: unaligned K must match the pre-tier dispatch"
                );
            }
        }
    }
}
