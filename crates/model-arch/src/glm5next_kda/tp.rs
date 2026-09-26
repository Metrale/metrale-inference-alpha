// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3-Flash KDA tensor-parallel shard plan: head-parallel, one all-reduce after
//! `o_proj`.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - [`KdaTpPlan::new`] refuses `tp_rank >= tp_size`, a head count not divisible by `tp_size`,
//!   and a local q|k width that is not a multiple of 256.
//! - Sharded tensors split into equal, contiguous per-rank ranges ordered by rank.
//!
//! The plan is pure data, so the per-rank row arithmetic is tested without a GPU; `tp_bind`
//! applies it. Each rank owns a contiguous head range, `o_proj` is row-parallel, and one
//! all-reduce follows it, as in the GDN head-parallel helpers (`crate::tp_shard::gdn`). Unlike
//! GDN, KDA's `q/k/v_proj` and `q/k/v_conv1d` are separate tensors on disk, so each is sliced on
//! its own and no segmented copy is needed.
//!
//! * `A_log` is per head and `dt_bias` per channel, so they shard in different units.
//! * `o_norm` is `[head_dim]`, shared by every head, so it is replicated.
//! * `f_a`/`g_a` are down-projections `[rank, hidden]` and are replicated; only the `_b`
//!   up-projections carry head structure.
//! * `o_proj` is `[hidden, heads * head_dim]` sliced on its input dimension; each rank's output is
//!   partial until the all-reduce.

use anyhow::{Result, bail};
use metrale_config::ModelConfig;

/// 2026-09-25: How one KDA tensor maps onto TP ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdaShard {
    /// 2026-09-25: Every rank holds the whole tensor.
    Replicated,
    /// 2026-09-25: Leading dim is `heads`: slice by head range.
    HeadRows,
    /// 2026-09-25: Leading dim is `heads * head_dim`: slice by channel range.
    ChannelRows,
    /// 2026-09-25: Trailing (input) dim is `heads * head_dim`: a row-parallel GEMM, sliced on the
    /// input dim, whose output is all-reduced.
    ChannelCols,
}

/// 2026-09-25: One tensor's placement, in `[rows, row_elems]` terms. For
/// [`KdaShard::ChannelCols`] the sharded axis is `row_elems`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KdaTensorPlan {
    pub name: &'static str,
    pub kind: KdaShard,
    pub elem_bytes: usize,
    /// 2026-09-25: Rows of the full, on-disk tensor.
    pub full_rows: usize,
    /// 2026-09-25: Elements per row of the full tensor.
    pub full_row_elems: usize,
    /// 2026-09-25: Rows this rank keeps.
    pub local_rows: usize,
    /// 2026-09-25: Elements per row this rank keeps (differs from full only for `ChannelCols`).
    pub local_row_elems: usize,
    /// 2026-09-25: Offset, in rows, of this rank's slice. 0 for `Replicated` and `ChannelCols`.
    pub src_row_offset: usize,
    /// 2026-09-25: Offset, in elements, of this rank's column slice. 0 except for `ChannelCols`.
    pub src_col_offset: usize,
}

impl KdaTensorPlan {
    /// 2026-09-25: Bytes this rank stores for this tensor.
    pub fn local_bytes(&self) -> usize {
        self.local_rows * self.local_row_elems * self.elem_bytes
    }
    /// 2026-09-25: Bytes the full tensor occupies on disk.
    pub fn full_bytes(&self) -> usize {
        self.full_rows * self.full_row_elems * self.elem_bytes
    }
}

const BF16: usize = 2;
const F32: usize = 4;

/// 2026-09-25: The per-rank shard plan for one KDA block.
#[derive(Debug, Clone)]
pub struct KdaTpPlan {
    pub tp_rank: usize,
    pub tp_size: usize,
    pub hidden: usize,
    pub head_dim: usize,
    /// 2026-09-25: Pre-shard head count (all ranks combined).
    pub full_heads: usize,
    /// 2026-09-25: Heads this rank owns.
    pub local_heads: usize,
    pub conv_kernel: usize,
    /// 2026-09-25: Low-rank width of the `f`/`g` gate bottleneck.
    pub gate_rank: usize,
    pub tensors: Vec<KdaTensorPlan>,
}

impl KdaTpPlan {
    /// 2026-09-25: Build from a `ModelConfig` whose linear-head counts are already per rank
    /// (`serve_phases::topology` divides them), so the full count is `local * tp_size`, as in
    /// `TpGdnDims::from_config`.
    ///
    /// `gate_rank` is not a config key; the loader reads it as the `f_a_proj` row count.
    pub fn from_config(config: &ModelConfig, gate_rank: usize) -> Result<Self> {
        let tp_size = config.tp_world_size.max(1);
        Self::new(
            config.tp_rank,
            tp_size,
            config.hidden_size,
            config.linear_key_head_dim,
            config.linear_num_key_heads * tp_size,
            config.linear_conv_kernel_dim,
            gate_rank,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tp_rank: usize,
        tp_size: usize,
        hidden: usize,
        head_dim: usize,
        full_heads: usize,
        conv_kernel: usize,
        gate_rank: usize,
    ) -> Result<Self> {
        if tp_rank >= tp_size {
            bail!("tp_rank {tp_rank} >= tp_size {tp_size}");
        }
        if tp_size == 0 || head_dim == 0 || full_heads == 0 {
            bail!("degenerate KDA TP geometry: heads={full_heads} head_dim={head_dim}");
        }
        if !full_heads.is_multiple_of(tp_size) {
            bail!("KDA TP requires heads ({full_heads}) divisible by tp_size ({tp_size})");
        }
        let local_heads = full_heads / tp_size;

        // 2026-09-25: `Glm5NextKdaConfig::validate`'s 256-channel rule, re-checked on the local
        // q|k width.
        let local_qk_channels = 2 * local_heads * head_dim;
        if !local_qk_channels.is_multiple_of(256) {
            bail!(
                "KDA TP: local qk_channels ({local_qk_channels}) must be a multiple of 256 \
                 (heads={local_heads}, head_dim={head_dim}); causal_conv1d_update_l2norm \
                 hardcodes 2 heads per 256-thread block"
            );
        }

        let full_ch = full_heads * head_dim;
        let local_ch = local_heads * head_dim;
        let ch_off = tp_rank * local_ch;
        let head_off = tp_rank * local_heads;

        let rows = |name, kind, elem_bytes, full_rows, full_row_elems| {
            let (local_rows, local_row_elems, src_row_offset, src_col_offset) = match kind {
                KdaShard::Replicated => (full_rows, full_row_elems, 0, 0),
                KdaShard::HeadRows => (full_rows / tp_size, full_row_elems, head_off, 0),
                KdaShard::ChannelRows => (full_rows / tp_size, full_row_elems, ch_off, 0),
                // 2026-09-25: Row-parallel: slice the input (column) dim, keep every row.
                KdaShard::ChannelCols => (full_rows, full_row_elems / tp_size, 0, ch_off),
            };
            KdaTensorPlan {
                name,
                kind,
                elem_bytes,
                full_rows,
                full_row_elems,
                local_rows,
                local_row_elems,
                src_row_offset,
                src_col_offset,
            }
        };

        let tensors = vec![
            // 2026-09-25: `[heads * head_dim, hidden]`, column-parallel by output channel.
            rows("q_proj", KdaShard::ChannelRows, BF16, full_ch, hidden),
            rows("k_proj", KdaShard::ChannelRows, BF16, full_ch, hidden),
            rows("v_proj", KdaShard::ChannelRows, BF16, full_ch, hidden),
            // 2026-09-25: `[heads * head_dim, conv_kernel]` each, separate on disk.
            rows(
                "q_conv1d",
                KdaShard::ChannelRows,
                BF16,
                full_ch,
                conv_kernel,
            ),
            rows(
                "k_conv1d",
                KdaShard::ChannelRows,
                BF16,
                full_ch,
                conv_kernel,
            ),
            rows(
                "v_conv1d",
                KdaShard::ChannelRows,
                BF16,
                full_ch,
                conv_kernel,
            ),
            // 2026-09-25: Low-rank gates: `_a` down-projects (replicated), `_b` up-projects into
            // channel space (sharded).
            rows("f_a_proj", KdaShard::Replicated, BF16, gate_rank, hidden),
            rows("f_b_proj", KdaShard::ChannelRows, BF16, full_ch, gate_rank),
            rows("g_a_proj", KdaShard::Replicated, BF16, gate_rank, hidden),
            rows("g_b_proj", KdaShard::ChannelRows, BF16, full_ch, gate_rank),
            // 2026-09-25: beta, one row per head.
            rows("b_proj", KdaShard::HeadRows, BF16, full_heads, hidden),
            rows("A_log", KdaShard::HeadRows, F32, full_heads, 1),
            rows("dt_bias", KdaShard::ChannelRows, F32, full_ch, 1),
            // 2026-09-25: `[head_dim]`, shared by every head, so replicated.
            rows("o_norm", KdaShard::Replicated, BF16, head_dim, 1),
            // 2026-09-25: `[hidden, heads * head_dim]`, row-parallel; all-reduce after.
            rows("o_proj", KdaShard::ChannelCols, BF16, hidden, full_ch),
        ];

        Ok(Self {
            tp_rank,
            tp_size,
            hidden,
            head_dim,
            full_heads,
            local_heads,
            conv_kernel,
            gate_rank,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Option<&KdaTensorPlan> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// 2026-09-25: Total bytes this rank stores for one KDA block.
    pub fn local_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.local_bytes()).sum()
    }

    /// 2026-09-25: Total bytes one KDA block occupies on disk.
    pub fn full_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.full_bytes()).sum()
    }

    /// 2026-09-25: Whether the layer must all-reduce after `o_proj`. False at `tp_size == 1`, where
    /// the row-parallel slice is the whole tensor.
    pub fn needs_output_all_reduce(&self) -> bool {
        self.tp_size > 1
    }
}

#[cfg(test)]
mod tests;
