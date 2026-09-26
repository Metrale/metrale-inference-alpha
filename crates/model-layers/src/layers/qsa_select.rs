// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-query prefill selection for the QSA indexer.
//!
//! Owner: model-layers (QSA).
//! Invariants:
//! - Only rows at global positions `>= inert_bound()` are overwritten; rows
//!   below keep the dense attention output.
//!
//! A child module of `qsa` (via `#[path]`), so the indexer's private fields
//! are reachable without widening their visibility.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{QsaIndexer, QsaSeqState};
use crate::layers::ops;

impl QsaIndexer {
    /// 2026-09-25: Per-query prefill selection for a prefill chunk. Chunk rows whose
    /// global position (`seq_start + row`) is at or past the inert bound get
    /// their attention context rows (pre-gate, pre-o_proj) overwritten with
    /// attention over their selected set, read from the paged KV cache, which
    /// must already hold every prior chunk plus this one. Rows below the bound
    /// keep the dense output, which is identical there. `prefill_ingest` must
    /// have run for this chunk (`prefill_inner.rs` calls it before attention).
    /// `METRALE_QSA_NO_PREFILL_SELECT=1` makes this a no-op.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_select(
        &self,
        st: &mut QsaSeqState,
        normed: DevicePtr,
        q_roped: DevicePtr,
        attn_ctx: DevicePtr,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        seq_block_table: &[u32],
        seq_start: usize,
        num_tokens: usize,
        nq: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        scratch: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let bound = self.inert_bound(); // 2026-09-25: first selective global position
        let total = seq_start + num_tokens;
        if total <= bound {
            return Ok(());
        }
        static S2_OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *S2_OFF
            .get_or_init(|| std::env::var("METRALE_QSA_NO_PREFILL_SELECT").as_deref() == Ok("1"))
        {
            return Ok(());
        }
        let diag = {
            static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *D.get_or_init(|| std::env::var("METRALE_QSA_S2_DIAG").as_deref() == Ok("1"))
        };
        // 2026-09-25: METRALE_QSA_S2_DIAG=1: keep the dense context of the last
        // row before the overwrite, and log cosine(dense, selected) after.
        let q_row = nq as usize * self.hd_attn as usize;
        let mut dense_last = Vec::new();
        if diag {
            dense_last = vec![0u8; q_row * 2];
            gpu.copy_d2h_on_stream(
                attn_ctx.offset((num_tokens - 1) * q_row * 2),
                &mut dense_last,
                stream,
            )?;
            // 2026-09-25: Norm probes: row 100, the first selective row and the
            // last row.
            let probe = |row: usize| -> Result<f64> {
                let mut b = vec![0u8; q_row * 2];
                gpu.copy_d2h_on_stream(attn_ctx.offset(row * q_row * 2), &mut b, stream)?;
                Ok(b.chunks_exact(2)
                    .map(|c| {
                        let v =
                            f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64;
                        v * v
                    })
                    .sum::<f64>()
                    .sqrt())
            };
            tracing::warn!(
                "QSA S2 DIAG norms: row100={:.3} first_sel(row {bound})={:.3} last={:.3} q_row={q_row}",
                probe(100)?,
                probe(bound)?,
                probe(num_tokens - 1)?
            );
            let probe_at = |base: DevicePtr, row: usize| -> Result<f64> {
                let mut b = vec![0u8; q_row * 2];
                gpu.copy_d2h_on_stream(base.offset(row * q_row * 2), &mut b, stream)?;
                Ok(b.chunks_exact(2)
                    .map(|c| {
                        let v =
                            f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64;
                        v * v
                    })
                    .sum::<f64>()
                    .sqrt())
            };
            let mut ctx_line = String::new();
            let mut q_line = String::new();
            for row in [128usize, 256, 512, 768, 1024, 1280, 1536, 1792, 1900, 2000] {
                ctx_line += &format!(" {row}:{:.2}", probe_at(attn_ctx, row)?);
            }
            tracing::warn!("QSA S2 DIAG wide:{ctx_line}");
            ctx_line = String::new();
            for row in (2040..2056).step_by(2) {
                ctx_line += &format!(" {row}:{:.2}", probe_at(attn_ctx, row)?);
                q_line += &format!(" {row}:{:.2}", probe_at(q_roped, row)?);
            }
            tracing::warn!("QSA S2 DIAG ctx rows:{ctx_line}");
            tracing::warn!("QSA S2 DIAG   q rows:{q_line}");
        }
        // 2026-09-25: Upload the physical block table for the full context (a
        // selective query attends blocks from every prior chunk).
        let pages_needed = total.div_ceil(block_size as usize);
        anyhow::ensure!(
            seq_block_table.len() >= pages_needed,
            "QSA: block table has {} pages for {} tokens",
            seq_block_table.len(),
            pages_needed
        );
        let tbytes: Vec<u8> = seq_block_table[..pages_needed]
            .iter()
            .flat_map(|b| (*b as i32).to_le_bytes())
            .collect();
        gpu.copy_h2d_async(&tbytes, self.prefill_table_dev, stream)?;
        let block_table_dev = self.prefill_table_dev;
        // 2026-09-25: Must equal ROWS in sizes.rs `qsa_select_scratch`.
        const ROWS: usize = 2048;
        let ratio = self.ratio as usize;
        let topk = self.block_topk as usize;
        let heads = self.n_heads as usize;
        let hd = self.hd as usize;
        let hd_attn = self.hd_attn as usize;
        let qkw = self.qk_width();
        let q_row = nq as usize * hd_attn;

        // 2026-09-25: Scratch layout. The per-call score stride is
        // `total.div_ceil(ratio)`; sizes.rs reserves `max_seq_len.div_ceil(ratio)`.
        let stride = total.div_ceil(ratio);
        let qk_buf = scratch;
        let qpost = scratch.offset(ROWS * qkw * 2);
        let scores = qpost.offset(ROWS * heads * hd * 4);
        let lists = scores.offset(ROWS * stride * 4);

        let first_sel_pos = bound.max(seq_start);
        let n_sel_total = total - first_sel_pos;
        let mut slab = 0usize;
        while slab < n_sel_total {
            let rows = ROWS.min(n_sel_total - slab);
            let first_pos = first_sel_pos + slab;
            let first_row = first_pos - seq_start;

            ops::cublas_bf16_proj_dense(
                normed.offset(first_row * self.hidden as usize * 2),
                self.qk_proj_w,
                qk_buf,
                rows as u32,
                qkw as u32,
                self.hidden,
                stream,
            )
            .context("QSA qk projection (prefill select)")?;
            ops::qsa_qprep_rows(
                gpu,
                self.k_qprep_rows_k,
                qk_buf,
                self.q_norm_w,
                qpost,
                rows as u32,
                first_pos as u32,
                qkw as u32,
                self.n_heads,
                self.hd,
                self.rot,
                self.theta,
                self.eps,
                stream,
            )?;
            // 2026-09-25: `n_blocks_max` is the last row's complete-block count.
            let n_blocks_max = (first_pos + rows) / ratio;
            // 2026-09-25: The tensor-core scorer is used when the target ships it and
            // the shape is 4 heads x 128; METRALE_QSA_SCORE_SCALAR=1 forces the
            // scalar kernel.
            let tc = self.k_score_rows_tc_k.0 != 0
                && self.n_heads == 4
                && self.hd == 128
                && std::env::var("METRALE_QSA_SCORE_SCALAR").as_deref() != Ok("1");
            if tc {
                ops::qsa_score_rows_tc(
                    gpu,
                    self.k_score_rows_tc_k,
                    qpost,
                    st.block_keys,
                    scores,
                    rows as u32,
                    n_blocks_max as u32,
                    first_pos as u32,
                    stride as u32,
                    self.ratio,
                    stream,
                )?;
            } else {
                ops::qsa_score_rows(
                    gpu,
                    self.k_score_rows_k,
                    qpost,
                    st.block_keys,
                    scores,
                    rows as u32,
                    n_blocks_max as u32,
                    first_pos as u32,
                    stride as u32,
                    self.ratio,
                    self.n_heads,
                    self.hd,
                    stream,
                )?;
            }

            // 2026-09-25: Host top-k per row; the D2H is ordered after the scoring
            // kernel on `stream`. Larger score first, lower index on ties.
            let mut raw = vec![0u8; rows * stride * 4];
            gpu.copy_d2h_on_stream(scores, &mut raw, stream)?;
            let sc: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut host_lists = vec![0u8; rows * topk * 4];
            for r in 0..rows {
                let complete = (first_pos + r + 1) / ratio;
                let row_sc = &sc[r * stride..r * stride + complete];
                let mut order: Vec<u32> = (0..complete as u32).collect();
                order.sort_by(|&a, &b| {
                    row_sc[b as usize]
                        .partial_cmp(&row_sc[a as usize])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });
                for (i, b) in order[..topk].iter().enumerate() {
                    host_lists[(r * topk + i) * 4..(r * topk + i) * 4 + 4]
                        .copy_from_slice(&(*b as i32).to_le_bytes());
                }
            }
            gpu.copy_h2d_async(&host_lists, lists, stream)?;

            ops::qsa_prefill_attn(
                gpu,
                self.k_prefill_attn_k,
                q_roped.offset(first_row * q_row * 2),
                k_pool,
                v_pool,
                block_table_dev,
                lists,
                attn_ctx.offset(first_row * q_row * 2),
                rows as u32,
                first_pos as u32,
                topk as u32,
                self.ratio,
                block_size,
                nq,
                self.nkv_attn,
                self.hd_attn,
                inv_sqrt_d,
                stream,
            )?;
            slab += rows;
        }
        if diag {
            let mut sel_last = vec![0u8; q_row * 2];
            gpu.copy_d2h_on_stream(
                attn_ctx.offset((num_tokens - 1) * q_row * 2),
                &mut sel_last,
                stream,
            )?;
            let f = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect()
            };
            let (a, b) = (f(&dense_last), f(&sel_last));
            let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            tracing::warn!(
                "QSA S2 DIAG: last-row ctx dense-vs-selected cos={:.6} |dense|={:.3} |sel|={:.3}",
                dot / (na * nb).max(1e-30),
                na,
                nb
            );
        }
        tracing::debug!(
            "QSA prefill select: {} selective rows over {} tokens",
            n_sel_total,
            num_tokens
        );
        Ok(())
    }
}
