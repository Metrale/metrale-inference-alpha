// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `METRALE_DFLASH_CONTIG_ATTN=1` attention of the paged drafter layer: it
//! gathers the ctx K/V through the host into contiguous buffers and runs `ops::prefill_attention`.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::PagedLayerArgs;
use crate::dflash_head::BlockDiffusionDraftHead;
use metrale_model_layers::layer::ForwardContext;

impl BlockDiffusionDraftHead {
    /// 2026-09-25: The `METRALE_DFLASH_CONTIG_ATTN=1` attention, single sequence only:
    ///   1. Copy the ctx K/V of slots `[0, ctx_count)` from the paged cache to the host.
    ///   2. Append the block K/V from scratch, giving contiguous
    ///      `[ctx_count + g, num_kv_heads, head_dim]` BF16 K and V.
    ///   3. Pad Q with zero rows for the ctx positions.
    ///   4. Run non-causal `ops::prefill_attention` over `ctx_count + g` rows.
    ///   5. Move the block rows' output to `attn_out[0..g)` for post_attn.
    ///
    /// It syncs the stream and copies through the host, so it is a diagnostic path.
    pub(super) fn forward_block_layer_attention_contig(
        &self,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
    ) -> Result<()> {
        use metrale_model_layers::layers::ops;

        let PagedLayerArgs {
            ctx_count,
            q_dim,
            kv_dim,
            block_table_dev,
            stream,
            inv_sqrt_d,
            ..
        } = *args;
        let gpu = ctx.gpu;
        // 2026-09-25: It addresses one contiguous ctx + block window, so a batch is
        // refused; `propose_batch` then falls back to per-sequence propose.
        anyhow::ensure!(
            args.n_seq.max(1) == 1,
            "CONTIG_ATTN: batched propose (n_seq={}) needs the indirect paged \
             attention path; set METRALE_DFLASH_CONTIG_ATTN=0",
            args.n_seq
        );
        let g = self.block_g() as u32;
        let ctx_us = ctx_count as usize;
        let g_us = g as usize;
        let seq_len = ctx_count + g;
        const BF16: usize = 2;
        const BLOCK_SIZE: usize = 16;
        let kv_slot = kv_dim as usize * BF16;
        let q_slot = q_dim as usize * BF16;

        // 2026-09-25: The scratch rows are at least `ctx_window + gamma`
        // (`from_weights`), so `ctx_count <= ctx_window` keeps the `ctx_count + g` rows
        // written below inside q_buf, k_buf, v_buf and attn_out.
        anyhow::ensure!(
            ctx_us <= self.ctx_window,
            "CONTIG_ATTN: ctx_count({ctx_us}) > ctx_window({}); \
             scratch buffers sized for {} rows — reduce ctx or raise METRALE_DFLASH_CTX_WINDOW",
            self.ctx_window,
            self.ctx_window + g_us,
        );

        // 2026-09-25: `copy_d2h` does not wait for `stream`, so sync it first.
        gpu.synchronize(stream)?;

        // 2026-09-25: With no ctx rows the block's Q, K and V are already contiguous.
        if ctx_count == 0 {
            ops::prefill_attention(
                gpu,
                self.kernels.prefill_attn,
                self.scratch.q_buf,
                self.scratch.k_buf,
                self.scratch.v_buf,
                self.scratch.attn_out,
                g,
                1,
                self.num_q_heads as u32,
                self.num_kv_heads as u32,
                self.head_dim as u32,
                inv_sqrt_d,
                false,
                0,
                stream,
            )?;
            if args.block_dump {
                self.block_dump_buf(
                    ctx,
                    self.scratch.attn_out,
                    args.layer_idx,
                    "attn_out",
                    g,
                    q_dim,
                    stream,
                )?;
            }
            return Ok(());
        }

        let mut noise_k = vec![0u8; g_us * kv_slot];
        let mut noise_v = vec![0u8; g_us * kv_slot];
        let mut noise_q = vec![0u8; g_us * q_slot];
        gpu.copy_d2h(self.scratch.k_buf, &mut noise_k)?;
        gpu.copy_d2h(self.scratch.v_buf, &mut noise_v)?;
        gpu.copy_d2h(self.scratch.q_buf, &mut noise_q)?;

        let num_ctx_blocks = ctx_us.div_ceil(BLOCK_SIZE);
        let mut bt_raw = vec![0u8; num_ctx_blocks * 4];
        gpu.copy_d2h(block_table_dev, &mut bt_raw)?;
        let block_table: Vec<u32> = bt_raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // 2026-09-25: Gather the ctx K/V, one copy per physical block. The pool is
        // `[num_phys_blocks, BLOCK_SIZE, kv_dim]` BF16; logical slot s lives in
        // physical block `block_table[s / BLOCK_SIZE]` at row `s % BLOCK_SIZE`.
        let phys_block_bytes = BLOCK_SIZE * kv_slot;
        let mut ctx_k = vec![0u8; ctx_us * kv_slot];
        let mut ctx_v = vec![0u8; ctx_us * kv_slot];
        for b in 0..num_ctx_blocks {
            let phys = block_table[b] as usize;
            let pool_off = phys * phys_block_bytes;
            let mut blk_k = vec![0u8; phys_block_bytes];
            let mut blk_v = vec![0u8; phys_block_bytes];
            gpu.copy_d2h(k_pool.offset(pool_off), &mut blk_k)?;
            gpu.copy_d2h(v_pool.offset(pool_off), &mut blk_v)?;
            let slot_start = b * BLOCK_SIZE;
            let slot_end = (slot_start + BLOCK_SIZE).min(ctx_us);
            for s in slot_start..slot_end {
                let src = (s - slot_start) * kv_slot;
                let dst = s * kv_slot;
                ctx_k[dst..dst + kv_slot].copy_from_slice(&blk_k[src..src + kv_slot]);
                ctx_v[dst..dst + kv_slot].copy_from_slice(&blk_v[src..src + kv_slot]);
            }
        }

        let total_kv = (ctx_us + g_us) * kv_slot;
        let mut k_contig = vec![0u8; total_kv];
        let mut v_contig = vec![0u8; total_kv];
        k_contig[..ctx_us * kv_slot].copy_from_slice(&ctx_k);
        k_contig[ctx_us * kv_slot..].copy_from_slice(&noise_k);
        v_contig[..ctx_us * kv_slot].copy_from_slice(&ctx_v);
        v_contig[ctx_us * kv_slot..].copy_from_slice(&noise_v);
        gpu.copy_h2d(&k_contig, self.scratch.k_buf)?;
        gpu.copy_h2d(&v_contig, self.scratch.v_buf)?;

        // 2026-09-25: Q is zero for the ctx rows, whose output is discarded, then the
        // block's Q.
        let total_q = (ctx_us + g_us) * q_slot;
        let mut q_contig = vec![0u8; total_q];
        q_contig[ctx_us * q_slot..].copy_from_slice(&noise_q);
        gpu.copy_h2d(&q_contig, self.scratch.q_buf)?;

        // 2026-09-25: Non-causal attention over the contiguous buffers:
        //   q_buf:    [ctx + g, q_dim]
        //   k_buf:    [ctx + g, kv_dim]
        //   v_buf:    [ctx + g, kv_dim]
        //   attn_out: [ctx + g, q_dim], of which rows [ctx, ctx + g) are kept
        ops::prefill_attention(
            gpu,
            self.kernels.prefill_attn,
            self.scratch.q_buf,
            self.scratch.k_buf,
            self.scratch.v_buf,
            self.scratch.attn_out,
            seq_len,
            1,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            inv_sqrt_d,
            false,
            0,
            stream,
        )?;

        // 2026-09-25: Move rows [ctx, ctx + g) of attn_out to [0, g), where post_attn
        // reads them.
        gpu.synchronize(stream)?;
        let mut noise_attn = vec![0u8; g_us * q_slot];
        gpu.copy_d2h(
            self.scratch.attn_out.offset(ctx_us * q_slot),
            &mut noise_attn,
        )?;
        gpu.copy_h2d(&noise_attn, self.scratch.attn_out)?;

        if args.block_dump {
            self.block_dump_buf(
                ctx,
                self.scratch.attn_out,
                args.layer_idx,
                "attn_out",
                g,
                q_dim,
                stream,
            )?;
        }

        Ok(())
    }
}
