// SPDX-License-Identifier: AGPL-3.0-only

//! Exact drafter KV across sequences (`ModelLevers::mtp_kv_exact`).
//!
//! After a batched verify, every ACCEPTED draft `d_k` of sequence `i` needs
//! the drafter row the reference proposer builds for it: the pair
//! `(embed(d_k), target_hidden)` where the hidden is the verify forward's
//! own row for the token before `d_k`, at RoPE position = `d_k`'s position.
//! Without it the drafter attends over a context with a hole at every
//! accepted token. The rows need no attention pass (a single-layer drafter's
//! K/V are pure functions of its input pair — see `prefill.rs`), so all
//! sequences' rows go through ONE fc GEMM and one K/V GEMM pair.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::{MtpHead, MtpProposerState, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::speculative::ProposerState;

impl MtpHead {
    /// Append the catch-up rows. `tokens[i]` are sequence i's accepted drafts
    /// in order, `hiddens[i][k]` the target hidden paired with
    /// `tokens[i][k]`, `first_pos[i]` the RoPE position of `tokens[i][0]`.
    /// Rows land at each drafter's current length. Returns rows written; 0
    /// when the head cannot build rows this way (non-BF16 head / no scratch),
    /// which leaves the drafter exactly as before (holes, never corruption).
    pub(crate) fn catchup_batch_impl(
        &self,
        tokens: &[Vec<u32>],
        hiddens: &[Vec<DevicePtr>],
        first_pos: &[usize],
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let n = tokens.len();
        ensure!(
            n == hiddens.len() && n == first_pos.len() && n == states.len(),
            "catchup_batch: length mismatch"
        );
        let c: usize = tokens.iter().map(Vec::len).sum();
        if c == 0 {
            return Ok(0);
        }
        let Some(scratch) = self.prefill_scratch.as_ref() else {
            return Ok(0);
        };
        let (fc_w, k_w, v_w) = match (&self.fc, &self.k_proj, &self.v_proj) {
            (ProjectionWeight::Bf16(fc), ProjectionWeight::Bf16(k), ProjectionWeight::Bf16(v))
                if self.kv_bf16 =>
            {
                (fc, k, v)
            }
            _ => return Ok(0),
        };
        ensure!(
            c <= super::prefill::PREFILL_CHUNK,
            "catchup_batch: {c} rows exceed the {}-row scratch",
            super::prefill::PREFILL_CHUNK
        );
        let gpu = ctx.gpu;
        let h = ctx.config.hidden_size;
        let nq = ctx.config.num_attention_heads as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let hd = ctx.config.head_dim as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let kv_dim = (nkv * hd) as usize;
        let bf16 = 2usize;

        // 1. Gather embeddings and target hiddens into contiguous [c, h]
        //    rows. `fc_out` holds the hiddens until the fc GEMM (step 4)
        //    overwrites it.
        let mut r = 0usize;
        for (toks, hs) in tokens.iter().zip(hiddens) {
            ensure!(toks.len() == hs.len(), "catchup_batch: token/hidden count");
            for (&t, &hp) in toks.iter().zip(hs) {
                let src = self.embed_tokens.weight.offset(t as usize * h * bf16);
                gpu.copy_d2d_async(src, scratch.embed.offset(r * h * bf16), h * bf16, stream)?;
                gpu.copy_d2d_async(hp, scratch.fc_out.offset(r * h * bf16), h * bf16, stream)?;
                r += 1;
            }
        }

        // 2. Pre-fc norms (target rows: the target final norm first, under
        //    `mtp_target_postnorm`, into `concat`, which step 3 rewrites).
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            scratch.embed,
            &self.pre_fc_norm_embedding,
            scratch.normed_embed,
            c as u32,
            h as u32,
            eps,
            stream,
        )?;
        let hidden_rows =
            self.target_postnorm_rows(ctx, true, scratch.fc_out, scratch.concat, c as u32, stream)?;
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            hidden_rows,
            &self.pre_fc_norm_hidden,
            scratch.normed_hidden,
            c as u32,
            h as u32,
            eps,
            stream,
        )?;

        // 3. Per-row concat [normed_embed | normed_hidden] -> [c, 2h].
        for r in 0..c {
            ops::bf16_concat(
                gpu,
                self.bf16_concat_k,
                scratch.normed_embed.offset(r * h * bf16),
                scratch.normed_hidden.offset(r * h * bf16),
                scratch.concat.offset(r * 2 * h * bf16),
                h as u32,
                stream,
            )?;
        }

        // 4. fc -> input layernorm -> K/V (+ k_norm). Q is not needed.
        self.gemm_rows(
            gpu,
            scratch.concat,
            fc_w,
            scratch.fc_out,
            c,
            h as u32,
            (2 * h) as u32,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            scratch.fc_out,
            &self.input_layernorm,
            scratch.normed2,
            c as u32,
            h as u32,
            eps,
            stream,
        )?;
        self.gemm_rows(
            gpu,
            scratch.normed2,
            k_w,
            scratch.k_out,
            c,
            kv_dim as u32,
            h as u32,
            stream,
        )?;
        self.gemm_rows(
            gpu,
            scratch.normed2,
            v_w,
            scratch.v_out,
            c,
            kv_dim as u32,
            h as u32,
            stream,
        )?;
        if !self.k_norm.weight.is_null() {
            ops::rms_norm(
                gpu,
                self.rms_norm_k,
                scratch.k_out,
                &self.k_norm,
                scratch.k_out,
                c as u32 * nkv,
                hd,
                eps,
                stream,
            )?;
        }

        // 5. Slots at each drafter's current length, RoPE at the tokens'
        //    positions; grow block tables first.
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let mut positions: Vec<u32> = Vec::with_capacity(c);
        let mut slots: Vec<i64> = Vec::with_capacity(c);
        for (i, toks) in tokens.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let st = states[i]
                .as_any_mut()
                .downcast_mut::<MtpProposerState>()
                .ok_or_else(|| anyhow::anyhow!("catchup_batch: invalid MTP proposer state"))?;
            let blocks_needed = (st.seq_len + toks.len() - 1) / bs + 1;
            while st.block_table.len() < blocks_needed {
                st.block_table.push(kv_cache.alloc_block()?);
            }
            for k in 0..toks.len() {
                let slot = st.seq_len + k;
                slots.push((st.block_table[slot / bs] as i64) * (bs as i64) + (slot % bs) as i64);
                positions.push((first_pos[i] + k) as u32);
            }
        }
        let pos_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_le_bytes()).collect();
        let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        gpu.copy_h2d_async(&pos_bytes, scratch.pos_dev, stream)?;
        gpu.copy_h2d_async(&slot_bytes, scratch.slot_dev, stream)?;
        ops::rope(
            gpu,
            self.rope_k,
            scratch.q_scratch,
            scratch.k_out,
            scratch.pos_dev,
            c as u32,
            nq,
            nkv,
            hd,
            ctx.config.rotary_dim() as u32,
            ctx.config.rope_theta as f32,
            stream,
        )?;
        ops::reshape_and_cache(
            gpu,
            self.reshape_cache_k,
            scratch.k_out,
            scratch.v_out,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            scratch.slot_dev,
            c as u32,
            nkv,
            hd,
            bs as u32,
            kv_dim as u32,
            kv_dim as u32,
            kv_cache.cache_stride() as u64,
            stream,
        )?;
        drop(kv_cache);

        for (i, toks) in tokens.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let st = states[i]
                .as_any_mut()
                .downcast_mut::<MtpProposerState>()
                .ok_or_else(|| anyhow::anyhow!("catchup_batch: invalid MTP proposer state"))?;
            st.seq_len += toks.len();
            st.last_pair_key = Some((first_pos[i] + toks.len() - 1).saturating_sub(1));
        }
        Ok(c)
    }
}
