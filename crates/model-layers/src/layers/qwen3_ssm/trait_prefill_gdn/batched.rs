// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Multi-stream GDN scans for the batched prefill layer:
//! equal-length streams (`prefill_gdn_full_batched_inner`) and a varlen FLA
//! call (`prefill_gdn_full_batched_fla_varlen_inner`).
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants:
//! - `prefill_gdn_full_batched_fla_varlen_inner` returns `Ok(false)` without
//!   launching anything when it is not eligible.

use super::super::*;

impl Qwen3SsmLayer {
    /// 2026-09-25: GDN recurrence over `batch_size` streams of `chunk_len`
    /// tokens each; `h_state_ptrs` is a device array of per-stream h-state
    /// pointers. The caller (`prefill_ssm_batched_layer`) uses it only when
    /// every stream has the same length, so stream `b`'s rows in `gdn_bufs`
    /// start at `b * chunk_len`.
    ///
    /// Ladder: FLA when `gdn_batched_fla` (METRALE_GDN_BATCHED_FLA=1) and both
    /// head dims are 128, then the `_batched` WY64, WY4, persistent
    /// (256..=4096 tokens) and split4 kernels. Returns an error when none of
    /// them is loaded.
    pub(crate) fn prefill_gdn_full_batched_inner(
        &self,
        h_state_ptrs: metrale_gpu_runtime::gpu::DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        chunk_len: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let q_ptr = gdn_bufs.qkv;
        let k_ptr = gdn_bufs.qkv.offset(key_dim * bf16);
        let v_ptr = gdn_bufs.qkv.offset(key_dim * 2 * bf16);
        let gate_ptr = gdn_bufs.gate_beta;
        let beta_ptr = gdn_bufs.gate_beta.offset(nv * fp32);
        let gb_stride = (nv * 2) as u32;

        // 2026-09-25: FLA over all streams, with `h_state_ptrs` passed as the
        // pointer table (`h_state_is_table`). The scratch spans the batch, so it
        // is sized by `total_nt = batch_size * num_chunks`.
        if ctx.levers.gdn_batched_fla && kd == 128 && vd == 128 {
            let fla_scratch = ctx.buffers.gdn_fla_scratch();
            if fla_scratch.0 != 0
                && self.gdn_prefill_fla_recompute_wu_k.0 != 0
                && self.gdn_prefill_fla_chunk_delta_h_k.0 != 0
                && self.gdn_prefill_fla_chunk_fwd_o_k.0 != 0
            {
                let num_chunks = chunk_len.div_ceil(64);
                let total_nt = (batch_size * num_chunks) as usize;
                let w_out = fla_scratch;
                let u_out = w_out.offset(total_nt * nv * 64 * kd * bf16);
                let s_out = u_out.offset(total_nt * nv * 64 * vd * bf16);
                let uc_out = s_out.offset(total_nt * nv * kd * vd * bf16);
                let gc_out = uc_out.offset(total_nt * nv * 64 * vd * bf16);
                return ops::gdn_prefill_fla(
                    ctx.gpu,
                    self.gdn_prefill_fla_recompute_wu_k,
                    self.gdn_prefill_fla_recompute_wu_hopper_k,
                    self.gdn_prefill_fla_chunk_fwd_o_hopper_k,
                    self.gdn_prefill_fla_chunk_delta_h_k,
                    self.gdn_prefill_fla_chunk_delta_h_tc_vblock_k,
                    self.gdn_prefill_fla_chunk_delta_h_tcfuse_k,
                    self.gdn_prefill_fla_chunk_delta_h_fused_k,
                    self.gdn_prefill_fla_chunk_delta_h_tma_k,
                    self.gdn_prefill_fla_chunk_fwd_o_k,
                    h_state_ptrs,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_bufs.output,
                    w_out,
                    u_out,
                    s_out,
                    uc_out,
                    gc_out,
                    batch_size,
                    chunk_len,
                    num_chunks,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    gb_stride,
                    true,
                    metrale_gpu_runtime::gpu::DevicePtr::NULL,
                    metrale_gpu_runtime::gpu::DevicePtr::NULL,
                    false,
                    ctx.profile,
                    stream,
                );
            }
        }

        if self.gdn_prefill_wy32_batched_k.0 != 0 && chunk_len > 32 {
            // 2026-09-25: The dynamic shared memory must cover the kernel's whole
            // layout (`gated_delta_rule_wy64_prefill.cu`, C = 32): H, smem_k,
            // smem_q, smem_warp[4], smem_kd[C*C], smem_g[C], smem_bt[C].
            let smem =
                (kd * vd * 4 + 32 * kd * 2 + 32 * kd * 2 + 32 * 32 * 4 + (4 + 32 + 32) * 4) as u32;
            ops::gdn_prefill_persistent_smem_batched(
                ctx.gpu,
                self.gdn_prefill_wy32_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if self.gdn_prefill_persistent_wy4_batched_k.0 != 0 {
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem_batched(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if (256..=4096).contains(&chunk_len) && self.gdn_prefill_persistent_batched_k.0 != 0
        {
            ops::gdn_prefill_persistent_batched(
                ctx.gpu,
                self.gdn_prefill_persistent_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else if self.gdn_prefill_split4_batched_k.0 != 0 {
            ops::gdn_prefill_split4_batched(
                ctx.gpu,
                self.gdn_prefill_split4_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else {
            anyhow::bail!(
                "Qwen3SsmLayer::prefill_gdn_full_batched_inner: no batched GDN \
                 kernel handle is loaded for this target — caller should fall \
                 back to per-stream prefill_gdn_full."
            );
        }

        Ok(())
    }

    /// 2026-09-25: One varlen FLA call over streams of differing lengths.
    /// `cu_seqlens` (device) holds the per-stream token offsets, `cu_chunks` is
    /// passed as NULL, and `total_nt` (the sum of the per-stream chunk counts)
    /// sizes the scratch. Returns `Ok(false)`, launching nothing, when
    /// `gdn_batched_fla` is off, a head dim is not 128, `cu_seqlens` or the
    /// FLA scratch is NULL, or an FLA kernel is missing; the caller then runs
    /// each stream alone.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_gdn_full_batched_fla_varlen_inner(
        &self,
        h_state_ptrs: metrale_gpu_runtime::gpu::DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        cu_seqlens: metrale_gpu_runtime::gpu::DevicePtr,
        max_num_chunks: u32,
        total_nt: usize,
        max_seqlen: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let fla_scratch = ctx.buffers.gdn_fla_scratch();
        if !ctx.levers.gdn_batched_fla
            || kd != 128
            || vd != 128
            || fla_scratch.0 == 0
            || cu_seqlens.0 == 0
            || self.gdn_prefill_fla_recompute_wu_k.0 == 0
            || self.gdn_prefill_fla_chunk_delta_h_k.0 == 0
            || self.gdn_prefill_fla_chunk_fwd_o_k.0 == 0
        {
            return Ok(false);
        }
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let bf16 = 2usize;
        let fp32 = 4usize;
        let q_ptr = gdn_bufs.qkv;
        let k_ptr = gdn_bufs.qkv.offset(key_dim * bf16);
        let v_ptr = gdn_bufs.qkv.offset(key_dim * 2 * bf16);
        let gate_ptr = gdn_bufs.gate_beta;
        let beta_ptr = gdn_bufs.gate_beta.offset(nv * fp32);
        let gb_stride = (nv * 2) as u32;
        let w_out = fla_scratch;
        let u_out = w_out.offset(total_nt * nv * 64 * kd * bf16);
        let s_out = u_out.offset(total_nt * nv * 64 * vd * bf16);
        let uc_out = s_out.offset(total_nt * nv * kd * vd * bf16);
        let gc_out = uc_out.offset(total_nt * nv * 64 * vd * bf16);
        ops::gdn_prefill_fla(
            ctx.gpu,
            self.gdn_prefill_fla_recompute_wu_k,
            self.gdn_prefill_fla_recompute_wu_hopper_k,
            self.gdn_prefill_fla_chunk_fwd_o_hopper_k,
            self.gdn_prefill_fla_chunk_delta_h_k,
            self.gdn_prefill_fla_chunk_delta_h_tc_vblock_k,
            self.gdn_prefill_fla_chunk_delta_h_tcfuse_k,
            self.gdn_prefill_fla_chunk_delta_h_fused_k,
            self.gdn_prefill_fla_chunk_delta_h_tma_k,
            self.gdn_prefill_fla_chunk_fwd_o_k,
            h_state_ptrs,
            q_ptr,
            k_ptr,
            v_ptr,
            gate_ptr,
            beta_ptr,
            gdn_bufs.output,
            w_out,
            u_out,
            s_out,
            uc_out,
            gc_out,
            batch_size,
            max_seqlen,
            max_num_chunks,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            conv_dim as u32,
            conv_dim as u32,
            gb_stride,
            true,
            cu_seqlens,
            metrale_gpu_runtime::gpu::DevicePtr::NULL,
            true,
            ctx.profile,
            stream,
        )?;
        Ok(true)
    }
}
