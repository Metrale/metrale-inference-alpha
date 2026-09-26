// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GDN recurrence over a whole staged range (`prefill_gdn_full`),
//! used by the two-phase prefill and the per-stream fallback of the batched
//! layer; `batched` holds the multi-stream scans.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants: none beyond the types.

use super::*;

mod batched;

impl Qwen3SsmLayer {
    /// 2026-09-25: GDN recurrence over the range staged in `gdn_bufs`.
    ///
    /// `prefill_h_begin` widens an f16-sized pool slot into an FP32 stage,
    /// the kernel ladder runs over that pointer, and `prefill_h_end` narrows
    /// it back. The ladder is its own function so that its early `return`s
    /// cannot skip the narrowing; an error from the ladder returns before
    /// `prefill_h_end`. `prefill_gdn_recurrence_staged` does the same for the
    /// chunked path.
    pub(super) fn prefill_gdn_full_inner(
        &self,
        state: &mut dyn LayerState,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
        let h = super::ssm_h_fp16::prefill_h_begin(
            ctx.gpu,
            self.ssm_h_f16_to_f32_k,
            ssm_state,
            self.h_state_bytes,
            stream,
        )?;
        self.prefill_gdn_full_over(h.ptr(), gdn_bufs, ctx, stream)?;
        super::ssm_h_fp16::prefill_h_end(
            ctx.gpu,
            self.ssm_h_f32_to_f16_k,
            h,
            self.h_state_bytes,
            stream,
        )
    }

    /// 2026-09-25: The GDN kernel ladder over an FP32 h-state; the first
    /// matching arm runs.
    fn prefill_gdn_full_over(
        &self,
        h_state: DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
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

        let total = gdn_bufs.total_len as u32;

        // 2026-09-25: Q, K and V sit at offsets 0, key_dim and 2 * key_dim of
        // each `conv_dim`-element row.
        let q_ptr = gdn_bufs.qkv;
        let k_ptr = gdn_bufs.qkv.offset(key_dim * bf16);
        let v_ptr = gdn_bufs.qkv.offset(key_dim * 2 * bf16);

        // 2026-09-25: Gates: per token gate[nv] then beta[nv], FP32.
        let gate_ptr = gdn_bufs.gate_beta;
        let beta_ptr = gdn_bufs.gate_beta.offset(nv * fp32);
        let gb_stride = (nv * 2) as u32;

        tracing::debug!(
            "GDN prefill: total={total} wy32_k={} wy4_k={} persistent_k={} split4_k={}",
            self.gdn_prefill_wy32_k.0 != 0,
            self.gdn_prefill_persistent_wy4_k.0 != 0,
            self.gdn_prefill_persistent_k.0 != 0,
            self.gdn_prefill_split4_k.0 != 0
        );
        // 2026-09-25: `metrale_scale` builds (gfx1151, 64 KB LDS): WY64 (C=32),
        // WY4 and persistent keep H in shared memory (69688 B and 67584 B at
        // 128-dim heads) and do not fit; split4 keeps H in global memory (2 KB
        // of shared memory) and takes any length, so every size goes there.
        if cfg!(metrale_scale) {
            return ops::gdn_prefill_split4(
                ctx.gpu,
                self.gdn_prefill_split4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            );
        }
        // 2026-09-25: FLA chunked GDN (`gdn_prefill_fla`), the same call the
        // single-stream `prefill_gdn_recurrence` makes. Skipped on an exact
        // replay (`gdn_exact_replay`) and for head dims other than 128.
        let fla_scratch = ctx.buffers.gdn_fla_scratch();
        if !ctx.gdn_exact_replay
            && kd == 128
            && vd == 128
            && fla_scratch.0 != 0
            && self.gdn_prefill_fla_recompute_wu_k.0 != 0
            && self.gdn_prefill_fla_chunk_delta_h_k.0 != 0
            && self.gdn_prefill_fla_chunk_fwd_o_k.0 != 0
        {
            let num_chunks = total.div_ceil(64);
            let nt = num_chunks as usize;
            let w_out = fla_scratch;
            let u_out = w_out.offset(nt * nv * 64 * kd * bf16);
            let s_out = u_out.offset(nt * nv * 64 * vd * bf16);
            let uc_out = s_out.offset(nt * nv * kd * vd * bf16);
            let gc_out = uc_out.offset(nt * nv * 64 * vd * bf16);
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
                h_state,
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
                1,
                total,
                num_chunks,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                false,
                metrale_gpu_runtime::gpu::DevicePtr::NULL,
                metrale_gpu_runtime::gpu::DevicePtr::NULL,
                false,
                ctx.profile,
                stream,
            );
        }
        if self.gdn_prefill_wy32_k.0 != 0 && total > 32 && !cfg!(metrale_scale) {
            // 2026-09-25: The dynamic shared memory must cover the kernel's whole
            // layout (`gated_delta_rule_wy64_prefill.cu`, C = 32): H, smem_k,
            // smem_q, smem_warp[4], smem_kd[C*C], smem_g[C], smem_bt[C].
            let smem =
                (kd * vd * 4 + 32 * kd * 2 + 32 * kd * 2 + 32 * 32 * 4 + (4 + 32 + 32) * 4) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_wy32_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
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
        } else if total > 4096 {
            // 2026-09-25: Above 4096 tokens without WY64, run 4096-token chunks
            // over the same h-state.
            let chunk_max = 4096u32;
            let mut offset = 0u32;
            while offset < total {
                let chunk = (total - offset).min(chunk_max);
                let q_chunk = q_ptr.offset(offset as usize * conv_dim * bf16);
                let k_chunk = k_ptr.offset(offset as usize * conv_dim * bf16);
                let v_chunk = v_ptr.offset(offset as usize * conv_dim * bf16);
                let gate_chunk = gate_ptr.offset(offset as usize * gb_stride as usize * fp32);
                let beta_chunk = beta_ptr.offset(offset as usize * gb_stride as usize * fp32);
                let out_chunk = gdn_bufs.output.offset(offset as usize * value_dim * bf16);

                if self.gdn_prefill_persistent_k.0 != 0 && chunk >= 256 {
                    ops::gdn_prefill_persistent(
                        ctx.gpu,
                        self.gdn_prefill_persistent_k,
                        h_state,
                        q_chunk,
                        k_chunk,
                        v_chunk,
                        gate_chunk,
                        beta_chunk,
                        out_chunk,
                        1,
                        chunk,
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
                    ops::gdn_prefill_split4(
                        ctx.gpu,
                        self.gdn_prefill_split4_k,
                        h_state,
                        q_chunk,
                        k_chunk,
                        v_chunk,
                        gate_chunk,
                        beta_chunk,
                        out_chunk,
                        1,
                        chunk,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        conv_dim as u32,
                        conv_dim as u32,
                        gb_stride,
                        stream,
                    )?;
                }
                offset += chunk;
            }
        } else if self.gdn_prefill_persistent_wy4_k.0 != 0 && !cfg!(metrale_scale) {
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
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
        } else if (256..=4096).contains(&total) && self.gdn_prefill_persistent_k.0 != 0 {
            ops::gdn_prefill_persistent(
                ctx.gpu,
                self.gdn_prefill_persistent_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
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
            ops::gdn_prefill_split4(
                ctx.gpu,
                self.gdn_prefill_split4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                1,
                total,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        }

        Ok(())
    }
}
