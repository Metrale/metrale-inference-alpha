// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The single-stream GDN prefill recurrence
//! (`prefill_gdn_recurrence`) and the prefill conv1d with mid-chunk tail
//! capture (`conv1d_prefill_capture`).
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants: none beyond the types.

use super::*;

impl Qwen3SsmLayer {
    /// 2026-09-25: GDN prefill recurrence over one chunk of one sequence. The
    /// first matching arm runs: split4 on `metrale_scale` builds; FLA chunked;
    /// the register-resident kernel; WY4; persistent (256..=4096 tokens);
    /// split4.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_gdn_recurrence(
        &self,
        h_state: DevicePtr,
        q_ptr: DevicePtr,
        k_ptr: DevicePtr,
        v_ptr: DevicePtr,
        gates_buf: DevicePtr,
        gdn_out_buf: DevicePtr,
        k: u32,
        nk: usize,
        nv: usize,
        kd: usize,
        vd: usize,
        conv_dim: usize,
        midcap_idx: Option<usize>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let fp32 = 4usize;
        let gb_stride = (nv * 2) as u32;

        // 2026-09-25: `metrale_scale` builds (gfx1151, 64 KB LDS) run split4
        // only: it keeps H in global memory and takes any length, while WY4 and
        // persistent ask for 69688 B and 67584 B of shared memory at 128-dim
        // heads.
        if cfg!(metrale_scale) {
            // 2026-09-25: Mid-chunk tail capture: run split4 up to
            // `cap_local_early` (when set) and to `cap_local`, copying h_state
            // into the reserved snapshot slots at each point, then finish the
            // chunk on the same h_state.
            if let (Some(cap), Some(idx)) = (ctx.midchunk_capture.as_ref(), midcap_idx) {
                let cl = cap.cap_local;
                if cl > 0 && (cl as u32) < k {
                    let bf16 = 2usize;
                    let value_dim = nv * vd;
                    // 2026-09-25: Split4 over local tokens [start, start + len)
                    // on the chained h_state.
                    let seg = |start: usize, len: u32| -> Result<()> {
                        let gate = gates_buf.offset(start * gb_stride as usize * fp32);
                        ops::gdn_prefill_split4(
                            ctx.gpu,
                            self.gdn_prefill_split4_k,
                            h_state,
                            q_ptr.offset(start * conv_dim * bf16),
                            k_ptr.offset(start * conv_dim * bf16),
                            v_ptr.offset(start * conv_dim * bf16),
                            gate,
                            gate.offset(nv * fp32),
                            gdn_out_buf.offset(start * value_dim * bf16),
                            1,
                            len,
                            nk as u32,
                            nv as u32,
                            kd as u32,
                            vd as u32,
                            conv_dim as u32,
                            conv_dim as u32,
                            gb_stride,
                            stream,
                        )
                    };
                    // 2026-09-25: Optional earlier capture at `cap_local_early`
                    // (token tb - block_size).
                    let mut start = 0usize;
                    if let Some(ce) = cap.cap_local_early {
                        seg(0, ce as u32)?;
                        ctx.gpu.copy_d2d_async(
                            h_state,
                            cap.h_dsts_early[idx],
                            cap.h_bytes,
                            stream,
                        )?;
                        start = ce;
                    }
                    // 2026-09-25: Capture h_state at the tail boundary tb.
                    seg(start, (cl - start) as u32)?;
                    ctx.gpu
                        .copy_d2d_async(h_state, cap.h_dsts[idx], cap.h_bytes, stream)?;
                    return seg(cl, k - cl as u32);
                }
            }
            return ops::gdn_prefill_split4(
                ctx.gpu,
                self.gdn_prefill_split4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
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

        // 2026-09-25: FLA chunked prefill (`gdn_prefill_fla`) runs for 128-dim
        // heads when its kernels and scratch are present, METRALE_NO_GDN_FLA is
        // not 1, and this is not an exact replay. Measured 2026-06-06: 1.75x
        // WY4 at 16k tokens, token-equal (cos 1.0 against the scalar kernel).
        // An exact replay (`gdn_exact_replay`, a warm hit restored from an SSM
        // snapshot) skips FLA: FLA groups the tokens into 64-token chunks from
        // the replay's start and keeps BF16 intermediates, so its result would
        // differ from the pass that produced the cached state. The replay takes
        // the register-resident arm below when it applies, else WY4.
        let fla_scratch = ctx.buffers.gdn_fla_scratch();
        // 2026-09-25: METRALE_NO_GDN_FLA=1 skips the FLA arm; a diagnostic lever,
        // read once per process.
        static NO_FLA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let no_fla =
            *NO_FLA.get_or_init(|| std::env::var("METRALE_NO_GDN_FLA").as_deref() == Ok("1"));
        if !no_fla
            && !ctx.gdn_exact_replay
            && kd == 128
            && vd == 128
            && fla_scratch.0 != 0
            && self.gdn_prefill_fla_recompute_wu_k.0 != 0
            && self.gdn_prefill_fla_chunk_delta_h_k.0 != 0
            && self.gdn_prefill_fla_chunk_fwd_o_k.0 != 0
        {
            // 2026-09-25: Log once (per `ctx.stats`) that the FLA path is active.
            if ctx.stats.once("log:gdn_fla_chunked") {
                tracing::info!(
                    "GDN prefill: FLA chunked path ACTIVE (baked default: recompute_wu → chunk_delta_h_ksplit → chunk_fwd_o)"
                );
            }
            let num_chunks = k.div_ceil(64);
            let nt = num_chunks as usize;
            let w_out = fla_scratch;
            let u_out = w_out.offset(nt * nv * 64 * kd * 2);
            let s_out = u_out.offset(nt * nv * 64 * vd * 2);
            let uc_out = s_out.offset(nt * nv * kd * vd * 2);
            let gc_out = uc_out.offset(nt * nv * 64 * vd * 2);
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
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                w_out,
                u_out,
                s_out,
                uc_out,
                gc_out,
                1,
                k,
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
            )?;
        } else if ctx.levers.gdn_regresident
            && kd == 128
            && vd == 128
            && self.gdn_prefill_regresident_k.0 != 0
        {
            // 2026-09-25: Register-resident token-sequential recurrence. When
            // the FLA arm did not run (an exact replay, METRALE_NO_GDN_FLA=1, or
            // a missing FLA kernel or scratch), this arm runs if
            // `gdn_regresident` is on (unless METRALE_NO_GDN_REGRESIDENT=1),
            // both head dims are 128, and its kernel is loaded.
            if ctx.stats.once("log:gdn_regresident") {
                tracing::info!(
                    "GDN prefill: REGISTER-RESIDENT warm-replay path ACTIVE (default; H in regs, no smem-H)"
                );
            }
            ops::gdn_prefill_regresident(
                ctx.gpu,
                self.gdn_prefill_regresident_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else if self.gdn_prefill_persistent_wy4_k.0 != 0 {
            // 2026-09-25: WY4: four tokens per iteration. The dynamic shared
            // memory covers H, the four k/q buffer pairs and the warp sums
            // (`gated_delta_rule_persistent.cu`).
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
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
        } else if (256..=4096).contains(&k) && self.gdn_prefill_persistent_k.0 != 0 {
            ops::gdn_prefill_persistent(
                ctx.gpu,
                self.gdn_prefill_persistent_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
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
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
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

    /// 2026-09-25: Prefill conv1d with optional mid-chunk tail capture. When
    /// capturing, the conv runs up to `cap_local_early` (when set) and to
    /// `cap_local`, copying conv_state into the reserved snapshot slots at
    /// each point, then finishes the chunk. conv_state carries the sliding
    /// window between the calls, as it does between prefill chunks.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn conv1d_prefill_capture(
        &self,
        ctx: &ForwardContext,
        conv_state: DevicePtr,
        input: DevicePtr,
        output: DevicePtr,
        conv_dim: usize,
        d_conv: usize,
        k: u32,
        qkvz_size: usize,
        midcap_idx: Option<usize>,
        stream: u64,
    ) -> Result<()> {
        if let (Some(cap), Some(idx)) = (ctx.midchunk_capture.as_ref(), midcap_idx) {
            let cl = cap.cap_local;
            if cl > 0 && (cl as u32) < k {
                let bf16 = 2usize;
                // 2026-09-25: Conv over local tokens [start, start + len) on the
                // chained conv_state.
                let seg = |start: usize, len: u32| -> Result<()> {
                    ops::conv1d_update_prefill(
                        ctx.gpu,
                        self.conv1d_prefill_k,
                        self.conv1d_prefill_tp_k,
                        conv_state,
                        input.offset(start * qkvz_size * bf16),
                        &self.ssm.conv1d,
                        DevicePtr::NULL,
                        output.offset(start * conv_dim * bf16),
                        conv_dim as u32,
                        d_conv as u32,
                        len,
                        qkvz_size as u32,
                        conv_dim as u32,
                        stream,
                    )
                };
                // 2026-09-25: Optional earlier capture at `cap_local_early`
                // (token tb - block_size).
                let mut start = 0usize;
                if let Some(ce) = cap.cap_local_early {
                    seg(0, ce as u32)?;
                    ctx.gpu.copy_d2d_async(
                        conv_state,
                        cap.conv_dsts_early[idx],
                        cap.conv_bytes,
                        stream,
                    )?;
                    start = ce;
                }
                // 2026-09-25: Capture conv_state at the tail boundary tb.
                seg(start, (cl - start) as u32)?;
                ctx.gpu
                    .copy_d2d_async(conv_state, cap.conv_dsts[idx], cap.conv_bytes, stream)?;
                seg(cl, k - cl as u32)?;
                return Ok(());
            }
        }
        ops::conv1d_update_prefill(
            ctx.gpu,
            self.conv1d_prefill_k,
            self.conv1d_prefill_tp_k,
            conv_state,
            input,
            &self.ssm.conv1d,
            DevicePtr::NULL,
            output,
            conv_dim as u32,
            d_conv as u32,
            k,
            qkvz_size as u32,
            conv_dim as u32,
            stream,
        )
    }
}
