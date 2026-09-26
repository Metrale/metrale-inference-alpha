// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Multi-sequence decode output gate and O projection: `attn_out`
//! `[n, q_dim]` to `o_out` `[n, h]`, choosing the O kernel by weight format and n.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - Every `Ok` return of `ms_phase_o_proj` goes through `ms_o_proj_lora`, so
//!   every O route applies the per-request LoRA delta when one is routed.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::HeadGateActivation;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::weight_map::WeightQuantFormat;

/// 2026-09-25: The shared signature of the contiguous FP8 o_proj kernels
/// (`w8a16_gemv_batch{4,16}`, `w8a16_gemm_m16`, `w8a16_gemm_pipelined_m32` and
/// the N-column rungs), so the route picks a function pointer and one loop
/// launches it. `qkv_fp8_batch.rs` does the same for the strided kernels.
type BatchGemv = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;

impl Qwen3AttentionLayer {
    /// 2026-09-25: Output gate (when the layer has one) and O projection into
    /// the `moe_output` buffer, which is returned.
    pub(in super::super) fn ms_phase_o_proj(
        &self,
        c: &MultiSeqCtx<'_>,
        attn_out: DevicePtr,
    ) -> Result<DevicePtr> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            hd,
            bf16,
            q_dim,
            per_seq_qkv,
            qkv_buf,
            normed,
            ..
        } = *c;
        if self.gated {
            // 2026-09-25: One launch for all n rows. `attn_out` is contiguous
            // `[n, q_dim]` and each row's gate sits `q_dim` elements into that
            // row's `qkv_buf` slice, so the gates are `per_seq_qkv` bytes apart:
            // the layout `sigmoid_gate_mul_batched` reads
            // (`gate[t * gate_stride + d]`, stride in elements).
            debug_assert_eq!(
                per_seq_qkv % bf16,
                0,
                "gate stride must be whole bf16 elements"
            );
            ops::sigmoid_gate_mul_batched(
                fwd.gpu,
                self.sigmoid_gate_mul_batched_k,
                attn_out,
                qkv_buf.offset(q_dim as usize * bf16),
                attn_out,
                q_dim,
                (per_seq_qkv / bf16) as u32,
                n as u32,
                stream,
            )?;
        }

        if let Some(ref g_proj) = self.head_gate_weight {
            let gate_buf = qkv_buf;
            // 2026-09-25: The per-head gate projection. n in 2..=8 uses the batched
            // GEMV (`ceil(nq / 4)` CTAs) when it is loaded and `bf16_batchm_enabled`.
            // Its kernel computes at most
            // `DENSE_GEMV_BATCHM_MAX_M` (16) rows, and `ops::dense_gemv_batchm`
            // refuses a larger m. Every other n takes `dense_gemm_tc`, which
            // handles any M.
            if (2..=8).contains(&n)
                && self.dense_gemv_batchm_k.0 != 0
                && super::super::qkv::bf16_batchm_enabled()
            {
                ops::dense_gemv_batchm(
                    fwd.gpu,
                    self.dense_gemv_batchm_k,
                    normed,
                    g_proj,
                    gate_buf,
                    n as u32,
                    nq,
                    h as u32,
                    nq,
                    stream,
                )?;
            } else {
                ops::dense_gemm_tc(
                    fwd.gpu,
                    self.dense_gemm_tc_k,
                    normed,
                    g_proj,
                    gate_buf,
                    n as u32,
                    nq,
                    h as u32,
                    stream,
                )?;
            }
            match self.head_gate_activation {
                HeadGateActivation::Sigmoid => ops::sigmoid_gate_mul_head_broadcast(
                    fwd.gpu,
                    self.sigmoid_gate_head_broadcast_k,
                    attn_out,
                    gate_buf,
                    attn_out,
                    nq,
                    hd,
                    n as u32,
                    stream,
                )?,
                HeadGateActivation::Softplus => ops::softplus_gate_mul_head_broadcast(
                    fwd.gpu,
                    self.softplus_gate_head_broadcast_k,
                    attn_out,
                    gate_buf,
                    attn_out,
                    nq,
                    hd,
                    n as u32,
                    stream,
                )?,
            }
        }

        let o_out = fwd.buffers.moe_output();
        if let Some(q2) = self.o_weight.as_ref().and_then(|w| w.as_packed_q2()) {
            // 2026-09-25: Packed Q2_0 weight: one 2-bit GEMV per row.
            for i in 0..n {
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                let o_out_i = o_out.offset(i * h * bf16);
                ops::q2_0_gemv_vec(fwd.gpu, self.q2_0_gemv_k, attn_out_i, q2, o_out_i, stream)?;
            }
        } else if let Some(o_bf16) = self.o_dense_bf16.as_ref() {
            // 2026-09-25: A BF16 O projection installed by the loader
            // (`set_o_dense_bf16`). Both operands are contiguous, so one launch
            // reads the weight once for all n rows: the batched GEMV for n in
            // 2..=8 (when loaded and `bf16_batchm_enabled`), otherwise
            // `dense_gemm` (grid `[ceil(N/16), ceil(M/16)]`).
            if (2..=8).contains(&n)
                && self.dense_gemv_batchm_k.0 != 0
                && super::super::qkv::bf16_batchm_enabled()
            {
                ops::dense_gemv_batchm(
                    fwd.gpu,
                    self.dense_gemv_batchm_k,
                    attn_out,
                    o_bf16,
                    o_out,
                    n as u32,
                    h as u32,
                    nq * hd,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    fwd.gpu,
                    self.dense_gemm_k,
                    attn_out,
                    o_bf16,
                    o_out,
                    n as u32,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(o_fp8) = self.o_weight.as_ref().and_then(|w| w.as_fp8()) {
            // 2026-09-25: The cuBLASLt W8A8 route (`w8a8_decode.rs`) goes first.
            // When it declines (`Ok(false)`) the W8A16 route below runs.
            if self.try_ms_o_proj_decode_w8a8(c, o_fp8, attn_out, o_out)? {
                return self.ms_o_proj_lora(c, attn_out, o_out);
            }
            // 2026-09-25: Each launch serves `step` contiguous rows from one pass
            // over the block-scaled weight. A single row, a weight that is not
            // block-scaled, or a target without the batched kernels runs the
            // scalar `w8a16_gemv` per row. `w8a16_gemv_batch4` and
            // `w8a16_gemv_batch16` are the MAX_M = 4 and 16 instantiations of one
            // template (`w8a16_gemv_batch4.cu`), whose header states each row is
            // byte-identical to the scalar `w8a16_gemv`.
            let block_scaled = o_fp8.scale_format == WeightQuantFormat::Fp8BlockScaled
                && h % 128 == 0
                && q_dim % 128 == 0;
            let wide = n > 4 && self.w8a16_gemv_batch16_k.0 != 0;
            // 2026-09-25: The tensor-core tier, in the same 16-row groups, under
            // the attention lever `self.m16_tc` (target default `attn_m16_tc`,
            // env `METRALE_ATTN_M16_TC` or `METRALE_M16_TC`), not the FFN's. The
            // MMA reassociates the K reduction, so it is not bit-identical to the
            // GEMV. `ops::w8a16_gemm_m16` requires `K = nq * hd` to be a multiple
            // of 128.
            let tc = wide
                && self.m16_tc
                && self.w8a16_gemm_m16_k.0 != 0
                && (nq * hd).is_multiple_of(128);
            let batched = n > 1 && block_scaled && (self.w8a16_gemv_batch4_k.0 != 0 || wide);
            let (gemv, kernel, step) = if !batched {
                (
                    ops::w8a16_gemv_batch4 as BatchGemv,
                    self.w8a16_gemv_batch4_k,
                    1,
                )
            } else if n > 16 && self.w8a16_gemm_pipelined_m32_k.0 != 0 {
                // 2026-09-25: 17 or more rows: one launch of the 32-row M-tile
                // kernel over all n rows (`grid.y = ceil(n / 32)`, one weight
                // pass per tile). `step = n` makes the loop below one iteration.
                // It has tensor-core numerics, like the `m16` tier. `block_scaled`
                // holds here, which gives the kernel's `K % 128 == 0`. It is checked
                // before `tc`, whose kernel takes at most 16 rows per launch.
                (
                    ops::w8a16_gemm_pipelined_m32 as BatchGemv,
                    self.w8a16_gemm_pipelined_m32_k,
                    n,
                )
            } else if tc {
                crate::layers::qwen3_attention::attn_m16_tc_route::log_o_proj_m16_tc_route(
                    fwd.stats,
                );
                (ops::w8a16_gemm_m16 as BatchGemv, self.w8a16_gemm_m16_k, 16)
            } else if let Some((gemv, kernel)) = self.ncol_contiguous_route(n) {
                // 2026-09-25: The N-column-blocked GEMV (`attn_ncol_gemv.rs`): the
                // batch16 GEMV's weight pass and per-row reduction order, with the
                // activation loads and converts shared by adjacent output columns.
                // Same 16-row groups. Only reached when `batched` holds, so
                // `block_scaled` holds too.
                (gemv as BatchGemv, kernel, 16)
            } else if wide {
                (
                    ops::w8a16_gemv_batch16 as BatchGemv,
                    self.w8a16_gemv_batch16_k,
                    16,
                )
            } else {
                (
                    ops::w8a16_gemv_batch4 as BatchGemv,
                    self.w8a16_gemv_batch4_k,
                    4,
                )
            };
            for i in (0..n).step_by(step) {
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                let o_out_i = o_out.offset(i * h * bf16);
                if batched {
                    gemv(
                        fwd.gpu,
                        kernel,
                        attn_out_i,
                        o_fp8.weight,
                        o_fp8.row_scale,
                        o_out_i,
                        (n - i).min(step) as u32,
                        h as u32,
                        nq * hd,
                        stream,
                    )?;
                } else {
                    ops::w8a16_gemv(
                        fwd.gpu,
                        self.w8a16_gemv_k,
                        attn_out_i,
                        o_fp8.weight,
                        o_fp8.row_scale,
                        o_out_i,
                        h as u32,
                        nq * hd,
                        stream,
                    )?;
                }
            }
        } else if n == 3 && !self.attn.o_proj.is_null() {
            ops::w4a16_gemv_batch3(
                fwd.gpu,
                self.w4a16_gemv_batch3_k,
                attn_out,
                &self.attn.o_proj,
                o_out,
                h as u32,
                nq * hd,
                stream,
            )?;
        } else if n == 2 && !self.attn.o_proj.is_null() {
            ops::w4a16_gemv_batch2(
                fwd.gpu,
                self.w4a16_gemv_batch2_k,
                attn_out,
                &self.attn.o_proj,
                o_out,
                h as u32,
                nq * hd,
                stream,
            )?;
        } else if !self.attn.o_proj.is_null() {
            // 2026-09-25: NVFP4 O projection for n = 1 and n >= 4: one
            // `wide_verify_gemm` call (`qkv.rs`) over all n rows. Both operands
            // are contiguous, so no scatter is needed.
            self.wide_verify_gemm(
                c,
                attn_out,
                &self.attn.o_proj,
                self.o_nvfp4_t.as_ref(),
                o_out,
                n as u32,
                h as u32,
                nq * hd,
                false,
            )?;
        } else {
            for i in 0..n {
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                let o_out_i = o_out.offset(i * h * bf16);
                self.nvfp4_decode_gemv(
                    fwd.gpu,
                    fwd.levers.gemv_sw,
                    attn_out_i,
                    &self.attn.o_proj,
                    o_out_i,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
            }
        }

        self.ms_o_proj_lora(c, attn_out, o_out)
    }

    /// 2026-09-25: Per-request O LoRA delta (batched bgmv): x is `attn_out`
    /// (after the gate, `[n, q_dim]`), folded in place into `o_out` (`[n, h]`).
    /// Does nothing unless the layer has LoRA weights with an O route and
    /// `seq_slot` is non-null. Returns `o_out`.
    ///
    /// A method rather than a tail block so the W8A8 route's early return goes
    /// through it too: every O route must fold the adapter.
    fn ms_o_proj_lora(
        &self,
        c: &MultiSeqCtx<'_>,
        attn_out: DevicePtr,
        o_out: DevicePtr,
    ) -> Result<DevicePtr> {
        let MultiSeqCtx {
            fwd,
            n,
            h,
            q_dim,
            stream,
            ..
        } = *c;
        if let Some(ref lw) = self.lora
            && c.seq_slot.0 != 0
            && let Some(ref route) = lw.o_route
        {
            ops::lora_delta::apply_lora_bgmv(
                fwd.gpu,
                &lw.kernels,
                route,
                attn_out,
                o_out,
                c.seq_slot,
                n as u32,
                q_dim,
                h as u32,
                fwd.buffers.lora_xa(),
                stream,
            )?;
        }
        Ok(o_out)
    }
}

#[cfg(test)]
#[path = "o_proj_tests.rs"]
mod tests;
