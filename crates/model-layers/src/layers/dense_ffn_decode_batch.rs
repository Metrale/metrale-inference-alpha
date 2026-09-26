// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Dense-FFN decode for small verify batches: `forward_k2`, `forward_k3`,
//! `forward_km` and the packed-Q2 `forward_km_q2`.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - On success the `[m, hidden]` output is in `ctx.buffers.moe_output()`.
//! - A layer with a BF16 or FP8 overlay is handed to `forward_prefill`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::{DenseFfnLayer, DenseFfnWeightsQ2, FfnActivation, native_small_batch_uses_prefill};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::PackedQ2Weight;

impl DenseFfnLayer {
    /// 2026-09-25: Packed-Q2 FFN for `m` rows, used by `forward_k2` and `forward_k3`:
    /// `q2_0_gemv_vec_batchm` for gate and up, `act_mul`, then down; SiLU only. Output
    /// `[m, hidden]` in `moe_output`.
    fn forward_km_q2(
        &self,
        q2w: &DenseFfnWeightsQ2,
        input: DevicePtr,
        ctx: &ForwardContext,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        if self.q2_0_gemv_batchm_k.0 == 0 {
            anyhow::bail!(
                "q2_0_gemv_vec_batchm kernel missing in this target build — packed-Q2 \
                 batched decode (METRALE_GGUF_NATIVE_Q2, C>=2) is unavailable"
            );
        }
        if self.activation != FfnActivation::SiLU {
            anyhow::bail!(
                "packed-Q2 FFN batched decode supports SiLU only (got {:?})",
                self.activation
            );
        }
        let inter = ctx.config.intermediate_size as u32;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let batchm = |w: &PackedQ2Weight, inp: DevicePtr, out: DevicePtr| -> Result<()> {
            ops::q2_0_gemv_vec_batchm(ctx.gpu, self.q2_0_gemv_batchm_k, inp, w, out, m, stream)
        };
        batchm(&q2w.gate_proj, input, gate_out)?;
        batchm(&q2w.up_proj, input, up_out)?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        batchm(&q2w.down_proj, gate_out, output)?;
        Ok(())
    }

    /// 2026-09-25: FFN for 2 rows. Packed-Q2 goes to `forward_km_q2`, FP8/BF16 to
    /// `forward_prefill`, and with `--w4a4-downcast` to `forward_km` when `can_forward_km(2)`;
    /// otherwise NVFP4 runs `w4a16_gemv_dual_batch2`, `act_mul` and `w4a16_gemv_batch2`, plus the
    /// LoRA deltas.
    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        if let Some(ref q2w) = self.q2_weights {
            return self.forward_km_q2(q2w, input, ctx, 2, stream);
        }
        if native_small_batch_uses_prefill(self.bf16_weights.is_some(), self.fp8_weights.is_some())
        {
            return self.forward_prefill(input, 2, ctx, stream);
        }
        // 2026-09-25: `--w4a4-downcast`: the batch2 kernels below are W4A16; `forward_km` routes
        // every projection through the W4A4 launcher.
        if crate::layers::ops::w4a4_proj::w4a4_downcast_enabled() && self.can_forward_km(2) {
            return self.forward_km(input, 2, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        ops::w4a16_gemv_dual_batch2(
            ctx.gpu,
            self.w4a16_gemv_dual_batch2,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;
        self.apply_lora_gate_up(ctx, input, gate_out, up_out, 2, stream)?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            2 * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_batch2(
            ctx.gpu,
            self.w4a16_gemv_batch2,
            gate_out,
            &self.weights.down_proj,
            output,
            h,
            inter,
            stream,
        )?;
        self.apply_lora_down(ctx, gate_out, output, 2, stream)?;

        Ok(())
    }

    /// 2026-09-25: FFN for 3 rows; the same routing as `forward_k2`, with the batch3 kernels.
    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        if let Some(ref q2w) = self.q2_weights {
            return self.forward_km_q2(q2w, input, ctx, 3, stream);
        }
        if native_small_batch_uses_prefill(self.bf16_weights.is_some(), self.fp8_weights.is_some())
        {
            return self.forward_prefill(input, 3, ctx, stream);
        }
        // 2026-09-25: `--w4a4-downcast`: the batch3 kernels below are W4A16; `forward_km` routes
        // every projection through the W4A4 launcher.
        if crate::layers::ops::w4a4_proj::w4a4_downcast_enabled() && self.can_forward_km(3) {
            return self.forward_km(input, 3, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        ops::w4a16_gemv_dual_batch3(
            ctx.gpu,
            self.w4a16_gemv_dual_batch3,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;
        self.apply_lora_gate_up(ctx, input, gate_out, up_out, 3, stream)?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            3 * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_batch3(
            ctx.gpu,
            self.w4a16_gemv_batch3,
            gate_out,
            &self.weights.down_proj,
            output,
            h,
            inter,
            stream,
        )?;
        self.apply_lora_down(ctx, gate_out, output, 3, stream)?;

        Ok(())
    }

    /// 2026-09-25: The `w4a16_gemv_batch{M}` handle for `m` rows, from `W4a16BatchmTiers::kernel`;
    /// zero when no tier serves `m`.
    pub(super) fn batchm_kernel(&self, m: u32) -> KernelHandle {
        self.w4a16_batchm.kernel(m)
    }

    /// 2026-09-25: Whether `forward_km` can serve `m` rows: a batchm tier resolved, and the layer
    /// has NVFP4 gate weights or an FP8 overlay.
    pub fn can_forward_km(&self, m: u32) -> bool {
        self.batchm_kernel(m).0 != 0
            && (!self.weights.gate_proj.weight.is_null()
                // 2026-09-25: An FP8 layer answers true: `forward_km` sends it to
                // `forward_prefill`, and callers such as
                // `qwen3_attention/trait_impl/multi_seq/ffn.rs` pick their verify branch by this.
                || self.fp8_weights.is_some())
    }

    /// 2026-09-25: FFN for `m` verify rows. FP8/BF16 layers go to `forward_prefill`. NVFP4 runs
    /// gate, up and down through `ops::w4a4_proj::nvfp4_proj_small_m` with the `batchm_kernel(m)`
    /// tier (W4A4 under `--w4a4-downcast`), plus `act_mul` and the LoRA deltas. There is no
    /// packed-Q2 arm; `can_forward_km` is false for a layer without NVFP4 or FP8 weights.
    pub fn forward_km(
        &self,
        input: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if native_small_batch_uses_prefill(self.bf16_weights.is_some(), self.fp8_weights.is_some())
        {
            return self.forward_prefill(input, m as usize, ctx, stream);
        }
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let kh = self.batchm_kernel(m);

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        ops::w4a4_proj::nvfp4_proj_small_m(
            ctx.gpu,
            kh,
            input,
            &self.weights.gate_proj,
            gate_out,
            m,
            inter,
            h,
            stream,
        )?;
        // 2026-09-25: Same `input` as gate: the W4A4 path reuses gate's quantisation.
        ops::w4a4_proj::nvfp4_proj_small_m_same_input(
            ctx.gpu,
            kh,
            input,
            &self.weights.up_proj,
            up_out,
            m,
            inter,
            h,
            stream,
        )?;
        self.apply_lora_gate_up(ctx, input, gate_out, up_out, m, stream)?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a4_proj::nvfp4_proj_small_m(
            ctx.gpu,
            kh,
            gate_out,
            &self.weights.down_proj,
            output,
            m,
            h,
            inter,
            stream,
        )?;
        self.apply_lora_down(ctx, gate_out, output, m, stream)?;

        Ok(())
    }
}
