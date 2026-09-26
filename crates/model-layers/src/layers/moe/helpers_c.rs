// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Expert precision setup (FP8 and BF16 expert tables, the BF16 shared
//! expert, FP8 prefill copies), the BF16 shared-expert pass, the router GEMM for
//! a BF16 gate, and the router pre-norm.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

impl MoeLayer {
    /// 2026-09-25: Make FP8 copies of the NVFP4 router gate and of the NVFP4 shared
    /// expert for the prefill FP8 GEMMs. Routed experts are untouched.
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let h = config.hidden_size;
        let shared_inter = config.shared_expert_intermediate_size;
        let num_experts = config.num_experts;
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;

        if let Some(ref nvfp4) = self.gate_nvfp4 {
            self.gate_fp8 =
                Some(nvfp4.predequant_to_fp8(gpu, predequant_k, num_experts, h, stream)?);
        }

        // 2026-09-25: An installed BF16 shared expert is used as is; no FP8 copy is
        // made of the NVFP4 placeholder beside it.
        if self.bf16_shared_expert.is_none()
            && !self.weights.shared_expert.gate_proj.is_null()
            && shared_inter > 0
        {
            self.shared_gate_fp8 = Some(self.weights.shared_expert.gate_proj.predequant_to_fp8(
                gpu,
                predequant_k,
                shared_inter,
                h,
                stream,
            )?);
            self.shared_up_fp8 = Some(self.weights.shared_expert.up_proj.predequant_to_fp8(
                gpu,
                predequant_k,
                shared_inter,
                h,
                stream,
            )?);
            self.shared_down_fp8 = Some(self.weights.shared_expert.down_proj.predequant_to_fp8(
                gpu,
                predequant_k,
                h,
                shared_inter,
                stream,
            )?);
        }

        Ok(())
    }

    /// 2026-09-25: Install FP8 routed experts: device pointer tables for gate, up
    /// and down, indexed by expert id in the FP8 kernels, plus the FP8 shared
    /// expert.
    pub fn set_fp8_experts(
        &mut self,
        experts: &[Fp8ExpertWeight],
        shared_expert: Fp8ExpertWeight,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.fp8_gate_weight_ptrs = Some(build_fp8_ptr_table(experts, |e| &e.gate_proj, gpu)?);
        self.fp8_up_weight_ptrs = Some(build_fp8_ptr_table(experts, |e| &e.up_proj, gpu)?);
        self.fp8_down_weight_ptrs = Some(build_fp8_ptr_table(experts, |e| &e.down_proj, gpu)?);
        self.fp8_shared_expert = Some(shared_expert);
        Ok(())
    }

    /// 2026-09-25: Install BF16 routed experts (FP8 experts dequantized at load;
    /// the qwen35 loader does this under METRALE_FP8_DEQUANT_MOE_TO_BF16=1).
    ///
    /// `shared_*` are the shared expert's BF16 gate/up/down pointers; all three
    /// NULL means no shared expert.
    pub fn set_bf16_experts(
        &mut self,
        gate_experts: &[crate::weight_map::DenseWeight],
        up_experts: &[crate::weight_map::DenseWeight],
        down_experts: &[crate::weight_map::DenseWeight],
        shared_gate: DevicePtr,
        shared_up: DevicePtr,
        shared_down: DevicePtr,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        use super::build_bf16_ptr_table;
        self.bf16_gate_weight_ptrs = Some(build_bf16_ptr_table(gate_experts, gpu)?);
        self.bf16_up_weight_ptrs = Some(build_bf16_ptr_table(up_experts, gpu)?);
        self.bf16_down_weight_ptrs = Some(build_bf16_ptr_table(down_experts, gpu)?);
        if shared_gate.is_null() && shared_up.is_null() && shared_down.is_null() {
            self.bf16_shared_expert = None;
        } else {
            self.set_bf16_shared_expert(
                DenseWeight {
                    weight: shared_gate,
                },
                DenseWeight { weight: shared_up },
                DenseWeight {
                    weight: shared_down,
                },
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: Install a BF16 shared expert, whatever the routed experts'
    /// precision.
    pub fn set_bf16_shared_expert(
        &mut self,
        gate_proj: DenseWeight,
        up_proj: DenseWeight,
        down_proj: DenseWeight,
    ) -> Result<()> {
        self.bf16_shared_expert = Some(Bf16SharedExpert::new(gate_proj, up_proj, down_proj)?);
        Ok(())
    }

    /// 2026-09-25: True when a BF16 shared expert sits beside non-BF16 routed
    /// experts, so the quantized fused kernels cannot compute the shared term.
    pub(super) fn has_mixed_bf16_shared_expert(&self) -> bool {
        self.bf16_shared_expert.is_some() && self.bf16_gate_weight_ptrs.is_none()
    }

    /// 2026-09-25: Run the installed BF16 shared expert (gate, up, `moe_act_mul`,
    /// down) for `num_tokens` rows into `down_out`. Callers pass the scratch
    /// buffers because the free aliases differ between decode and prefill.
    ///
    /// Returns `Ok(false)` without launching when no BF16 shared expert is
    /// installed. Errors if `num_tokens` or `shared_intermediate` is 0.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_bf16_shared_expert(
        &self,
        input: DevicePtr,
        num_tokens: u32,
        hidden_size: u32,
        shared_intermediate: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        down_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(shared) = self.bf16_shared_expert else {
            return Ok(false);
        };
        anyhow::ensure!(
            num_tokens > 0 && shared_intermediate > 0,
            "BF16 shared expert requires non-zero token and intermediate dimensions"
        );

        let project = |activation: DevicePtr,
                       weight: &DenseWeight,
                       output: DevicePtr,
                       n: u32,
                       k: u32|
         -> Result<()> {
            if num_tokens == 1 {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv,
                    activation,
                    weight,
                    output,
                    n,
                    k,
                    stream,
                )
            } else if ctx.dispatch.cublas.ffn {
                ops::cublas_bf16_proj_dense(
                    activation,
                    weight.weight,
                    output,
                    num_tokens,
                    n,
                    k,
                    stream,
                )
            } else {
                ops::dense_gemm_prefill(
                    ctx.gpu,
                    self.dense_gemm,
                    self.dense_gemm_pipelined,
                    activation,
                    weight,
                    output,
                    num_tokens,
                    n,
                    k,
                    stream,
                )
            }
        };

        project(
            input,
            &shared.gate_proj,
            gate_out,
            shared_intermediate,
            hidden_size,
        )?;
        project(
            input,
            &shared.up_proj,
            up_out,
            shared_intermediate,
            hidden_size,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            gate_out,
            up_out,
            gate_out,
            num_tokens * shared_intermediate,
            stream,
        )?;
        project(
            gate_out,
            &shared.down_proj,
            down_out,
            hidden_size,
            shared_intermediate,
        )?;
        Ok(true)
    }

    /// 2026-09-25: Router GEMM for a BF16 gate weight:
    /// `gate_logits[num_tokens, num_experts] = router_in @ gate^T`.
    ///
    /// It keeps the scalar kernel's accumulation order: `dense_gemm_bf16_router`
    /// when the target has it (same per-output order as `dense_gemm_bf16`, see
    /// dense_gemm_bf16.cu), else `dense_gemm_bf16` itself. Router logits select
    /// experts after a BF16 store, so an order change can flip near-tied
    /// selections. Measured 2026-08-12: routing this GEMM to
    /// `dense_gemm_bf16_pipelined` moved BFCL on the FP8 MoE flagship from 86.55
    /// to 84.76.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn router_gate_gemm_dense(
        &self,
        router_in: DevicePtr,
        gate_logits: DevicePtr,
        num_tokens: u32,
        num_experts: u32,
        hidden_size: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.dense_gemm_router.0 != 0 {
            return ops::dense_gemm_router(
                ctx.gpu,
                self.dense_gemm_router,
                router_in,
                &self.weights.gate,
                gate_logits,
                num_tokens,
                num_experts,
                hidden_size,
                stream,
            );
        }
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm,
            router_in,
            &self.weights.gate,
            gate_logits,
            num_tokens,
            num_experts,
            hidden_size,
            stream,
        )
    }

    /// 2026-09-25: The router input. With `router_pre_norm` (the Gemma-4 weight map
    /// stores `scale * hidden_size^-0.5` in it), rms-norm `input` with that
    /// weight into `ctx.buffers.qkv_output()` and return it; without, return
    /// `input`. The caller must not need `qkv_output` across the MoE block.
    pub(super) fn router_input(
        &self,
        input: DevicePtr,
        num_tokens: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let Some(ref weight) = self.weights.router_pre_norm else {
            return Ok(input);
        };
        let eps = ctx.config.rms_norm_eps as f32;
        let normed = ctx.buffers.qkv_output();
        ops::rms_norm(
            ctx.gpu,
            self.pre_expert_norm_k,
            input,
            weight,
            normed,
            num_tokens,
            h,
            eps,
            stream,
        )?;
        Ok(normed)
    }
}
