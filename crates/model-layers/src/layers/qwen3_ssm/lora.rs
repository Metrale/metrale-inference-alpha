// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: LoRA on the GDN layer: install of the MoE router/expert, dense
//! FFN and GDN `out_proj` deltas, and the `out_proj` delta's apply step.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::Qwen3SsmLayer;
use crate::layers::FfnComponent;
use crate::layers::ops::lora_delta::{LoraKernels, LoraPair};
use crate::lora::ExpertLoraLayer;

impl Qwen3SsmLayer {
    /// 2026-09-25: Install router and routed-expert LoRA on this layer's MoE FFN
    /// (`MoeLayer::set_lora_weights`). Errors when the FFN is not MoE.
    pub fn set_moe_lora_weights(
        &mut self,
        router: Option<LoraPair>,
        experts: ExpertLoraLayer,
        kernels: LoraKernels,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if let FfnComponent::Moe(m) = &mut self.ffn {
            return m.set_lora_weights(router, experts, kernels, gpu);
        }
        anyhow::bail!(
            "LoRA: router/expert deltas installed on a linear-attention layer whose \
             FFN is not MoE (loader/adapter mismatch)"
        )
    }
}

impl Qwen3SsmLayer {
    /// 2026-09-25: Install dense-FFN LoRA on this layer's `FfnComponent::Dense`
    /// (`DenseFfnLayer::set_lora_weights`). Errors when the FFN is MoE or
    /// absent, rather than dropping the deltas.
    pub fn set_ffn_lora_weights(
        &mut self,
        ffn: crate::layers::ops::lora_delta::LoraFfnWeights,
    ) -> Result<()> {
        match &mut self.ffn {
            FfnComponent::Dense(d) => d.set_lora_weights(ffn),
            FfnComponent::Moe(_) => anyhow::bail!(
                "LoRA: dense-FFN delta on a linear-attention layer whose FFN is MoE — \
                 routed-expert deltas belong on set_moe_lora_weights"
            ),
            FfnComponent::None => {
                anyhow::bail!("LoRA: dense-FFN delta on a linear-attention layer that has no FFN")
            }
        }
    }
}

impl Qwen3SsmLayer {
    /// 2026-09-25: Install this layer's GDN `out_proj` delta, which
    /// `apply_lora_out_proj` adds.
    pub fn set_out_proj_lora(&mut self, pair: LoraPair, kernels: LoraKernels) {
        self.lora_out_proj = Some((pair, kernels));
    }

    /// 2026-09-25: `out += pair.scale * (normed_out @ A^T) @ B^T`, in place.
    ///
    /// Returns without launching when no delta is installed or
    /// `METRALE_LORA_NO_FFN=1` (`lora_no_ffn`). Its caller,
    /// `ssm_tp_all_reduce`, runs it after the TP all-reduce: before that `out`
    /// is a per-rank partial, and a delta added there would be summed once per
    /// rank.
    pub(super) fn apply_lora_out_proj(
        &self,
        ctx: &crate::layers::ForwardContext,
        normed_out: metrale_gpu_runtime::gpu::DevicePtr,
        out: metrale_gpu_runtime::gpu::DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        if crate::layers::ops::lora_delta::lora_no_ffn() {
            return Ok(());
        }
        let Some((ref pair, ref kernels)) = self.lora_out_proj else {
            return Ok(());
        };
        crate::layers::ops::lora_delta::apply_lora_delta(
            ctx.gpu,
            kernels,
            pair,
            normed_out,
            out,
            m,
            ctx.buffers.lora_xa(),
            ctx.buffers.lora_delta(),
            stream,
        )
    }
}
