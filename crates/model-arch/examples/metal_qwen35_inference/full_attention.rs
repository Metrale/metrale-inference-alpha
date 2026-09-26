// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Full-attention decoder layer of the Metal Qwen3.5 example: the MLX int8 weight
//! loader and a `view()` that borrows it as the `qwen3_5::FullAttentionLayer` that
//! `metrale_model_layers::forward::qwen3_5::forward_full_attention` takes.
//!
//! Owner: model-arch examples (Metal Qwen3.5 driver).
//! Invariants: none beyond the types.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::metal_backend::MetalGpuBackend;
use metrale_gpu_runtime::mlx_int8::MlxInt8Weight;
use metrale_model_layers::forward::qwen3_5;
use safetensors::SafeTensors;

pub(crate) struct FullAttentionLayer {
    pub(crate) input_ln: DevicePtr,
    pub(crate) q_norm: DevicePtr,
    pub(crate) k_norm: DevicePtr,
    pub(crate) post_ln: DevicePtr,
    pub(crate) q_proj: MlxInt8Weight,
    pub(crate) k_proj: MlxInt8Weight,
    pub(crate) v_proj: MlxInt8Weight,
    pub(crate) o_proj: MlxInt8Weight,
    pub(crate) gate_proj: MlxInt8Weight,
    pub(crate) up_proj: MlxInt8Weight,
    pub(crate) down_proj: MlxInt8Weight,
}

impl FullAttentionLayer {
    pub(crate) fn load(
        backend: &MetalGpuBackend,
        st: &SafeTensors,
        layer_idx: u32,
        group_size: u32,
    ) -> Result<Self> {
        let prefix = format!("language_model.model.layers.{layer_idx}");
        let load_bf16 = |name: &str| -> Result<DevicePtr> {
            let t = st.tensor(name).with_context(|| format!("missing {name}"))?;
            let p = backend.alloc(t.data().len())?;
            backend.copy_h2d(t.data(), p)?;
            Ok(p)
        };
        let load_q = |suffix: &str| {
            MlxInt8Weight::load(backend, st, &format!("{prefix}.{suffix}"), group_size)
        };
        Ok(Self {
            input_ln: load_bf16(&format!("{prefix}.input_layernorm.weight"))?,
            q_norm: load_bf16(&format!("{prefix}.self_attn.q_norm.weight"))?,
            k_norm: load_bf16(&format!("{prefix}.self_attn.k_norm.weight"))?,
            post_ln: load_bf16(&format!("{prefix}.post_attention_layernorm.weight"))?,
            q_proj: load_q("self_attn.q_proj")?,
            k_proj: load_q("self_attn.k_proj")?,
            v_proj: load_q("self_attn.v_proj")?,
            o_proj: load_q("self_attn.o_proj")?,
            gate_proj: load_q("mlp.gate_proj")?,
            up_proj: load_q("mlp.up_proj")?,
            down_proj: load_q("mlp.down_proj")?,
        })
    }

    /// 2026-09-25: Borrow this layer as the struct the shared forward takes.
    pub(crate) fn view(&self) -> qwen3_5::FullAttentionLayer<'_, MlxInt8Weight> {
        qwen3_5::FullAttentionLayer {
            input_ln: self.input_ln,
            q_norm: self.q_norm,
            k_norm: self.k_norm,
            post_ln: self.post_ln,
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: &self.v_proj,
            o_proj: &self.o_proj,
            gate_proj: &self.gate_proj,
            up_proj: &self.up_proj,
            down_proj: &self.down_proj,
        }
    }
}
