// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The per-layer state the MLA load steps share: inputs, config scalars, and the weights each step fills in.
//!
//! Owner: model-arch (Mistral loader).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::weight_map::DenseWeight;

/// 2026-09-25: One layer's load state, built by `load_layers_inner` (and by
/// the LongCat loader). Each step reads the fields an earlier step set and
/// sets its own; `phase_assemble` consumes the context.
pub(crate) struct MistralLayerCtx<'a> {
    pub store: &'a WeightStore,
    pub config: &'a ModelConfig,
    pub gpu: &'a dyn GpuBackend,
    pub absmax_k: metrale_gpu_runtime::gpu::KernelHandle,
    pub quantize_k: metrale_gpu_runtime::gpu::KernelHandle,
    pub stream: u64,
    pub layer_idx: usize,

    // 2026-09-25: Copied from the config by `new`. A caller may override
    // them: the LongCat loader sets padded `hd`, `nope` and `v_dim`.
    pub h: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub nope: usize,
    pub rope: usize,
    pub v_dim: usize,
    pub bf16: usize,

    // 2026-09-25: Set by `phase_lora_qkv`.
    pub wq_a_dense: Option<DenseWeight>,
    pub wq_a_nvfp4: Option<metrale_model_layers::weight_map::QuantizedWeight>,
    pub wq_b: Option<DenseWeight>,
    pub wq_b_nvfp4: Option<metrale_model_layers::weight_map::QuantizedWeight>,
    pub q_a_norm: Option<DenseWeight>,
    pub wkv_a_dense: Option<DenseWeight>,
    pub wkv_a_nvfp4: Option<metrale_model_layers::weight_map::QuantizedWeight>,
    pub wkv_a_rope_dense: Option<DenseWeight>,
    pub wkv_b: Option<DenseWeight>,
    pub kv_a_norm: Option<DenseWeight>,

    // 2026-09-25: Set by `phase_per_head`; `w_uk_host` is read again by
    // `phase_block_diag`.
    pub w_uk_t: Option<DenseWeight>,
    pub w_uv: Option<DenseWeight>,
    pub wq_b_rope: Option<DenseWeight>,
    pub w_uk_host: Vec<u8>,

    // 2026-09-25: Set by `phase_qk_absorbed`.
    pub w_qk_absorbed: Option<DenseWeight>,

    // 2026-09-25: Set by `phase_block_diag`.
    pub w_uk_block_diag: Option<DenseWeight>,
    pub w_uv_block_diag: Option<DenseWeight>,

    // 2026-09-25: Set by `phase_o_proj`.
    pub o_dense_bf16: Option<DenseWeight>,
    pub o_nvfp4: Option<metrale_model_layers::weight_map::QuantizedWeight>,
}

impl<'a> MistralLayerCtx<'a> {
    pub(crate) fn new(
        store: &'a WeightStore,
        config: &'a ModelConfig,
        gpu: &'a dyn GpuBackend,
        absmax_k: metrale_gpu_runtime::gpu::KernelHandle,
        quantize_k: metrale_gpu_runtime::gpu::KernelHandle,
        stream: u64,
        layer_idx: usize,
    ) -> Self {
        Self {
            store,
            config,
            gpu,
            absmax_k,
            quantize_k,
            stream,
            layer_idx,
            h: config.hidden_size,
            n_heads: config.num_attention_heads,
            n_kv: config.num_key_value_heads,
            hd: config.head_dim,
            q_lora: config.q_lora_rank,
            kv_lora: config.kv_lora_rank,
            nope: config.qk_nope_head_dim,
            rope: config.qk_rope_head_dim,
            v_dim: config.v_head_dim,
            bf16: 2,
            wq_a_dense: None,
            wq_a_nvfp4: None,
            wq_b: None,
            wq_b_nvfp4: None,
            q_a_norm: None,
            wkv_a_dense: None,
            wkv_a_nvfp4: None,
            wkv_a_rope_dense: None,
            wkv_b: None,
            kv_a_norm: None,
            w_uk_t: None,
            w_uv: None,
            wq_b_rope: None,
            w_uk_host: Vec::new(),
            w_qk_absorbed: None,
            w_uk_block_diag: None,
            w_uv_block_diag: None,
            o_dense_bf16: None,
            o_nvfp4: None,
        }
    }

    pub(crate) fn ap(&self) -> String {
        format!("layers.{}.attention", self.layer_idx)
    }
}

/// 2026-09-25: The shared YaRN inv_freq table: computed and stored in
/// `*cached` on the first call (while it is null), then returned as is.
pub(crate) fn ensure_yarn_inv_freq(
    cached: &mut DevicePtr,
    config: &ModelConfig,
    rope: usize,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    if !cached.is_null() {
        return Ok(*cached);
    }
    let ptr = super::yarn::compute_yarn_inv_freq(config, rope, gpu)?;
    *cached = ptr;
    Ok(ptr)
}
