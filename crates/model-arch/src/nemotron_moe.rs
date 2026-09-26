// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Nemotron-H standalone MoE FFN layer, in two shapes:
//!   - direct (`moe_latent_size == 0`): the routed experts work on the hidden
//!     vector;
//!   - LatentMoE (`moe_latent_size > 0`): the routed experts work in a latent
//!     space of `moe_latent_size`, with the fc1 / fc2 latent projections
//!     between hidden and latent.
//!
//! Forward: RMS norm, gate, sigmoid top-k routing, (fc1 if latent), routed
//! up, relu² and down, weighted sum, (fc2 if latent), plus the shared expert
//! (up, relu², down), summed and added to `hidden`. Experts are selected on
//! the device through pointer tables; routing reads nothing back to the host.
//!
//! Owner: model-arch (Nemotron-H).
//! Invariants:
//! - `new` refuses a `top_k` outside `1..=num_experts` or over the routing
//!   kernels' bounds (`MOE_TOPK_SIGMOID_MAX_TOP_K`,
//!   `MOE_TOPK_SIGMOID_MAX_EXPERTS`).

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use metrale_model_layers::layer::{EmptyLayerState, ForwardContext, LayerState, TransformerLayer};
use metrale_model_layers::layer::{
    LayerAuxState, LayerCapabilities, LayerGraphHooks, LayerSplitPrefill, LayerWeightSetup,
    LayerWriteOnAccept,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, NemotronMoeWeights, QuantizedWeight};

/// 2026-09-25: Device-side pointer table for one projection across all experts.
struct ExpertPtrTable {
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
}

/// 2026-09-25: Nemotron-H standalone MoE FFN layer.
pub struct NemotronMoeLayer {
    weights: NemotronMoeWeights,
    input_norm: DenseWeight,
    /// 2026-09-25: LatentMoE dimension (0 = direct, >0 = latent).
    moe_latent_size: usize,
    /// 2026-09-25: Routed expert intermediate size for this layer: the
    /// constructor's `moe_inter`, or `config.moe_intermediate_size` when it is 0.
    moe_inter: usize,
    /// 2026-09-25: Experts per token for this layer: the constructor's `top_k`,
    /// or `config.num_experts_per_tok` when it is 0.
    top_k: usize,
    rms_norm_residual_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    topk_sigmoid_k: KernelHandle,
    moe_expert_gemv_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    /// 2026-09-25: Single-warp `w4a16_gemv_sw`; `KernelHandle(0)` when the
    /// kernel is missing, and then decode uses the base GEMV.
    w4a16_gemv_sw_k: KernelHandle,
    /// 2026-09-25: Native-FP8 GEMV for the shared expert's up_proj and
    /// down_proj (`NemotronMoeWeights::shared_up_fp8`, `shared_down_fp8`). 0
    /// when unavailable.
    w8a16_gemv_k: KernelHandle,
    /// 2026-09-25: Native-FP8 prefill GEMM for the shared expert. 0 when unavailable.
    w8a16_gemm_k: KernelHandle,
    w8a16_gemm_pipelined_k: KernelHandle,
    relu2_down_shared_k: KernelHandle,
    weighted_sum_scale_k: KernelHandle,
    residual_add_k: KernelHandle,
    dense_gemm_k: KernelHandle,
    /// 2026-09-25: `dense_gemm_bf16_pipelined`, which `dense_gemm_prefill`
    /// prefers over `dense_gemm_k`; 0 when the target lacks it.
    dense_gemm_pipelined_k: KernelHandle,
    w4a16_gemm_k: KernelHandle,
    topk_sigmoid_batched_k: KernelHandle,
    moe_up_prefill_k: KernelHandle,
    moe_relu2_down_prefill_k: KernelHandle,
    moe_weighted_sum_prefill_k: KernelHandle,
    moe_sort_k: KernelHandle,
    moe_grouped_gemm_k: KernelHandle,
    moe_relu2_elementwise_k: KernelHandle,
    moe_grouped_gemm_relu2_k: KernelHandle,
    moe_w4a4_grouped_k: KernelHandle,
    moe_unpermute_reduce_k: KernelHandle,
    moe_grouped_gemm_n128_k: KernelHandle,
    up_ptrs: ExpertPtrTable,
    down_ptrs: ExpertPtrTable,
    // 2026-09-25: Transposed expert pointer tables, for `moe_grouped_gemm_n128_k`.
    up_ptrs_t: Option<ExpertPtrTable>,
    down_ptrs_t: Option<ExpertPtrTable>,
    // 2026-09-25: Transposed shared-expert weights, for the `w4a16_gemm_t` prefill arms.
    shared_up_t: Option<QuantizedWeight>,
    shared_down_t: Option<QuantizedWeight>,
    // 2026-09-25: Pre-dequantized FP8 E4M3 [N, K] copies of the shared-expert
    // projections, for `fp8_gemm_t_m128_mfast`.
    shared_up_pd_fp8: Option<DevicePtr>,
    shared_down_pd_fp8: Option<DevicePtr>,
    // 2026-09-25: FP8 E4M3 copies of the BF16 latent projections, so prefill
    // runs `fp8_gemm_t_m128_mfast` for fc1 / fc2 instead of `dense_gemm_prefill`.
    fc1_pd_fp8: Option<DevicePtr>,
    fc2_pd_fp8: Option<DevicePtr>,
    // 2026-09-25: The shared expert's transposed-NVFP4, pre-dequant-FP8 and W4A4
    // prefill kernels.
    w4a16_gemm_t_k: KernelHandle,
    w4a16_gemm_t_m128_k: KernelHandle,
    fp8_gemm_m128_k: KernelHandle,
    w4a4_gemm_k: KernelHandle,
    quantize_nvfp4_k: KernelHandle,
}

impl NemotronMoeLayer {
    pub fn new(
        weights: NemotronMoeWeights,
        input_norm: DenseWeight,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        moe_inter: usize,
        top_k: usize,
    ) -> Result<Self> {
        let up_ptrs = build_ptr_table(&weights.experts, |e| &e.up_proj, gpu)?;
        let down_ptrs = build_ptr_table(&weights.experts, |e| &e.down_proj, gpu)?;
        let moe_inter = if moe_inter > 0 {
            moe_inter
        } else {
            config.moe_intermediate_size
        };
        let top_k = if top_k > 0 {
            top_k
        } else {
            config.num_experts_per_tok
        };
        // 2026-09-25: The loader passes each layer's own `top_k`
        // (`num_experts_per_tok_for`), so the routing kernels' bounds are
        // checked per MoE layer.
        let num_experts = weights.experts.len();
        anyhow::ensure!(
            top_k > 0
                && top_k <= num_experts
                && top_k <= metrale_model_layers::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K
                && num_experts <= metrale_model_layers::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
            "Nemotron MoE config invalid: top_k={} must be in 1..={} and within \
             the routing kernels' bounds (top_k max {}, num_experts={} max {})",
            top_k,
            num_experts,
            metrale_model_layers::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K,
            num_experts,
            metrale_model_layers::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
        );

        Ok(Self {
            weights,
            input_norm,
            moe_latent_size: config.moe_latent_size,
            moe_inter,
            top_k,
            rms_norm_residual_k: gpu.kernel("norm", "rms_norm_residual")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            topk_sigmoid_k: gpu.kernel("moe_topk_sig", "moe_topk_sigmoid")?,
            moe_expert_gemv_k: gpu.kernel("moe_expert_gemv", "moe_expert_gemv")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_sw",
            ),
            w8a16_gemv_k: metrale_model_layers::layers::try_kernel(gpu, "w8a16_gemv", "w8a16_gemv"),
            w8a16_gemm_k: metrale_model_layers::layers::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemm_pipelined_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            relu2_down_shared_k: gpu.kernel("moe_relu2_fused", "moe_expert_relu2_down_shared")?,
            weighted_sum_scale_k: gpu.kernel("relu2", "moe_weighted_sum_scale")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            dense_gemm_pipelined_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "gemm",
                "dense_gemm_bf16_pipelined",
            ),
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            topk_sigmoid_batched_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_topk_sigmoid_batched",
            ),
            moe_up_prefill_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_up_prefill",
            ),
            moe_relu2_down_prefill_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_relu2_down_prefill",
            ),
            moe_weighted_sum_prefill_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_weighted_sum_prefill",
            ),
            moe_sort_k: metrale_model_layers::layers::try_kernel(gpu, "moe", "moe_sort_by_expert"),
            moe_grouped_gemm_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable",
            ),
            moe_relu2_elementwise_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "relu2",
                "relu_squared_inplace",
            ),
            moe_grouped_gemm_relu2_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_relu2",
            ),
            moe_w4a4_grouped_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "moe_w4a4",
                "moe_w4a4_grouped_gemm_relu2",
            ),
            moe_unpermute_reduce_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "moe",
                "moe_unpermute_reduce_indexed",
            ),
            moe_grouped_gemm_n128_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_t",
            ),
            up_ptrs,
            down_ptrs,
            up_ptrs_t: None,
            down_ptrs_t: None,
            shared_up_t: None,
            shared_down_t: None,
            shared_up_pd_fp8: None,
            shared_down_pd_fp8: None,
            fc1_pd_fp8: None,
            fc2_pd_fp8: None,
            w4a16_gemm_t_k: metrale_model_layers::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_t"),
            w4a16_gemm_t_m128_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m128",
            ),
            fp8_gemm_m128_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "w4a16",
                "fp8_gemm_t_m128_mfast",
            ),
            w4a4_gemm_k: metrale_model_layers::layers::try_kernel(gpu, "w4a4", "w4a4_gemm_mfast"),
            quantize_nvfp4_k: metrale_model_layers::layers::try_kernel(
                gpu,
                "quantize_nvfp4",
                "quantize_bf16_to_nvfp4",
            ),
        })
    }
}

mod decode_helpers;
mod prefill_fallback;
mod prefill_shared_up;
mod prefill_sorted;
mod prefill_weights;
mod ptr_tables;

use prefill_sorted::SortedPrefillCtx;
use ptr_tables::{build_ptr_table, build_ptr_table_from_weights};

impl TransformerLayer for NemotronMoeLayer {
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        _state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_inner(hidden, residual, ctx, stream)
    }

    /// 2026-09-25: Prefill over all tokens: batched RMS norm, gate GEMM, shared
    /// expert up GEMM and (LatentMoE) fc1 GEMM, then the sorted grouped-GEMM
    /// expert path (`prefill_sorted.rs`) or the per-token fallback
    /// (`prefill_fallback.rs`).
    #[allow(clippy::overly_complex_bool_expr)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        _state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let inter = self.moe_inter as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = self.top_k as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let scale = ctx.config.routed_scaling_factor as f32;
        let n = num_tokens as u32;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;

        let gate_logits = ctx.buffers.gate_logits();
        self.dense_gemm_prefill(
            ctx.gpu,
            normed,
            &self.weights.gate,
            gate_logits,
            n,
            num_experts,
            h as u32,
            stream,
        )?;

        let has_batched = self.topk_sigmoid_batched_k.0 != 0
            && self.moe_up_prefill_k.0 != 0
            && self.moe_relu2_down_prefill_k.0 != 0
            && self.moe_weighted_sum_prefill_k.0 != 0;

        let shared_up_out_base = ctx.buffers.ssm_qkvz();
        let use_batched_moe = has_batched && num_tokens > 1;
        // 2026-09-25: The shared expert's up projection for every token, into
        // `ssm_qkvz`; both expert paths read it. Arms in `prefill_shared_up.rs`.
        self.prefill_shared_up(normed, shared_up_out_base, n, h, shared_inter, ctx, stream)?;

        // 2026-09-25: LatentMoE: fc1 into `attn_output`. `moe_output` is not
        // used here because the expert paths write their routed output there.
        let latent = self.moe_latent_size as u32;
        let latent_base = if latent > 0 {
            let latent_buf = ctx.buffers.attn_output();
            if let Some(w_fp8) = self.fc1_pd_fp8 {
                ops::fp8_gemm_m128_mfast(
                    ctx.gpu,
                    self.fp8_gemm_m128_k,
                    normed,
                    w_fp8,
                    latent_buf,
                    n,
                    latent,
                    h as u32,
                    stream,
                )?;
            } else {
                let fc1 = self.weights.fc1_latent_proj.as_ref().unwrap();
                self.dense_gemm_prefill(
                    ctx.gpu, normed, fc1, latent_buf, n, latent, h as u32, stream,
                )?;
            }
            Some(latent_buf)
        } else {
            None
        };

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(n as usize * top_k as usize * 4);

        // 2026-09-25: The sorted path (sort the routed rows by expert, then grouped
        // GEMMs) needs more than one token, the four `nemotron_moe_prefill`
        // kernels (`has_batched`), and the sort, grouped-GEMM and unpermute kernels.
        let use_sorted = use_batched_moe
            && self.moe_sort_k.0 != 0
            && self.moe_grouped_gemm_k.0 != 0
            && self.moe_unpermute_reduce_k.0 != 0;

        let p = SortedPrefillCtx {
            n,
            num_tokens,
            h,
            inter,
            shared_inter,
            num_experts,
            top_k,
            scale,
            latent,
            gate_logits,
            indices_dev,
            weights_dev,
            normed,
            hidden,
            latent_base,
            shared_up_out_base,
        };
        if use_sorted {
            self.prefill_sorted_path(&p, ctx, stream)?;
        } else {
            self.prefill_fallback_path(&p, ctx, stream)?;
        }

        Ok(())
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(EmptyLayerState))
    }
}

impl LayerCapabilities for NemotronMoeLayer {}
impl LayerWeightSetup for NemotronMoeLayer {}
impl LayerWriteOnAccept for NemotronMoeLayer {}
impl LayerGraphHooks for NemotronMoeLayer {}
impl LayerAuxState for NemotronMoeLayer {}
impl LayerSplitPrefill for NemotronMoeLayer {}
