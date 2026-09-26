// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Kernel handles for the served NLLB bf16 runtime, looked up once at model
//! construction from the `nllb_encoder` module (`kernels/gb10/common/nllb_encoder.cu`)
//! and the shared `gemm` and `argmax` modules.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// 2026-09-25: The NLLB kernel handles.
pub(super) struct NllbKernels {
    pub embed: KernelHandle,
    pub scale: KernelHandle,
    pub add: KernelHandle,
    pub relu: KernelHandle,
    pub ln: KernelHandle,
    /// 2026-09-25: Out-of-place layer norm (reads `src`, writes `dst`); the beam
    /// decode uses it so no `dh -> normed` copy precedes each layer norm.
    pub ln_oop: KernelHandle,
    pub bias: KernelHandle,
    pub attn: KernelHandle,
    /// 2026-09-25: Tensor-core pipelined GEMM (`gemm` module) for multi-row
    /// projections: the encoder, the beam decode and the beam lm_head.
    pub gemm: KernelHandle,
    /// 2026-09-25: GEMV (`nllb_encoder`) for the single-token decode projections
    /// and lm_head.
    pub gemv: KernelHandle,
    /// 2026-09-25: On-device argmax over bf16 logits.
    pub argmax: KernelHandle,
    /// 2026-09-25: Batched decode attention over B rows, per-row key length `tk[b]`.
    pub attn_bdecode: KernelHandle,
    /// 2026-09-25: Write `src[B,d]` into a batch-major cache `[B,stride,d]` at row `pos`.
    pub scatter: KernelHandle,
    /// 2026-09-25: Reorder beam caches: `dst[i] = src[perm[i]]`.
    pub gather: KernelHandle,
    /// 2026-09-25: Broadcast-add one position row across a `[B,d]` batch.
    pub add_row: KernelHandle,
    /// 2026-09-25: On-device beam candidate reduction: per row, the log-sum-exp
    /// over the full vocab and the top-K `(value, token)` pairs.
    pub beam_topk: KernelHandle,
}

impl NllbKernels {
    pub(super) fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            embed: gpu.kernel("nllb_encoder", "nllb_embed_bf16")?,
            scale: gpu.kernel("nllb_encoder", "nllb_scale_bf16")?,
            add: gpu.kernel("nllb_encoder", "nllb_add_bf16")?,
            relu: gpu.kernel("nllb_encoder", "nllb_relu_bf16")?,
            ln: gpu.kernel("nllb_encoder", "nllb_layernorm_bf16")?,
            ln_oop: gpu.kernel("nllb_encoder", "nllb_layernorm_oop_bf16")?,
            bias: gpu.kernel("nllb_encoder", "nllb_bias_bf16")?,
            attn: gpu.kernel("nllb_encoder", "nllb_attn_kv_bf16")?,
            gemm: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            gemv: gpu.kernel("nllb_encoder", "nllb_gemv_bf16")?,
            argmax: gpu.kernel("argmax", "argmax_bf16")?,
            attn_bdecode: gpu.kernel("nllb_encoder", "nllb_attn_bdecode")?,
            scatter: gpu.kernel("nllb_encoder", "nllb_scatter_batched")?,
            gather: gpu.kernel("nllb_encoder", "nllb_gather_batched")?,
            add_row: gpu.kernel("nllb_encoder", "nllb_add_row_bf16")?,
            beam_topk: gpu.kernel("nllb_encoder", "nllb_beam_topk")?,
        })
    }
}
