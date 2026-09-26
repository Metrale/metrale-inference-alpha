// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-call context for multi-sequence batched decode: the scalars
//! and buffer pointers the phases share.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - `per_seq_qkv` is the byte width of one sequence's `[Q | K | V]` row in
//!   BF16, with Q `q_proj_dim` wide (twice `q_dim` on a gated layer).
//! - `seq_slot` is `DevicePtr(0)` until the caller installs the step's slot buffer.

use metrale_gpu_runtime::gpu::DevicePtr;

use crate::layer::ForwardContext;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// 2026-09-25: Built once per call by `decode_multi_seq_inner` and passed to
/// each phase by reference.
#[allow(dead_code)]
pub(super) struct MultiSeqCtx<'a> {
    /// 2026-09-25: The forward-pass context (backend, buffers, config).
    pub fwd: &'a ForwardContext<'a>,
    /// 2026-09-25: Hidden state, one row per sequence.
    pub hidden: DevicePtr,
    /// 2026-09-25: Residual, one row per sequence.
    pub residual: DevicePtr,
    /// 2026-09-25: Number of sequences, one decode token each.
    pub n: usize,
    pub stream: u64,

    // 2026-09-25: Config and per-layer scalars, resolved once in `new`.
    pub h: usize,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub eps: f32,
    pub bs: u32,
    pub bf16: usize,
    pub q_dim: u32,
    pub q_proj_dim: u32,
    pub q_proj_bytes: usize,
    pub per_seq_qkv: usize,

    /// 2026-09-25: The `norm_output` buffer: the RMS-normed hidden rows that
    /// the projections read.
    pub normed: DevicePtr,
    /// 2026-09-25: The `qkv_output` buffer: one `[Q | K | V]` row per sequence,
    /// `per_seq_qkv` bytes apart.
    pub qkv_buf: DevicePtr,
    /// 2026-09-25: Per-request LoRA routing: the step's `[n]` i32 adapter-slot
    /// buffer, copied from `AttnMetadataDev::seq_slot` by `decode_multi_seq_inner`.
    /// The Q/K/V (`ms_qkv_apply_lora`) and O (`ms_o_proj_lora`) bgmv sites skip
    /// their work when it is `DevicePtr(0)`.
    pub seq_slot: DevicePtr,
}

impl<'a> MultiSeqCtx<'a> {
    pub(super) fn new(
        layer: &Qwen3AttentionLayer,
        fwd: &'a ForwardContext<'a>,
        hidden: DevicePtr,
        residual: DevicePtr,
        n: usize,
        bs: u32,
        stream: u64,
    ) -> Self {
        let h = fwd.config.hidden_size;
        let nq = layer
            .num_q_heads_override
            .unwrap_or(fwd.config.num_attention_heads) as u32;
        let nkv = layer
            .num_kv_heads_override
            .unwrap_or(fwd.config.num_key_value_heads) as u32;
        let hd = layer.head_dim_override.unwrap_or(fwd.config.head_dim) as u32;
        let eps = fwd.config.rms_norm_eps as f32;
        let bf16 = 2usize;
        let q_dim = nq * hd;
        let q_proj_dim = if layer.gated { q_dim * 2 } else { q_dim };
        let q_proj_bytes = q_proj_dim as usize * bf16;
        let per_seq_qkv = q_proj_bytes + (nkv * hd) as usize * bf16 * 2;
        let normed = fwd.buffers.norm_output();
        let qkv_buf = fwd.buffers.qkv_output();
        Self {
            fwd,
            hidden,
            residual,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bs,
            bf16,
            q_dim,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            seq_slot: DevicePtr(0),
        }
    }
}
