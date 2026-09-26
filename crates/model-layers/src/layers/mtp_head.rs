// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The MTP (multi-token prediction) draft head, a [`DraftProposer`]. One
//! decoder layer drafts from the token embedding and the target's hidden:
//! pre-fc norms, concat, fc, gated attention over its own KV cache, MoE or
//! dense FFN, final norm, the NVFP4 LM head and argmax. [`MtpQuantization`]
//! selects the projection weights' precision.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - `mtp_rows_to_trim` never returns more than `num_drafted`.

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::ForwardContext;
use crate::layers::MoeLayer;
use crate::layers::ops;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_map::{
    DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight, quantize_to_fp8, quantize_to_nvfp4,
};

/// 2026-09-25: Whether the drafter prefill is on (`levers.drafter.prefill`, on unless
/// `METRALE_NO_MTP_DRAFTER_CONTEXT` turns it off; policy in
/// `crate::drafter_context`). The prefill builds the drafter's KV over the whole
/// prompt before the first propose. The drafter's K/V depend only on its input
/// pair, not on attention, so it runs the norms, fc, K/V projections, RoPE and
/// the cache write, with no attention pass.
pub fn mtp_drafter_prefill_enabled(levers: &crate::layers::ops::ModelLevers) -> bool {
    levers.drafter.prefill
}

/// 2026-09-25: Scratch for the batched drafter-row writer (drafter prefill and
/// catch-up), allocated by `MtpHead::new` when either is enabled. Every buffer
/// holds `PREFILL_CHUNK` rows, and none aliases the shared arena.
pub(crate) struct MtpPrefillScratch {
    pub embed: DevicePtr,
    pub normed_embed: DevicePtr,
    pub normed_hidden: DevicePtr,
    pub concat: DevicePtr,
    pub fc_out: DevicePtr,
    pub normed2: DevicePtr,
    pub k_out: DevicePtr,
    pub v_out: DevicePtr,
    /// 2026-09-25: RoPE rotates Q and K in one launch; the row writer discards Q, but
    /// the kernel still needs a writable `[chunk, nq * hd]` region.
    pub q_scratch: DevicePtr,
    /// 2026-09-25: u32 RoPE positions, one per row.
    pub pos_dev: DevicePtr,
    /// 2026-09-25: i64 KV slot mapping, one per row.
    pub slot_dev: DevicePtr,
}

/// 2026-09-25: Precision of the MTP head's projection weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpQuantization {
    Nvfp4,
    Fp8,
    Bf16,
}

impl MtpQuantization {
    /// 2026-09-25: Whether the batched drafter prefill can run at this precision. The
    /// prefill itself re-checks the weight variants and kernel handles; this
    /// lets the caller skip the `max_seq_len x hidden` BF16 prompt-hidden buffer
    /// for a head that could never use it. `quantize_proj` produces
    /// `ProjectionWeight::Bf16` only for [`MtpQuantization::Bf16`].
    pub fn supports_drafter_prefill(self) -> bool {
        matches!(self, Self::Bf16)
    }

    /// 2026-09-25: The precision the head's forward runs at (KV dtype, forward arms,
    /// drafter prefill). A dense-FFN head under [`Self::Nvfp4`] runs the BF16
    /// forward, so it reports [`Self::Bf16`]; every other combination reports
    /// itself. `MtpHead::new`, the MTP KV-pool reserve and the prompt-capture
    /// allocation all ask here.
    pub fn effective_for_head(self, dense_ffn_head: bool) -> Self {
        if dense_ffn_head && matches!(self, Self::Nvfp4) {
            Self::Bf16
        } else {
            self
        }
    }
}

impl std::str::FromStr for MtpQuantization {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "nvfp4" | "fp4" => Ok(Self::Nvfp4),
            "fp8" => Ok(Self::Fp8),
            "bf16" => Ok(Self::Bf16),
            _ => anyhow::bail!("Unknown MTP quantization: {s}. Expected: nvfp4, fp8, bf16"),
        }
    }
}

/// 2026-09-25: A projection weight in one of the head's precisions.
#[allow(dead_code)]
enum ProjectionWeight {
    Nvfp4(QuantizedWeight),
    Fp8(Fp8DenseWeight),
    /// 2026-09-25: FP8 block-scaled weight, run by `w8a16_gemv`. No constructor in
    /// the head builds this variant.
    Fp8BlockScaled(Fp8Weight),
    Bf16(DenseWeight),
}

/// 2026-09-25: Per-sequence MTP proposer state.
pub struct MtpProposerState {
    /// 2026-09-25: Block table in the head's own KV cache.
    pub block_table: Vec<u32>,
    /// 2026-09-25: Drafter rows written in the head's KV cache.
    pub seq_len: usize,
    /// 2026-09-25: Drafts produced by the last propose; `after_verify` trims from it.
    pub last_num_drafted: usize,
    /// 2026-09-25: Sequence-space pair key of the newest drafter row: a `forward_one`
    /// at RoPE position `p` writes key `p - 1`. `seq_len` counts rows, which need
    /// not match sequence positions. `None` until a row is written.
    pub last_pair_key: Option<usize>,
    /// 2026-09-25: The drafts the last propose returned, in order.
    pub last_drafts: Vec<u32>,
    pub pending_catchup: Vec<u32>,
}

impl ProposerState for MtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// 2026-09-25: The MTP draft head.
#[allow(dead_code)]
pub struct MtpHead {
    pre_fc_norm_embedding: DenseWeight,
    pre_fc_norm_hidden: DenseWeight,
    input_layernorm: DenseWeight,
    post_attn_layernorm: DenseWeight,
    norm: DenseWeight,

    fc: ProjectionWeight,
    q_proj: ProjectionWeight,
    k_proj: ProjectionWeight,
    v_proj: ProjectionWeight,
    o_proj: ProjectionWeight,

    q_norm: DenseWeight,
    k_norm: DenseWeight,

    // 2026-09-25: FFN storage. An NVFP4 MoE head uses `moe_nvfp4`; an FP8/BF16 MoE
    // head uses `moe_fp8` when the checkpoint ships FP8 experts, else the
    // per-expert `*_generic` weights. A dense-FFN head leaves all of them `None`.
    moe_nvfp4: Option<MoeLayer>,
    moe_experts_generic: Option<Vec<(ProjectionWeight, ProjectionWeight, ProjectionWeight)>>,
    moe_shared_generic: Option<(ProjectionWeight, ProjectionWeight, ProjectionWeight)>,
    /// 2026-09-25: The checkpoint's FP8 block-scaled routed and shared experts as a
    /// [`MoeLayer`] with FP8 pointer tables, built for an FP8/BF16 head when
    /// `MtpWeights::fp8_experts` is present. `forward_one` runs it through
    /// `MoeLayer::forward`, and the batched propose through
    /// `forward_fp8_grouped_decode`. `None` means `moe_forward_generic` runs.
    moe_fp8: Option<MoeLayer>,
    moe_gate: DenseWeight,
    shared_expert_gate: DenseWeight,

    /// 2026-09-25: Dense FFN `(gate_proj, up_proj, down_proj)` for a head whose
    /// checkpoint has a dense FFN. When `Some`, the forward runs this MLP and
    /// the `Option` MoE fields are `None`.
    dense_ffn_generic: Option<(ProjectionWeight, ProjectionWeight, ProjectionWeight)>,

    quant: MtpQuantization,

    /// 2026-09-25: LM-head rows the drafter scores; 0 means the full vocab.
    mtp_vocab_size: u32,

    embed_tokens: DenseWeight,
    /// 2026-09-25: The target's final norm weight, applied to target-hidden rows before
    /// `pre_fc_norm_hidden` under `ModelLevers::mtp_target_postnorm`.
    target_final_norm: DenseWeight,
    lm_head_nvfp4: QuantizedWeight,

    // 2026-09-25: The head's own one-layer KV cache, separate from the target's.
    kv_cache: Mutex<PagedKvCache>,
    attn_layer_idx: usize,

    rms_norm_k: KernelHandle,
    rms_norm_residual_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    /// 2026-09-25: `w4a16_gemv_sw`; while it is 0, `w4a16_decode_gemv` runs the base
    /// GEMV.
    w4a16_gemv_sw_k: KernelHandle,
    /// 2026-09-25: False when `METRALE_NO_GEMV_SW=1`; read at construction because
    /// `gemv` has no `ForwardContext`.
    gemv_sw: bool,
    w4a16_gemv_qg_k: KernelHandle,
    w4a16_gemv_dual_k: KernelHandle,
    rope_k: KernelHandle,
    reshape_cache_k: KernelHandle,
    paged_decode_k: KernelHandle,
    /// 2026-09-25: KV cache dtype: BF16 for BF16 and FP8 heads, FP8 for NVFP4 heads.
    kv_bf16: bool,
    /// 2026-09-25: `ModelLevers::mtp_kv_exact`, stored at construction because
    /// `after_verify` has no `ForwardContext`.
    kv_exact: bool,
    residual_add_k: KernelHandle,
    residual_add_rms_norm_k: KernelHandle,
    sigmoid_gate_mul_k: KernelHandle,
    bf16_concat_k: KernelHandle,
    argmax_k: KernelHandle,
    embed_from_argmax_k: KernelHandle,
    /// 2026-09-25: 4-byte device buffer that `embed_from_argmax` writes the draft id to,
    /// for deferred readback.
    draft_token_id_dev: DevicePtr,
    /// 2026-09-25: Chain confidence of the last propose, as f32 bits: the minimum top-1
    /// softmax probability across its drafts. Reset to 1.0 at each propose and
    /// lowered by the forward when `draft_conf_tau > 0`.
    pub(super) last_conf_bits: std::sync::atomic::AtomicU32,
    dense_gemv_k: Option<KernelHandle>,
    dense_gemv_fp8w_k: Option<KernelHandle>,
    w8a16_gemv_k: Option<KernelHandle>,
    deinterleave_qg_k: Option<KernelHandle>,
    moe_topk_k: Option<KernelHandle>,
    moe_silu_mul_k: Option<KernelHandle>,
    moe_weighted_sum_blend_k: Option<KernelHandle>,
    /// 2026-09-25: Batched BF16 GEMM for the drafter prefill; 0 when absent.
    dense_gemm_k: KernelHandle,
    /// 2026-09-25: `dense_gemm_bf16_pipelined` for the batched propose; 0 when absent,
    /// and then the batched propose is out of scope.
    dense_gemm_pipelined_k: KernelHandle,
    /// 2026-09-25: `dense_gemv_bf16_batchm`, one pass over each BF16 weight for all M
    /// rows, for batched-propose widths 2..=8; 0 when absent.
    /// `row_dispatch::drafter_row_kernel` chooses between it and the others.
    dense_gemv_batchm_k: KernelHandle,
    /// 2026-09-25: Batched LM-head kernels: the `w4a16_batchm` tiers (widths 4 to
    /// 8) here and `w4a16_gemv_batch16`/`32` below (0 when absent);
    /// [`MtpHead::lm_head_batch_kernel`] picks one per batch width.
    w4a16_batchm: W4a16BatchmTiers,
    w4a16_gemv_batch16_k: KernelHandle,
    w4a16_gemv_batch32_k: KernelHandle,
    /// 2026-09-25: `(weight_t, ldb)`: the transposed LM-head copy the caller passes in,
    /// with row stride `ldb`. The batched propose runs a tile GEMM on it for
    /// n >= 5 when `w4a16_gemm_t_k` is non-zero. `None` when the caller passed
    /// none or `METRALE_NO_MTP_LMHEAD_TGEMM` is set to any value.
    pub(super) lm_head_nvfp4_t: Option<(QuantizedWeight, u32)>,
    /// 2026-09-25: Tile GEMM for `lm_head_nvfp4_t`, from `tgemm_kernel`; while it is 0
    /// the batched LM head uses `lm_head_batch_kernel`.
    pub(super) w4a16_gemm_t_k: KernelHandle,
    /// 2026-09-25: Drafter attention metadata for the batched propose,
    /// `PROPOSE_META_SEQS * propose_meta_stride` bytes, one slab per sequence.
    /// Its own allocation, not part of the shared scratch arena.
    propose_meta: DevicePtr,
    /// 2026-09-25: Per-sequence stride of `propose_meta`, from
    /// `batch_caps::propose_meta_stride_env` at construction.
    propose_meta_stride: usize,
    /// 2026-09-25: `argmax_bf16_batch` for the batched propose; while it is 0, the
    /// propose runs `argmax_bf16` once per row.
    argmax_batch_k: KernelHandle,
    /// 2026-09-25: `argmax_bf16_batch_lp`, the batched argmax that also writes each
    /// row's top-1 log-probability; used instead of `argmax_batch_k` when the
    /// caller asks for confidences and this handle is non-zero.
    argmax_batch_lp_k: KernelHandle,
    /// 2026-09-25: Drafter-row scratch; `None` unless the drafter prefill or the
    /// catch-up is enabled.
    prefill_scratch: Option<MtpPrefillScratch>,
}

impl MtpHead {
    /// 2026-09-25: Lock the head's KV cache; `mtp_multi` frees blocks through it.
    /// `parking_lot::Mutex` does not poison, so this cannot fail.
    pub(crate) fn kv_cache_lock(&self) -> parking_lot::MutexGuard<'_, PagedKvCache> {
        self.kv_cache.lock()
    }

    /// 2026-09-25: One-row GEMV with the kernel for `proj`'s precision.
    fn gemv(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        proj: &ProjectionWeight,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        match proj {
            ProjectionWeight::Nvfp4(w) => ops::w4a16_decode_gemv(
                gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                self.gemv_sw,
                input,
                w,
                output,
                n,
                k,
                stream,
            ),
            ProjectionWeight::Fp8(w) => ops::dense_gemv_fp8w(
                gpu,
                self.dense_gemv_fp8w_k.unwrap(),
                input,
                w,
                output,
                n,
                k,
                stream,
            ),
            ProjectionWeight::Fp8BlockScaled(w) => ops::w8a16_gemv(
                gpu,
                self.w8a16_gemv_k.unwrap(),
                input,
                w.weight,
                w.row_scale,
                output,
                n,
                k,
                stream,
            ),
            ProjectionWeight::Bf16(w) => ops::dense_gemv(
                gpu,
                self.dense_gemv_k.unwrap(),
                input,
                w,
                output,
                n,
                k,
                stream,
            ),
        }
    }

    /// 2026-09-25: Quantize a BF16 weight to `quant` (a copy of the handle for BF16).
    fn quantize_proj(
        bf16: &DenseWeight,
        n: usize,
        k: usize,
        quant: MtpQuantization,
        gpu: &dyn GpuBackend,
        absmax_k: KernelHandle,
        nvfp4_k: KernelHandle,
        fp8_k: KernelHandle,
        stream: u64,
    ) -> Result<ProjectionWeight> {
        match quant {
            MtpQuantization::Nvfp4 => Ok(ProjectionWeight::Nvfp4(quantize_to_nvfp4(
                bf16, n, k, gpu, absmax_k, nvfp4_k, stream,
            )?)),
            MtpQuantization::Fp8 => Ok(ProjectionWeight::Fp8(quantize_to_fp8(
                bf16, n, k, gpu, fp8_k, stream,
            )?)),
            MtpQuantization::Bf16 => Ok(ProjectionWeight::Bf16(*bf16)),
        }
    }
}

mod batch_caps;
mod catchup_batch;
mod chain_hidden;
mod draft_proposer;
mod forward;
mod forward_batch;
mod forward_batch_ffn;
mod moe_forward;
mod new;
mod new_native_fp8_moe;
mod prefill;
pub(crate) mod row_dispatch;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtp_proposer_state_downcast() {
        let state: Box<dyn ProposerState> = Box::new(MtpProposerState {
            block_table: vec![0, 1, 2],
            seq_len: 42,
            last_num_drafted: 0,
            last_pair_key: None,
            last_drafts: Vec::new(),
            pending_catchup: Vec::new(),
        });
        let mtp = state.as_any().downcast_ref::<MtpProposerState>().unwrap();
        assert_eq!(mtp.seq_len, 42);
        assert_eq!(mtp.block_table.len(), 3);
    }
}

/// 2026-09-25: How many drafter KV rows `after_verify` drops when exact KV is off.
///
/// * Rejected rows always go: `num_drafted - num_accepted`.
/// * With `refeed_accepted` (`mtp_refeed_accepted_enabled`), the accepted rows
///   built from a drafter hidden also go: every accepted draft except the first,
///   whose row was built from the target's hidden.
///
/// Never more than `num_drafted`.
pub(crate) fn mtp_rows_to_trim(
    num_drafted: usize,
    num_accepted: usize,
    refeed_accepted: bool,
) -> usize {
    let rejected = num_drafted.saturating_sub(num_accepted);
    let accepted_with_drafter_hidden = if refeed_accepted {
        num_accepted.saturating_sub(1)
    } else {
        0
    };
    (rejected + accepted_with_drafter_hidden).min(num_drafted)
}

#[cfg(test)]
mod refeed_trim_tests;
