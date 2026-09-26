// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: DeepSeek-V4.1 Flash attention on the GPU: one forward for every layer.
//!
//! A layer's `LayerRole::ratio` selects its kind. Ratio 0 attends over a
//! sliding window of its own fp8-rounded kv rows. Ratio > 0 attends over that
//! window plus compressed positions chosen by an indexer from the latent cache
//! the kv-source layers (`is_kv_source`) publish in `SharedV41`. The index
//! selection is computed by the index-source layers and reused by the ratio > 0
//! layers after them; the candidate-source layer also publishes a block-level
//! candidate mask that later index sources apply.
//!
//! This is the device form of `deepseek_v41_ref::compress::attention_any`, and
//! `attn_v41_tests.rs` compares it with the reference layer by layer:
//! * projections on `dense_gemm_bf16` / `dense_gemv_bf16` (bf16 weights) or the
//!   `Q2_K` K-quant kernels;
//! * RMSNorm, RoPE, the fp8/fp4 quantisers, the compressor pooling, the
//!   indexer scores, sparse attention with the sink and the `wo_a` column
//!   slices on `kernels/gb10/deepseek-v4-flash/nvfp4/attn_v41.cu`;
//! * the candidate-block and index top-k selections on the CPU through the
//!   reference's `torch_cpu_topk_set`, which reproduces torch's CPU tie order
//!   among equal scores; the scores come back in one stream-ordered download.
//!
//! Layer state (window ring, compressor partial group, the latent and
//! index-key caches) lives on the device; the shared index selection and
//! candidate mask in `SharedV41` are host vectors.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants:
//! - A ratio > 0 layer fails (does not attend) when no kv source has published
//!   a latent cache (`forward`, `decode_prep_role`).
//! - `forward` refuses a selection wider than `window + index_topk` or 2048,
//!   the `sparse_attn` score buffer.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// 2026-09-25: An attention projection on the device: bf16, or raw `Q2_K`
/// blocks. `Q3_K` is refused by `AttnV41::gemm`.
pub use metrale_model_layers::layers::ops::ResidentMat as AttnMat;

const MODULE: &str = "attn_v41";
const GEMM_MODULE: &str = "gemm";
/// 2026-09-25: Quantiser blocks per launch block: one warp per block, 8 warps
/// in the 256-thread launch block (`attn_v41_act_quant_fp8`, `attn_v41_fp4_quant`).
const QUANT_BLOCKS_PER_LAUNCH: u32 = 8;

/// 2026-09-25: Model-wide attention geometry, built from `ModelConfig` and the
/// `METRALE_DS41_*` limits by the DeepSeek-V4.1 weight loader.
#[derive(Clone, Debug)]
pub struct AttnV41Cfg {
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_rank: usize,
    pub o_rank: usize,
    pub groups: usize,
    pub window: usize,
    pub eps: f32,
    pub index_heads: usize,
    pub index_hd: usize,
    pub index_topk: usize,
    pub cand_topk_blocks: usize,
    pub cand_block: usize,
    pub max_seq: usize,
    pub max_tokens: usize,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub rope_factor: f32,
    pub orig_seq: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl AttnV41Cfg {
    pub fn gw(&self) -> usize {
        self.n_heads * self.head_dim / self.groups
    }
}

/// 2026-09-25: A layer's compress ratio and which shared slots it writes or reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerRole {
    pub ratio: usize,
    pub is_kv_source: bool,
    pub is_index_source: bool,
    pub is_candidate_source: bool,
    pub uses_candidates: bool,
}

/// 2026-09-25: Compressor weights (kv sources only). At ratio > 1 `kv` and
/// `gate` are f32 `[hd, dim]`; at ratio 1 `kv` is bf16 and `gate` is `None`.
/// `norm` is f32 `[hd]`.
pub struct CompressorWeightsGpu {
    pub kv: DevicePtr,
    pub gate: Option<DevicePtr>,
    pub norm: DevicePtr,
}

/// 2026-09-25: Indexer weights (index sources only): bf16 `wq_b`
/// `[index_heads * index_hd, q_rank]` and `weights_proj` `[index_heads, dim]`;
/// on a layer that is also a kv source, bf16 `wk` `[index_hd, hd]` and f32
/// `k_norm` `[index_hd]`.
pub struct IndexerWeightsGpu {
    pub wq_b: DevicePtr,
    pub weights_proj: DevicePtr,
    pub wk: Option<DevicePtr>,
    pub k_norm: Option<DevicePtr>,
}

/// 2026-09-25: One layer's resident attention weights. Projections are
/// `[out, in]`: `wo_a` is `[groups * o_rank, gw]`, `wo_b` `[dim, groups *
/// o_rank]`. `sink` (`[n_heads]`), `q_norm` (`[q_rank]`) and `kv_norm`
/// (`[hd]`) are f32.
pub struct AttnV41LayerWeights {
    pub role: LayerRole,
    pub sink: DevicePtr,
    pub wq_a: AttnMat,
    pub q_norm: DevicePtr,
    pub wq_b: AttnMat,
    pub wkv: AttnMat,
    pub kv_norm: DevicePtr,
    pub wo_a: AttnMat,
    pub wo_b: AttnMat,
    pub comp: Option<CompressorWeightsGpu>,
    pub idx: Option<IndexerWeightsGpu>,
}

/// 2026-09-25: One sequence's attention state for one layer, across prefill
/// and decode: the bf16 window ring `[window, hd]`; on kv sources also the
/// compressor's partial group (f32 kv and score, `[ratio, hd]` each, used at
/// ratio > 1), the bf16 latent cache `[max_seq / ratio, hd]` and the bf16
/// index-key cache `[max_seq / ratio, index_hd]`.
pub struct AttnV41LayerState {
    window: DevicePtr,
    comp_state: Option<(DevicePtr, DevicePtr)>,
    compress_kv: Option<DevicePtr>,
    index_k: Option<DevicePtr>,
}

impl AttnV41LayerState {
    pub fn new(gpu: &dyn GpuBackend, c: &AttnV41Cfg, role: LayerRole) -> Result<Self> {
        let window = gpu.alloc(c.window * c.head_dim * 2)?;
        gpu.memset(window, 0, c.window * c.head_dim * 2)?;
        let (comp_state, compress_kv, index_k) = if role.is_kv_source {
            let ratio = role.ratio.max(1);
            let groups = c.max_seq / ratio;
            let a = gpu.alloc(ratio * c.head_dim * 4)?;
            let b = gpu.alloc(ratio * c.head_dim * 4)?;
            gpu.memset(a, 0, ratio * c.head_dim * 4)?;
            // 2026-09-25: The score state starts at -inf, as
            // `CompressorState::new` in the reference.
            let neg: Vec<u8> =
                std::iter::repeat_n(f32::NEG_INFINITY.to_le_bytes(), ratio * c.head_dim)
                    .flatten()
                    .collect();
            gpu.copy_h2d(&neg, b)?;
            let ckv = gpu.alloc(groups.max(1) * c.head_dim * 2)?;
            gpu.memset(ckv, 0, groups.max(1) * c.head_dim * 2)?;
            let ik = gpu.alloc(groups.max(1) * c.index_hd * 2)?;
            gpu.memset(ik, 0, groups.max(1) * c.index_hd * 2)?;
            (Some((a, b)), Some(ckv), Some(ik))
        } else {
            (None, None, None)
        };
        Ok(AttnV41LayerState {
            window,
            comp_state,
            compress_kv,
            index_k,
        })
    }

    /// 2026-09-25: The window ring, bf16 `[window, hd]`: a pointer the captured
    /// decode steps bake.
    pub fn window(&self) -> DevicePtr {
        self.window
    }

    /// 2026-09-25: Free this sequence's device buffers (the window ring and,
    /// on kv sources, the two compressor-state buffers and both caches) and
    /// clear the pointers. A second call frees nothing: the window is NULL
    /// (`GpuBackend::free` ignores NULL) and the options are `None`.
    pub fn free(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let window = std::mem::replace(&mut self.window, DevicePtr::NULL);
        gpu.free(window)?;
        if let Some((a, b)) = self.comp_state.take() {
            gpu.free(a)?;
            gpu.free(b)?;
        }
        if let Some(p) = self.compress_kv.take() {
            gpu.free(p)?;
        }
        if let Some(p) = self.index_k.take() {
            gpu.free(p)?;
        }
        Ok(())
    }
}

/// 2026-09-25: What source layers publish for the layers after them: the
/// latest kv source's latent cache (`compress_kv`) and index-key cache
/// (`index_k`); the latest index source's selection `topk_idxs`
/// (`[queries][topk]`, offset past the window rows, -1 = absent); the
/// candidate source's mask `candidates` (`[queries][cand_width]`).
/// `compress_len` is the largest group count written so far and only grows.
#[derive(Default)]
pub struct SharedV41 {
    pub compress_kv: Option<DevicePtr>,
    pub compress_len: usize,
    pub index_k: Option<DevicePtr>,
    pub topk_idxs: Vec<i32>,
    pub topk: usize,
    pub candidates: Vec<bool>,
    pub cand_width: usize,
}

/// 2026-09-25: The intermediates of one layer's forward. `q` and `o` are bf16
/// `[tokens, n_heads, hd]` (`q` after RoPE, `o` before the inverse rotation);
/// `rows_a` are the window rows (the chunk's own rows at `start_pos == 0`,
/// the ring otherwise); `rows_b` the kv source's latent cache, of which
/// `rows_b_len` rows are attended; `idx` is `[tokens][topk]`, window slots
/// then compressed positions offset by `rows_a_len`; `out` is bf16 `[tokens, dim]`.
pub struct AttnV41Run {
    pub q: DevicePtr,
    pub rows_a: DevicePtr,
    pub rows_a_len: usize,
    pub rows_b: Option<DevicePtr>,
    pub rows_b_len: usize,
    pub idx: Vec<i32>,
    pub topk: usize,
    pub o: DevicePtr,
    pub out: DevicePtr,
}

struct Kernels {
    gemm: KernelHandle,
    // 2026-09-25: `gemv` serves bf16 projections at m = 1. For `AttnMat::Q2K`:
    // q8_1 row quantisation + MMVQ at m <= 8, D2S6 tile quantisation + MMQ
    // above; `mmvq_q2k_groups_w` does every `wo_a` output group in one launch
    // and `mmvq_q2k_pair_w` does `wq_a` and `wkv` in one launch (both m <= 8).
    gemv: KernelHandle,
    q8_rows: KernelHandle,
    mmvq_q2k_w: KernelHandle,
    mmvq_q2k_groups_w: KernelHandle,
    mmvq_q2k_pair_w: KernelHandle,
    quant_d2s6: KernelHandle,
    mmq_q2k_nc: KernelHandle,
    mmq_q2k_wc: KernelHandle,
    rmsnorm_bf16: KernelHandle,
    rmsnorm_f32: KernelHandle,
    rope: KernelHandle,
    act_quant: KernelHandle,
    fp4_quant: KernelHandle,
    gemm_f32: KernelHandle,
    // 2026-09-25: The compressor's f32 product at m = 1 when `dim % 8 == 0`;
    // `gemm_f32` otherwise.
    gemv_f32_staged: KernelHandle,
    pool: KernelHandle,
    index_score: KernelHandle,
    sparse_attn: KernelHandle,
    slice_cols: KernelHandle,
    scatter_cols: KernelHandle,
    scale_bf16: KernelHandle,
    // 2026-09-25: The captured decode step's window ring write; the slot is
    // `pos[0] % window`, read on the device.
    ring_put: KernelHandle,
}

/// 2026-09-25: The attention runtime: kernels, RoPE tables, and workspaces for
/// up to `max_tokens` positions per call.
pub struct AttnV41 {
    pub cfg: AttnV41Cfg,
    k: Kernels,
    fc_plain: DevicePtr,
    fc_yarn: DevicePtr,
    // 2026-09-25: q8_1 activations for the Q2_K projections, sized for the
    // larger of the m <= 8 row layout and the MMQ tile layout.
    a_q8: DevicePtr,
    qr_raw: DevicePtr,
    qr: DevicePtr,
    q: DevicePtr,
    kv_raw: DevicePtr,
    kv: DevicePtr,
    o: DevicePtr,
    o_rot: DevicePtr,
    og: DevicePtr,
    slice_in: DevicePtr,
    slice_out: DevicePtr,
    out: DevicePtr,
    pos: DevicePtr,
    head_pos: DevicePtr,
    idx_pos: DevicePtr,
    grp_pos: DevicePtr,
    idx_dev: DevicePtr,
    // 2026-09-25: The captured step's selection for ratio-0 layers (window
    // only); `idx_dev` holds the ratio > 0 selection, so one multi-layer
    // capture can read both classes.
    idx_dev_win: DevicePtr,
    ckv: DevicePtr,
    cscore: DevicePtr,
    pooled: DevicePtr,
    latent_raw: DevicePtr,
    latent: DevicePtr,
    ik_raw: DevicePtr,
    ik: DevicePtr,
    iq: DevicePtr,
    iw_raw: DevicePtr,
    iw: DevicePtr,
    score: DevicePtr,
    // 2026-09-25: What `decode_prep_role` last uploaded into `pos` /
    // `head_pos`, `idx_dev` and `idx_dev_win`, so a step re-uploads only what
    // changed. `None` = unknown, upload.
    decode_pos: Option<usize>,
    decode_idx: Option<Vec<i32>>,
    decode_idx_win: Option<Vec<i32>>,
}

fn upload_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len().max(4))?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

fn upload_i32(gpu: &dyn GpuBackend, dst: DevicePtr, v: &[i32]) -> Result<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.copy_h2d(&bytes, dst)
}

/// 2026-09-25: Enqueue an upload of `v` on `stream` (`GpuBackend::copy_h2d_async`,
/// so `v` may be dropped on return), for the captured decode step's per-token inputs.
fn upload_i32_async(gpu: &dyn GpuBackend, dst: DevicePtr, v: &[i32], stream: u64) -> Result<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.copy_h2d_async(&bytes, dst, stream)
}

fn at(p: DevicePtr, byte_off: usize) -> DevicePtr {
    DevicePtr(p.0 + byte_off as u64)
}

mod decode;
mod forward;
mod init;
mod primitives;
mod sources;

#[cfg(test)]
#[path = "attn_v41_tests.rs"]
mod tests;
