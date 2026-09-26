// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Served NLLB-200 / M2M-100 encoder-decoder translation model.
//!
//! `NllbGpuModel` implements [`crate::traits::Model`], so an NLLB checkpoint is served
//! through the same scheduler and API as the decoder-only models, with its weights from
//! the standard `--model` store. The factory builds it for `model_type` `m2m_100` or
//! `nllb`.
//!
//! The model owns all its KV and does not use the scheduler's paged block cache: per
//! sequence, a decoder self-attention cache that grows one row per token and a
//! cross-attention cache computed once from the encoder (see `kv`). Logits are bf16.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

mod beam;
mod beam_compute;
mod beam_multi;
mod compute;
mod kernels;
mod kv;
mod lang;
mod lora;
mod model_impl;
mod util;

pub use lang::NllbLang;

use compute::DecScratch;
use kernels::NllbKernels;
use kv::NllbSeqKv;
use lora::NllbLora;

/// 2026-09-25: Floor on the decoder self-attention cache depth:
/// `cache_rows = max(DEFAULT_CACHE_ROWS, min(max_seq_len, 2048))`. One sequence's
/// self-attention KV is `cache_rows · d · 2 bytes · 2 (K and V) · dec_layers`.
const DEFAULT_CACHE_ROWS: usize = 512;

/// 2026-09-25: The served NLLB encoder-decoder model. Fields are immutable after
/// construction, behind a `Mutex`, or atomic. The decode scratch (`dec`) and the
/// LoRA gate (`lora_active`) are shared by every sequence, so callers must not
/// forward two sequences at once.
pub struct NllbGpuModel {
    gpu: Box<dyn GpuBackend>,
    kernels: NllbKernels,
    weights: HashMap<String, DevicePtr>,
    embed_table: DevicePtr,
    d: usize,
    heads: usize,
    head_dim: usize,
    ffn: usize,
    enc_layers: usize,
    dec_layers: usize,
    vocab: usize,
    embed_scale: f32,
    attn_scale: f32,
    cache_rows: usize,
    max_batch: usize,
    lang: NllbLang,
    dec: DecScratch,
    /// 2026-09-25: bf16 decode logits `[max_batch, vocab]`. `decode_batch` writes
    /// contiguous rows `0..n` (batch position `i` ↔ `seqs[i]`), the scheduler's contract.
    decode_logits: DevicePtr,
    /// 2026-09-25: bf16 prefill logits `[max_batch, vocab]`. A prefill writes row
    /// `slot_idx % max_batch`.
    prefill_logits: DevicePtr,
    /// 2026-09-25: Decoder sinusoidal position table `[cache_rows, d]` bf16.
    pos_table: DevicePtr,
    kv: Mutex<HashMap<usize, NllbSeqKv>>,
    slots: Mutex<SlotAlloc>,
    /// 2026-09-25: Optional PEFT LoRA adapter, applied after each biased linear
    /// projection it targets.
    lora: Option<NllbLora>,
    /// 2026-09-25: Per-request LoRA gate, set from the sequence's `adapter_slot`
    /// before each forward (`>= 0` applies the adapter).
    lora_active: AtomicBool,
}

/// 2026-09-25: Monotonic slot allocator with a free list for reuse.
#[derive(Default)]
struct SlotAlloc {
    next: usize,
    free: Vec<usize>,
}

impl SlotAlloc {
    fn claim(&mut self) -> usize {
        self.free.pop().unwrap_or_else(|| {
            let s = self.next;
            self.next += 1;
            s
        })
    }
    fn release(&mut self, slot: usize) {
        self.free.push(slot);
    }
}

impl NllbGpuModel {
    /// 2026-09-25: Build from the standard `--model` weight store and GPU
    /// backend. `lang` carries the language ids the server's tokenizer
    /// resolved. `max_seq_len` sets the decoder KV depth, clamped to
    /// `DEFAULT_CACHE_ROWS..=2048`. Fails unless `model.shared.weight` is bf16.
    pub fn new(
        config: &ModelConfig,
        store: &WeightStore,
        gpu: Box<dyn GpuBackend>,
        lang: NllbLang,
        max_seq_len: usize,
        max_batch: usize,
        lora_dir: Option<&std::path::Path>,
    ) -> Result<Self> {
        let d = config.hidden_size;
        let heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let ffn = config.intermediate_size;
        let dec_layers = config.num_hidden_layers;
        // 2026-09-25: The encoder uses the decoder's layer count
        // (`num_hidden_layers`), head count and FFN width.
        let enc_layers = dec_layers;
        let vocab = config.vocab_size;
        // 2026-09-25: Embeddings are always scaled by √d_model; the config's
        // `scale_embedding` is not read.
        let embed_scale = (d as f32).sqrt();
        let attn_scale = (head_dim as f32).powf(-0.5);
        let cache_rows = DEFAULT_CACHE_ROWS.max(max_seq_len.min(2048));

        ensure!(
            store.get("model.shared.weight")?.dtype == WeightDtype::BF16,
            "nllb serving requires a bf16 checkpoint; convert with \
             scripts/convert-safetensors-to-bf16.py"
        );
        let weights: HashMap<String, DevicePtr> = store
            .names()
            .map(|n| Ok((n.to_string(), store.get(n)?.ptr)))
            .collect::<Result<_>>()?;
        let embed_table = *weights
            .get("model.shared.weight")
            .context("nllb: missing tied embedding model.shared.weight")?;

        // 2026-09-25: Make the CUDA context current on this thread before the
        // kernel lookups and allocations below.
        gpu.bind_to_thread()?;
        let kernels = NllbKernels::new(gpu.as_ref())?;
        let dec = DecScratch::new(gpu.as_ref(), d, ffn, vocab)?;
        let max_batch = max_batch.max(1);
        let decode_logits = gpu.alloc(max_batch * vocab * 2)?;
        let prefill_logits = gpu.alloc(max_batch * vocab * 2)?;
        let pos_table = gpu.alloc(cache_rows * d * 2)?;
        let pos_host = util::decoder_pos_table_bf16(cache_rows, d);
        gpu.copy_h2d(util::bf16_bytes(&pos_host), pos_table)?;

        let lora = match lora_dir {
            Some(dir) => Some(NllbLora::load(dir, gpu.as_ref(), cache_rows)?),
            None => None,
        };

        tracing::info!(
            "NLLB served model ready: d={d} heads={heads} enc={enc_layers} dec={dec_layers} \
             vocab={vocab} src_lang_id={} tgt_lang_id={} cache_rows={cache_rows}",
            lang.src_lang_id,
            lang.tgt_lang_id,
        );

        Ok(Self {
            gpu,
            kernels,
            weights,
            embed_table,
            d,
            heads,
            head_dim,
            ffn,
            enc_layers,
            dec_layers,
            vocab,
            embed_scale,
            attn_scale,
            cache_rows,
            max_batch,
            lang,
            dec,
            decode_logits,
            prefill_logits,
            pos_table,
            kv: Mutex::new(HashMap::new()),
            slots: Mutex::new(SlotAlloc::default()),
            lora,
            lora_active: AtomicBool::new(false),
        })
    }

    /// 2026-09-25: Device pointer for weight `name`. Panics when the store has
    /// no such weight; construction copies every name in the store.
    #[inline]
    pub(super) fn w(&self, name: &str) -> DevicePtr {
        self.weights[name]
    }

    /// 2026-09-25: Arm or disarm the LoRA delta for the sequence about to be
    /// forwarded (`adapter_slot >= 0` arms it). Always disarmed when no adapter
    /// is loaded.
    #[inline]
    pub(super) fn set_lora_active(&self, adapter_slot: i32) {
        self.lora_active
            .store(self.lora.is_some() && adapter_slot >= 0, Ordering::Relaxed);
    }

    #[inline]
    pub(super) fn lora_is_active(&self) -> bool {
        self.lora_active.load(Ordering::Relaxed)
    }
}
