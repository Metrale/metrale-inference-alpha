// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `build_model`: loads the weights with the loader for the
//! config's `model_type`, sizes and allocates the buffer arena and the KV
//! cache, and assembles a `TransformerModel` with its optional MTP, DFlash and
//! LoRA parts. NLLB / M2M-100 configs get an `NllbGpuModel` instead.
//!
//! Owner: metrale-model-engine.
//! Invariants:
//! - The LoRA adapters and the buffer arena are allocated before the KV budget
//!   is computed, so the budget is sized after them.

use anyhow::Result;
use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;
use metrale_telemetry::prefix_cache::PrefixCache;

use super::loader_for_config;
use super::m2_setup::maybe_run_minimax_m2_moe_transpose;
use super::{DflashBuildArgs, LoraBuildArgs};
use crate::model::TransformerModel;
use metrale_model_layers::layers::MtpQuantization;

mod kv_blocks;
mod kv_budget;
mod kv_sizing;
mod kv_summary;
mod mem_trace;
mod mtp_modules;
mod proposers;
pub mod slots;
mod weights_prep;

use mem_trace::MemTrace;

pub fn build_model(
    mut config: ModelConfig,
    // 2026-09-25: By value: the model adopts the store at the end, so its
    // teardown can free the weights. `mut` for `loader.prune_after_load`.
    mut store: WeightStore,
    gpu: Box<dyn GpuBackend>,
    max_batch_tokens: usize,
    kv_block_size: usize,
    max_seq_len: usize,
    // 2026-10-01: The slot request and the pre-load reserve as a function of the slot count;
    // the count is resolved at KV sizing (`kv_sizing.rs`) and returned with the model.
    slots: slots::SlotPlan<'_>,
    mtp_quant: MtpQuantization,
    use_speculative: bool,
    prefix_cache: Box<dyn PrefixCache>,
    mtp_vocab_size: u32,
    comm: Option<std::sync::Arc<dyn metrale_comm::CommBackend>>,
    self_speculative: bool,
    num_drafts: usize,
    kv_dtype: KvCacheDtype,
    gpu_memory_utilization: f64,
    ssm_cache_slots: usize,
    layer_dtypes: Vec<KvCacheDtype>,
    ssm_checkpoint_interval: usize,
    // 2026-09-25: Per-sequence HBM cache cap for `--high-speed-swap`; `None`
    // for no cap.
    hss_cache_blocks_per_seq: Option<u32>,
    // 2026-09-25: DFlash drafter; `None` for no DFlash.
    dflash_args: Option<DflashBuildArgs<'_>>,
    // 2026-09-25: LoRA adapters (`--lora-adapter`); `None` for the base model.
    lora_args: Option<LoraBuildArgs<'_>>,
    // 2026-09-25: NLLB / M2M-100 `(src_lang_id, tgt_lang_id)`, resolved by
    // the server. Required for those model types.
    nllb_lang: Option<(u32, u32)>,
    // 2026-09-25: NLLB / M2M-100 PEFT LoRA adapter directory; `None` for the
    // base model.
    nllb_lora_dir: Option<std::path::PathBuf>,
) -> Result<slots::BuiltModel> {
    // 2026-09-25: NLLB / M2M-100 is an encoder-decoder model, served by
    // `NllbGpuModel` from the same `store`. This returns before
    // `loader_for_config`, so the decoder-only loader never runs for it.
    #[cfg(feature = "cuda")]
    if matches!(config.model_type.as_str(), "m2m_100" | "nllb") {
        let (src, tgt) = nllb_lang.ok_or_else(|| {
            anyhow::anyhow!(
                "NLLB serving requires --src-lang and --tgt-lang (translation language pair)"
            )
        })?;
        let lang = crate::model::nllb::NllbLang {
            src_lang_id: src,
            tgt_lang_id: tgt,
            decoder_start_id: config.eos_token_id,
            eos_id: config.eos_token_id,
            pad_id: 1,
        };
        let nllb_slots = match slots.request {
            slots::SlotRequest::Count(n) => n,
            slots::SlotRequest::Auto => anyhow::bail!(
                "--max-batch-size auto balances recurrent-state slots against the KV pool; \
                 an encoder-decoder model has neither. Pass a count."
            ),
        };
        let model = crate::model::nllb::NllbGpuModel::new(
            &config,
            &store,
            gpu,
            lang,
            max_seq_len,
            nllb_slots,
            nllb_lora_dir.as_deref(),
        )?;
        return Ok(slots::BuiltModel {
            max_batch_size: nllb_slots,
            model: Box::new(model),
        });
    }
    #[cfg(not(feature = "cuda"))]
    let _ = (nllb_lang, nllb_lora_dir);

    // 2026-09-25: Step 1: select the weight loader.
    let loader = loader_for_config(&config)?;

    // 2026-09-25: Free memory before the LoRA load and the buffer arena. The
    // baseline-free-bytes path of the KV budget subtracts the budget-time
    // sample from it to get what `build_model` itself allocated.
    let build_entry_free = gpu.free_memory().ok();

    // 2026-09-25: LoRA adapters load before `BufferArena::new` and before the
    // free-memory sample the KV budget is computed from, so their pool is
    // counted as used. `config.adapter_max_rank` is set first because the
    // buffer arena sizes the LoRA scratch from it.
    let lora_weights: Option<metrale_model_layers::lora::LoraWeights> =
        if let Some(ref la) = lora_args {
            config.adapter_max_rank = la.max_lora_rank;
            loader.load_lora_adapters(
                &la.adapters,
                &config,
                gpu.as_ref(),
                la.max_loras,
                la.max_lora_rank,
            )?
        } else {
            None
        };

    // 2026-09-25: With DFlash, the target's capture layers are the drafter's
    // `dflash_config.target_layer_ids`, used as they are, set before
    // `TransformerModel::new` so it allocates the capture buffer.
    weights_prep::apply_dflash_capture_config(&mut config, &dflash_args);

    // 2026-09-25: Step 2: load the weights.
    let attn_layer_dtypes: Vec<KvCacheDtype> = if layer_dtypes.is_empty() {
        vec![kv_dtype; config.num_attention_layers()]
    } else {
        layer_dtypes.clone()
    };

    // 2026-09-25: Per-layer KV dims, from loaders whose layers differ in
    // attention geometry (Gemma-4). The default is empty: every layer uses the
    // global `num_kv_heads` and `head_dim`.
    config.kv_layer_dims = loader.kv_layer_dims(&config);

    // 2026-09-25: `MemTrace` logs the free-memory change at each build step;
    // the free_before/after pair adds a per-layer average for layer
    // construction.
    let mut mem = MemTrace::new(gpu.as_ref());
    let free_before_layers = gpu.free_memory().unwrap_or(0);
    let mut layers = loader.load_layers(&store, &config, gpu.as_ref(), &attn_layer_dtypes)?;
    let free_after_layers = gpu.free_memory().unwrap_or(0);
    tracing::info!(
        "Layer construction: {:.2} GB consumed ({:.2} GB free -> {:.2} GB free) \
         across {} layers, {:.1} MB/layer average",
        (free_before_layers.saturating_sub(free_after_layers)) as f64 / 1e9,
        free_before_layers as f64 / 1e9,
        free_after_layers as f64 / 1e9,
        config.num_hidden_layers,
        (free_before_layers.saturating_sub(free_after_layers)) as f64
            / 1e6
            / config.num_hidden_layers.max(1) as f64,
    );
    mem.mark("load_layers");
    let embed = loader.load_embedding(&store, &config, gpu.as_ref())?;
    mem.mark("load_embedding");
    // 2026-09-25: n-gram fused embedding: LongCat's loader builds one, the
    // default returns `None`. Sized for `max_batch_tokens`, the widest embed.
    let ngram_embed =
        loader.load_ngram_embedding(&store, &config, gpu.as_ref(), max_batch_tokens)?;
    let final_norm = loader.load_final_norm(&store, &config, gpu.as_ref())?;
    mem.mark("load_final_norm");
    let lm_head = loader.load_lm_head(&store, &config, gpu.as_ref())?;
    mem.mark("load_lm_head");
    let mtp_weights = loader.load_mtp_weights_multi(&store, &config, gpu.as_ref())?;
    mem.mark("load_mtp_weights_multi");

    // 2026-09-25: GLM-5.3 and DeepSeek-V4 have their own MTP modules, not the
    // generic `MtpWeights`; each is loaded only with `--speculative`, and its
    // proposer is built after the model (Steps 6b and 6c). The GLM-5.3 module
    // (`layers.{num_hidden_layers}`) is loaded on every EP rank; the
    // DeepSeek-V4 module only on rank 0.
    let glm_mtp_module =
        mtp_modules::load_glm_mtp_module(&store, &config, gpu.as_ref(), use_speculative);
    let glm_mtp_embed = embed;
    let glm_mtp_lm_head = lm_head;

    let v4_mtp_module = mtp_modules::load_v4_mtp_module(
        &store,
        &config,
        gpu.as_ref(),
        &attn_layer_dtypes,
        use_speculative,
    );

    // 2026-09-25: `--speculative` with no MTP module bound by any of the three
    // paths (generic, GLM-5.3, DeepSeek-V4). `detect_in_store` then tells a
    // checkpoint without an MTP head (warn) from one whose head no loader bound
    // (error).
    if use_speculative
        && mtp_weights.is_empty()
        && glm_mtp_module.is_none()
        && v4_mtp_module.is_none()
    {
        mtp_modules::warn_unbound_mtp_head(&store, &config);
    }
    mem.mark("mtp modules (glm/v4)");
    let vision_encoder = loader.load_vision_encoder(&store, &config, gpu.as_ref())?;
    mem.mark("load_vision_encoder");

    // 2026-09-25: When no vision encoder was bound, free the store's
    // vision-tower tensors before the KV budget is computed. The decision keys
    // off the bind result, not a model list, so an encoder that binds these
    // pointers keeps them.
    if vision_encoder.is_none() {
        weights_prep::release_unbound_vision_tower(&mut store, gpu.as_ref(), &config)?;
    }
    mem.mark("vision reclaim");

    // 2026-09-25: When the checkpoint's quantization config ignores the MTP
    // weights (`mtp.fc.weight` or layer 0's q_proj), they stay BF16 whatever
    // `--mtp-quantization` says.
    let effective_mtp_quant =
        mtp_modules::effective_mtp_quantization(&mtp_weights, &config, &store, mtp_quant)?;

    // 2026-09-25: Step 3: LM-head quantization (NVFP4, FP8 or BF16) and the
    // draft-only NVFP4 head for MTP, in lm_head_setup.rs.
    let (lm_head_nvfp4, lm_head_fp8, mtp_lm_head_nvfp4) = super::lm_head_setup::setup_lm_heads(
        &store,
        &lm_head,
        &config,
        gpu.as_ref(),
        use_speculative,
        !mtp_weights.is_empty(),
    )?;

    // 2026-10-01: Step 3a: the transposed twin of the NVFP4 LM head, built here rather than in
    // `TransformerModel::new` so the KV sizing finds it on the allocation ledger.
    let lm_head_nvfp4_t =
        crate::model::impl_a1::build_lm_head_nvfp4_t(&lm_head_nvfp4, &config, gpu.as_ref())?;

    // 2026-09-25: Copies (DenseWeight is `Copy`) of the embedding and the BF16
    // LM head for the DeepSeek-V4 MTP proposer, taken before both move into
    // `TransformerModel::new`.
    let v4_mtp_embed = embed;
    let v4_mtp_lm_head = lm_head;

    // 2026-09-25: Step 3b: the post-load MoE prefill transpose for minimax_m2,
    // step3p7 and deepseek_v4 (m2_setup.rs); a no-op for other model types.
    maybe_run_minimax_m2_moe_transpose(&config, gpu.as_ref(), &mut layers)?;

    // 2026-09-25: Step 3c: the loader drops store tensors it has uploaded its
    // own copies of (the default is a no-op). This runs after every `load_*`
    // above and before the KV budget is computed, so the freed memory counts.
    loader.prune_after_load(&mut store, &config, gpu.as_ref())?;
    mem.mark("prune_after_load");
    tracing::info!(
        "WeightStore after prune: {} tensors, {:.3} GiB still resident",
        store.len(),
        store.resident_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    // 2026-09-25: Step 4: the buffer arena. 2026-10-01: Sized for the slot ceiling (`auto` resolves
    // after it, at KV sizing); the preflight sized the arena reserve for the same ceiling.
    let buffers = BufferArena::new(
        &config,
        max_batch_tokens,
        max_seq_len,
        kv_block_size,
        slots.request.ceiling(),
        gpu.as_ref(),
    )?;

    // 2026-09-25: Step 5: size the KV cache. With MLA (`kv_lora_rank > 0`) it
    // caches one head of `kv_lora_rank + qk_rope_head_dim` dims per token, the
    // compressed latent, not the expanded heads.
    let (kv_num_heads, kv_head_dim) = if config.kv_lora_rank > 0 {
        let mla_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim;
        tracing::info!(
            "MLA absorbed KV cache: 1 head × {} dims ({}+{}) per token (vs {} heads × {})",
            mla_cache_dim,
            config.kv_lora_rank,
            config.qk_rope_head_dim,
            config.num_key_value_heads,
            config.head_dim,
        );
        (1, mla_cache_dim)
    } else {
        (config.num_key_value_heads, config.head_dim)
    };
    let kv_config = KvCacheConfig {
        block_size: kv_block_size,
        num_kv_heads: kv_num_heads,
        head_dim: kv_head_dim,
        num_layers: config.num_attention_layers(),
        dtype: kv_dtype,
        layer_dtypes: layer_dtypes.clone(),
        layer_dims: config.kv_layer_dims.clone(),
        cache_blocks_per_seq: hss_cache_blocks_per_seq,
    };

    if hss_cache_blocks_per_seq.is_some() {
        kv_summary::log_hss_kv_summary(&kv_config);
    }
    // 2026-09-25: `total_memory × gpu_memory_utilization` is the budget for
    // everything this process holds. The KV cache gets what is left of it
    // after `used_so_far`, `inference_reserve` and the DFlash and MTP
    // reserves, clamped to the free memory left after those reserves.
    // 2026-10-01: Sized with the slot count (`kv_sizing.rs`).
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    // 2026-09-25: The DFlash drafter head allocates at Step 7, after this
    // sizing, so its estimate is reserved here: drafter KV
    // (max_seq_len × layers × 2 × kv_dim × BF16), fused KV, the hidden-state
    // capture, FP8 weight mirrors (half the drafter store) and 300 MiB of
    // scratch.
    let dflash_reserve: usize =
        kv_budget::dflash_reserve_bytes(&dflash_args, &config, max_seq_len, gib);
    let sized = kv_sizing::size_kv(
        &kv_sizing::KvInputs {
            gpu: gpu.as_ref(),
            store: &store,
            build_entry_free,
            config: &config,
            kv_config: &kv_config,
            prefix_cache: prefix_cache.as_ref(),
            dflash_reserve,
            use_speculative,
            mtp_weights: &mtp_weights,
            effective_mtp_quant,
            max_seq_len,
            kv_block_size,
            hss_cache_blocks_per_seq,
            gpu_memory_utilization,
        },
        &slots,
    )?;
    let (max_batch_size, num_kv_blocks) = (sized.slots, sized.num_kv_blocks);
    let total_mem = gpu.total_memory()?;
    let total_budget = (total_mem as f64 * gpu_memory_utilization) as usize;
    let kv_cache = PagedKvCache::new(kv_config, num_kv_blocks, gpu.as_ref())?;

    // 2026-09-25: Step 6: assemble the model. The DFlash drafter shares the
    // target's embedding and LM head, so their pointers are copied first.
    let target_embed_for_dflash = embed.weight;
    let target_lm_head_for_dflash = lm_head.weight;
    // 2026-09-25: The NVFP4 LM head, shared with the DFlash drafter.
    let target_lm_head_nvfp4_for_dflash = lm_head_nvfp4;
    let target_hidden_for_dflash = config.hidden_size;
    // 2026-09-25: For the DFlash drafter, the checkpoint's own FP8 LM head if
    // it ships one (`lm_head_setup::native_fp8_lm_head_share`).
    let target_lm_head_native_fp8_for_dflash = if dflash_args.is_some() {
        super::lm_head_setup::native_fp8_lm_head_share(&store, &config, gpu.as_ref())?
    } else {
        None
    };

    let mut model = TransformerModel::new(
        config,
        embed,
        final_norm,
        lm_head,
        lm_head_nvfp4,
        lm_head_fp8,
        mtp_lm_head_nvfp4,
        lm_head_nvfp4_t,
        layers,
        buffers,
        kv_cache,
        mtp_weights,
        gpu,
        max_seq_len,
        max_batch_size,
        effective_mtp_quant,
        use_speculative,
        prefix_cache,
        mtp_vocab_size,
        comm,
        self_speculative,
        num_drafts,
        vision_encoder,
        ssm_cache_slots,
        ssm_checkpoint_interval,
    )?;

    // 2026-09-25: A Q6_K `lm_head.weight` in the store (a DeepSeek-V4.1 GGUF,
    // kept Q6_K by the sidecar's `v41_kquant_resident_dtype`) is served by the
    // K-quant GEMV in model/lm_head_q6k.rs.
    if store
        .get("lm_head.weight")
        .is_ok_and(|w| w.dtype == metrale_model_weights::weights::WeightDtype::Q6K)
    {
        model.set_lm_head_q6k(lm_head.weight)?;
    }
    // 2026-09-28: Declared-W8A8 FP8 head (model/lm_head_fp8_rows.rs).
    model.install_declared_lm_head_w8a8()?;
    // 2026-10-02: Declared-W4A16 NVFP4 head on the row tiles (model/lm_head_nvfp4_rows.rs).
    model.install_declared_lm_head_w4a16_rows()?;

    // 2026-09-25: Step 6b: the DeepSeek-V4 MTP proposer, built after `new()`
    // because it needs the model's GPU backend. `--dflash` conflicts with
    // `--speculative` in the CLI, so Step 7 does not also install one.
    proposers::install_v4_mtp_proposer(
        &mut model,
        v4_mtp_module,
        v4_mtp_embed,
        v4_mtp_lm_head,
        mtp_vocab_size,
        max_seq_len,
    );

    // 2026-09-25: Step 6c: the GLM-5.3 MTP proposer, built after `new()` for
    // the same reason.
    proposers::install_glm_mtp_proposer(
        &mut model,
        glm_mtp_module,
        glm_mtp_embed,
        glm_mtp_lm_head,
        max_seq_len,
    );

    // 2026-09-25: Step 7: the DFlash drafter, which uses the target's
    // embedding and LM head.
    proposers::install_dflash_drafter(
        &mut model,
        dflash_args,
        target_embed_for_dflash,
        target_lm_head_for_dflash,
        target_lm_head_nvfp4_for_dflash,
        target_lm_head_native_fp8_for_dflash,
        target_hidden_for_dflash,
        max_seq_len,
        max_batch_size,
    )?;

    // 2026-09-25: Step 8: the n-gram embedding, and the LoRA adapters loaded
    // before KV sizing, into the model's layers.
    if let Some(ngram) = ngram_embed {
        model.set_ngram_embedding(ngram);
    }
    model.set_lora_weights(lora_weights)?;

    // 2026-09-25: The model adopts the store, so its teardown frees the
    // weights (last, after everything that reads them).
    model.adopt_weight_store(store);

    proposers::log_alloc_report(&model, total_budget, gpu_memory_utilization, total_mem, gib);
    Ok(slots::BuiltModel {
        model: Box::new(model),
        max_batch_size,
    })
}
