// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The PLE layer of `qwen4_exp`: `key_proj`, `value_proj`, three
//! norms and `conv1d` under `{lp}.ple`, the I64 id tables under
//! `ple_embedding`, and the n-gram table, whose shards stay deferred on disk
//! and are served through a segmented row cache.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - `load` returns a layer only for a layer that `ple_layer_ids` lists.
//! - It refuses a table whose shards differ in shape or dtype, whose id tables
//!   reach past the rows present, or whose dtype is FP8 E4M3.
//!
//! The shards `ngram_embedding.shard_{0,1,..}.weight` form one table of
//! `shards * rows_per` rows. Each is opened at its own file and byte offset,
//! so they need not be consecutive or in one file.

#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

#[cfg(feature = "cuda")]
use metrale_model_layers::layers::ngram_embed::NgramTable;
use metrale_model_layers::layers::ple::PleLayer;
#[cfg(feature = "cuda")]
use metrale_model_layers::layers::ple::{PleIdDims, PleWeights};
#[cfg(feature = "cuda")]
use metrale_model_layers::weight_map::dense;

#[cfg(feature = "cuda")]
fn slots_from_env() -> usize {
    std::env::var("METRALE_PLE_CACHE_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(65536)
}

/// 2026-09-25: Read an I64 device tensor back to the host; `PleIdDims` takes
/// the id tables as host vectors.
#[cfg(feature = "cuda")]
fn i64_host(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<Vec<u64>> {
    let t = store.get(name).with_context(|| format!("PLE: {name}"))?;
    let n = t.num_elements();
    let mut raw = vec![0u8; n * 8];
    gpu.copy_d2h(t.ptr, &mut raw)
        .with_context(|| format!("PLE: reading {name} back to host"))?;
    Ok(raw
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

/// 2026-09-25: Read a one-element BF16 or FP32 device tensor back to the host
/// as f32. Any other element count or dtype is an error.
#[cfg(feature = "cuda")]
fn f32_scalar(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<f32> {
    let t = store.get(name).with_context(|| format!("PLE: {name}"))?;
    anyhow::ensure!(
        t.num_elements() == 1,
        "PLE: {name} has {} elements; a table-wide scale is one",
        t.num_elements()
    );
    match t.dtype {
        metrale_model_weights::weights::WeightDtype::FP32 => {
            let mut raw = [0u8; 4];
            gpu.copy_d2h(t.ptr, &mut raw)?;
            Ok(f32::from_le_bytes(raw))
        }
        metrale_model_weights::weights::WeightDtype::BF16 => {
            let mut raw = [0u8; 2];
            gpu.copy_d2h(t.ptr, &mut raw)?;
            // 2026-09-25: BF16 is the top 16 bits of an f32.
            Ok(f32::from_bits(u32::from(u16::from_le_bytes(raw)) << 16))
        }
        other => anyhow::bail!("PLE: {name} is {other:?}; expected a BF16 or FP32 scalar"),
    }
}

/// 2026-09-25: The PLE layer for `layer_idx`, or `None` when `ple_layer_ids`
/// does not list it.
#[cfg(feature = "cuda")]
pub(super) fn load(
    store: &WeightStore,
    config: &ModelConfig,
    layer_idx: usize,
    max_tokens: usize,
    gpu: &dyn GpuBackend,
) -> Result<Option<PleLayer>> {
    if config.ple_layer_ids.is_empty() {
        return Ok(None);
    }
    // 2026-09-25: `ple_layer_ids` is 1-indexed: model layer `layer_idx` is
    // listed as `layer_idx + 1`.
    if !config.ple_layer_ids.contains(&(layer_idx + 1)) {
        return Ok(None);
    }
    let lp = format!("{}.ple", config.layer_prefix(layer_idx));
    let h = config.hidden_size;
    let hc = config.hc_mult;
    let eos = config.eos_token_id;

    let dims = PleIdDims {
        ngram_size: config.emb_neighbor_num,
        heads_per_ngram: config.emb_split_num,
        multipliers: i64_host(store, &format!("{lp}.ple_embedding.layer_multipliers"), gpu)?,
        head_vocab_sizes: i64_host(
            store,
            &format!("{lp}.ple_embedding.ngram_heads_vocab_sizes"),
            gpu,
        )?,
        head_offsets: i64_host(
            store,
            &format!("{lp}.ple_embedding.ngram_heads_offsets"),
            gpu,
        )?,
        eos_token_id: eos,
    };
    dims.validate().context("PLE: checkpoint id geometry")?;
    let heads = dims.ngram_heads();

    // 2026-09-25: `(file, byte offset)` per shard, walked from `shard_0` up to
    // the first missing index.
    let mut shards: Vec<(std::path::PathBuf, u64)> = Vec::new();
    let mut rows_per = 0usize;
    let mut head_dim = 0usize;
    let mut dtype = None;
    for i in 0.. {
        let name = format!("{lp}.ple_embedding.ngram_embedding.shard_{i}.weight");
        let Some(d) = store.deferred(&name) else {
            break;
        };
        anyhow::ensure!(
            d.shape.len() == 2,
            "PLE: shard {i} has shape {:?}, expected 2-D",
            d.shape
        );
        if i == 0 {
            rows_per = d.shape[0];
            head_dim = d.shape[1];
            dtype = Some(d.dtype);
        } else {
            anyhow::ensure!(
                Some(d.dtype) == dtype,
                "PLE: shard {i} is {:?} but shard 0 is {:?}; one row stride covers \
                 the whole table",
                d.dtype,
                dtype
            );
            // 2026-09-25: Equal row counts, because the cache maps a global id
            // to its shard with one divide.
            anyhow::ensure!(
                d.shape[0] == rows_per && d.shape[1] == head_dim,
                "PLE: shard {i} is {:?} but shard 0 is [{rows_per}, {head_dim}]. \
                 The segmented row cache maps a global id with one divide, which \
                 requires every shard to hold the same number of rows.",
                d.shape
            );
        }
        shards.push((d.path.clone(), d.offset));
    }
    anyhow::ensure!(
        !shards.is_empty(),
        "PLE: no `{lp}.ple_embedding.ngram_embedding.shard_*` was deferred. Either \
         the checkpoint has none, or they were UPLOADED whole — which for this \
         table is 102 GB of BF16 and would not have fit."
    );
    // 2026-09-25: A missing shard would end the walk early and shorten the
    // table, and the row cache's `resolve` would then fail a request for a row
    // past the end. The id tables give the highest row the hash can produce,
    // so the check is made here, at load.
    let rows_total = rows_per as u64 * shards.len() as u64;
    let highest_id = dims
        .head_offsets
        .iter()
        .zip(dims.head_vocab_sizes.iter())
        .map(|(off, vocab)| off + vocab)
        .max()
        .unwrap_or(0);
    anyhow::ensure!(
        highest_id <= rows_total,
        "PLE: the checkpoint's id tables reach row {highest_id}, but only \
         {rows_total} rows are present ({} shards x {rows_per}). Either a \
         `shard_*` tensor is missing from this checkpoint — the walk stops at the \
         first gap — or the id tables belong to a different conversion.",
        shards.len()
    );

    let dtype = dtype.context("PLE: no shard dtype")?;
    let elem = match dtype {
        metrale_model_weights::weights::WeightDtype::BF16 => 2,
        metrale_model_weights::weights::WeightDtype::FP8E4M3 => 1,
        other => anyhow::bail!(
            "PLE: n-gram table is {other:?}; the row cache reads raw rows and the \
             gather kernel has a path for BF16 and FP8 E4M3 only"
        ),
    };
    let slots = slots_from_env();
    let cache = metrale_storage::NgramRowCache::open_segmented(
        &shards,
        rows_per as u64,
        None,
        head_dim * elem,
        slots,
    )
    .context("PLE: n-gram row cache")?;

    // 2026-09-25: An FP8 table is refused, after its one-element
    // `ngram_embedding.weight_scale` is read and checked.
    if elem == 1 {
        let name = format!("{lp}.ple_embedding.ngram_embedding.weight_scale");
        let scale = f32_scalar(store, &name, gpu)
            .with_context(|| format!("PLE: FP8 table needs {name}"))?;
        anyhow::ensure!(
            scale.is_finite() && scale > 0.0,
            "PLE: n-gram weight_scale is {scale}, which cannot dequantize anything"
        );
        // 2026-09-25: `PleLayer` gathers with `embed_from_argmax`'s
        // `batched_embed` (`ple/layer.rs`), which reads BF16 rows and applies no
        // scale. The same kernel file has `batched_embed_fp8`, which `PleLayer`
        // does not use.
        anyhow::bail!(
            "PLE: this checkpoint stores the n-gram table in FP8 (1 byte/element, \
             scale {scale:.6e}), but the gather kernel wired to PleLayer reads BF16 \
             (2 bytes/element) and applies no scale -- loading it would produce \
             silently wrong output rather than an error. Use a BF16 conversion of \
             this model, or wire `batched_embed_fp8` into PleLayer first."
        );
    }

    let weights = PleWeights {
        key_proj: dense(store, &format!("{lp}.key_proj.weight"))?,
        value_proj: dense(store, &format!("{lp}.value_proj.weight"))?,
        norm_key: dense(store, &format!("{lp}.norm_key.weight"))?,
        norm_query: dense(store, &format!("{lp}.norm_query.weight"))?,
        norm_conv: dense(store, &format!("{lp}.norm_conv.weight"))?,
        conv1d: dense(store, &format!("{lp}.conv1d.weight"))?,
    };

    let dilation = config.emb_neighbor_num; // 2026-09-25: the n-gram size.
    tracing::info!(
        "PLE at MODEL LAYER {layer_idx} (ple_layer_ids={:?}, 1-indexed): \
         {} shards x {rows_per} rows x {head_dim} dims = {} rows ({:.1} GB BF16) \
         served off NVMe with {slots} cached slots ({:.1} MB); {heads} heads, \
         conv k={} dilation={dilation} (state {} steps)",
        config.ple_layer_ids,
        shards.len(),
        shards.len() * rows_per,
        (shards.len() * rows_per * head_dim * 2) as f64 / 1e9,
        (slots * head_dim * 2) as f64 / 1e6,
        config.ple_conv_kernel_size,
        (config.ple_conv_kernel_size - 1) * dilation,
    );

    PleLayer::new(
        dims,
        head_dim,
        h,
        hc,
        config.ple_conv_kernel_size,
        dilation,
        config.rms_norm_eps as f32,
        weights,
        NgramTable::Cached(Box::new(cache)),
        max_tokens,
        gpu,
    )
    .map(Some)
    .context("PLE: layer construction")
}

/// 2026-09-25: Without the `cuda` feature there is no row cache, so a model
/// with PLE is refused. `None` would mean the model has no PLE.
#[cfg(not(feature = "cuda"))]
pub(super) fn load(
    _store: &WeightStore,
    config: &ModelConfig,
    _layer_idx: usize,
    _max_tokens: usize,
    _gpu: &dyn GpuBackend,
) -> Result<Option<PleLayer>> {
    if config.ple_layer_ids.is_empty() {
        return Ok(None);
    }
    anyhow::bail!(
        "qwen4_exp PLE: this checkpoint has n-gram embeddings, but the row \
         cache that serves them needs the `cuda` feature; this build cannot \
         serve it"
    )
}
