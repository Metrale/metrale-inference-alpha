// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Builds the LongCat n-gram embedding from the lookup tables the weight loader
//! deferred.
//!
//! Owner: model-arch weight loader (LongCat).
//! Invariants:
//! - Every table `model.ngram_embeddings.embedders.{i}.weight` must be deferred (recorded
//!   with its shard path and offset, never uploaded); a missing or uploaded table is an error.
//! - Each table is served by an `NgramTable::Cached` row cache of `slots` BF16 rows read from
//!   its shard, and its row count and width are checked against `NgramDims` first.

#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

#[cfg(feature = "cuda")]
use metrale_model_layers::layers::ngram_embed::NgramTable;
use metrale_model_layers::layers::ngram_embed::{NgramDims, NgramEmbedding};
#[cfg(feature = "cuda")]
use metrale_model_layers::weight_map::dense;

/// 2026-09-25: Cached rows per table, unless `METRALE_NGRAM_CACHE_SLOTS` holds a
/// positive integer.
#[cfg(feature = "cuda")]
const DEFAULT_SLOTS: usize = 65536;

#[cfg(feature = "cuda")]
fn slots_from_env() -> usize {
    std::env::var("METRALE_NGRAM_CACHE_SLOTS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_SLOTS)
}

/// 2026-09-25: Build the n-gram embedding, or `None` when
/// `METRALE_NGRAM_DISABLE` is set or the config has no n-gram geometry
/// (`NgramDims::from_config`).
#[cfg(feature = "cuda")]
pub(super) fn build(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    max_tokens: usize,
) -> Result<Option<NgramEmbedding>> {
    // 2026-09-25: Diagnostic lever: `METRALE_NGRAM_DISABLE` with any value
    // serves the plain `embed_tokens` gather instead of the fused embedding,
    // which gives wrong output.
    if std::env::var("METRALE_NGRAM_DISABLE").is_ok() {
        tracing::warn!(
            "METRALE_NGRAM_DISABLE set — n-gram embedding NOT installed;              output will be incorrect. Diagnostic use only."
        );
        return Ok(None);
    }
    let Some(dims) = NgramDims::from_config(config) else {
        return Ok(None);
    };
    let n_tables = dims.num_tables();
    let slots = slots_from_env();

    let word = dense(store, "model.embed_tokens.weight").context("ngram: base embedding")?;

    let mut tables = Vec::with_capacity(n_tables);
    let mut projs = Vec::with_capacity(n_tables);
    for i in 0..n_tables {
        let tname = format!("model.ngram_embeddings.embedders.{i}.weight");
        let d = store.deferred(&tname).ok_or_else(|| {
            anyhow::anyhow!(
                "ngram: table {tname} was not deferred by the loader — it is either \
                 missing from the checkpoint or was uploaded whole (62.8 GB of BF16)"
            )
        })?;
        anyhow::ensure!(
            d.shape.len() == 2,
            "ngram: table {tname} has shape {:?}, expected 2-D",
            d.shape
        );
        let rows_total = d.shape[0] as u64;
        let expected = dims.table_rows(i);
        anyhow::ensure!(
            rows_total == expected,
            "ngram: table {i} has {rows_total} rows but the config's hash geometry \
             wants {expected} (ratio*vocab + 2*i + 1) — the id space and the table \
             would disagree and every lookup would be wrong"
        );
        anyhow::ensure!(
            d.shape[1] == dims.table_dim(),
            "ngram: table {i} dim {} != hidden/num_tables {}",
            d.shape[1],
            dims.table_dim()
        );
        // 2026-09-25: BF16 rows, no per-row scale file.
        let row_stride = d.shape[1] * 2;
        let cache = metrale_storage::NgramRowCache::open_at(
            &d.path, d.offset, None, rows_total, row_stride, slots,
        )
        .with_context(|| format!("ngram: row cache for table {i}"))?;
        tables.push(NgramTable::Cached(Box::new(cache)));

        let pname = format!("model.ngram_embeddings.post_projs.{i}.weight");
        projs.push(dense(store, &pname).with_context(|| format!("ngram: proj {i}"))?);
    }

    tracing::info!(
        "LongCat n-gram embedding: {n_tables} tables x {} rows x {} dims, NVMe row cache \
         ({slots} slots = {:.0} MB total, vs {:.1} GB BF16-resident); fusion = \
         (base + sum proj_i(table_i)) / {}",
        dims.table_rows(0),
        dims.table_dim(),
        (n_tables * slots * dims.table_dim() * 2) as f64 / 1e6,
        (n_tables as f64 * dims.table_rows(0) as f64 * dims.table_dim() as f64 * 2.0) / 1e9,
        n_tables + 1,
    );

    Ok(Some(NgramEmbedding::new(
        dims, word, tables, projs, max_tokens, gpu,
    )?))
}

/// 2026-09-25: Without the `cuda` feature there is no row cache, so a config
/// with n-gram geometry is an error. `None` would mean the architecture has no
/// n-gram embedding and serve the plain gather.
#[cfg(not(feature = "cuda"))]
pub(super) fn build(
    _store: &WeightStore,
    config: &ModelConfig,
    _gpu: &dyn GpuBackend,
    _max_tokens: usize,
) -> Result<Option<NgramEmbedding>> {
    if NgramDims::from_config(config).is_none() {
        return Ok(None);
    }
    anyhow::bail!(
        "ngram: this checkpoint has n-gram embeddings, but the row cache that \
         serves them needs the `cuda` feature; this build cannot serve it"
    )
}
