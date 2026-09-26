// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The KV pool's block count for `build_model`, and the check of
//! how many full-length sequences that pool holds.
//!
//! Owner: metrale-model-engine.
//! Invariants:
//! - These functions read and log; none allocates device memory.

use anyhow::Result;
use metrale_cache::kv_cache::{KvCacheConfig, PagedKvCache};
use metrale_telemetry::prefix_cache::PrefixCache;

/// 2026-09-26: The block count under `--high-speed-swap`, sized from the
/// per-sequence cap `cap`.
pub(super) fn hss_kv_blocks(
    cap: u32,
    max_seq_len: usize,
    kv_block_size: usize,
    max_batch_size: usize,
) -> usize {
    // 2026-09-25: Per sequence `max(cap + 1, ceil(max_seq_len /
    // block_size))` blocks, so a prefill up to `max_seq_len` fits,
    // plus one dummy block for out-of-bounds-safe paged-kernel reads.
    let max_seq_blocks = max_seq_len.div_ceil(kv_block_size);
    let per_seq = (cap as usize + 1).max(max_seq_blocks);
    let n = max_batch_size * per_seq + 1;
    tracing::info!(target: "metrale_model_engine::factory::build", "--high-speed-swap: HBM cache sized to {n} blocks ({} batch × max(cap={cap}+1, max_seq_len_blocks={max_seq_blocks}) + 1 dummy); \
         prefill grows monotonically, decode shrinks to cap × bs and streams older blocks from disk via the orchestrator",
        max_batch_size
    );
    n
}

/// 2026-09-26: The block count sized from `kv_budget`. An error when the
/// budget is `0`.
pub(super) fn budget_kv_blocks(
    kv_config: &KvCacheConfig,
    kv_budget: usize,
    prefix_cache: &dyn PrefixCache,
    max_seq_len: usize,
    kv_block_size: usize,
    max_batch_size: usize,
    total_mem: usize,
    gpu_memory_utilization: f64,
    total_budget: usize,
    used_so_far: usize,
    inference_reserve: usize,
) -> Result<usize> {
    if kv_budget == 0 {
        anyhow::bail!(
            "No memory left for KV cache: total GPU = {:.1} GB, \
             --gpu-memory-utilization {:.0}% → budget {:.1} GB, \
             but {:.1} GB already consumed + {:.1} GB inference reserve \
             = {:.1} GB committed.  Raise --gpu-memory-utilization or \
             use a smaller model.",
            total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
            gpu_memory_utilization * 100.0,
            total_budget as f64 / (1024.0 * 1024.0 * 1024.0),
            used_so_far as f64 / (1024.0 * 1024.0 * 1024.0),
            inference_reserve as f64 / (1024.0 * 1024.0 * 1024.0),
            (used_so_far + inference_reserve) as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
    let budget_blocks = PagedKvCache::compute_num_blocks(kv_config, kv_budget)?;
    // 2026-09-25: `compute_num_blocks` spends the whole residual
    // budget, and the `max_concurrent` check below only warns or
    // refuses. With the prefix cache inactive, blocks beyond
    // `max_batch_size × blocks per sequence` can never be addressed, so
    // the pool is clamped to that plus one spare block per sequence and
    // the dummy block. An active prefix cache keeps the budget-sized
    // pool, as does setting `METRALE_KV_POOL_UNCLAMPED` (presence: `=0`
    // also counts).
    let n = if prefix_cache.is_active() || std::env::var("METRALE_KV_POOL_UNCLAMPED").is_ok() {
        budget_blocks
    } else {
        let per_seq = max_seq_len.div_ceil(kv_block_size);
        let reachable = max_batch_size
            .saturating_mul(per_seq)
            .saturating_add(max_batch_size)
            .saturating_add(1);
        let clamped = budget_blocks.min(reachable);
        if clamped < budget_blocks {
            let freed = (budget_blocks - clamped) * kv_config.block_bytes_kv_all_layers();
            tracing::info!(target: "metrale_model_engine::factory::build", "KV pool clamped to reachable demand: {} -> {} blocks \
                 ({} seq x {} blocks/seq + {} spare + 1 dummy); \
                 {:.2} GB not allocated (prefix caching inactive, so surplus \
                 blocks are unreachable). Restore with --enable-prefix-caching \
                 or METRALE_KV_POOL_UNCLAMPED.",
                budget_blocks,
                clamped,
                max_batch_size,
                per_seq,
                max_batch_size,
                freed as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        clamped
    };
    let max_kv_tokens = n * kv_block_size;
    tracing::info!(target: "metrale_model_engine::factory::build", "KV cache: {:.1} GB total × {:.0}% util = {:.1} GB budget; \
         {:.1} GB pre-KV + {:.1} GB reserve → {:.1} GB for KV \
         → {} blocks × {} tok/block = {} max KV tokens",
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        gpu_memory_utilization * 100.0,
        total_budget as f64 / (1024.0 * 1024.0 * 1024.0),
        used_so_far as f64 / (1024.0 * 1024.0 * 1024.0),
        inference_reserve as f64 / (1024.0 * 1024.0 * 1024.0),
        kv_budget as f64 / (1024.0 * 1024.0 * 1024.0),
        n,
        kv_block_size,
        max_kv_tokens,
    );
    Ok(n)
}

/// 2026-09-26: Warns, or with `METRALE_KV_OVERCOMMIT=0` or `=false` returns an
/// error, when the pool holds fewer than `max_batch_size` sequences.
pub(super) fn check_kv_concurrency(
    num_kv_blocks: usize,
    hss_cache_blocks_per_seq: Option<u32>,
    max_seq_len: usize,
    kv_block_size: usize,
    max_batch_size: usize,
) -> Result<()> {
    // 2026-09-25: With `--high-speed-swap` a sequence needs only its cap of
    // blocks resident, so the concurrency check uses the cap.
    let blocks_per_seq = match hss_cache_blocks_per_seq {
        Some(cap) => cap as usize,
        None => max_seq_len.div_ceil(kv_block_size),
    };
    let max_concurrent = num_kv_blocks / blocks_per_seq.max(1);
    if max_concurrent < max_batch_size {
        // 2026-09-25: A max_seq_len at which the requested batch size fits.
        let suggested_max_seq_len = (num_kv_blocks / max_batch_size.max(1)) * kv_block_size;
        // 2026-09-25: The check assumes every sequence reaches
        // `--max-seq-len`. Paged KV allocates blocks on demand, so by default
        // (overcommit) it only warns; `METRALE_KV_OVERCOMMIT=0` or `=false`
        // makes it refuse to start.
        let overcommit = !matches!(
            std::env::var("METRALE_KV_OVERCOMMIT").as_deref(),
            Ok("0") | Ok("false")
        );
        if overcommit {
            tracing::warn!(target: "metrale_model_engine::factory::build", "KV OVERCOMMIT: pool fits {} seq(s) at full --max-seq-len={} but \
                 --max-batch-size={} requested ({} block(s)/seq, {} block(s) total). \
                 Paged KV allocates on demand; long-context bursts are back-pressured \
                 at the block allocator, not refused at boot.",
                max_concurrent,
                max_seq_len,
                max_batch_size,
                blocks_per_seq,
                num_kv_blocks,
            );
        } else {
            anyhow::bail!(
                "KV cache can hold at most {} concurrent sequence(s) at --max-seq-len={}, \
                 but --max-batch-size={} was requested. \
                 KV pool has {} block(s) of {} tokens each; each sequence needs {} block(s). \
                 Try --max-seq-len {} (keeps max_batch_size={}), reduce --max-batch-size, \
                 or unset METRALE_KV_OVERCOMMIT=0 to allow on-demand paged allocation (default).",
                max_concurrent,
                max_seq_len,
                max_batch_size,
                num_kv_blocks,
                kv_block_size,
                blocks_per_seq,
                suggested_max_seq_len.max(kv_block_size),
                max_batch_size,
            );
        }
    }
    Ok(())
}
