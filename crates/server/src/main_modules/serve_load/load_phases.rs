// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Steps of `load_model` around the weight load (phases 4 to 6): the
//! reserve preflight's inputs, the weight load with its checkpoint checks and
//! memory audit, and the KV kernel resolution for every per-layer KV dtype.
//!
//! Owner: server startup (`met serve`).
//! Invariants: the GPU calls run in `load_model`'s order; the `tracing` events
//! keep `load_model`'s target (`met::main_modules::serve_load`).

use std::path::Path;

use anyhow::{Context, Result};
use metrale_config::ModelConfig;

use crate::cli;
use crate::main_modules::serve_phases;

pub(super) fn reserve_preflight(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    free_mem: usize,
    model_dir: &Path,
    ptx_set: &metrale_kernels::TargetPtxSet,
) -> Result<serve_phases::ReservePreflight> {
    // 2026-09-26: Inputs the preflight's decode-ring auto-fit needs. The KV
    // dtype comes from `resolve_kv_dtype_str`, the resolver the cache uses
    // below; a value that does not parse is taken as BF16 here, and the
    // cache's own parse reports the error.
    let (preflight_kv_dtype_str, _) = serve_phases::kv_cache::resolve_kv_dtype_str(
        args.kv_cache_dtype.as_deref(),
        ptx_set.behavior.default_kv_dtype,
    );
    let post_load_inputs = serve_phases::PostLoadInputs {
        total_mem: gpu.total_memory().unwrap_or(0),
        model_dir,
        kv_dtype: preflight_kv_dtype_str
            .parse()
            .unwrap_or(metrale_cache::kv_cache::KvCacheDtype::Bf16),
        w8a8_prefill_kernels:
            metrale_model_layers::layers::qwen3_attention::w8a8_prefill_kernels_loaded(gpu),
    };
    serve_phases::preflight_reserve(args, config, free_mem, &post_load_inputs)
}

pub(super) fn load_weights(
    args: &cli::ServeArgs,
    config: &mut ModelConfig,
    model_dir: &Path,
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    ep_rank: usize,
    ep_size: usize,
    ptx_set: &metrale_kernels::TargetPtxSet,
    free_mem: usize,
    inference_reserve: usize,
    total_reserve: usize,
    gdn_two_phase_bytes: usize,
    max_batch_tokens_pre: usize,
) -> Result<metrale_model_weights::weights::WeightStore> {
    metrale_telemetry::progress::phase(5, "weight load");
    let oom_reserve_bytes = args.oom_guard_mb * 1024 * 1024;
    tracing::info!(target: "met::main_modules::serve_load", "OOM guard reserve: {} MB", args.oom_guard_mb);
    let store = serve_phases::load_weight_store(
        args,
        config,
        model_dir,
        gpu,
        ep_rank,
        ep_size,
        oom_reserve_bytes,
    )?;

    metrale_model_weights::weights::auto_detect_weight_prefix(&store, config);

    // 2026-09-26: Checkpoint and config consistency check, before
    // `init_nccl_comm`: a mismatch fails this rank before any collective setup.
    let (kv_dtype, _) = serve_phases::kv_cache::resolve_kv_dtype_str(
        args.kv_cache_dtype.as_deref(),
        ptx_set.behavior.default_kv_dtype,
    );
    metrale_model_weights::preflight::preflight(
        &store,
        config,
        args.speculative,
        // 2026-09-26: The resolved KV dtype (`resolve_kv_dtype_str`: the flag,
        // then MODEL.toml `default_kv_dtype`, then the engine default), the
        // same resolver the cache uses.
        Some(kv_dtype.as_str()),
    )
    .context("Checkpoint pre-flight check failed")?;

    // 2026-09-26: Logged so the quant-format decision shows in the serve log.
    let quant_format = metrale_model_layers::quant_format::detect_quant_format(config, &store);
    tracing::info!(target: "met::main_modules::serve_load", "Quantization format: {} (base variant {:?}), ignored globs = {}",
        quant_format.name(),
        quant_format.base_variant(),
        match &config.quantization_config {
            Some(qc) => qc.ignore_modules.len(),
            None => 0,
        },
    );

    // 2026-09-26: Pre-warm cuBLASLt so the first request does not pay its lazy
    // init. Measured 2026-08-22 on the 35B on GB10: without it the first
    // request was about 0.9 s slower than warm ones. A failure is logged and
    // does not fail the serve.
    #[cfg(feature = "cuda")]
    metrale_gpu_runtime::cublaslt::prewarm(0);

    serve_phases::post_load_memory_audit(
        args,
        config,
        gpu,
        store.total_bytes(),
        free_mem,
        inference_reserve,
        total_reserve,
        gdn_two_phase_bytes,
        max_batch_tokens_pre,
    )?;
    Ok(store)
}

pub(super) fn validate_kv_kernels(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    kv_dtype: metrale_cache::kv_cache::KvCacheDtype,
    layer_dtypes: &[metrale_cache::kv_cache::KvCacheDtype],
    config: &ModelConfig,
) -> Result<()> {
    // 2026-09-26: Resolve now, for each distinct per-layer KV dtype, every
    // kernel its dispatch needs, so a missing one fails the boot instead of a
    // request.
    let mut distinct: Vec<metrale_cache::kv_cache::KvCacheDtype> = vec![kv_dtype];
    for d in layer_dtypes {
        if !distinct.contains(d) {
            distinct.push(*d);
        }
    }
    for d in distinct {
        metrale_model_layers::layers::qwen3_attention::validate_required_kv_kernels(
            gpu,
            d,
            config.head_dim,
        )
        .context("kv-cache kernel preflight failed")?;
    }
    Ok(())
}
