// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The model-building half of startup (`load_engine`): resolve, configure, load and
//! build one checkpoint, apply `--forward`, run the kernel audit, and hand back the model with
//! what the scheduler half of `load_model` reads. `met circuit diff` calls it alone, to drive a
//! built model without a scheduler.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - The steps and their order are `load_model`'s; this file only ends where the scheduler
//!   half begins.

use anyhow::Result;

use super::{act_quant_support, adapters, load_phases, model_setup};
use crate::cli;
use crate::main_modules::serve_phases;

/// 2026-09-28: A built model and the startup values the scheduler half reads.
pub(crate) struct Engine {
    pub args: cli::ServeArgs,
    pub model: Box<dyn metrale_model_engine::traits::Model>,
    pub config: metrale_config::ModelConfig,
    pub config_json: String,
    pub model_dir: std::path::PathBuf,
    pub ptx_set: metrale_kernels::TargetPtxSet,
    pub sampling_presets: metrale_kernels::SamplingPresets,
    pub vision_max_pixels: Option<usize>,
    pub remote_image_policy: crate::api::chat::remote_image::RemoteImagePolicy,
    pub video_ffmpeg: metrale_model_layers::video_decode_ffmpeg::FfmpegPolicy,
    pub world_size: usize,
    pub prefill_budget: usize,
    pub max_batch_tokens: usize,
    pub lora_states: Vec<serve_phases::LoraAdapterState>,
    pub nllb_adapter_name: Option<String>,
    pub early_high_speed_swap_cfg: Option<metrale_storage::HighSpeedSwapConfig>,
    pub forward: metrale_model_engine::traits::ForwardDisclosure,
    /// 2026-10-01: The slot count `--max-batch-size auto` resolved to; `None` for a count.
    pub auto_max_batch_size: Option<usize>,
    /// 2026-10-01: The memory budget and ledger reader `GET /memory` reports.
    pub device_budget: crate::main_modules::memory_probe::DeviceBudget,
}

/// 2026-09-28: Build the model `args` names. `Ok(None)` means this rank is an EP worker: it ran
/// its command loop and has nothing to serve.
pub(crate) fn load_engine(mut args: cli::ServeArgs) -> Result<Option<Engine>> {
    metrale_telemetry::progress::phase(1, "model resolve");
    let model_dir = serve_phases::resolve_model_dir(&args)?;

    tracing::info!("Port: {}", args.port);

    tracing::info!(
        "SSM decode h-state dtype: {} (--ssm-h-dtype)",
        if metrale_model_layers::layers::qwen3_ssm::ssm_h_f16_pool_enabled() {
            "f16 + f16-sized pools (stage 3)"
        } else if metrale_model_layers::layers::qwen3_ssm::ssm_h_fp16_enabled() {
            "f16"
        } else {
            "f32 (full precision)"
        }
    );

    let (mut config, config_json) = model_setup::configure_model(&args, &model_dir)?;

    let (vision_max_pixels, remote_image_policy, video_ffmpeg) =
        model_setup::resolve_media_policies(&args, &model_dir, &mut config)?;

    model_setup::log_model_config(&config);

    let ptx_set = model_setup::select_kernel_target(&args, &config, &model_dir)?;
    let sampling_presets = ptx_set.sampling;
    model_setup::check_kernel_target(&ptx_set, &mut config)?;
    model_setup::publish_row_tiers(&args, &config);
    act_quant_support::check(&config, &args.lm_head_dtype)?;
    model_setup::publish_moe_expert_act(&config);

    // 2026-09-26: After this call `args.num_drafts` is `Some`, so
    // `args.resolved_num_drafts()` is valid.
    serve_phases::apply_model_default_num_drafts(&mut args, &ptx_set);
    serve_phases::publish_mtp_max_seqs(&args, &ptx_set)?;
    serve_phases::publish_copy_tier(&args)?;
    // 2026-09-30: One γ for the reserve, the pools and the scheduler.
    serve_phases::apply_dflash_gamma(&mut args, serve_phases::model_default_drafter(&ptx_set))?;

    // 2026-10-01: Host memory before the first backend exists: the baseline `GET /memory`
    // measures the serve's device footprint from.
    crate::main_modules::memory_probe::record_mem_available_at_start();
    let (gpu, free_mem, device_budget) = serve_phases::init_gpu_backend(&args, &ptx_set)?;

    // 2026-09-26: Topology runs before `preflight_reserve`: `resolve_topology`
    // divides the attention and linear-attention head counts by `tp_size`
    // (with a divisibility check), and the reserve is sized from those fields.
    // At `--tp-size 1` it divides nothing.
    metrale_telemetry::progress::phase(4, "topology");
    let serve_phases::Topology {
        world_size,
        tp_size: _tp_size,
        ep_size,
        tp_rank: _tp_rank,
        ep_rank,
    } = serve_phases::resolve_topology(&args, &mut config)?;

    let serve_phases::ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        ssm_prefill_chunk,
        max_batch_tokens_pre,
        plan: reserve_plan,
    } = load_phases::reserve_preflight(
        &args,
        &config,
        gpu.as_ref(),
        free_mem,
        &model_dir,
        &ptx_set,
    )?;
    let total_reserve = inference_reserve + buffer_arena_bytes;

    // 2026-09-26: OOM watchdog (CUDA builds): every 2 s it reads free device
    // memory and, after three consecutive readings below 2048 MiB, exits the
    // process with status 1. Spawned once per process.
    #[cfg(feature = "cuda")]
    let _oom_watchdog = metrale_gpu_runtime::cuda_backend::spawn_oom_watchdog(
        2048,
        std::time::Duration::from_secs(2),
    );
    #[cfg(feature = "cuda")]
    tracing::info!("OOM watchdog started (threshold: 2 GB, interval: 2s)");

    // 2026-09-26: An explicit `--fp8-kv-calibration-tokens` wins, including 0,
    // which turns calibration off for a model whose MODEL.toml enables it.
    // Omitted: MODEL.toml `[behavior].fp8_kv_calibration_tokens`, 0 when absent.
    config.fp8_kv_calibration_tokens = args
        .fp8_kv_calibration_tokens
        .unwrap_or(ptx_set.behavior.fp8_kv_calibration_tokens);
    // 2026-09-26: Set on every load: config parsing leaves it at 0.0
    // (`#[serde(skip)]`), and `validate_serve_args` has checked the flag is at
    // least 1.0.
    config.fp8_kv_headroom = args.fp8_kv_headroom;

    let store = load_phases::load_weights(
        &args,
        &mut config,
        &model_dir,
        gpu.as_ref(),
        ep_rank,
        ep_size,
        &ptx_set,
        free_mem,
        inference_reserve,
        total_reserve,
        gdn_two_phase_bytes,
        max_batch_tokens_pre,
    )?;

    metrale_telemetry::progress::phase(6, "kv cache");
    let serve_phases::PrefillBudget {
        prefill_budget,
        max_batch_tokens,
        spec_tokens: _spec_tokens,
    } = serve_phases::resolve_prefill_budget(&args, ssm_prefill_chunk);
    let prefix_cache = serve_phases::build_prefix_cache(&args, &config);
    let comm = serve_phases::init_nccl_comm(
        &args,
        gpu.as_ref(),
        world_size,
        max_batch_tokens,
        config.hidden_size,
        config.vocab_size,
    )?;
    config.profile = args.profile;
    serve_phases::cap_vocab_size_to_tokenizer(&model_dir, &mut config);
    let serve_phases::KvCacheConfig {
        effective_kv_dtype_str: _,
        kv_dtype,
        layer_dtypes,
        hss_cache_blocks_per_seq,
    } = serve_phases::resolve_kv_cache_config(
        &args,
        &config,
        ptx_set.behavior.default_kv_dtype,
        &store.kv_scale_census()?,
    )?;

    load_phases::validate_kv_kernels(gpu.as_ref(), kv_dtype, &layer_dtypes, &config)?;
    let dflash_drafter_state = serve_phases::load_dflash_drafter(&args, &ptx_set, gpu.as_ref())?;
    // 2026-09-26: LoRA adapters load before `gpu` moves into `build_model`.
    // `lora_states` outlives it: `lora_args` borrows each `store`, and
    // `AppState` takes the adapter names. NLLB loads its adapter through its
    // own path (`nllb_lora_dir` below).
    let is_nllb = matches!(config.model_type.as_str(), "m2m_100" | "nllb");
    let lora_states = if is_nllb {
        Vec::new()
    } else {
        serve_phases::load_lora_adapters(&args, gpu.as_ref())?
    };
    if !lora_states.is_empty() && world_size > 1 {
        anyhow::bail!(
            "--lora-adapter requires world_size=1 in v0 (got {world_size}); \
             TP adapter sharding is M3"
        );
    }
    let lora_args = adapters::lora_build_args(&args, &lora_states);
    let dflash_args = adapters::dflash_build_args(&args, &dflash_drafter_state);
    let nllb_lang = adapters::resolve_nllb_lang(&args, &config, &model_dir)?;
    let (nllb_lora_dir, nllb_adapter_name) = adapters::resolve_nllb_adapter(&args, is_nllb)?;
    let built = serve_phases::build_model(
        &args,
        &config,
        // 2026-09-26: Moved: the model owns the weight store from here.
        store,
        gpu,
        max_batch_tokens,
        kv_dtype,
        &reserve_plan,
        layer_dtypes,
        hss_cache_blocks_per_seq,
        prefix_cache,
        comm,
        dflash_args,
        lora_args,
        nllb_lang,
        nllb_lora_dir,
    )?;
    // 2026-10-01: From here `--max-batch-size` is the count the model was built with (what `auto`
    // resolved to), for the scheduler, the EP worker and the disclosure.
    let auto_max_batch_size = (args.max_batch_size
        == metrale_model_engine::factory::SlotRequest::Auto)
        .then_some(built.max_batch_size);
    args.max_batch_size = metrale_model_engine::factory::SlotRequest::Count(built.max_batch_size);
    let model = built.model;

    // 2026-09-28: `--forward`, applied before the audit so the gate sees the executor's lookups.
    model.set_forward(&serve_phases::forward_select(
        &args,
        &ptx_set,
        &config_json,
    )?)?;
    let forward = model.forward_disclosure();

    // 2026-09-26: Kernel load audit and the boot gate (`kernel_gate`). Under
    // `--check-kernels` this call does not return: it prints the report and
    // exits with the unresolved count as the status.
    metrale_telemetry::progress::phase(7, "kernel audit");
    serve_phases::audit_and_gate(&args, &ptx_set)?;

    // 2026-09-26: Built before `maybe_run_ep_worker`, which installs it on
    // worker ranks.
    let early_high_speed_swap_cfg = serve_phases::build_high_speed_swap_config(&args)?;

    let mut model_opt = Some(model);
    if serve_phases::maybe_run_ep_worker(&args, &mut model_opt, &early_high_speed_swap_cfg)? {
        return Ok(None);
    }
    let model = model_opt.expect("head retains model on rank 0");
    Ok(Some(Engine {
        args,
        model,
        config,
        config_json,
        model_dir,
        ptx_set,
        sampling_presets,
        vision_max_pixels,
        remote_image_policy,
        video_ffmpeg,
        world_size,
        prefill_budget,
        max_batch_tokens,
        lora_states,
        nllb_adapter_name,
        early_high_speed_swap_cfg,
        forward,
        auto_max_batch_size,
        device_budget,
    }))
}
