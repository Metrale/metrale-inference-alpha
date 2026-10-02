// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The config, kernel-target and tokenizer steps of `load_model`
//! (phases 2, 3 and 8): the lm_head dtype, the media policies, the kernel
//! target and its quant check, the chat tokenizer, the server's default chat
//! template kwargs, and the MODEL.toml behavior with its CLI overrides.
//!
//! Owner: server startup (`met serve`).
//! Invariants: each function runs its steps in `load_model`'s order; the
//! `tracing` events keep `load_model`'s target (`met::main_modules::serve_load`).

use std::path::Path;

use anyhow::{Context, Result};
use metrale_config::ModelConfig;

use crate::cli;
use crate::main_modules::serve::{
    DefaultChatTemplateKwargs, canonicalize_model_quant, describe_quant_source,
    parse_default_chat_template_kwargs, quant_pair_compatible, resolve_vision_max_pixels,
};
use crate::main_modules::serve_phases;
use crate::tokenizer::ChatTokenizer;

pub(super) fn configure_model(
    args: &cli::ServeArgs,
    model_dir: &Path,
) -> Result<(ModelConfig, String)> {
    metrale_telemetry::progress::phase(2, "config");
    let (mut config, config_json) = serve_phases::load_model_config(model_dir)?;

    // 2026-09-26: A sibling `hf_quant_config.json` fills `quantization_config`
    // when config.json has none; its top level is read as the quantization
    // block.
    serve_phases::merge_sidecar_quant_config(model_dir, &mut config)?;

    // 2026-09-28: The head `--lm-head-dtype default` takes under the published tier.
    let choice = metrale_config::WeightQuantPolicy::for_checkpoint(
        args.weight_quant_tier()?,
        config.quantization_config.as_ref(),
        metrale_model_layers::layers::kernel_caps(),
    )
    .lm_head();
    let (lm_head_bf16_override, lm_head_fp8) = lm_head_flags(&args.lm_head_dtype, choice)?;
    config.lm_head_bf16_override = lm_head_bf16_override;
    config.lm_head_fp8 = lm_head_fp8;
    Ok((config, config_json))
}

/// 2026-09-26: `--lm-head-dtype` as `(lm_head_bf16_override, lm_head_fp8)`. The first is read
/// first by `skip_lm_head_quantization`; `setup_lm_heads` reads both, and uses a checkpoint's
/// pre-packed NVFP4 head whatever they say. An unknown value fails the boot. An explicit value
/// always wins; `default` takes `choice` (2026-09-28: `WeightQuantPolicy::lm_head`), and
/// logs it when it is not the engine's per-model default.
fn lm_head_flags(
    lm_head_dtype: &str,
    choice: metrale_config::weight_quantization::LmHeadChoice,
) -> Result<(Option<bool>, bool)> {
    use metrale_config::weight_quantization::{LmHeadChoice, LmHeadFormat};
    Ok(match lm_head_dtype {
        "default" => match choice {
            LmHeadChoice::EngineDefault => (None, false),
            LmHeadChoice::PendingFp8Kernel => {
                tracing::info!(
                    "--weight-quantization declared: the checkpoint declares an FP8 lm_head; the \
                     engine's default head runs instead until the batched FP8 head kernel lands \
                     (the FP8 head launches once per row); --lm-head-dtype fp8 serves it now"
                );
                (None, false)
            }
            LmHeadChoice::Declared(head) => {
                tracing::info!(
                    "--weight-quantization declared: lm_head {head:?}, as the checkpoint \
                     declares (--lm-head-dtype overrides)"
                );
                match head {
                    LmHeadFormat::Bf16 => (Some(true), false),
                    LmHeadFormat::Fp8 => (Some(false), true),
                    LmHeadFormat::Nvfp4 => (Some(false), false),
                }
            }
        },
        "bf16" => (Some(true), false),
        // 2026-09-26: Quantize the lm_head to NVFP4 whatever the model's own
        // default is.
        "nvfp4" => (Some(false), false),
        // 2026-09-26: Quantize the lm_head to FP8 E4M3 with per-row scales,
        // or use the checkpoint's own FP8 head when it ships one.
        "fp8" => (Some(false), true),
        other => {
            anyhow::bail!(
                "--lm-head-dtype must be 'default', 'bf16', 'nvfp4', or 'fp8', got '{other}'"
            )
        }
    })
}

pub(super) fn resolve_media_policies(
    args: &cli::ServeArgs,
    model_dir: &Path,
    config: &mut ModelConfig,
) -> Result<(
    Option<usize>,
    crate::api::chat::remote_image::RemoteImagePolicy,
    metrale_model_layers::video_decode_ffmpeg::FfmpegPolicy,
)> {
    // 2026-09-26: The vision area bound is resolved once and installed on the
    // config before `build_model`, which builds the vision encoder; the
    // encoder sizes its patch buffers from it (`derive_max_patches`).
    let vision_max_pixels = resolve_vision_max_pixels(args, model_dir)?;
    if let Some(v) = config.vision.as_mut() {
        v.max_pixels = vision_max_pixels;
    }
    match vision_max_pixels {
        Some(px) => {
            tracing::info!(target: "met::main_modules::serve_load", "Vision area bound: {} px ({})",
                px,
                if args.vision_max_pixels > 0 {
                    "--vision-max-pixels"
                } else {
                    // 2026-09-26: Names no file: `read_preprocessor_max_pixels`
                    // logs the file and key it read.
                    "checkpoint processor config / METRALE_VISION_MAX_PIXELS"
                }
            )
        }
        None => {
            tracing::info!(target: "met::main_modules::serve_load", "Vision area bound: none declared — falling back to the 1280px long-side clamp"
            )
        }
    }

    // 2026-09-26: Logged at WARN when on: it lets the server make outbound HTTP
    // requests to URLs that clients supply.
    let remote_image_policy = crate::api::chat::remote_image::RemoteImagePolicy {
        enabled: args.vision_allow_remote_images,
        max_bytes: args.vision_remote_image_max_mb.saturating_mul(1024 * 1024),
        timeout_secs: args.vision_remote_image_timeout_s,
        allow_private: args.vision_remote_image_allow_private,
    };
    if remote_image_policy.enabled {
        tracing::warn!(target: "met::main_modules::serve_load", "Remote image fetching ENABLED (--vision-allow-remote-images): this server will \
             issue outbound HTTP to URLs supplied in chat requests. Cap {} MiB, timeout {} s, \
             private/loopback/link-local destinations {}.",
            args.vision_remote_image_max_mb,
            remote_image_policy.timeout_secs,
            if remote_image_policy.allow_private {
                "ALLOWED (--vision-remote-image-allow-private)"
            } else {
                "refused"
            }
        );
    } else {
        tracing::info!(target: "met::main_modules::serve_load", "Remote image fetching disabled (default); image_url parts carrying an http(s) \
             URL are refused with a 400. Send base64 data: URIs, or pass \
             --vision-allow-remote-images."
        );
    }

    // 2026-09-26: Probed at boot, so a missing ffmpeg shows in the startup log
    // instead of on the first video request.
    let video_ffmpeg = metrale_model_layers::video_decode_ffmpeg::FfmpegPolicy {
        enabled: args.video_allow_ffmpeg,
        binary: args.video_ffmpeg_path.clone(),
        max_frames: args.video_max_frames,
        timeout_secs: args.video_decode_timeout_s,
        ..Default::default()
    };
    match metrale_model_layers::video_decode_ffmpeg::probe(&video_ffmpeg) {
        metrale_model_layers::video_decode_ffmpeg::Availability::Ready(v) => {
            tracing::info!(target: "met::main_modules::serve_load", "Video decoding ENABLED via {} ({}); sampling at {} fps, max {} frames",
                args.video_ffmpeg_path,
                v,
                args.video_fps,
                args.video_max_frames,
            )
        }
        // 2026-09-26: A warning, not a boot failure: text and image serving do
        // not need ffmpeg.
        metrale_model_layers::video_decode_ffmpeg::Availability::Missing(why) => {
            tracing::warn!(target: "met::main_modules::serve_load", "Video decoding was ENABLED (--video-allow-ffmpeg) but the decoder is NOT \
                 USABLE: {why}. Every video request will fail. Install ffmpeg (apt install \
                 ffmpeg) or point --video-ffmpeg-path at it. Animated GIF still decodes \
                 in-process; text and image serving are unaffected.",
            )
        }
        metrale_model_layers::video_decode_ffmpeg::Availability::Disabled => {
            tracing::info!(target: "met::main_modules::serve_load", "Video decoding disabled (default). Animated GIF decodes in-process; every \
                 other container needs ffmpeg — pass --video-allow-ffmpeg to enable it."
            )
        }
    }
    Ok((vision_max_pixels, remote_image_policy, video_ffmpeg))
}

pub(super) fn log_model_config(config: &ModelConfig) {
    if let Some(ref qc) = config.quantization_config {
        tracing::info!(target: "met::main_modules::serve_load", "Quantization config: method={:?}, algo={:?}, format={:?}, {} module(s) in ignore list",
            qc.quant_method,
            qc.quant_algo,
            qc.format,
            qc.ignore_modules.len(),
        );
    }

    tracing::info!(target: "met::main_modules::serve_load", "Model config: {} layers, {} attention, {} SSM, {} experts, rope_theta={}, head_dim={}, rotary_dim={}",
        config.num_hidden_layers,
        config.num_attention_layers(),
        config.num_ssm_layers(),
        config.num_experts,
        config.rope_theta,
        config.head_dim,
        config.rotary_dim(),
    );
}

pub(super) fn select_kernel_target(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    model_dir: &Path,
) -> Result<metrale_kernels::TargetPtxSet> {
    metrale_telemetry::progress::phase(3, "gpu init");
    // 2026-09-26: Target resolution rules are in `metrale_kernels::resolve`:
    // exact `hidden_size` before wildcards, a collision broken by each
    // target's `match_names` against `model_refs`, an unbroken tie an error,
    // and `--kernel-target` pins the target.
    let model_dir_str = model_dir.display().to_string();
    let model_refs: Vec<&str> = [
        args.model.as_deref(),
        args.model_name.as_deref(),
        Some(model_dir_str.as_str()),
    ]
    .into_iter()
    .flatten()
    .collect();
    let ptx_set = metrale_kernels::ptx_for_config(
        &config.model_type,
        config.hidden_size,
        &model_refs,
        args.kernel_target.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?
    .with_context(|| {
        format!(
            "No compiled kernel target matches model_type '{}' / hidden_size={}. \
             Available targets: {:?}",
            config.model_type,
            config.hidden_size,
            metrale_kernels::available_targets()
                .iter()
                .map(|t| &t.target.model)
                .collect::<Vec<_>>(),
        )
    })?;
    // 2026-09-26: kimi_k3 MXFP4 weights need the exact mxfp4 target: for several
    // quant variants of one target, `ptx_for_config` returns the first.
    let ptx_set = if config.model_type == "kimi_k3" && canonicalize_model_quant(config) == "mxfp4" {
        metrale_kernels::ptx_for_exact_target(ptx_set.target.model, "mxfp4")
            .context("K3 MXFP4 requires its compiled mxfp4 target; rebuild with METRALE_TARGET_QUANT=mxfp4 or *")?
    } else {
        ptx_set
    };
    Ok(ptx_set)
}

pub(super) fn check_kernel_target(
    ptx_set: &metrale_kernels::TargetPtxSet,
    config: &mut ModelConfig,
) -> Result<()> {
    // 2026-09-26: Record the resolved target for the dashboard's kernel table.
    crate::tui::data::kernels::publish_loaded_target(ptx_set.target.model, ptx_set.target.quant);

    // 2026-09-26: `ptx_for_config` does not select on quant, so the kernel and
    // model quant pair is checked against `quant_pair_compatible` here, before
    // any weight loads.
    let model_quant = canonicalize_model_quant(config);
    let kernel_quant = ptx_set.target.quant;
    if !quant_pair_compatible(kernel_quant, &model_quant) {
        anyhow::bail!(
            "Kernel/model QUANT MISMATCH. Kernel target: {} (quant={kernel_quant}). \
             Model declares quant={model_quant} ({}). \
             The compiled kernel set has no known dispatch path for \
             quant '{model_quant}' — loading would produce silent garbage. \
             Rebuild with METRALE_TARGET_QUANT={model_quant} (or =* to bundle multiple \
             variants) and restart.",
            ptx_set.target,
            describe_quant_source(config),
        );
    }
    tracing::info!(target: "met::main_modules::serve_load", "Selected kernel target: {} ({} modules) — quant compat: kernel={kernel_quant} \
         model={model_quant} OK",
        ptx_set.target,
        ptx_set.modules.len(),
    );

    // 2026-09-26: A checkpoint that declares a vision tower, on a target that
    // ships neither vision module (`vision_encoder`, `glm_vit`), is served
    // text-only: `config.vision` is cleared.
    const VISION_MODULES: [&str; 2] = ["vision_encoder", "glm_vit"];
    if config.vision.is_some()
        && !ptx_set
            .modules
            .iter()
            .any(|(name, _)| VISION_MODULES.contains(name))
    {
        tracing::warn!(target: "met::main_modules::serve_load", "Checkpoint declares a vision tower but kernel target {} ships neither a \
             vision_encoder nor a glm_vit module — serving TEXT-ONLY (image inputs \
             ignored). Rebuild the target with vision to enable images.",
            ptx_set.target,
        );
        config.vision = None;
    }
    Ok(())
}

/// 2026-09-27: Publish the model's row-tier policy before it is built:
/// canonical for an FP8 MoE checkpoint unless `--no-canonical-tiers`, with the two
/// `METRALE_*` levers over the checkpoint default (`row_tiers`).
/// 2026-09-30: A fixed `--activation-quantization` for any family publishes canonical: its
/// single-order kernels are that mode's W8A16 and NVFP4-head paths (`--no-canonical-tiers` is
/// refused beside it by `validate_serve_args`).
/// 2026-10-02: A MoE checkpoint whose attention or GDN projections the `--weight-quantization`
/// policy serves at declared FP8 weights (nvidia/Qwen3.6-35B-A3B-NVFP4 under `declared`) counts
/// as FP8 here: its W8A16 projections take the tile order the FP8 MoE validated, and its NVFP4
/// experts are row-invariant on the grouped decode either way.
pub(super) fn publish_row_tiers(args: &cli::ServeArgs, config: &ModelConfig) {
    use metrale_model_layers::layers::{RowTiers, publish_row_tiers, resolve_row_tiers};
    if metrale_model_layers::layers::any_fixed() {
        publish_row_tiers(RowTiers::Canonical);
        return;
    }
    let policy = metrale_config::WeightQuantPolicy::for_checkpoint(
        metrale_model_layers::layers::weight_quantization(),
        config.quantization_config.as_ref(),
        metrale_model_layers::layers::kernel_caps(),
    );
    // 2026-10-02: `weight_prefix` is set when the weights are opened, after this runs, so a
    // Qwen3.5/3.6 checkpoint's `model.language_model` names are tried beside the default ones.
    let fp8_projections = (0..config.num_hidden_layers).any(|i| {
        [
            config.layer_prefix(i),
            format!("model.language_model.layers.{i}"),
        ]
        .iter()
        .any(|lp| {
            policy.wants_fp8_weights(&format!("{lp}.self_attn.q_proj"))
                || policy.wants_fp8_weights(&format!("{lp}.linear_attn.in_proj_qkv"))
        })
    });
    publish_row_tiers(resolve_row_tiers(
        args.no_canonical_tiers,
        std::env::var_os("METRALE_ROW_EXACT_TIERS").is_some(),
        std::env::var_os("METRALE_CANONICAL_TIERS").is_some(),
        config.num_experts > 0 && (canonicalize_model_quant(config) == "fp8" || fp8_projections),
    ));
}

/// 2026-09-28: Publish whether the MoE experts decode W8A8 (`set_moe_expert_fp8_act`) before the
/// model is built. The cell is process-wide, so it holds FP8 only when the
/// `--weight-quantization` policy asks for FP8 activations on every layer's experts; a dense
/// model, the `nvfp4` tier, or a checkpoint that declares 16-bit activations keeps W8A16.
pub(super) fn publish_moe_expert_act(config: &ModelConfig) {
    use metrale_config::weight_quantization::ActFormat;
    let policy = metrale_config::WeightQuantPolicy::for_checkpoint(
        metrale_model_layers::layers::weight_quantization(),
        config.quantization_config.as_ref(),
        metrale_model_layers::layers::kernel_caps(),
    );
    let declared_fp8 = config.num_experts > 0
        && (0..config.num_hidden_layers).all(|i| {
            let module = format!("{}.mlp.experts.0.gate_proj", config.layer_prefix(i));
            policy.fp8_decode_act(&module) == Some(ActFormat::Fp8)
        });
    // 2026-09-30: A fixed `--activation-quantization` for the MoE names the expert activations
    // itself (`act_quant_support` has refused what the experts cannot run, and a ladder).
    let fp8 = match metrale_model_layers::layers::fixed_act(metrale_config::ProjFamily::Moe, 1) {
        Some(metrale_config::ActQuantFormat::Bf16) => false,
        Some(metrale_config::ActQuantFormat::Fp8) => config.num_experts > 0,
        _ => declared_fp8,
    };
    let published = metrale_model_layers::layers::set_moe_expert_fp8_act(fp8);
    let declared_a8 = config.num_experts > 0
        && (0..config.num_hidden_layers).any(|i| {
            let module = format!("{}.mlp.experts.0.gate_proj", config.layer_prefix(i));
            policy.declares_fp8_activations(&module)
        });
    if !fp8 && declared_a8 {
        tracing::info!(
            "--weight-quantization declared: the checkpoint declares FP8 activations for some of \
             its MoE experts; they decode W8A16, as the MoE W8A8 decode is process-wide and needs \
             every layer's experts at FP8 activations (or its kernel cap)"
        );
    }
    if published != fp8 {
        tracing::warn!(
            "MoE expert decode activations were fixed before the serve published them: \
             FP8 = {published}, the policy asks for FP8 = {fp8}"
        );
    } else if fp8 {
        tracing::info!("MoE experts: W8A8 decode (declared FP8 weights and activations)");
    }
}

pub(super) fn resolve_behavior(
    ptx_set: &metrale_kernels::TargetPtxSet,
    args: &cli::ServeArgs,
    default_kwargs: &DefaultChatTemplateKwargs,
) -> metrale_kernels::ModelBehavior {
    let mut b = ptx_set.behavior.clone();
    if let Some(cli_budget) = args.max_thinking_budget {
        b.max_thinking_budget = cli_budget;
    }
    if let Some(grammar_on) =
        crate::cli::flag_values::Tristate::validated(&args.tool_grammar).pinned()
    {
        b.disable_tool_grammar = !grammar_on;
    }
    // 2026-09-26: The server-level `preserve_thinking` overrides the
    // MODEL.toml value; a request's own value still wins
    // (`api/chat/prepare.rs`).
    if let Some(p) = default_kwargs.preserve_thinking {
        b.preserve_thinking = Some(p);
    }
    b
}

pub(super) fn load_tokenizer(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    model_dir: &Path,
    eos_tokens: &[u32],
) -> Result<(ChatTokenizer, bool)> {
    metrale_telemetry::progress::phase(8, "tokenizer");
    // 2026-09-26: `supports_thinking` comes from `ModelCapabilities`.
    let caps = config.capabilities();
    let supports_thinking = caps.supports_thinking;
    let tokenizer = ChatTokenizer::from_model_dir(
        model_dir,
        eos_tokens[0],
        supports_thinking,
        &config.model_type,
        Some(std::path::Path::new(".")),
        args.disable_template_overrides,
    )?;
    Ok((tokenizer, supports_thinking))
}

pub(super) fn parse_default_kwargs(args: &cli::ServeArgs) -> Result<DefaultChatTemplateKwargs> {
    // 2026-09-26: An unknown key or `reasoning_effort` value in
    // `--default-chat-template-kwargs` fails the boot.
    let default_kwargs = args
        .default_chat_template_kwargs
        .as_deref()
        .map(parse_default_chat_template_kwargs)
        .transpose()?
        .unwrap_or_default();
    Ok(default_kwargs)
}

#[cfg(test)]
#[path = "model_setup_tests.rs"]
mod tests;
