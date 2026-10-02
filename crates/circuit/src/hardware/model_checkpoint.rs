// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The model source over the checkpoint itself: `config.json` (and the sidecar
//! `hf_quant_config.json`) resolved by [`crate::resolve_checkpoint`] at the declared formats,
//! with the serving policy derived from the checkpoint and the engine's stated defaults. A
//! `recipe` request goes to the recipe's golden instance ([`InstancesSource`]), whose formats
//! are the recipe's.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Every derived setting names where its value comes from ([`POLICY_SOURCES`]); the class's
//!   `[defaults]` settings are placeholders here and are always re-read from the device's
//!   class ([`super::model::policy_on_class`]).
//! - A checkpoint the model axis refuses is refused here with its reason, never planned from a
//!   guess.

use std::collections::{BTreeMap, BTreeSet};

use super::HwError;
use super::model::{CircuitSource, InstancesSource, ModelSpec, ModelUnderPlan, PrecisionChoice};
use super::sources::KernelTree;
use crate::format::Format;
use crate::fuser::Policy;
use crate::ir::{Circuit, OpKind};
use crate::{QuantMetadata, ServePrecision, resolve_checkpoint};

/// 2026-09-30: The kernel quant axis every target is built for: the build compiles
/// `METRALE_TARGET_QUANT=nvfp4` unless told otherwise, and the FP8 35B-A3B golden instance
/// targets `gb10/qwen3.6-35b-a3b/nvfp4` too.
pub const KERNEL_QUANT: &str = "nvfp4";

/// 2026-09-30: Where each derived setting comes from, printed in the report.
pub const POLICY_SOURCES: [(&str, &str); 11] = [
    (
        "row_tiers",
        "canonical for FP8 routed experts or a MoE with FP8 projections, else by_rows \
         (ml/row_tiers.rs:64-78)",
    ),
    (
        "kv_cache_dtype",
        "the checkpoint's declared KV-cache format (bf16 when none)",
    ),
    ("lm_head_dtype", "the lm_head's declared weight format"),
    (
        "ssm_h_dtype",
        "f32, the --ssm-h-dtype default (crates/server/src/cli/serve_args.rs)",
    ),
    (
        "gemv_sw",
        "on unless METRALE_NO_GEMV_SW=1 (ml/ops/model_levers_resolve.rs:53)",
    ),
    (
        "w4a16_tc",
        "on unless METRALE_NO_W4A16_TC (ml/ops/gemv_tc.rs:79-82)",
    ),
    (
        "ssm_batched_recurrent",
        "the device class's HARDWARE.toml [defaults]",
    ),
    (
        "ssm_ba_gates_hopper",
        "the device class's HARDWARE.toml [defaults]",
    ),
    (
        "decode_split_silu",
        "the device class's HARDWARE.toml [defaults]",
    ),
    (
        "rms_norm_act_quant",
        "off: the executor has no fused norm-quantize launch (2026-09-30)",
    ),
    (
        "activation_quantization",
        "adaptive: the per-row-count routing FUSIONS.toml encodes, which the rules plan; a fixed \
         --activation-quantization is not modelled by the rules yet (2026-10-02)",
    ),
];

/// 2026-09-30: The source over checkpoints, falling back to the recipes for `recipe` formats.
pub struct CheckpointSource<'a> {
    /// 2026-09-30: The kernel tree (MODEL.toml target resolution).
    pub tree: &'a dyn KernelTree,
}

impl CircuitSource for CheckpointSource<'_> {
    fn model(&self, spec: &ModelSpec<'_>) -> Result<ModelUnderPlan, HwError> {
        if spec.precision == PrecisionChoice::Recipe {
            return InstancesSource {
                repo: self.tree.as_repo(),
            }
            .model(spec);
        }
        let config = spec.config_json.ok_or_else(|| {
            HwError::Model(format!(
                "{}: declared formats need the checkpoint's config.json (a checkpoint directory, \
                 the local Hugging Face cache, or --allow-network)",
                spec.checkpoint
            ))
        })?;
        let rc = resolve_checkpoint(
            config,
            QuantMetadata {
                hf_quant_config: spec.hf_quant,
            },
            &ServePrecision::Declared,
        )
        .map_err(|e| HwError::Model(format!("{}: {e}", spec.checkpoint)))?;
        // 2026-09-30: The engine's own resolution. A config the engine cannot parse has no
        // kernel target: the plan says why and reads the class's common layer only; it never
        // falls back to the raw config.json model_type (which the engine may rewrite).
        let (target, why_none) = match self.tree.kernel_target(config, &[spec.checkpoint]) {
            Ok(Some(t)) => (Some(t), String::new()),
            Ok(None) => (None, "none: no MODEL.toml claims it".to_string()),
            Err(e) => (
                None,
                format!("unresolved: the engine's config parse refuses this checkpoint ({e})"),
            ),
        };
        let policy = derive_policy(&rc.circuit, rc.kv_cache)?;
        let kernel_model = target.clone().unwrap_or_else(|| "(none)".into());
        let settings: Vec<String> = policy
            .settings
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        Ok(ModelUnderPlan {
            label: spec.checkpoint.to_string(),
            checkpoint: spec.checkpoint.to_string(),
            header: vec![
                ("recipe".into(), "none (planned from the checkpoint)".into()),
                ("checkpoint".into(), spec.checkpoint.to_string()),
                (
                    "target".into(),
                    format!("gb10/{kernel_model}/{KERNEL_QUANT}"),
                ),
                ("settings".into(), settings.join(" ")),
                ("opt-in levers".into(), "none".into()),
            ],
            kernel_model,
            kernel_quant: KERNEL_QUANT.into(),
            circuit: rc.circuit,
            policy,
            precision: format!(
                "declared by the checkpoint (arch `{}`, model_type `{}`; kernel target {})",
                rc.arch,
                rc.model_type,
                target.map_or(why_none, |t| format!("`{t}`"))
            ),
            precision_choice: PrecisionChoice::Declared,
            policy_sources: POLICY_SOURCES
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            settings_class: None,
        })
    }
}

fn dtype_name(f: Format) -> Result<&'static str, HwError> {
    match f {
        Format::Bf16 => Ok("bf16"),
        Format::Fp8E4m3 { .. } => Ok("fp8"),
        Format::Nvfp4 { .. } => Ok("nvfp4"),
        other => Err(HwError::Model(format!(
            "{} is no serving dtype",
            other.name()
        ))),
    }
}

/// 2026-09-30: The serving policy of a checkpoint-derived circuit ([`POLICY_SOURCES`]).
pub fn derive_policy(c: &Circuit, kv_cache: Option<Format>) -> Result<Policy, HwError> {
    let fp8_experts = c
        .nodes
        .iter()
        .any(|n| n.op == OpKind::ExpertGateUp && matches!(n.weight, Some(Format::Fp8E4m3 { .. })));
    // 2026-10-02: A MoE whose other projections are FP8 (nvidia/Qwen3.6-35B-A3B-NVFP4) is
    // canonical too (crates/server/src/main_modules/serve_load/model_setup.rs `publish_row_tiers`).
    let moe = c.nodes.iter().any(|n| n.op == OpKind::ExpertGateUp);
    let fp8_projections = c.nodes.iter().any(|n| {
        matches!(n.op, OpKind::Linear(_)) && matches!(n.weight, Some(Format::Fp8E4m3 { .. }))
    });
    let canonical = fp8_experts || (moe && fp8_projections);
    let head = c
        .nodes
        .iter()
        .find(|n| n.op == OpKind::LmHead)
        .and_then(|n| n.weight)
        .ok_or_else(|| HwError::Model("the circuit has no lm_head weight".into()))?;
    let kv = match kv_cache {
        None => "bf16",
        Some(f) => dtype_name(f)?,
    };
    // 2026-09-30: The two class settings are re-read from the device's class before any plan;
    // "class" never reaches a rule (policy_on_class refuses a class that does not state them).
    let settings: BTreeMap<String, String> = [
        ("row_tiers", if canonical { "canonical" } else { "by_rows" }),
        ("kv_cache_dtype", kv),
        ("lm_head_dtype", dtype_name(head)?),
        ("ssm_h_dtype", "f32"),
        ("gemv_sw", "on"),
        ("w4a16_tc", "on"),
        ("ssm_batched_recurrent", "class"),
        ("ssm_ba_gates_hopper", "class"),
        ("decode_split_silu", "class"),
        ("rms_norm_act_quant", "off"),
        ("activation_quantization", "adaptive"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    Ok(Policy {
        opt_in_levers: BTreeSet::new(),
        settings,
    })
}
