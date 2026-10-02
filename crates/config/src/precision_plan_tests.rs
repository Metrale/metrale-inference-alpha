// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The declared precision plan against real `quantization_config` blocks
//! (fixtures trimmed from the local HF cache: ignore lists and per-layer lists cut to a
//! few entries, schemes untouched).

use super::*;

fn fixture(name: &str) -> serde_json::Value {
    let text = match name {
        "unsloth" => include_str!("precision_plan/fixtures/unsloth_qwen3_8_27b_nvfp4.json"),
        "nvidia27" => include_str!("precision_plan/fixtures/nvidia_qwen3_6_27b_nvfp4.json"),
        "nvidia35" => include_str!("precision_plan/fixtures/nvidia_qwen3_6_35b_a3b_nvfp4.json"),
        "fp8" => include_str!("precision_plan/fixtures/qwen3_6_35b_a3b_fp8.json"),
        "ct_all" => include_str!("precision_plan/fixtures/kbenkhaled_qwen3_5_27b_nvfp4.json"),
        "next80" => include_str!("precision_plan/fixtures/nvidia_qwen3_next_80b_nvfp4.json"),
        other => panic!("no fixture {other}"),
    };
    serde_json::from_str(text).expect("fixture parses")
}

fn plan(name: &str) -> DeclaredPrecisionPlan {
    DeclaredPrecisionPlan::from_quantization_config(&fixture(name)).expect("plan")
}

const L: &str = "model.language_model.layers";

/// 2026-09-28: unsloth/Qwen3.8-27B-NVFP4: MLP 0-55 W4A4 (FP4 activations, group 16,
/// dynamic local), MLP 56-63 and every attention/GDN projection and lm_head W8A8 per token.
/// Layers 56-63 match both groups' patterns; the layer-numbered pattern sorts first, as in
/// compressed-tensors, and the on-disk tensors agree (layer 56's gate_proj is F8_E4M3).
#[test]
fn unsloth_dense_declares_w4a4_mlp_and_w8a8_projections() {
    let p = plan("unsloth");
    assert_eq!(p.source, PlanSource::CompressedTensors);
    for layer in [0, 31, 55] {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let got = p.resolve(&format!("{L}.{layer}.mlp.{proj}"));
            assert_eq!(got.label(), "W4A4", "layer {layer} {proj}");
            assert!(got.activation_is_fp4());
            let a = got.activation.expect("activation");
            assert_eq!(a.granularity, Granularity::TensorGroup(16));
            assert_eq!(a.timing, ScaleTiming::Local);
        }
    }
    let fp8_modules = [
        format!("{L}.56.mlp.gate_proj"),
        format!("{L}.63.mlp.down_proj"),
        format!("{L}.3.self_attn.q_proj"),
        format!("{L}.3.self_attn.o_proj"),
        format!("{L}.0.linear_attn.in_proj_qkv"),
        format!("{L}.0.linear_attn.in_proj_z"),
        format!("{L}.0.linear_attn.out_proj"),
        "lm_head".to_string(),
    ];
    for m in &fp8_modules {
        let got = p.resolve(m);
        assert_eq!(got.label(), "W8A8", "{m}");
        assert_eq!(
            got.weight.expect("w").granularity,
            Granularity::Channel,
            "{m}"
        );
        let a = got.activation.expect("a");
        assert_eq!(
            (a.granularity, a.timing),
            (Granularity::Token, ScaleTiming::Dynamic)
        );
    }
    // 2026-09-28: Layer 5 and 65 are not in the 56-63 alternation.
    assert_eq!(p.resolve(&format!("{L}.5.mlp.up_proj")).label(), "W4A4");
    assert_eq!(p.resolve(&format!("{L}.156.mlp.up_proj")).label(), "W4A4");
    // 2026-09-28: Ignored and unmatched modules are unquantized.
    assert_eq!(
        p.resolve("model.visual.blocks.0.attn.qkv"),
        LayerPrecision::UNQUANTIZED
    );
    assert_eq!(
        p.resolve(&format!("{L}.0.linear_attn.in_proj_a")).label(),
        "W16A16"
    );
}

/// 2026-09-28: nvidia/Qwen3.6-27B-NVFP4 (ModelOpt MIXED_PRECISION): the MLP is
/// `W4A16_NVFP4`, so its declared activations are 16-bit and W4A4 there would go below
/// the checkpoint. GDN/attention are per-tensor static FP8.
#[test]
fn nvidia_dense_declares_weight_only_mlp() {
    let p = plan("nvidia27");
    assert_eq!(p.source, PlanSource::ModelOpt);
    let mlp = p.resolve(&format!("{L}.0.mlp.gate_proj"));
    assert_eq!(mlp.label(), "W4A16");
    assert!(!mlp.activation_is_fp4());
    assert_eq!(p.resolve("lm_head").label(), "W4A16");
    let qkv = p.resolve(&format!("{L}.0.linear_attn.in_proj_qkv"));
    assert_eq!(qkv.label(), "W8A8");
    assert_eq!(qkv.activation.expect("a").timing, ScaleTiming::Static);
    assert_eq!(
        p.resolve("mtp.layers.0.mlp.gate_proj"),
        LayerPrecision::UNQUANTIZED
    );
}

/// 2026-09-28: nvidia/Qwen3.6-35B-A3B-NVFP4 lists FP4 activations in `config_groups` and
/// `W4A16_NVFP4` in `quantized_layers`; the per-layer list wins.
#[test]
fn modelopt_quantized_layers_outrank_config_groups() {
    let p = plan("nvidia35");
    let experts = p.resolve(&format!("{L}.0.mlp.experts"));
    assert_eq!(experts.label(), "W4A16");
    // 2026-09-28: Without `quantized_layers`, the same block's `config_groups` says W4A4.
    let mut raw = fixture("nvidia35");
    raw.as_object_mut().expect("obj").remove("quantized_layers");
    let groups_only = DeclaredPrecisionPlan::from_quantization_config(&raw).expect("plan");
    assert_eq!(
        groups_only.resolve(&format!("{L}.0.mlp.experts")).label(),
        "W4A4"
    );
}

/// 2026-10-02: What the qwen35 loader reads from nvidia/Qwen3.6-35B-A3B-NVFP4 under
/// `declared`: FP8 attention and GDN projections (per-tensor, static activations), NVFP4 W4A16
/// routed and shared experts, and an unquantized router, shared-expert gate and GDN `a`/`b`.
#[test]
fn nvidia35_declares_fp8_projections_nvfp4_experts_bf16_router() {
    let p = plan("nvidia35");
    for m in [
        format!("{L}.3.self_attn.q_proj"),
        format!("{L}.3.self_attn.o_proj"),
        format!("{L}.0.linear_attn.in_proj_qkv"),
        format!("{L}.0.linear_attn.in_proj_z"),
        format!("{L}.0.linear_attn.out_proj"),
    ] {
        let got = p.resolve(&m);
        assert_eq!(got.label(), "W8A8", "{m}");
        assert!(got.weight.expect("w").is_fp8(), "{m}");
    }
    for m in [
        format!("{L}.0.mlp.experts"),
        format!("{L}.0.mlp.shared_expert.down_proj"),
    ] {
        assert_eq!(p.resolve(&m).label(), "W4A16", "{m}");
    }
    for m in [
        format!("{L}.0.mlp.gate"),
        format!("{L}.0.mlp.shared_expert_gate"),
        format!("{L}.0.linear_attn.in_proj_a"),
    ] {
        assert_eq!(p.resolve(&m), LayerPrecision::UNQUANTIZED, "{m}");
    }
}

/// 2026-09-28: Qwen/Qwen3.6-35B-A3B-FP8: block-scaled E4M3 weights, dynamic E4M3
/// activations per token group of 128; `modules_to_not_convert` stays BF16.
#[test]
fn fp8_block_checkpoint_declares_w8a8_group_128() {
    let p = plan("fp8");
    assert_eq!(p.source, PlanSource::Fp8);
    let got = p.resolve(&format!("{L}.0.mlp.experts.3.down_proj"));
    assert_eq!(got.label(), "W8A8");
    assert_eq!(
        got.weight.expect("w").granularity,
        Granularity::Block(128, 128)
    );
    let a = got.activation.expect("a");
    assert_eq!(
        (a.granularity, a.timing),
        (Granularity::Group(128), ScaleTiming::Dynamic)
    );
    let ignored = fixture("fp8")["modules_to_not_convert"][0]
        .as_str()
        .expect("str")
        .to_string();
    assert_eq!(p.resolve(&ignored), LayerPrecision::UNQUANTIZED);
}

/// 2026-09-28: A compressed-tensors checkpoint that targets `Linear` declares W4A4 for every
/// linear layer not ignored (Kbenkhaled/Qwen3.5-27B-NVFP4), and a ModelOpt global
/// `quant_algo: NVFP4` does the same (nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4, whose
/// exclusions are globs and exact names).
#[test]
fn class_and_global_targets_cover_every_linear() {
    let ct = plan("ct_all");
    assert_eq!(
        ct.resolve(&format!("{L}.3.self_attn.q_proj")).label(),
        "W4A4"
    );
    assert_eq!(ct.resolve(&format!("{L}.3.mlp.down_proj")).label(), "W4A4");
    let next = plan("next80");
    assert_eq!(
        next.resolve("model.layers.7.mlp.experts.0.up_proj").label(),
        "W4A4"
    );
    assert_eq!(next.resolve("lm_head"), LayerPrecision::UNQUANTIZED);
    assert_eq!(
        next.resolve("model.layers.0.linear_attn.in_proj_qkvz"),
        LayerPrecision::UNQUANTIZED
    );
}

/// 2026-09-28: A group with no `input_activations` (null, absent or `{}`) is weight-only:
/// the default for its layers is 16-bit activations.
#[test]
fn no_activation_scheme_means_weight_only() {
    for act in [serde_json::Value::Null, serde_json::json!({})] {
        let mut raw = fixture("unsloth");
        raw["config_groups"]["group_1"]["input_activations"] = act.clone();
        let p = DeclaredPrecisionPlan::from_quantization_config(&raw).expect("plan");
        let mlp = p.resolve(&format!("{L}.10.mlp.gate_proj"));
        assert_eq!(mlp.label(), "W4A16", "input_activations = {act}");
        assert!(mlp.activation.is_none());
    }
    let mut raw = fixture("unsloth");
    raw["config_groups"]["group_1"]
        .as_object_mut()
        .expect("obj")
        .remove("input_activations");
    let p = DeclaredPrecisionPlan::from_quantization_config(&raw).expect("plan");
    assert_eq!(p.resolve(&format!("{L}.10.mlp.gate_proj")).label(), "W4A16");
}

/// 2026-09-28: No block, or a method the plan does not read, declares nothing.
#[test]
fn unknown_or_absent_methods_declare_nothing() {
    let p = DeclaredPrecisionPlan::from_quantization_config(&serde_json::json!({
        "quant_method": "gptq", "bits": 4
    }))
    .expect("plan");
    assert!(p.is_undeclared());
    assert_eq!(
        p.resolve(&format!("{L}.0.mlp.gate_proj")),
        LayerPrecision::UNQUANTIZED
    );
}

/// 2026-09-28: Malformed blocks are refused, naming the field, never guessed at.
#[test]
fn malformed_blocks_are_refused() {
    let cases: Vec<(&str, Box<dyn Fn(&mut serde_json::Value)>)> = vec![
        (
            "num_bits",
            Box::new(|r| {
                r["config_groups"]["group_1"]["input_activations"]["num_bits"] = "four".into()
            }),
        ),
        (
            "type",
            Box::new(|r| r["config_groups"]["group_1"]["weights"]["type"] = "posit".into()),
        ),
        (
            "strategy",
            Box::new(|r| r["config_groups"]["group_0"]["weights"]["strategy"] = "diagonal".into()),
        ),
        (
            "group_size",
            Box::new(|r| {
                r["config_groups"]["group_1"]["weights"]
                    .as_object_mut()
                    .expect("o")
                    .remove("group_size");
            }),
        ),
        (
            "dynamic",
            Box::new(|r| {
                r["config_groups"]["group_1"]["input_activations"]["dynamic"] = "global".into()
            }),
        ),
        (
            "targets",
            Box::new(|r| r["config_groups"]["group_0"]["targets"] = serde_json::json!([])),
        ),
        (
            "bad target pattern",
            Box::new(|r| {
                r["config_groups"]["group_0"]["targets"] = serde_json::json!(["re:(unclosed"])
            }),
        ),
        (
            "config_groups",
            Box::new(|r| r["config_groups"] = serde_json::json!(["group_0"])),
        ),
        (
            "weights",
            Box::new(|r| {
                r["config_groups"]["group_0"]
                    .as_object_mut()
                    .expect("o")
                    .remove("weights");
            }),
        ),
        (
            "ignore",
            Box::new(|r| r["ignore"] = serde_json::json!("lm_head")),
        ),
    ];
    for (field, mutate) in cases {
        let mut raw = fixture("unsloth");
        mutate(&mut raw);
        let err = DeclaredPrecisionPlan::from_quantization_config(&raw).expect_err(field);
        let err = format!("{err:#}");
        assert!(err.contains(field), "{field}: {err}");
    }
    let mixed = serde_json::json!({"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"});
    assert!(DeclaredPrecisionPlan::from_quantization_config(&mixed).is_err());
    let fp8 = serde_json::json!({"quant_method": "fp8", "activation_scheme": "sometimes"});
    assert!(DeclaredPrecisionPlan::from_quantization_config(&fp8).is_err());
    assert!(DeclaredPrecisionPlan::from_quantization_config(&serde_json::json!("fp8")).is_err());
}

/// 2026-09-30: Qwen/Qwen3.6-27B-FP8 lists MoE router names (`...mlp.gate`) among its
/// `modules_to_not_convert`. Matched on whole dotted segments they exclude a router, never
/// the dense `...mlp.gate_proj`, which the checkpoint stores as block-scaled E4M3 (its
/// `weight_scale_inv` is in the safetensors index). A substring match read it as BF16.
#[test]
fn an_hf_fp8_exclusion_names_whole_module_segments() {
    let text = include_str!("precision_plan/fixtures/qwen3_6_27b_fp8.json");
    let p = DeclaredPrecisionPlan::from_quantization_config(&serde_json::from_str(text).unwrap())
        .expect("plan");
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        assert_eq!(
            p.resolve(&format!("{L}.0.mlp.{proj}")).label(),
            "W8A8",
            "{proj}"
        );
    }
    // 2026-09-30: The listed modules themselves, and the children of a listed parent.
    assert_eq!(
        p.resolve(&format!("{L}.0.mlp.gate")),
        LayerPrecision::UNQUANTIZED
    );
    assert_eq!(
        p.resolve(&format!("{L}.0.linear_attn.in_proj_b")),
        LayerPrecision::UNQUANTIZED
    );
    assert_eq!(p.resolve("lm_head"), LayerPrecision::UNQUANTIZED);
}

/// 2026-09-30: The HF entry forms: exact, parent, trailing segments, and regex; a malformed
/// regex or an empty entry is refused.
#[test]
fn hf_module_entries_match_by_segment_and_refuse_bad_patterns() {
    let t = |e: &str| Target::hf_module(e).expect("entry");
    let m = "model.layers.3.mlp.gate_proj";
    assert!(t(m).matches_name(m));
    assert!(t("model.layers.3").matches_name(m), "a parent");
    assert!(t("mlp.gate_proj").matches_name(m), "trailing segments");
    assert!(!t("mlp.gate").matches_name(m), "not a segment of it");
    assert!(!t("layers.3.mlp.gate").matches_name(m));
    assert!(t("mlp.gate").matches_name("model.layers.3.mlp.gate"));
    assert!(
        t(r"model\.layers\.\d+\.mlp\.gate_proj").matches_name(m),
        "a regex"
    );
    assert!(
        !t(r"model\.layers\.\d+\.mlp\.gate").matches_name(m),
        "a regex ends on a segment"
    );
    assert!(t("model.layers.*").matches_name(m));
    assert!(Target::hf_module("model.layers.(").is_err());
    assert!(Target::hf_module("").is_err());
    let bad = serde_json::json!({
        "quant_method": "fp8", "activation_scheme": "dynamic",
        "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head", "model.(visual"],
    });
    assert!(DeclaredPrecisionPlan::from_quantization_config(&bad).is_err());
}
