// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Exact-integer tests of `predicted_derived_bytes`. No GPU,
//! checkpoint or environment: each `Fp8RouteInputs` is built by hand.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants: none beyond the types.

use super::*;
use metrale_config::{LayerType, ModelConfig, QuantizationConfig};

use metrale_model_layers::layers::ops::GemmDispatch;

/// 2026-09-25: A dense `qwen3_5` config with the shapes
/// `kernels/gb10/qwen3.8-27b/MODEL.toml` declares: 64 layers (16 full
/// attention, 48 linear attention), hidden 5120, intermediate 17408, head_dim
/// 256, 24 q heads, 4 kv heads, output-gated attention. That file has no
/// linear-attention head fields; the 16x128 key and 48x128 value heads here
/// are set by this fixture.
fn qwen38_27b() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3_5".to_string();
    c.num_experts = 0;
    c.num_experts_per_tok = 0;
    c.moe_intermediate_size = 0;
    c.hidden_size = 5120;
    c.intermediate_size = 17408;
    c.num_hidden_layers = 64;
    c.num_attention_heads = 24;
    c.num_key_value_heads = 4;
    c.head_dim = 256;
    c.attn_gated = true;
    c.linear_num_key_heads = 16;
    c.linear_key_head_dim = 128;
    c.linear_num_value_heads = 48;
    c.linear_value_head_dim = 128;
    c.full_attention_interval = 4;
    c.layer_types = (0..64)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    c.quantization_config = Some(QuantizationConfig {
        quant_method: "fp8".to_string(),
        quant_algo: String::new(),
        format: String::new(),
        ignore_modules: Vec::new(),
    });
    c
}

/// 2026-09-25: `METRALE_DENSE_FP8=1`, tp 1, a config-declared FP8 checkpoint,
/// the GDN FP8 arm on, both W8A8 prefill kernels present, no NVFP4 lever, and
/// the gate+up fusion off.
fn round6_route() -> Fp8RouteInputs {
    Fp8RouteInputs {
        dense_fp8: true,
        tp_size: 1,
        declared_variant: Some(Nvfp4Variant::Fp8Dequanted),
        gdn_fp8: true,
        w8a8_prefill_kernels: true,
        route: RouteEnv {
            keep_nvfp4: false,
            dispatch: GemmDispatch::defaults(),
            attn_w4a4: false,
            attn_prefill_q_t: false,
        },
        ffn_gateup_fused: false,
    }
}

fn predicted(route: &Fp8RouteInputs) -> PredictedDerived {
    match predicted_derived_bytes(&qwen38_27b(), route) {
        DerivedBytesEstimate::NativeFp8Dense(p) => p,
        DerivedBytesEstimate::Unavailable(why) => panic!("expected a prediction, got: {why}"),
    }
}

/// 2026-09-25: 16 x (K twin + V twin) + 48 x (fused `[QKV|Z]` + its scales +
/// `out_proj` scales + interleaved `in_proj_ba`) = 4,242,882,560 B, within 5%
/// of `measured`.
#[test]
fn the_27b_prediction_matches_the_round6_residency_summary() {
    let p = predicted(&round6_route());
    assert_eq!(
        p.attn_fp8_twins, 167_813_120,
        "16 layers x (K twin + V twin)"
    );
    assert_eq!(
        p.ssm_fp8_concat, 4_075_069_440,
        "48 GDN layers x 84,897,280"
    );
    assert_eq!(p.total(), 4_242_882_560);

    let measured = 4.24_f64;
    let predicted_gb = p.total() as f64 / 1e9;
    let err = (predicted_gb - measured).abs() / measured;
    assert!(
        err < 0.05,
        "predicted {predicted_gb:.3} GB vs the round-6 log's {measured} GB ({:.2}% off)",
        err * 100.0,
    );
}

/// 2026-09-25: Each term recomputed here from the shapes, independently of
/// `ssm_concat_bytes`.
#[test]
fn each_term_is_the_loaders_shape_arithmetic() {
    let c = qwen38_27b();
    let kv_twin = fp8_residency::fp8_twin_bytes(4 * 256, 5120);
    assert_eq!(kv_twin, 5_244_160);
    assert_eq!(
        predicted(&round6_route()).attn_fp8_twins,
        16 * 2 * kv_twin as u64
    );

    assert_eq!(c.ssm_qkvz_size(), 16384);
    assert_eq!(c.ssm_qkv_size(), 10240);
    assert_eq!(c.ssm_z_size(), 6144);
    let per_layer = 83_886_080 // 2026-09-25: fused `[QKV|Z]` E4M3 weight, 16384 x 5120
        + (80 * 40 * 4 + 48 * 40 * 4)
        + 40 * 48 * 4
        + 48 * 2 * 5120 * 2;
    assert_eq!(per_layer, 84_897_280);
    assert_eq!(
        predicted(&round6_route()).ssm_fp8_concat,
        48 * per_layer as u64,
    );
}

/// 2026-09-25: Arming the gate+up fusion leaves `total()` unchanged. The
/// prediction feeds `headroom.rs`, whose `weights` term is the on-disk size and
/// still counts the gate and up tensors that `prune_after_load` releases, so
/// the fused and pruned terms must cancel.
#[test]
fn the_gateup_fusion_is_residency_neutral() {
    let mut route = round6_route();
    route.ffn_gateup_fused = true;
    let p = predicted(&route);
    let base = predicted(&round6_route());

    // 2026-09-25: 2 x 17408 x 5120 = 178,257,920 B per layer, 64 layers.
    assert_eq!(p.ffn_gateup_fused, 64 * 178_257_920);
    assert_eq!(p.ffn_gateup_fused, 11_408_506_880, "11.4 GB, as briefed");
    assert_eq!(
        p.ffn_gateup_fused, p.ffn_gateup_pruned,
        "the fused weight IS the two store tensors copied side by side"
    );
    assert_eq!(
        p.total(),
        base.total(),
        "arming the fusion must not move the preflight yardstick by one byte"
    );
    assert!(p.twins.ffn_gateup_fused, "but the log still names it");
    assert!(!base.twins.ffn_gateup_fused);
}

/// 2026-09-25: An `intermediate_size` that is not a multiple of 128 is not
/// priced for the fusion, because `qwen35_dense::ffn_gateup_fused_selected`
/// declines it.
#[test]
fn a_width_that_is_not_a_whole_block_grid_is_never_priced() {
    let mut c = qwen38_27b();
    c.intermediate_size = 17408 - 64;
    let mut route = round6_route();
    route.ffn_gateup_fused = true;
    let p = match predicted_derived_bytes(&c, &route) {
        DerivedBytesEstimate::NativeFp8Dense(p) => p,
        DerivedBytesEstimate::Unavailable(why) => panic!("expected a prediction, got: {why}"),
    };
    assert_eq!(p.ffn_gateup_fused, 0);
    assert_eq!(p.ffn_gateup_pruned, 0);
    assert!(!p.twins.ffn_gateup_fused);
}

/// 2026-09-25: Without both W8A8 prefill kernels, the Q and O twins are
/// priced too.
#[test]
fn a_target_without_the_w8a8_prefill_kernels_pays_for_the_q_and_o_twins() {
    let mut route = round6_route();
    route.w8a8_prefill_kernels = false;
    let p = predicted(&route);
    assert!(
        p.attn_twin_set.q && p.attn_twin_set.o,
        "{:?}",
        p.attn_twin_set
    );
    let q_twin = fp8_residency::fp8_twin_bytes(24 * 256 * 2, 5120) as u64;
    let o_twin = fp8_residency::fp8_twin_bytes(5120, 24 * 256) as u64;
    assert_eq!(
        p.attn_fp8_twins,
        predicted(&round6_route()).attn_fp8_twins + 16 * (q_twin + o_twin),
    );
    assert!(p.total() > predicted(&round6_route()).total());
}

/// 2026-09-25: `METRALE_ATTN_PREFILL_Q_T=1` adds the Q twin, which
/// `prefill/cache_skip_qkv.rs` then reads.
#[test]
fn the_q_transpose_lever_adds_exactly_the_q_twin() {
    let mut route = round6_route();
    route.route.attn_prefill_q_t = true;
    let p = predicted(&route);
    assert!(p.attn_twin_set.q && !p.attn_twin_set.o);
    let q_twin = fp8_residency::fp8_twin_bytes(24 * 256 * 2, 5120) as u64;
    assert_eq!(
        p.attn_fp8_twins,
        predicted(&round6_route()).attn_fp8_twins + 16 * q_twin,
    );
}

/// 2026-09-25: `METRALE_NO_GDN_FP8` takes the GDN layers off the fused-concat
/// arm, so the SSM term is zero.
#[test]
fn disabling_the_gdn_fp8_arm_drops_the_fused_concat_term() {
    let mut route = round6_route();
    route.gdn_fp8 = false;
    let p = predicted(&route);
    assert_eq!(p.ssm_fp8_concat, 0);
    assert!(!p.twins.ssm_fp8_concat);
    assert_eq!(p.total(), 167_813_120);
}

/// 2026-09-25: Each gate that returns `Unavailable`, with its reason string.
#[test]
fn the_prediction_declines_rather_than_guessing() {
    let cases: Vec<(&str, Box<dyn Fn(&mut Fp8RouteInputs)>, &str)> = vec![
        (
            "flag off",
            Box::new(|r: &mut Fp8RouteInputs| r.dense_fp8 = false),
            "METRALE_DENSE_FP8 is not 1",
        ),
        (
            "tp > 1",
            Box::new(|r: &mut Fp8RouteInputs| r.tp_size = 2),
            "--tp-size > 1 takes the NVFP4 route",
        ),
        (
            "config declares nothing",
            Box::new(|r: &mut Fp8RouteInputs| r.declared_variant = None),
            "config.json does not declare a block-scaled FP8 checkpoint",
        ),
        (
            "config declares NVFP4",
            Box::new(|r: &mut Fp8RouteInputs| {
                r.declared_variant = Some(Nvfp4Variant::CompressedTensors)
            }),
            "config.json does not declare a block-scaled FP8 checkpoint",
        ),
        (
            "keep-nvfp4 escape hatch",
            Box::new(|r: &mut Fp8RouteInputs| r.route.keep_nvfp4 = true),
            "METRALE_DENSE_FP8_KEEP_NVFP4 restores the pre-#915 fallback copies",
        ),
        (
            "W4A4 o_proj lever",
            Box::new(|r: &mut Fp8RouteInputs| r.route.attn_w4a4 = true),
            "an NVFP4 fallback lever (METRALE_CUTLASS_NVFP4_* / METRALE_ATTN_W4A4) is set",
        ),
    ];
    for (name, mutate, want) in cases {
        let mut route = round6_route();
        mutate(&mut route);
        assert_eq!(
            predicted_derived_bytes(&qwen38_27b(), &route),
            DerivedBytesEstimate::Unavailable(want),
            "case: {name}",
        );
    }
}

/// 2026-09-25: A config that is not `is_qwen35_dense()` is declined first.
#[test]
fn another_architecture_is_declined_on_the_loader_gate() {
    let other = ModelConfig::qwen3_next_80b_nvfp4();
    assert_eq!(
        predicted_derived_bytes(&other, &round6_route()),
        DerivedBytesEstimate::Unavailable("not the Qwen3.5-dense loader"),
    );
}

/// 2026-09-25: `bytes()` and `reason()` each return `None` for the other
/// variant.
#[test]
fn the_estimate_reports_either_bytes_or_a_reason_never_both() {
    let ok = predicted_derived_bytes(&qwen38_27b(), &round6_route());
    assert_eq!(ok.bytes(), Some(4_242_882_560));
    assert_eq!(ok.reason(), None);
    let no = DerivedBytesEstimate::Unavailable("because");
    assert_eq!(no.bytes(), None);
    assert_eq!(no.reason(), Some("because"));
}
