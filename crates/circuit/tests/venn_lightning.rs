// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The Nemotron-3.5-Lightning report classifies every opportunity the `new-model`
//! method expects against Qwen3.6-35B-A3B and Qwen3.8-27B: attention head_dim, BF16
//! projections at many rows, FP8 per-tensor scales, the tensor-core grouped expert kernel,
//! routing scoring, the Mamba2 kernels, and the missing multi-row paths as the top flags.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;
mod venn_common;

use std::collections::BTreeSet;

use metrale_circuit::Mode;
use metrale_circuit::venn::{Class, ParamKind};
use venn_common::*;

#[test]
fn lightning_attention_head_dim_is_a_compile_time_parameter() {
    let r = report();
    for (mode, rows) in [
        (Mode::Decode, 1),
        (Mode::MultiSeq, 16),
        (Mode::MultiSeq, 128),
    ] {
        let f = row(&r, mode, rows, "attn.attend")
            .primary
            .as_ref()
            .expect("classified");
        assert_eq!(f.family, "paged_decode_attn");
        assert_eq!(f.class, Class::ParameterizationOpportunity);
        assert_eq!(
            diff(f, "head_dim"),
            ("128".into(), "256".into(), ParamKind::Compile)
        );
        assert_eq!(
            diff(f, "kv_dtype"),
            ("fp8".into(), "bf16".into(), ParamKind::Policy)
        );
    }
}

#[test]
fn lightning_bf16_projections_at_many_rows_are_a_wxay_w16a16_policy_variant() {
    let r = report();
    for rows in [16, 128] {
        for site in ["attn.q", "attn.k", "attn.v", "attn.o"] {
            let f = finding(row(&r, Mode::MultiSeq, rows, site), "wxay");
            assert_eq!(f.class, Class::PolicyVariant, "{site} n={rows}");
            assert_eq!(diff(f, "weight").0, "bf16");
            assert_eq!(diff(f, "activation").0, "bf16");
        }
    }
}

#[test]
fn lightning_fp8_per_tensor_scales_are_wxay_policy_variants() {
    let r = report();
    for site in ["mamba.in_proj", "mamba.out_proj"] {
        // 2026-10-02: The tensor-core row tiles (`tc_rows`, W8A16 on a block-128 grid) are now the
        // primary match: a per-tensor scale is their weight-scale policy (the loader repeats it
        // over the grid). The W8A8 family remains the activation-policy match.
        let p = row(&r, Mode::Decode, 1, site)
            .primary
            .as_ref()
            .expect("classified");
        assert_eq!(
            (p.family.as_str(), p.class),
            ("tc_rows", Class::PolicyVariant),
            "{site}"
        );
        assert_eq!(
            diff(p, "weight"),
            (
                "fp8/tensor".into(),
                "fp8/block128x128".into(),
                ParamKind::Policy
            )
        );
        let f = finding(row(&r, Mode::Decode, 1, site), "wxay");
        assert_eq!(f.class, Class::PolicyVariant, "{site}");
        assert_eq!(
            diff(f, "weight"),
            ("fp8/tensor".into(), "fp8/channel".into(), ParamKind::Policy)
        );
        assert_eq!(
            diff(f, "activation"),
            ("fp8/tensor".into(), "fp8/token".into(), ParamKind::Policy)
        );
    }
    // 2026-09-29: The static per-tensor quantizer: no family computes it; the dynamic ones are
    // the policy it would join.
    let q = row(&r, Mode::Decode, 1, "mamba.in_proj_quant")
        .primary
        .as_ref()
        .expect("classified");
    assert_eq!(q.class, Class::PolicyVariant);
    assert_eq!(diff(q, "format").0, "fp8/tensor");
}

#[test]
fn lightning_experts_are_a_format_and_activation_policy_of_the_tc_grouped_kernel() {
    let r = report();
    for rows in [16, 2] {
        let mode = if rows == 2 {
            Mode::Verify
        } else {
            Mode::MultiSeq
        };
        // 2026-10-02: The family now has an NVFP4 g16 point (moe_nvfp4_grouped_tc.cu), so the
        // weight format no longer differs; only the epilogue (ReLU² against SiLU·mul) does.
        let f = finding(row(&r, mode, rows, "moe.experts_up"), "moe_grouped_tc");
        assert_eq!(f.class, Class::PolicyVariant);
        assert!(f.diffs.iter().all(|d| d.param != "weight"), "{:?}", f.diffs);
        assert_eq!(
            diff(f, "epilogue"),
            ("relu2".into(), "silu_mul".into(), ParamKind::Policy)
        );
        let down = finding(row(&r, mode, rows, "moe.experts_down"), "moe_grouped_tc");
        assert!(
            down.diffs.iter().all(|d| d.param != "weight"),
            "{:?}",
            down.diffs
        );
    }
}

#[test]
fn lightning_routing_is_a_scoring_policy_variant() {
    let r = report();
    let f = row(&r, Mode::Decode, 1, "moe.top_k")
        .primary
        .as_ref()
        .expect("classified");
    assert_eq!(
        (f.family.as_str(), f.class),
        ("moe_topk", Class::PolicyVariant)
    );
    assert_eq!(
        diff(f, "scoring"),
        ("sigmoid_bias".into(), "softmax".into(), ParamKind::Policy)
    );
    // 2026-09-29: top-6 vs top-8 is a runtime argument and does not make it an opportunity.
    assert_eq!(diff(f, "top_k").2, ParamKind::Runtime);
}

#[test]
fn lightning_mamba2_kernels_are_reused_unmeasured_or_novel() {
    let r = report();
    for t in &r.tables {
        if t.run.mode == Mode::Draft {
            continue;
        }
        let ssm = row(&r, t.run.mode, t.run.rows, "mamba.ssm")
            .primary
            .as_ref()
            .expect("classified");
        assert_eq!(
            (ssm.family.as_str(), ssm.class),
            ("mamba2_ssm", Class::SharedUnmeasured)
        );
        let conv = row(&r, t.run.mode, t.run.rows, "mamba.conv")
            .primary
            .as_ref()
            .expect("classified");
        assert_eq!(
            (conv.family.as_str(), conv.class),
            ("causal_conv1d", Class::SharedUnmeasured)
        );
        // 2026-09-29: No kernel writes the Mamba2 per-row rollback snapshots.
        assert!(
            row(&r, t.run.mode, t.run.rows, "mamba.ssm_ckpt")
                .primary
                .is_none()
        );
        assert!(
            row(&r, t.run.mode, t.run.rows, "mamba.conv_ckpt")
                .primary
                .is_none()
        );
    }
}

#[test]
fn lightning_missing_multi_row_paths_are_the_top_flags() {
    let r = report();
    let top = &r.flags[0];
    assert!(top.source.starts_with("legacy: "), "{}", top.source);
    assert!(
        top.share > 1.0,
        "the worst fallback more than doubles the step: {}",
        top.share
    );
    let seen: BTreeSet<(String, &str)> = r
        .flags
        .iter()
        .map(|f| {
            (
                f.layer_kind.map_or("-", |k| k.name()).to_string(),
                f.run.mode.name(),
            )
        })
        .collect();
    for want in [
        ("mamba", "multi_seq"),
        ("mamba", "verify"),
        ("moe", "multi_seq"),
        ("moe", "verify"),
    ] {
        assert!(
            seen.contains(&(want.0.to_string(), want.1)),
            "no flag for {want:?}"
        );
    }
    assert!(r.flags.windows(2).all(|w| w[0].added_us >= w[1].added_us));
}
