// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of the split-K policy and the GQA-pack gate as pure
//! functions, with no GPU, environment or baked constant. The SM counts are
//! the `sm_count` of `kernels/hopper` (132) and `kernels/gb10` (48); the head
//! shape is `kernels/hopper/qwen3.8-27b/MODEL.toml` (24 q heads, 4 KV heads,
//! head dim 256).
//!
//! Owner: kernels crate (paged-decode attention policy).
//! Invariants: none beyond the types.

use super::*;

const NQ: u32 = 24;
const HOPPER_SMS: u32 = 132;
const GB10_SMS: u32 = 48;
/// 2026-09-25: The reference batch the legacy rule is fed: a max batch of 16.
const PIN: u32 = 16;

/// 2026-09-25: For 24 q heads and a max batch of 16 the legacy rule gives one
/// split with either SM count, 48 or 132; on 132 SMs that is 24 CTAs.
#[test]
fn the_legacy_rule_leaves_an_h100_at_twenty_four_ctas() {
    assert_eq!(legacy_splits(GB10_SMS, NQ, PIN), 1);
    // 2026-09-25: With the Hopper count it is still 1, because the reference
    // batch is the max batch (16), not the one sequence in flight.
    assert_eq!(legacy_splits(HOPPER_SMS, NQ, PIN), 1);
    assert!(NQ * legacy_splits(HOPPER_SMS, NQ, PIN) < HOPPER_SMS);
}

/// 2026-09-25: `clamp(ceil(2 * sm_count / num_q_heads), 1, 16)`.
#[test]
fn auto_fills_two_waves_at_the_single_stream_shape() {
    assert_eq!(auto_splits(HOPPER_SMS, NQ), 11);
    assert!(NQ * auto_splits(HOPPER_SMS, NQ) >= SPLITK_TARGET_WAVES * HOPPER_SMS);
    // 2026-09-25: GB10 declares `legacy`; this checks the formula only.
    assert_eq!(auto_splits(GB10_SMS, NQ), 4);
    assert_eq!(auto_splits(148, NQ), 13);
    assert_eq!(auto_splits(HOPPER_SMS, 512), 1);
    assert_eq!(auto_splits(HOPPER_SMS, 1), MAX_DECODE_SPLITS);
}

/// 2026-09-25: Under `auto` and a pin, the split count, and so the online-softmax
/// reduction tree, does not change with the co-batched sequence count.
#[test]
fn the_auto_split_count_does_not_move_with_the_co_batched_count() {
    let at = |n: u32| num_splits(SplitkPolicy::Auto, HOPPER_SMS, NQ, n);
    for n in [1u32, 4, 15, 16] {
        assert_eq!(
            at(n),
            at(1),
            "num_splits moved at num_seqs={n}: the reduction tree is not fixed \
             per serve, which is the determinism defect this policy must not \
             re-create"
        );
    }
    assert_eq!(at(1), 11);
    let pinned = |n: u32| num_splits(SplitkPolicy::Pinned(6), HOPPER_SMS, NQ, n);
    for n in [1u32, 4, 15, 16] {
        assert_eq!(pinned(n), 6, "a pinned split count is a constant");
    }
}

/// 2026-09-25: `legacy` does depend on its reference batch; the dispatch
/// passes `split_ref_seqs(num_seqs, max_decode_seqs)`, the larger of the two.
#[test]
fn legacy_is_the_old_rule_including_its_reference_batch() {
    assert_eq!(legacy_splits(GB10_SMS, 8, 1), 6);
    assert_eq!(legacy_splits(GB10_SMS, 8, 6), 1);
    assert_eq!(
        num_splits(SplitkPolicy::Legacy, GB10_SMS, 8, 6),
        legacy_splits(GB10_SMS, 8, 6)
    );
}

/// 2026-09-25: The policy grammar, including spellings that must parse to
/// `None`.
#[test]
fn the_grammar_parses_every_documented_spelling() {
    assert_eq!(parse("legacy"), Some(SplitkPolicy::Legacy));
    assert_eq!(parse(" AUTO "), Some(SplitkPolicy::Auto));
    assert_eq!(parse("4"), Some(SplitkPolicy::Pinned(4)));
    // 2026-09-25: `0`/`off` is one split, which is no split-K.
    assert_eq!(parse("0"), Some(SplitkPolicy::Pinned(1)));
    assert_eq!(parse("off"), Some(SplitkPolicy::Pinned(1)));
    // 2026-09-25: An over-large pin is clamped to the cap the workspace is
    // sized for, not rejected.
    assert_eq!(parse("999"), Some(SplitkPolicy::Pinned(MAX_DECODE_SPLITS)));
    assert_eq!(parse("aut0"), None);
    assert_eq!(parse(""), None);
    for p in [
        SplitkPolicy::Legacy,
        SplitkPolicy::Auto,
        SplitkPolicy::Pinned(6),
    ] {
        assert_eq!(parse(&p.label()), Some(p), "label must round-trip");
    }
}

/// 2026-09-25: A parseable environment value overrides the declaration and is
/// reported as such.
#[test]
fn the_environment_overrides_the_declaration_and_says_so() {
    assert_eq!(resolve_policy("auto", None), (SplitkPolicy::Auto, false));
    assert_eq!(
        resolve_policy("auto", Some("0")),
        (SplitkPolicy::Pinned(1), true),
        "the A/B control an H100 round runs against the new default"
    );
    assert_eq!(
        resolve_policy("legacy", Some("auto")),
        (SplitkPolicy::Auto, true)
    );
    assert_eq!(
        resolve_policy("auto", Some("yes-please")),
        (SplitkPolicy::Auto, false)
    );
    // 2026-09-25: An unparseable declaration resolves to `legacy`.
    assert_eq!(resolve_policy("", None), (SplitkPolicy::Legacy, false));
}

/// 2026-09-25: The kernel addresses `((seq * heads) + head) * num_splits +
/// split`, so the workspace must cover every slot a split launch can reach,
/// under every policy, or the kernel writes out of bounds with no error.
#[test]
fn the_workspace_covers_every_slot_the_grid_can_address() {
    // 2026-09-25: `DecodeMetaLayout::rows()` is `max(32, max_batch_size)`, the
    // widest batch the decode metadata accepts.
    let rows = 32u32;
    for policy in [
        SplitkPolicy::Legacy,
        SplitkPolicy::Auto,
        SplitkPolicy::Pinned(MAX_DECODE_SPLITS),
    ] {
        let slots = workspace_slots(policy, HOPPER_SMS, NQ, rows, PIN);
        for num_seqs in [1u32, 4, 16, rows] {
            let ref_seqs = num_seqs.max(PIN);
            let splits = num_splits(policy, HOPPER_SMS, NQ, ref_seqs);
            // 2026-09-25: At one split the dispatch runs the non-split kernel,
            // which does not touch the workspace.
            if splits == 1 {
                continue;
            }
            let used = num_seqs * NQ * splits;
            assert!(
                used <= slots,
                "{policy:?} at num_seqs={num_seqs}: grid addresses {used} slots, \
                 arena holds {slots}"
            );
        }
    }
    // 2026-09-25: `legacy` sizes `sm_count` slots.
    assert_eq!(
        workspace_slots(SplitkPolicy::Legacy, GB10_SMS, NQ, rows, PIN),
        GB10_SMS
    );
    assert_eq!(
        workspace_slots(SplitkPolicy::Auto, HOPPER_SMS, NQ, rows, PIN),
        32 * 24 * 11
    );
}

const NKV: u32 = 4;
const HD: u32 = 256;

/// 2026-09-25: The text of a file in `kernels/gb10/common/`.
fn kernel_src(name: &str) -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/common")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// 2026-09-25: The integer value of `#define <name> <int>` in a CUDA source.
fn cuda_define(src: &str, name: &str) -> u32 {
    let needle = format!("\n#define {name} ");
    let at = src
        .find(&needle)
        .unwrap_or_else(|| panic!("no `#define {name}` in source"));
    let rest = &src[at + needle.len()..];
    let end = rest.find('\n').unwrap_or(rest.len());
    rest[..end]
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("`#define {name} {}`: {e}", &rest[..end]))
}

/// 2026-09-25: The text of a `__device__ __forceinline__ void` helper, from its
/// signature through its closing brace at column 0.
fn device_helper(src: &str, name: &str) -> String {
    let needle = format!("__device__ __forceinline__ void {name}(");
    let at = src
        .find(&needle)
        .unwrap_or_else(|| panic!("no helper `{name}` in source"));
    let rest = &src[at..];
    let end = rest
        .find("\n}\n")
        .unwrap_or_else(|| panic!("helper `{name}` has no closing brace at column 0"));
    rest[..end + 3].to_string()
}

/// 2026-09-25: The packed kernel takes its query heads from `kv_head * PD_GQA`,
/// so any shape other than the compiled one would read and write the wrong
/// heads with output of the right size. The gate must refuse every such shape.
#[test]
fn the_pack_gate_admits_only_the_shape_the_kernel_is_compiled_for() {
    assert!(gqa_pack_shape_ok(NQ, NKV, HD));
    assert_eq!(NQ / NKV, DECODE_GQA_PACK_WIDTH);

    assert!(!gqa_pack_shape_ok(32, NKV, HD), "gqa 8 must be refused");
    assert!(!gqa_pack_shape_ok(16, NKV, HD), "gqa 4 must be refused");
    assert!(!gqa_pack_shape_ok(NQ, NQ, HD), "MHA must be refused");
    assert!(!gqa_pack_shape_ok(NQ, 1, HD), "MQA must be refused");
    // 2026-09-25: 25 / 4 is 6 in integer division, but 25 != 4 * 6.
    assert!(
        !gqa_pack_shape_ok(25, NKV, HD),
        "a non-exact ratio must be refused, not truncated"
    );
    assert!(!gqa_pack_shape_ok(NQ, NKV, 128), "hd 128 must be refused");
    assert!(!gqa_pack_shape_ok(NQ, NKV, 512), "hd 512 must be refused");
    assert!(!gqa_pack_shape_ok(0, 0, HD), "nkv 0 must be refused");
}

/// 2026-09-25: The packed kernels are declared off, and an unparseable
/// `METRALE_ATTN_DECODE_GQA_PACK` value keeps the declaration either way.
#[test]
fn the_pack_lever_is_declared_off_and_a_typo_keeps_the_declaration() {
    const { assert!(!DECODE_GQA_PACK_DECLARED) };
    assert!(!resolve_gqa_pack(DECODE_GQA_PACK_DECLARED, None));
    for on in ["1", "on", "true", "YES", " on "] {
        assert!(resolve_gqa_pack(false, Some(on)), "{on:?} must arm");
    }
    for off in ["0", "off", "false", "NO"] {
        assert!(!resolve_gqa_pack(true, Some(off)), "{off:?} must disarm");
    }
    for junk in ["auto", "2", "", "yes please"] {
        assert!(
            resolve_gqa_pack(true, Some(junk)),
            "{junk:?} must keep the declaration, not disarm"
        );
        assert!(
            !resolve_gqa_pack(false, Some(junk)),
            "{junk:?} must keep the declaration, not arm"
        );
    }
}

/// 2026-09-25: The pack width and head dim are defined in this crate and in
/// both packed CUDA sources; the CUDA copies size the register arrays, so the
/// three must agree.
#[test]
fn cuda_sources_declare_the_pack_width_rust_dispatches_on() {
    for name in [
        "paged_decode_attn_fp8_gqa.cu",
        "paged_decode_attn_bf16_gqa.cu",
    ] {
        let src = kernel_src(name);
        assert_eq!(
            cuda_define(&src, "PD_GQA"),
            DECODE_GQA_PACK_WIDTH,
            "{name}: PD_GQA must equal DECODE_GQA_PACK_WIDTH"
        );
        assert_eq!(
            cuda_define(&src, "PD_HDIM"),
            DECODE_GQA_PACK_HEAD_DIM,
            "{name}: PD_HDIM must equal DECODE_GQA_PACK_HEAD_DIM"
        );
        // 2026-09-25: Neither may sit under `#ifndef`, or a KERNEL.toml `-D`
        // could reshape the kernel without the host gate changing. Several
        // KERNEL.toml files pass `-DHDIM=`, which is why the name is PD_HDIM.
        assert!(
            !src.contains("#ifndef PD_GQA"),
            "{name}: PD_GQA must not be overridable from the build"
        );
        assert!(
            !src.contains("#ifndef PD_HDIM"),
            "{name}: PD_HDIM must not be overridable from the build"
        );
        assert!(
            !src.contains("#define HDIM"),
            "{name}: must not define HDIM — model builds pass -DHDIM on the \
             command line and nvcc treats the redefinition as an error"
        );
    }
}

/// 2026-09-25: The packed kernels carry their own copies of the unpacked
/// kernels' unpack helpers; this test keeps each copy identical to its source.
#[test]
fn gqa_kernels_copy_the_unpack_helpers_verbatim() {
    let bf16 = kernel_src("paged_decode_attn.cu");
    let bf16_gqa = kernel_src("paged_decode_attn_bf16_gqa.cu");
    assert_eq!(
        device_helper(&bf16, "unpack2_pd"),
        device_helper(&bf16_gqa, "unpack2_pd"),
        "unpack2_pd drifted between paged_decode_attn.cu and its packed twin"
    );

    let fp8 = kernel_src("paged_decode_attn_fp8.cu");
    let fp8_gqa = kernel_src("paged_decode_attn_fp8_gqa.cu");
    for helper in ["unpack2_bf16", "unpack4_fp8_raw"] {
        assert_eq!(
            device_helper(&fp8, helper),
            device_helper(&fp8_gqa, helper),
            "{helper} drifted between paged_decode_attn_fp8.cu and its packed twin"
        );
    }
}

/// 2026-09-25: Packing divides the CTA count by the pack width; with 4 KV
/// heads on 48 SMs the packed grid covers the SMs only from 12 sequences.
#[test]
fn packing_divides_the_cta_count_and_moves_the_fill_threshold() {
    for seqs in [1u32, 2, 8, 16, 128] {
        assert_eq!(
            gqa_pack_ctas(NKV, seqs) * DECODE_GQA_PACK_WIDTH,
            NQ * seqs,
            "packed CTAs x pack width must equal the unpacked CTA count"
        );
    }
    assert!(gqa_pack_ctas(NKV, 8) < GB10_SMS);
    assert!(gqa_pack_ctas(NKV, 11) < GB10_SMS);
    assert!(gqa_pack_ctas(NKV, 12) >= GB10_SMS);
    // 2026-09-25: The unpacked grid covers 48 SMs from two sequences, where
    // `legacy_splits` gives one split.
    const { assert!(NQ * 2 >= GB10_SMS) };
    assert_eq!(legacy_splits(GB10_SMS, NQ, 2), 1);
}
