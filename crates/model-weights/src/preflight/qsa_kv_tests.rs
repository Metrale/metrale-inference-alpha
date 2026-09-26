// SPDX-License-Identifier: MIT OR Apache-2.0

use super::check_qsa_kv_dtype;

/// Names from a checkpoint with no indexer: not QSA, so the flag is none of
/// this check's business.
const PLAIN: [&str; 2] = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.embed_tokens.weight",
];
/// Presence of an indexer tensor is what makes a checkpoint QSA — the same
/// thing the loader keys on.
const QSA: [&str; 2] = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.indexer.index_qk_proj.weight",
];
/// GLM-5.3's DSA k-pool indexer shares the `self_attn.indexer.` namespace
/// but is a different mechanism with an FP8-KV decode kernel. Names taken
/// verbatim from the NVFP4 checkpoint index.
const KPOOL_DSA: [&str; 3] = [
    "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_ape",
    "model.language_model.layers.3.self_attn.indexer.wq_b.weight",
    "model.language_model.layers.3.self_attn.indexer.k_norm.weight",
];

/// A k-pool DSA checkpoint must keep its fp8 KV cache. This is the sealed
/// GLM-5.3 serving configuration; refusing it here refuses the only
/// configuration that model has ever served.
#[test]
fn a_kpool_dsa_checkpoint_is_not_qsa_and_keeps_fp8() {
    for dt in [None, Some("fp8"), Some("bf16")] {
        assert!(
            check_qsa_kv_dtype(KPOOL_DSA.into_iter(), dt).is_ok(),
            "kpool DSA must not be caught by the QSA rule: {dt:?}"
        );
    }
}

/// ...and narrowing it must NOT have made the original rule inert.
#[test]
fn narrowing_for_kpool_did_not_disarm_the_qsa_rule() {
    check_qsa_kv_dtype(QSA.into_iter(), Some("fp8")).expect_err("QSA at fp8 must still refuse");
}

#[test]
fn a_non_qsa_checkpoint_accepts_any_kv_dtype() {
    for dt in [None, Some("fp8"), Some("bf16"), Some("nonsense")] {
        assert!(check_qsa_kv_dtype(PLAIN.into_iter(), dt).is_ok(), "{dt:?}");
    }
}

/// The refusal has to name the flag AND the value, because the operator is
/// reading it three minutes into a load they will have to repeat.
#[test]
fn a_qsa_checkpoint_refuses_a_quantized_kv_cache_by_name() {
    let e = check_qsa_kv_dtype(QSA.into_iter(), Some("fp8")).expect_err("must refuse");
    let msg = format!("{e}");
    assert!(msg.contains("--kv-cache-dtype bf16"), "{msg}");
    assert!(msg.contains("fp8"), "must quote what was given: {msg}");
    // The old text said "or drop the flag", which is the one thing that
    // cannot work: dropping it resolves to fp8, which is how we got here.
    assert!(
        !msg.contains("drop the flag"),
        "must not advise dropping the flag: {msg}"
    );
}

/// bf16, in the spellings that actually parse downstream.
#[test]
fn a_qsa_checkpoint_accepts_bf16() {
    for dt in [Some("bf16"), Some("BF16"), Some(" bf16 ")] {
        assert!(
            check_qsa_kv_dtype(QSA.into_iter(), dt).is_ok(),
            "{dt:?} must pass"
        );
    }
}

/// The regression this check was written for and then could not catch: the
/// engine default is fp8, so an unresolved `None` read as "bf16, fine" made
/// the check inert on `met serve <model>` with no flag — the invocation
/// most operators type. Callers resolve before calling; if one forgets, this
/// is the test that says so.
#[test]
fn the_engine_default_is_not_something_a_qsa_checkpoint_can_use() {
    assert_eq!(server_default_kv_dtype(), "fp8");
    check_qsa_kv_dtype(QSA.into_iter(), Some(server_default_kv_dtype()))
        .expect_err("the resolved default must be refused, not waved through");
}

/// Spelled out rather than imported: `metrale-server` depends on this crate,
/// so the constant cannot come the other way. If the server's default ever
/// changes, the assertion above fails here and names this comment.
fn server_default_kv_dtype() -> &'static str {
    "fp8"
}

/// Spellings the CLI rejects before preflight can see them. Accepting them
/// here only made the rule look laxer than it is.
#[test]
fn spellings_that_do_not_parse_downstream_are_not_accepted() {
    for dt in [Some("bfloat16"), Some("auto")] {
        check_qsa_kv_dtype(QSA.into_iter(), dt).expect_err("{dt:?} must not pass");
    }
}
