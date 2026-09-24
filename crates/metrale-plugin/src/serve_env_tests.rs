// SPDX-License-Identifier: AGPL-3.0-only

//! The lever contract behind #1242: what is a lever, what a declaration
//! means, and what the harness may inherit (nothing it did not declare).

use super::*;

fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn a_lever_is_any_metrale_name_that_is_not_a_harness_variable() {
    assert!(is_lever("METRALE_FP8_ROWWISE"));
    assert!(is_lever("METRALE_MTP_K_LADDER"));
    assert!(is_lever("METRALE_PREFILL_CODISPATCH"));
    // A read the server grows tomorrow is a lever by default.
    assert!(is_lever("METRALE_SOMETHING_NEW"));
    for harness in HARNESS_VARS {
        assert!(!is_lever(harness), "{harness} is the box's, not a lever");
    }
    // A sibling namespace is not a lever.
    assert!(!is_lever("METRALECTL_CONFIG_DIR"));
    assert!(!is_lever("PATH"));
    assert!(!is_lever("CUDARC_CUDA_VERSION"));
}

#[test]
fn levers_filters_an_environment_and_keeps_only_what_the_server_reads() {
    let env = [
        ("PATH", "/usr/bin"),
        ("METRALE_HOME", "/workspace/.metrale"),
        ("METRALE_FP8_ROWWISE", "1"),
        ("CUDARC_CUDA_VERSION", "13000"),
        ("METRALE_MTP_DCUT_RATIO", "1.0"),
    ];
    assert_eq!(
        levers(env),
        map(&[
            ("METRALE_FP8_ROWWISE", "1"),
            ("METRALE_MTP_DCUT_RATIO", "1.0")
        ])
    );
}

/// The fingerprint is a function of the SET: order-free, value-sensitive,
/// and unable to confuse `A=1 B=2` with any other spelling.
#[test]
fn the_fingerprint_separates_sets_that_differ_and_nothing_else() {
    let dense = map(&[
        ("METRALE_FP8_ROWWISE", "1"),
        ("METRALE_MTP_DCUT_RATIO", "1.0"),
        ("METRALE_MTP_K_LADDER", "1:3,2:1,4:2,8:2,16:1"),
    ]);
    let same = map(&[
        ("METRALE_MTP_K_LADDER", "1:3,2:1,4:2,8:2,16:1"),
        ("METRALE_MTP_DCUT_RATIO", "1.0"),
        ("METRALE_FP8_ROWWISE", "1"),
    ]);
    assert_eq!(fingerprint(&dense), fingerprint(&same));
    assert_eq!(fingerprint(&dense).len(), 64);
    let mut off = dense.clone();
    off.insert("METRALE_FP8_ROWWISE".into(), "0".into());
    assert_ne!(fingerprint(&dense), fingerprint(&off), "a value moved");
    let mut fewer = dense.clone();
    fewer.remove("METRALE_MTP_K_LADDER");
    assert_ne!(fingerprint(&dense), fingerprint(&fewer), "a lever dropped");
    assert_ne!(
        fingerprint(&BTreeMap::new()),
        fingerprint(&map(&[("METRALE_X", "")])),
        "an empty value is not an absent lever"
    );
    assert_ne!(
        fingerprint(&map(&[("METRALE_A", "1B=2")])),
        fingerprint(&map(&[("METRALE_A", "1"), ("B", "2")])),
    );
}

#[test]
fn a_declaration_is_validated() {
    let ok = declared(
        "recipe q/x",
        &map(&[
            ("METRALE_FP8_ROWWISE", "1"),
            ("METRALE_MTP_DCUT_RATIO", "1.0"),
        ]),
    )
    .unwrap();
    assert_eq!(
        ok,
        map(&[
            ("METRALE_FP8_ROWWISE", "1"),
            ("METRALE_MTP_DCUT_RATIO", "1.0")
        ])
    );

    for (bad, why) in [
        (("CUDA_VISIBLE_DEVICES", "0"), "not a METRALE_* serve lever"),
        (("METRALE_HOME", "/tmp/x"), "harness variable"),
        (("METRALE_FP8_ROWWISE", "  "), "empty value"),
    ] {
        let e = declared("recipe q/x", &map(&[bad]))
            .unwrap_err()
            .to_string();
        assert!(e.contains(why), "{bad:?}: {e}");
        assert!(e.contains(bad.0), "names the key: {e}");
    }
}

#[test]
fn the_gate_pin_wins_over_the_recipe_on_a_clash_and_adds_otherwise() {
    let merged = merge_declared(
        map(&[("METRALE_A", "recipe"), ("METRALE_B", "recipe")]),
        map(&[("METRALE_B", "pin"), ("METRALE_C", "pin")]),
    );
    assert_eq!(
        merged,
        map(&[
            ("METRALE_A", "recipe"),
            ("METRALE_B", "pin"),
            ("METRALE_C", "pin")
        ])
    );
}

/// The #1242 case, verbatim: a node's bench.yaml exported the dense
/// concurrency levers, and `bfcl-subset`'s recipe declares none of them.
#[test]
fn an_undeclared_lever_in_the_harness_is_refused_by_name() {
    let node = map(&[
        ("METRALE_FP8_ROWWISE", "1"),
        ("METRALE_MTP_DCUT_RATIO", "1.0"),
        ("METRALE_MTP_K_LADDER", "1:3,2:1,4:2,8:2,16:1"),
    ]);
    let e = reconcile(
        "recipe qwen3.8/qwen3.8-27b-nvfp4-unsloth-bfcl",
        &BTreeMap::new(),
        &node,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("3 serve lever(s)"), "{e}");
    assert!(e.contains("METRALE_FP8_ROWWISE=1"), "{e}");
    assert!(
        e.contains("METRALE_MTP_K_LADDER=1:3,2:1,4:2,8:2,16:1"),
        "{e}"
    );
    assert!(
        e.contains("recipe qwen3.8/qwen3.8-27b-nvfp4-unsloth-bfcl"),
        "names the recipe: {e}"
    );
    assert!(e.contains("bench.yaml"), "names the remedy: {e}");
    // One stray lever beside a fully declared set is still refused.
    let declared = map(&[("METRALE_FP8_ROWWISE", "1")]);
    let mut present = declared.clone();
    present.insert("METRALE_PREFILL_CODISPATCH".into(), "1".into());
    let e = reconcile("recipe r", &declared, &present)
        .unwrap_err()
        .to_string();
    assert!(e.contains("1 serve lever(s)"), "{e}");
    // The BLAMED list is what is asserted, not the whole message: its fixed
    // prose cites METRALE_FP8_ROWWISE=1 as the #1242 example whatever was
    // refused, so a whole-message `!contains` would fail on the example.
    let blamed = e
        .split("does not declare: ")
        .nth(1)
        .and_then(|rest| rest.split(". A lever").next())
        .unwrap_or_else(|| panic!("the refusal names what it refuses: {e}"));
    assert_eq!(
        blamed, "METRALE_PREFILL_CODISPATCH=1",
        "only the undeclared lever is blamed, never the declared one: {e}"
    );
}

/// A declared lever is APPLIED: absent from the harness it is what the
/// child must be given; present at the declared value there is nothing to
/// give; present at another value it is a contradiction, not a silent win.
#[test]
fn a_declared_lever_is_applied_and_a_contradicted_one_is_refused() {
    let declared = map(&[
        ("METRALE_FP8_ROWWISE", "1"),
        ("METRALE_MTP_DCUT_RATIO", "1.0"),
    ]);
    let fresh = reconcile("recipe r", &declared, &BTreeMap::new()).unwrap();
    assert_eq!(
        fresh.env, declared,
        "the server runs under the whole declaration"
    );
    assert_eq!(fresh.missing, declared, "and none of it was already there");

    let half = reconcile("recipe r", &declared, &map(&[("METRALE_FP8_ROWWISE", "1")])).unwrap();
    assert_eq!(half.env, declared);
    assert_eq!(half.missing, map(&[("METRALE_MTP_DCUT_RATIO", "1.0")]));

    let all = reconcile("recipe r", &declared, &declared).unwrap();
    assert!(all.missing.is_empty(), "nothing to apply, nothing refused");

    let e = reconcile(
        "recipe r",
        &declared,
        &map(&[
            ("METRALE_FP8_ROWWISE", "0"),
            ("METRALE_MTP_DCUT_RATIO", "1.0"),
        ]),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("contradicts recipe r"), "{e}");
    assert!(
        e.contains("METRALE_FP8_ROWWISE: harness \"0\", recipe r declares \"1\""),
        "{e}"
    );
    assert!(
        !e.contains("METRALE_MTP_DCUT_RATIO"),
        "the agreeing one is not blamed: {e}"
    );
}

/// A harness with no levers and a recipe declaring none is the common case
/// and must cost nothing: no refusal, nothing to apply, the empty digest.
#[test]
fn no_levers_anywhere_reconciles_to_nothing() {
    let r = reconcile("recipe r", &BTreeMap::new(), &BTreeMap::new()).unwrap();
    assert!(r.env.is_empty() && r.missing.is_empty());
    assert_eq!(fingerprint(&r.env), fingerprint(&BTreeMap::new()));
}
