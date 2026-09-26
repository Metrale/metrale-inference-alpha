// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `Recipe` parsing and argv rendering over the vendored recipes.
//!
//! Owner: server (recipe).
//! Invariants: none beyond the types.

use super::*;

fn fixtures() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recipes")
}

/// 2026-09-26: Every vendored recipe, parsed with the id `family/stem` from its path.
fn all() -> Vec<Recipe> {
    let mut out = Vec::new();
    let mut stack = vec![fixtures()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("fixtures dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "yaml") {
                continue;
            }
            let family = path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let text = std::fs::read_to_string(&path).expect("read");
            out.push(
                Recipe::parse(format!("{family}/{stem}"), &text)
                    .unwrap_or_else(|e| panic!("{}: {e:#}", path.display())),
            );
        }
    }
    out
}

#[test]
fn the_whole_corpus_reads() {
    let all = all();
    assert_eq!(all.len(), 28);
    assert_eq!(all.iter().filter(|r| r.is_metrale()).count(), 26);
    assert_eq!(
        all.iter().filter(|r| r.version == "1").count(),
        2,
        "the two vLLM recipes"
    );
    for r in &all {
        assert!(!r.model.is_empty(), "{}: model", r.id);
        assert!(!r.container.is_empty(), "{}: container", r.id);
        assert!(!r.description.is_empty(), "{}: description", r.id);
        assert!(!r.defaults.is_empty(), "{}: defaults", r.id);
    }
}

#[test]
fn metadata_is_read_from_where_each_version_puts_it() {
    let all = all();
    let v2 = all.iter().find(|r| r.is_metrale()).expect("a v2 recipe");
    assert!(!v2.maintainer.is_empty(), "v2 metadata block");
    // 2026-09-26: A version 1 recipe has no metadata block; its description is
    // top-level.
    let v1 = all.iter().find(|r| r.version == "1").expect("a v1 recipe");
    assert!(!v1.description.is_empty(), "v1 top-level description");
    assert!(v1.maintainer.is_empty(), "v1 genuinely has no maintainer");
}

/// 2026-09-26: Every vendored `runtime: metrale` recipe renders to argv that clap
/// parses and `validate_serve_args` approves. This covers the vendored fixtures
/// only, not the live index.
#[test]
fn every_metrale_recipe_produces_a_valid_serve_config() {
    let no_overrides = BTreeMap::new();
    let mut checked = 0;
    for r in all().iter().filter(|r| r.is_metrale()) {
        r.serve_args(&no_overrides)
            .unwrap_or_else(|e| panic!("{}: {e:#}", r.id));
        checked += 1;
    }
    assert_eq!(checked, 26);
}

#[test]
fn a_multi_node_recipe_carries_its_world_size() {
    // 2026-09-26: The EP=2 recipes put `ep_size: 2` in defaults and `min_nodes: 2`
    // at the top level. Reading only `defaults` yields "--ep-size 2 exceeds
    // --world-size 1".
    let all = all();
    let ep = all
        .iter()
        .find(|r| r.defaults.contains_key("ep_size"))
        .expect("an EP recipe");
    assert_eq!(ep.min_nodes, 2);
    let argv = ep.argv(&BTreeMap::new()).expect("renders");
    let world = argv
        .iter()
        .position(|a| a == "--world-size")
        .expect("present");
    assert_eq!(argv[world + 1], "2");
    ep.serve_args(&BTreeMap::new()).expect("validates");
}

#[test]
fn a_single_node_recipe_does_not_pass_world_size() {
    let all = all();
    let solo = all
        .iter()
        .find(|r| r.is_metrale() && r.min_nodes == 1)
        .expect("a single-node recipe");
    let argv = solo.argv(&BTreeMap::new()).expect("renders");
    assert!(!argv.iter().any(|a| a == "--world-size"));
}

#[test]
fn an_override_replaces_rather_than_appends() {
    let all = all();
    let r = all
        .iter()
        .find(|r| r.is_metrale() && r.defaults.contains_key("max_model_len"))
        .expect("a recipe with a context length");
    let overrides = BTreeMap::from([("max_model_len".to_string(), "4096".to_string())]);
    let argv = r.argv(&overrides).expect("renders");
    let hits: Vec<usize> = argv
        .iter()
        .enumerate()
        .filter(|(_, a)| *a == "--max-seq-len")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(hits.len(), 1, "specified once, not twice: {argv:?}");
    assert_eq!(argv[hits[0] + 1], "4096");
    let args = r.serve_args(&overrides).expect("validates");
    assert_eq!(args.max_seq_len, 4096);
}

/// 2026-09-26: A key that is no flag is refused by clap, which names it, when
/// `serve_args` parses the rendered argv. `argv` itself accepts a key absent
/// from `defaults:`, as the next test adds one.
#[test]
fn an_unknown_override_is_refused_by_the_clap_round_trip() {
    let all = all();
    let r = all
        .iter()
        .find(|r| r.is_metrale())
        .expect("a metrale recipe");
    let overrides = BTreeMap::from([("nonsense".to_string(), "1".to_string())]);
    let err = format!("{:#}", r.serve_args(&overrides).expect_err("refused"));
    assert!(err.contains("nonsense"), "names the bad key: {err}");
}

/// 2026-09-26: A key the recipe does not list can be added by an override.
#[test]
fn a_setting_the_recipe_does_not_list_can_be_added() {
    let all = all();
    let r = all
        .iter()
        .find(|r| r.is_metrale() && !r.defaults.contains_key("fp8_kv_calibration_tokens"))
        .expect("a metrale recipe without the key");
    let overrides = BTreeMap::from([
        ("kv_cache_dtype".to_string(), "fp8".to_string()),
        ("fp8_kv_calibration_tokens".to_string(), "512".to_string()),
    ]);
    let args = r.serve_args(&overrides).expect("validates");
    assert_eq!(args.kv_cache_dtype.as_deref(), Some("fp8"));
    assert_eq!(args.fp8_kv_calibration_tokens, Some(512));
}

/// 2026-09-26: `NOT_FLAGS` is empty, so no key reaches the no-flag refusal in
/// `argv_edited`. This pins only that `port` and `kv_cache_dtype` still map to a
/// flag; nothing here exercises the refusal.
#[test]
fn an_addition_that_maps_to_no_flag_is_refused() {
    use crate::recipe::schema;
    for key in ["port", "kv_cache_dtype"] {
        assert!(
            schema::flag_for(key).is_some(),
            "{key} must still render, or the guard below changes meaning"
        );
    }
}

#[test]
fn a_vllm_recipe_is_readable_but_not_launchable() {
    // 2026-09-26: A recipe without `runtime: metrale` parses, so it can be listed,
    // but `argv` refuses it.
    let all = all();
    let v1 = all.iter().find(|r| !r.is_metrale()).expect("a vLLM recipe");
    assert!(!v1.model.is_empty(), "still readable for the list");
    let err = format!("{:#}", v1.argv(&BTreeMap::new()).expect_err("refused"));
    assert!(err.contains("runtime: metrale"), "{err}");
}

#[test]
fn an_updated_date_is_read_from_metadata() {
    let text = "\
recipe_version: \"2\"
model: org/model
container: metrale
metadata:
  updated: \"2026-08-01\"
defaults:
  max-batch-size: \"8\"
";
    let r = Recipe::parse("fam/stem", text).expect("parses");
    assert_eq!(r.updated, "2026-08-01");
}

#[test]
fn a_recipe_without_a_date_still_parses_and_reports_none() {
    let text = "\
recipe_version: \"2\"
model: org/model
container: metrale
metadata:
  maintainer: someone
defaults:
  max-batch-size: \"8\"
";
    let r = Recipe::parse("fam/stem", text).expect("parses without a date");
    assert!(r.updated.is_empty());
    assert_eq!(r.maintainer, "someone", "other metadata is unaffected");
}

#[test]
fn the_whole_vendored_corpus_still_parses_with_the_new_field() {
    // 2026-09-26: No vendored recipe carries `metadata.updated`.
    for r in all() {
        assert!(
            r.updated.is_empty(),
            "{} unexpectedly carries a date: {:?}",
            r.id,
            r.updated
        );
    }
}

/// 2026-09-26: Network test: the commit-date fallback resolves against the recipe
/// repository. Ignored by default so the suite stays offline.
#[test]
#[ignore = "network"]
fn the_commit_date_fallback_resolves_against_the_real_repo() {
    let d = super::fetch_github::commit_date("qwen3-coder-next/qwen3-coder-next-fp8")
        .expect("the recipe exists in the repo");
    assert_eq!(d.len(), 10, "YYYY-MM-DD, got {d:?}");
    assert!(d.starts_with("20"), "{d}");
}

/// 2026-09-26: Every vendored `runtime: metrale` recipe is servable under
/// `--hermetic`.
///
/// Validation refuses `--hermetic` beside a flag it closes, and a recipe's
/// defaults can set such a flag, so `hermetic::expand` must override it before
/// rendering. `serve_args` validates internally, so rendering is the whole check.
#[test]
fn every_recipe_can_be_served_hermetically() {
    // 2026-09-26: Only `runtime: metrale` recipes can be served at all; a
    // non-metrale one is refused before any flag is looked at, which is a
    // different rule.
    let recipes: Vec<Recipe> = all().into_iter().filter(|r| r.is_metrale()).collect();
    // 2026-09-26: Vacuity guard: the sweep must have read the fixtures.
    assert!(
        recipes.len() > 5,
        "only {} recipes found — the sweep is not reading the fixtures",
        recipes.len()
    );
    // 2026-09-26: Power guard: at least one recipe sets a key `--hermetic` closes,
    // or this test would pass with the expansion removed.
    let closed: Vec<&str> = crate::cli::hermetic::CLOSED_KEYS
        .iter()
        .map(|(k, _)| *k)
        .collect();
    let conflicting = recipes
        .iter()
        .filter(|r| {
            let argv = r.argv(&Default::default()).unwrap_or_default().join(" ");
            closed
                .iter()
                .any(|k| argv.contains(&format!("--{}", k.replace('_', "-"))))
        })
        .count();
    assert!(
        conflicting > 0,
        "no fixture recipe sets any of {closed:?}, so this test cannot see the \
         failure it exists for — add such a fixture rather than deleting this"
    );

    let overrides = crate::cli::hermetic::expand(
        [("hermetic".to_string(), "true".to_string())]
            .into_iter()
            .collect(),
    );
    for r in &recipes {
        r.serve_args(&overrides)
            .unwrap_or_else(|e| panic!("{} cannot be served hermetically: {e:#}", r.id));
    }
}

/// 2026-09-26: A recipe's `env:` block is read verbatim: absent and `{}` both mean
/// none, and a non-scalar entry or a non-mapping block is a parse error.
#[test]
fn the_env_block_is_read_verbatim_and_absent_means_none() {
    let head = "recipe_version: \"2\"\nmodel: org/m\ncontainer: c\nruntime: metrale\n\
                defaults:\n  port: 8888\n";
    assert!(Recipe::parse("f/s", head).unwrap().env.is_empty());
    assert!(
        Recipe::parse("f/s", &format!("{head}env: {{}}\n"))
            .unwrap()
            .env
            .is_empty()
    );
    let declared = Recipe::parse(
        "f/s",
        &format!(
            "{head}env:\n  METRALE_FP8_ROWWISE: \"1\"\n  METRALE_MTP_K_LADDER: 1:3,2:1,4:2,8:2,16:1\n"
        ),
    )
    .unwrap();
    assert_eq!(
        declared.env.iter().collect::<Vec<_>>(),
        [
            (&"METRALE_FP8_ROWWISE".to_string(), &"1".to_string()),
            (
                &"METRALE_MTP_K_LADDER".to_string(),
                &"1:3,2:1,4:2,8:2,16:1".to_string()
            ),
        ]
    );
    // 2026-09-26: In the vendored corpus, the three DeepSeek-V4.1 Flash recipes
    // declare the same four `METRALE_DS41_*` levers and every other recipe
    // declares none (the two diffusion-gemma files write `env: {}`). Asserted per
    // recipe, so a fixture that grows a block fails here.
    let ds41: BTreeMap<String, String> = [
        ("METRALE_DS41_EXPERT_CACHE_GIB", "88"),
        ("METRALE_DS41_MAX_SEQ", "4096"),
        ("METRALE_DS41_MAX_TOKENS", "512"),
        ("METRALE_DS41_READER_THREADS", "8"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    for r in all() {
        if r.id
            .starts_with("deepseek-v4.1/deepseek-v4.1-flash-q2k-b200")
        {
            assert_eq!(r.env, ds41, "{}", r.id);
            metrale_bench::serve_env::declared(&r.id, &r.env)
                .unwrap_or_else(|e| panic!("{} declares a block the gate refuses: {e:#}", r.id));
        } else {
            assert!(r.env.is_empty(), "{} declares {:?}", r.id, r.env);
        }
    }
    let e = Recipe::parse("f/s", &format!("{head}env:\n  METRALE_X:\n    nested: 1\n"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("env.METRALE_X is not a scalar"), "{e}");
    let e = Recipe::parse("f/s", &format!("{head}env: 1\n"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("`env:` must be a mapping"), "{e}");
}

/// 2026-09-26: A recipe that writes a boolean flag's default is refused, by recipe
/// id, when it renders; an override's `false` removes the recipe's `true`.
#[test]
fn a_recipe_restating_a_boolean_default_is_refused() {
    let text = |line: &str| {
        format!(
            "recipe_version: \"2\"\nmodel: org/m\nruntime: metrale\ncontainer: c\n\
             defaults:\n  port: 8888\n  {line}\n"
        )
    };
    let none = BTreeMap::new();
    let r = Recipe::parse("t/restated", &text("speculative: false")).expect("parses");
    let err = format!("{:#}", r.argv(&none).expect_err("refused"));
    assert!(err.contains("t/restated"), "names the recipe: {err}");
    assert!(err.contains("restates the default"), "{err}");

    let r = Recipe::parse("t/on", &text("enable_prefix_caching: true")).expect("parses");
    let flag = "--enable-prefix-caching".to_string();
    assert!(r.argv(&none).expect("renders").contains(&flag));
    let off = BTreeMap::from([("enable_prefix_caching".to_string(), "false".to_string())]);
    assert!(!r.argv(&off).expect("renders").contains(&flag));
}
