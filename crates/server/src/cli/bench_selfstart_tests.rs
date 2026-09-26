// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `bench_selfstart`: the start-slot refusals, the
//! free-memory preflight, the lever refusal, `Drop` teardown, and a baseline
//! serve pin reaching the rendered serve args.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants:
//! - No test here trips the shutdown latch: it is process-global with no
//!   reset, and `model_swap` reads it. So `SelfServed::shutdown`, which trips
//!   it, is not called here; `Drop`, which does not, is.

use super::*;

#[test]
fn the_start_slot_is_claimable_exactly_once() {
    // 2026-09-26: A local flag, so the real `STARTED` is not spent by this test.
    let started = AtomicBool::new(false);
    claim_start_slot(&started, false).expect("the first claim takes the slot");
    let err = claim_start_slot(&started, false).expect_err("the second is refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("already started a server"), "{msg}");
    assert!(
        msg.contains("one benchmark per invocation"),
        "says what to do instead: {msg}"
    );
}

#[test]
fn an_already_requested_shutdown_refuses_before_the_wait() {
    // 2026-09-26: A distinct message from the spent-slot refusal, and the slot
    // stays unclaimed.
    let started = AtomicBool::new(false);
    let err = claim_start_slot(&started, true).expect_err("refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("shutdown has already been requested"), "{msg}");
    assert!(
        !started.load(Ordering::SeqCst),
        "and the slot is not spent by a claim that never started anything"
    );
}

#[test]
fn a_clean_box_serves_at_the_recipes_utilisation() {
    // 2026-09-26: The line repeats the recipe's utilisation as written.
    let line =
        headroom_verdict(121.0, 114.0, 0.90, "qwen3.6/27b", 0.85).expect("a clean box passes");
    assert!(line.contains("0.90"), "{line}");
    assert!(line.contains("94 %"), "{line}");
}

#[test]
fn a_co_tenanted_box_is_refused_with_the_remedies() {
    // 2026-09-26: 98 of 121 GiB available is 81 %, under the 85 % floor.
    let err = headroom_verdict(121.0, 98.0, 0.90, "qwen3.6/27b", 0.85).expect_err("refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("qwen3.6/27b"), "names the recipe: {msg}");
    assert!(msg.contains("docker ps"), "names a remedy: {msg}");
    assert!(msg.contains("nvidia-smi"), "and the other one: {msg}");
    assert!(
        msg.contains("not a judgement on the recipe"),
        "and says what it is NOT refusing: {msg}"
    );
}

#[test]
fn the_threshold_itself_is_inclusive() {
    // 2026-09-26: Exactly at the floor passes; just under does not.
    let total = 100.0;
    assert!(headroom_verdict(total, total * 0.85, 0.9, "r", 0.85).is_ok());
    assert!(headroom_verdict(total, total * 0.85 - 0.1, 0.9, "r", 0.85).is_err());
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("a current-thread runtime")
}

/// 2026-09-26: A `SelfServed` around a task that never finishes on its own,
/// plus a receiver that resolves with `Err` once that task is dropped. The
/// sender lives inside the task, so the channel closing proves the drop.
fn served_forever() -> (SelfServed, tokio::sync::oneshot::Receiver<()>) {
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        let _tx = tx;
        std::future::pending::<()>().await;
        Ok(())
    });
    let served = SelfServed {
        target: TargetEndpoint::local(1, "m"),
        recipe_id: "r".to_string(),
        overrides: Default::default(),
        resolved: Default::default(),
        serve_env: Default::default(),
        baseline_entry: Default::default(),
        server: Some(server),
    };
    (served, rx)
}

/// 2026-09-26: A declared lever the process lacks is refused with the exact
/// export line.
#[test]
fn a_declared_lever_this_process_lacks_is_refused_with_the_export_line() {
    use metrale_bench::serve_env::Reconciled;
    let declared: BTreeMap<String, String> = [
        ("METRALE_FP8_ROWWISE", "1"),
        ("METRALE_MTP_DCUT_RATIO", "1.0"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    let err = refuse_unapplied_levers(
        "qwen3.8/qwen3.8-27b-nvfp4-unsloth",
        &Reconciled {
            env: declared.clone(),
            missing: declared.clone(),
        },
    )
    .expect_err("refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("2 serve lever(s)"), "{msg}");
    assert!(
        msg.contains("env METRALE_FP8_ROWWISE=1 METRALE_MTP_DCUT_RATIO=1.0 met benchmark run"),
        "the export line is verbatim: {msg}"
    );
    assert!(msg.contains("--serve-reuse"), "names the child path: {msg}");
    assert!(msg.contains("qwen3.8/qwen3.8-27b-nvfp4-unsloth"), "{msg}");
    // 2026-09-26: Negative control: nothing missing, nothing refused.
    refuse_unapplied_levers(
        "r",
        &Reconciled {
            env: declared,
            missing: BTreeMap::new(),
        },
    )
    .expect("nothing missing, nothing refused");
}

#[test]
fn dropping_a_self_served_tears_the_server_down() {
    // 2026-09-26: A dropped `JoinHandle` detaches its task; `Drop` must abort it.
    runtime().block_on(async {
        let (served, rx) = served_forever();
        drop(served);
        let waited = tokio::time::timeout(Duration::from_secs(5), rx).await;
        assert!(
            matches!(waited, Ok(Err(_))),
            "the server task must be aborted, not detached: {waited:?}"
        );
    });
}

#[test]
fn a_torn_down_server_is_not_torn_down_twice() {
    // 2026-09-26: `Drop` runs after the handle was taken, as `shutdown` leaves
    // it; it finds no task, and the task aborted by hand still ends.
    runtime().block_on(async {
        let (mut served, rx) = served_forever();
        let handle = served.server.take().expect("constructed as Some");
        handle.abort();
        drop(served);
        let waited = tokio::time::timeout(Duration::from_secs(5), rx).await;
        assert!(matches!(waited, Ok(Err(_))), "{waited:?}");
    });
}

/// 2026-09-26: `merge_serve_overrides` keeps an unclashed baseline pin and
/// gives a clash to the operator's `--serve-override`.
#[test]
fn baseline_pins_are_applied_and_the_operator_wins_a_clash() {
    let baseline = BTreeMap::from([
        ("ssm_cache_slots".to_string(), "256".to_string()),
        ("kv_cache_dtype".to_string(), "bf16".to_string()),
    ]);
    let requested = BTreeMap::from([("kv_cache_dtype".to_string(), "fp8".to_string())]);
    let merged = metrale_bench::gate::merge_serve_overrides(baseline, requested);
    assert_eq!(
        merged.get("ssm_cache_slots").map(String::as_str),
        Some("256"),
        "an unclashed pin survives the merge"
    );
    assert_eq!(
        merged.get("kv_cache_dtype").map(String::as_str),
        Some("fp8"),
        "the operator's value wins the clash"
    );
}

/// 2026-09-26: A serve pin the committed BENCH.toml declares for
/// `concurrency-sweep`, with no `--serve-override`, reaches the rendered serve
/// args. Walks `plan_serve`'s steps (read_baseline, resolve, merge with an
/// empty CLI map, `Recipe::serve_args`) over the real tree, so a pin that TOML
/// attaches to the wrong `[[benchmarks]]` entry fails here.
#[test]
fn a_baseline_declared_serve_pin_reaches_the_rendered_serve_args_without_cli_flags() {
    let root = crate::cli::bench_run::repo_root().expect("inside the repo");
    let baseline = gate::read_baseline(&root, "concurrency-sweep").expect("baseline assembles");
    let resolved = crate::cli::bench_resolve::resolve(&baseline, "concurrency-sweep", None, None)
        .expect("the default variant resolves");
    // 2026-09-26: Compared with the entry's own `recipe`, not a literal; which
    // recipe the ladder serves is asserted in `gate::bench_serve_pin_tests`.
    assert_eq!(
        Some(resolved.recipe_id.as_str()),
        resolved.entry.recipe.as_deref(),
        "resolution must serve the recipe the baseline declares"
    );

    let merged =
        gate::merge_serve_overrides(resolved.entry.serve_overrides.clone(), BTreeMap::new());
    assert!(
        !merged.is_empty(),
        "the baseline-declared serve pin was lost before the merge: with no CLI flags the \
         gate would serve the recipe verbatim and print no OVERRIDES line"
    );

    // 2026-09-26: Render through a committed recipe fixture; it stands in for
    // the rendering only, and the pin itself comes from the real BENCH.toml.
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-27b-nvfp4.yaml");
    let text = std::fs::read_to_string(&fixture).expect("fixture recipe");
    let recipe = crate::recipe::Recipe::parse("qwen3.6/qwen3.6-27b-nvfp4", &text).expect("parses");
    let args = recipe
        .serve_args(&merged)
        .expect("pins render to valid serve args");
    // 2026-09-26: Expected values are read from the committed pin, not re-typed.
    assert_eq!(
        args.max_batch_size.to_string(),
        merged["max_batch_size"],
        "the batching pin reached the serve"
    );
    assert_eq!(
        args.kv_cache_dtype.as_deref(),
        Some(merged["kv_cache_dtype"].as_str()),
        "the KV pin reached the serve"
    );
    // 2026-09-26: Negative control. The fixture declares `max_batch_size: 1`
    // and `kv_cache_dtype: bf16`, so a pin with another value passes the
    // assertions above only by reaching argv. `ssm_cache_slots` is not pinned,
    // so it must render the fixture's own value.
    assert!(
        !merged.contains_key("ssm_cache_slots"),
        "the ladder inherits the recipe's pool; re-pinning it re-opens the last \
         disagreement with the published leg"
    );
    assert_eq!(
        args.ssm_cache_slots,
        recipe.defaults["ssm_cache_slots"]
            .parse::<usize>()
            .expect("the fixture declares a numeric pool size"),
        "an unpinned key must render the recipe's own value"
    );
    assert_eq!(
        args.max_seq_len.to_string(),
        merged["max_model_len"],
        "the context pin reached the serve"
    );
}
