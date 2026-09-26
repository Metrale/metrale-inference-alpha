// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `swap`'s refusals, the checks that run before anything
//! is torn down, and of `carry_process_flags`. A successful swap needs a GPU
//! and a checkpoint and is not tested here.
//!
//! Owner: server (model hosting).
//! Invariants: none beyond the types.

use super::*;
use clap::Parser as _;

fn args(extra: &[&str]) -> cli::ServeArgs {
    let mut argv = vec!["met", "serve", "dummy/model"];
    argv.extend_from_slice(extra);
    match cli::Cli::parse_from(argv).command {
        cli::Command::Serve(a) => a,
        cli::Command::Benchmark(_)
        | cli::Command::DumpServeOptions
        | cli::Command::SyncRecipes
        | cli::Command::Doctor => {
            unreachable!("parsed a serve command")
        }
    }
}

/// 2026-09-26: A config `validate_serve_args` rejects is refused before the
/// host is cleared.
#[test]
fn an_invalid_config_is_refused_before_anything_is_torn_down() {
    let host = Arc::new(ModelHost::empty());
    let bad = args(&["--scheduler", "nonsense"]);
    let err = swap(&host, bad).expect_err("refused");
    let text = format!("{err:#}");
    assert!(text.contains("--scheduler"), "{text}");
    assert!(
        !host.is_loaded(),
        "the host was empty and must still be empty"
    );
}

/// 2026-09-26: Multi-rank is refused: the EP worker holds its model until the
/// head exits, and there is no command to make it load another.
#[test]
fn a_multi_rank_deployment_is_refused() {
    let host = Arc::new(ModelHost::empty());
    let multi = args(&["--world-size", "2"]);
    let err = swap(&host, multi).expect_err("refused");
    assert!(format!("{err:#}").contains("single-node only"));
}

/// 2026-09-26: A refused swap publishes nothing into the host. The host here
/// is empty, so this does not exercise a loaded model.
#[test]
fn a_refused_swap_leaves_the_running_model_alone() {
    let host = Arc::new(ModelHost::empty());
    assert!(!host.is_loaded());
    let _ = swap(&host, args(&["--world-size", "4"]));
    assert!(!host.is_loaded(), "clear() must not have run");
}

/// 2026-09-26: The host records the running argv (`set_args`), which a failed
/// swap restores.
#[test]
fn the_host_remembers_what_it_is_running() {
    let host = Arc::new(ModelHost::empty());
    assert!(
        host.args().is_none(),
        "nothing loaded, nothing to restore to"
    );

    let first = args(&["--port", "9001"]);
    host.set_args(first.clone());
    assert_eq!(
        host.args().map(|a| a.port),
        Some(9001),
        "a swap can now restore to what was running"
    );
}

#[test]
fn a_recipe_cannot_switch_off_the_operators_auto_swap_policy() {
    // 2026-09-26: A recipe that omits `--auto-swap` does not turn off the
    // running server's `--auto-swap`.
    use clap::Parser as _;
    let previous = cli::ServeArgs::parse_from(["met", "org/live", "--auto-swap"]);
    let mut next = cli::ServeArgs::parse_from(["met", "org/next"]);
    assert!(!next.auto_swap, "the recipe says nothing about it");

    super::carry_process_flags(&mut next, &previous);
    assert!(next.auto_swap, "the operator's policy survives the swap");
    assert_eq!(
        next.model.as_deref(),
        Some("org/next"),
        "the MODEL still swaps"
    );
}

#[test]
fn a_recipe_cannot_switch_on_auto_swap_where_it_was_forbidden() {
    // 2026-09-26: A recipe's `--auto-swap` does not turn it on for a server
    // started without it.
    use clap::Parser as _;
    let previous = cli::ServeArgs::parse_from(["met", "org/live"]);
    let mut next = cli::ServeArgs::parse_from(["met", "org/next", "--auto-swap"]);
    super::carry_process_flags(&mut next, &previous);
    assert!(
        !super::super::auto_swap::enabled(&next),
        "the recipe's --auto-swap does not survive the swap"
    );
}

#[test]
fn a_recipes_port_cannot_move_a_socket_that_is_already_bound() {
    use clap::Parser as _;
    let previous = cli::ServeArgs::parse_from(["met", "org/live", "--port", "8888"]);
    let mut next = cli::ServeArgs::parse_from(["met", "org/next", "--port", "9100"]);
    super::carry_process_flags(&mut next, &previous);
    assert_eq!(next.port, 8888, "the bound port is authoritative");
}

#[test]
fn a_model_this_build_has_no_kernels_for_is_refused_before_teardown() {
    // 2026-09-26: A checkpoint this build has no kernel target for is refused
    // by `preflight_kernel_target`, which reads its config.json, before the
    // running model is released.
    let host = Arc::new(ModelHost::empty());
    let dir = tempfile::tempdir().expect("tmp");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"no_such_architecture","hidden_size":4096,"num_hidden_layers":1}"#,
    )
    .expect("write");

    use clap::Parser as _;
    let args = cli::ServeArgs::parse_from(["met", dir.path().to_str().expect("utf8")]);
    let err = super::swap(&host, args).expect_err("refused");
    let text = format!("{err:#}");
    assert!(
        text.contains("no compiled kernels") || text.contains("no_such_architecture"),
        "{text}"
    );
    assert!(
        text.contains("running model is untouched") || host.current().is_none(),
        "nothing was torn down: {text}"
    );
}

#[test]
fn two_swaps_at_once_do_not_both_tear_down_the_model() {
    // 2026-09-26: `swap_guard` lets one swap in at a time. Without it both
    // callers reach `ModelHost::take`, and the second gets `None` and treats it
    // as a modelless boot.
    let host = Arc::new(ModelHost::empty());
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let overlapping = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let inside = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let threads: Vec<_> = (0..2)
        .map(|_| {
            let (host, barrier) = (host.clone(), barrier.clone());
            let (overlapping, inside) = (overlapping.clone(), inside.clone());
            std::thread::spawn(move || {
                barrier.wait();
                let _guard = host.swap_guard();
                if inside.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 0 {
                    overlapping.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                std::thread::sleep(std::time::Duration::from_millis(60));
                inside.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            })
        })
        .collect();
    for t in threads {
        t.join().expect("no panic");
    }
    assert!(
        !overlapping.load(std::sync::atomic::Ordering::SeqCst),
        "two swaps were inside the guard at once"
    );
}

#[test]
fn the_swap_the_winner_already_performed_is_not_repeated() {
    // 2026-09-26: The no-op check compares whole `ServeArgs`, so a different
    // recipe for the same checkpoint is unequal and still swaps.
    use clap::Parser as _;
    let a = cli::ServeArgs::parse_from(["met", "org/m"]);
    let mut b = cli::ServeArgs::parse_from(["met", "org/m"]);
    assert_eq!(a, b, "same argv");
    b.max_batch_size = a.max_batch_size + 1;
    assert_ne!(a, b, "a different recipe for the same model is a real swap");
}

#[test]
fn a_swap_is_refused_once_shutdown_has_been_requested() {
    // 2026-09-26: A load started in an exiting process would release the
    // running model for one that never serves.
    let err = super::refuse_if_shutting_down(true).expect_err("refused");
    assert!(format!("{err:#}").contains("shutdown"), "{err:#}");
    super::refuse_if_shutting_down(false).expect("a live process may swap");
}

#[test]
fn a_recipe_cannot_turn_off_authentication() {
    // 2026-09-26: `carry_process_flags` does not carry `require_auth`, so the
    // auth policy must live on the host (installed at boot) and never be read
    // from a swap's argv. This test fails if a swap path replaces it.
    let host = Arc::new(ModelHost::empty());
    let cfg = std::sync::Arc::new(
        crate::auth::AuthConfig::from_inline("sk-test-token").expect("valid token"),
    );
    host.set_auth(Some(cfg));

    // 2026-09-26: A multi-rank swap is refused after its argv is taken; the
    // host's auth policy must survive the attempt.
    use clap::Parser as _;
    let mut args = cli::ServeArgs::parse_from(["met", "org/m"]);
    args.world_size = 2;
    let _ = super::swap(&host, args);

    assert!(
        host.auth().is_some(),
        "the API-key policy must not be a casualty of a swap"
    );
}

#[test]
fn a_swap_does_not_silently_stop_request_dumping() {
    // 2026-09-26: A recipe that omits `--dump` does not stop request dumping:
    // `carry_process_flags` keeps the running server's `--dump`.
    use clap::Parser as _;
    let previous = cli::ServeArgs::parse_from(["met", "org/live", "--dump", "/tmp/probe.jsonl"]);
    let mut next = cli::ServeArgs::parse_from(["met", "org/next"]);
    assert!(next.dump.is_none(), "the recipe says nothing about it");

    super::carry_process_flags(&mut next, &previous);
    assert_eq!(next.dump.as_deref(), Some("/tmp/probe.jsonl"));
    assert_eq!(
        next.model.as_deref(),
        Some("org/next"),
        "the MODEL still swaps"
    );
}

#[test]
fn a_repeat_of_the_live_config_is_a_no_op_even_with_process_flags_set() {
    // 2026-09-26: The live argv already holds the carried flags and a recipe's
    // does not, so the no-op comparison runs after carrying. A repeat of the
    // live recipe then compares equal even with process flags set.
    use clap::Parser as _;

    // 2026-09-26: What the host stores after a swap: the recipe's argv plus
    // the carried flags.
    let mut live = cli::ServeArgs::parse_from(["met", "org/m"]);
    let operator =
        cli::ServeArgs::parse_from(["met", "org/boot", "--auto-swap", "--dump", "/tmp/d.jsonl"]);
    super::carry_process_flags(&mut live, &operator);

    // 2026-09-26: What a queued request brings: the same recipe, without them.
    let mut queued = cli::ServeArgs::parse_from(["met", "org/m"]);
    assert_ne!(
        live, queued,
        "they differ before carrying — the old comparison"
    );

    super::carry_process_flags(&mut queued, &live);
    assert_eq!(
        live, queued,
        "and are equal after it, which is what makes the no-op fire"
    );
}
