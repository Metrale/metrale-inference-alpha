// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The plain decode lane one step ahead of the host, driven through the real core with the asynchronous router over the recording model.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! Driven under `ScriptedDeviceIo` (max depth 2) and the effect-trace
//! decorator. What is proved here, on the CPU:
//! * every golden scenario emits byte-identical client streams under the
//!   asynchronous router, and the lane ran ahead in `plain_decode` and
//!   `slai_policy`;
//! * the negative control: with the over-run discard switched off, an
//!   early-stop scenario's client output changes;
//! * the depth-2 cases: an over-run after a stop is never emitted, its
//!   `Rollback` names exactly that sequence, the KV block table
//!   reconciles, readiness order does not change emissions, and a device
//!   fault mid-pipeline fails only the rows it hit.

use std::sync::Arc;

use metrale_scheduler::ScriptedDeviceIo;

use super::model::ModelCfg;
use super::runner::{EOS, ReqSpec, RunOptions, Scenario, run_scenario, run_scenario_with};
use super::scenarios;
use super::tests::{SERIAL, golden_dir};
use crate::scheduler::PipelineFaults;
use crate::scheduler::io::{AsyncDeviceIo, TracingDeviceIo};

/// 2026-09-25: Rows of the readback ring; wider than any scenario's batch (the
/// scenarios keep `RunOptions::default()`'s `max_batch_size` of 8).
const RING_ROWS: usize = 16;

fn async_run(sc: &Scenario) -> Vec<String> {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let vocab = sc.cfg.vocab;
    run_scenario_with(
        sc,
        Arc::new(move |model, sink| {
            let dev = AsyncDeviceIo::new(model, RING_ROWS)
                .expect("the recording model feeds tokens on the device");
            let dev = ScriptedDeviceIo::new(
                dev,
                |seq: &metrale_model_engine::traits::SequenceState| {
                    if seq.session_hash != 0 {
                        seq.session_hash
                    } else {
                        u64::from(seq.tokens.first().copied().unwrap_or(0))
                    }
                },
                vocab,
                2,
            );
            Box::new(TracingDeviceIo::new(dev, sink))
        }),
    )
}

fn sync_run(sc: &Scenario) -> Vec<String> {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run_scenario(sc)
}

/// 2026-09-25: What the clients saw: the streams and the rotation ack.
fn client_lines(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|l| l.starts_with("out s") || l.starts_with("rotation ack"))
        .cloned()
        .collect()
}

fn count(lines: &[String], needle: &str) -> usize {
    lines.iter().filter(|l| l.contains(needle)).count()
}

fn output(lines: &[String], id: u64) -> String {
    let key = format!("out s{id}: ");
    lines
        .iter()
        .find_map(|l| l.strip_prefix(&key))
        .unwrap_or_else(|| panic!("no output for s{id}"))
        .to_string()
}

/// 2026-09-25: Three streams that stop by EOS well before their token budget of 12.
fn early_stops(name: &'static str, cfg: ModelCfg) -> Scenario {
    early_stops_with_prompts(name, cfg, [5, 5, 7])
}

/// 2026-09-25: `early_stops` with the three prompt lengths chosen by the test.
fn early_stops_with_prompts(name: &'static str, cfg: ModelCfg, prompts: [usize; 3]) -> Scenario {
    let mut reqs = vec![
        ReqSpec::new(1, prompts[0], vec![10, 70, 71, 72, EOS]),
        ReqSpec::new(2, prompts[1], vec![20, 80, 81, EOS]),
        ReqSpec::new(3, prompts[2], vec![30, 90, 91, 92, 93, EOS]),
    ];
    for r in &mut reqs {
        r.max_tokens = 12;
    }
    Scenario {
        name,
        cfg,
        opts: RunOptions::default(),
        reqs,
    }
}

// 2026-09-25: the equivalence suite.

#[test]
fn every_golden_scenario_emits_the_same_client_streams_under_the_async_router() {
    let mut fed_by_scenario = Vec::new();
    for sc in scenarios::all() {
        let golden = std::fs::read_to_string(golden_dir().join(format!("{}.trace", sc.name)))
            .unwrap_or_else(|e| panic!("golden for {}: {e}", sc.name));
        let expected: Vec<String> = golden
            .lines()
            .filter(|l| l.starts_with("out s") || l.starts_with("rotation ack"))
            .map(str::to_string)
            .collect();
        let live = async_run(&sc);
        assert_eq!(
            client_lines(&live),
            expected,
            "{}: the async router changed what a client saw",
            sc.name
        );
        fed_by_scenario.push((sc.name, count(&live, "fx launch DecodeFed")));
    }
    eprintln!("ahead launches per scenario: {fed_by_scenario:?}");
    let fed = |name: &str| {
        fed_by_scenario
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, c)| *c)
            .unwrap_or_else(|| panic!("no scenario {name}"))
    };
    // 2026-09-25: the lane ran ahead in plain_decode and slai_policy, and
    // never in the scenarios listed below (host-logits rows, the hybrid
    // model, the spec lanes, the mid-stream cancel).
    assert!(fed("plain_decode") > 0, "plain decode never ran ahead");
    assert!(fed("slai_policy") > 0);
    for depth_one in [
        "host_logits_paths",
        "watchdog_rollback",
        "verify_k2",
        "verify_k3",
        "verify_k4",
        "batched_verify_k4",
        "dflash",
        "dflash_batched",
        "ngram",
        "self_spec",
        "cancel_mid_stream",
    ] {
        assert_eq!(fed(depth_one), 0, "{depth_one} ran a step ahead");
    }
}

#[test]
fn the_suite_fails_when_the_over_run_discard_is_off() {
    let sc = early_stops("pipeline_negative_control", ModelCfg::default());
    let expected = client_lines(&sync_run(&sc));
    let mut control = sc.clone();
    control.opts.pipeline_faults = PipelineFaults { keep_overrun: true };
    let live = async_run(&control);
    assert_ne!(
        client_lines(&live),
        expected,
        "with the discard off the over-run token must reach a client"
    );
    // 2026-09-25: the script repeats its last token, so the over-run of a
    // stopped row is a second EOS: one more token counted.
    assert!(
        output(&live, 2).contains("n=5") && expected[1].contains("n=4"),
        "{}\n{}",
        output(&live, 2),
        expected[1]
    );
    assert!(count(&live, "fx launch DecodeFed") > 0);
}

// 2026-09-25: depth-2 cases.

#[test]
fn b_an_over_run_after_a_stop_is_never_emitted() {
    let sc = early_stops("pipeline_overrun", ModelCfg::default());
    let live = async_run(&sc);
    assert_eq!(
        output(&live, 1),
        "T10 T70 T71 T72 Done(stop, n=5, think=0, cached=0, acc=0, guard=None)"
    );
    assert_eq!(
        output(&live, 2),
        "T20 T80 T81 Done(stop, n=4, think=0, cached=0, acc=0, guard=None)"
    );
    assert_eq!(
        output(&live, 3),
        "T30 T90 T91 T92 T93 Done(stop, n=6, think=0, cached=0, acc=0, guard=None)"
    );
    assert_eq!(client_lines(&live), client_lines(&sync_run(&sc)));
    // 2026-09-25: the over-runs happened and were discarded: s2 and s3
    // stop with a step in flight (one rollback each), s1 does not
    // (`b_the_rollback_names_exactly_the_stopped_sequence`), and every
    // finish is sent exactly once.
    assert!(count(&live, "fx launch DecodeFed") >= 3);
    assert_eq!(count(&live, "fx apply Rollback"), 2);
    for id in 1..=3 {
        assert_eq!(output(&live, id).matches("Done(").count(), 1);
    }
}

#[test]
fn b_the_rollback_names_exactly_the_stopped_sequence() {
    let sc = early_stops("pipeline_rollback_target", ModelCfg::default());
    let live = async_run(&sc);
    // 2026-09-25: the router releases the reservation of the row it
    // rolls back, and the recording model names the sequence: s2, then
    // s3, and never s1.
    assert_eq!(count(&live, "release_decode_blocks(s2@"), 1);
    assert_eq!(count(&live, "release_decode_blocks(s3@"), 1);
    assert_eq!(count(&live, "release_decode_blocks(s1@"), 0);
    let at = |id: u64| {
        live.iter()
            .position(|l| l.contains(&format!("release_decode_blocks(s{id}@")))
            .expect("release")
    };
    assert!(at(2) < at(3));
}

#[test]
fn b_the_kv_block_table_reconciles_after_an_over_run() {
    // 2026-09-25: small blocks, so an over-run crosses a block
    // boundary and the ahead launch had to reserve one.
    let cfg = ModelCfg {
        block_size: 4,
        ..ModelCfg::default()
    };
    // 2026-09-25: s2's over-run lands on position 7 (prompt 4 plus
    // three generated), the last of a block, so its ahead launch
    // reserved the next one.
    let sc = early_stops_with_prompts("pipeline_kv_reconcile", cfg, [5, 4, 7]);
    let live = async_run(&sc);
    let sync = sync_run(&sc);
    assert_eq!(client_lines(&live), client_lines(&sync));
    let released: Vec<&String> = live
        .iter()
        .filter(|l| l.contains("release_decode_blocks("))
        .collect();
    assert_eq!(released.len(), 2);
    assert!(
        released.iter().any(|l| l.ends_with("n=1)")),
        "no over-run crossed a block boundary: {released:?}"
    );
    // 2026-09-25: the blocks freed at retirement match the synchronous
    // run's; the frees are compared sorted because the retirement order
    // can differ.
    let frees = |lines: &[String]| -> Vec<String> {
        let mut v: Vec<String> = lines
            .iter()
            .filter(|l| l.starts_with("free_sequence("))
            .cloned()
            .collect();
        v.sort();
        v
    };
    assert_eq!(frees(&live), frees(&sync));
}

#[test]
fn b_readiness_order_does_not_change_emissions() {
    // 2026-09-25: the newest step completes first: the older ticket
    // the core awaits is not ready when polled, so the router takes
    // its blocking arm — and the streams are the synchronous ones
    // regardless.
    let reversed = early_stops(
        "pipeline_reversed_completion",
        ModelCfg {
            reversed_completion: true,
            ..ModelCfg::default()
        },
    );
    let live = async_run(&reversed);
    assert_eq!(client_lines(&live), client_lines(&sync_run(&reversed)));
    assert!(count(&live, "fx launch DecodeFed") > 0);
    assert!(
        count(&live, "event_synchronize(") > 0,
        "the older step was never pending"
    );
    // 2026-09-25: a step that completes after a few polls is read without blocking.
    let lagging = early_stops(
        "pipeline_lagging_completion",
        ModelCfg {
            event_lag: 3,
            ..ModelCfg::default()
        },
    );
    let live = async_run(&lagging);
    assert_eq!(client_lines(&live), client_lines(&sync_run(&lagging)));
    assert_eq!(count(&live, "event_synchronize("), 0);
}

// 2026-09-25: device errors.

#[test]
fn c_a_device_fault_mid_pipeline_fails_only_the_rows_it_hit() {
    // 2026-09-25: two rows decode with a step in flight; a third
    // arrives at tick 1. The first event poll faults: rows 1 and 2 get
    // the error, and row 3 still completes.
    let mut late = ReqSpec::new(3, 5, vec![30, 90, 91, EOS]);
    late.arrive_at_tick = Some(1);
    let mut reqs = vec![
        ReqSpec::new(1, 5, vec![10, 70, 71, 72, 73, EOS]),
        ReqSpec::new(2, 5, vec![20, 80, 81, 82, 83, EOS]),
        late,
    ];
    for r in &mut reqs {
        r.max_tokens = 12;
    }
    let sc = Scenario {
        name: "pipeline_device_fault",
        cfg: ModelCfg {
            event_query_fault_at: Some(1),
            ..ModelCfg::default()
        },
        opts: RunOptions::default(),
        reqs,
    };
    let live = async_run(&sc);
    assert_eq!(output(&live, 1), "T10 Error(scripted device fault)");
    assert_eq!(output(&live, 2), "T20 Error(scripted device fault)");
    assert_eq!(
        output(&live, 3),
        "T30 T90 T91 Done(stop, n=4, think=0, cached=0, acc=0, guard=None)"
    );
    assert_eq!(
        count(&live, "fx apply ReleaseSeq{cache=false, send_error}"),
        2
    );
    assert_eq!(
        count(&live, "fx apply ReleaseSeq{cache=true, finish_sequence}"),
        1
    );
}

#[test]
fn a_model_without_a_feed_refuses_the_async_router() {
    let model = super::model::RecordingModel::new(ModelCfg {
        device_token_feed: false,
        ..ModelCfg::default()
    });
    let err = AsyncDeviceIo::new(Arc::new(model), RING_ROWS)
        .err()
        .expect("refused");
    assert!(
        format!("{err:#}").contains("no device token feed"),
        "{err:#}"
    );
}
