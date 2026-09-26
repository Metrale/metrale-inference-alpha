// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: End-to-end tests: real benchmarks against the loopback mock
//! endpoint, over a socket, through the chunked framing the mock splits
//! mid-line.
//!
//! Owner: bench (integration tests).
//! Invariants: none beyond the types.

// 2026-09-26: Each test binary uses a different subset of the shared mock's
// helpers, so what one does not call is dead code in that binary.
#[allow(dead_code)]
mod mock_endpoint;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::StreamExt;
use metrale_bench::benchmarks::concurrency::ConcurrencySweep;
use metrale_bench::benchmarks::ttft::TtftGate;
use metrale_bench::{
    ArtifactStore, Benchmark, BenchmarkResult, ParamValue, ParamValues, Plugin, PluginHandle,
    RunStatus, TargetEndpoint, VerdictKind,
};

fn temp_store(name: &str) -> ArtifactStore {
    let dir = std::env::temp_dir().join(format!("metrale-e2e-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    ArtifactStore::with_root(dir)
}

/// 2026-09-26: A handle whose event receiver is kept alive for the test's
/// duration.
fn handle(port: u16, store: ArtifactStore) -> (PluginHandle, Arc<AtomicBool>) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || while rx.recv().is_ok() {});
    let cancel = Arc::new(AtomicBool::new(false));
    (
        PluginHandle::new(
            1,
            TargetEndpoint::local(port, "mock"),
            store,
            tx,
            cancel.clone(),
        ),
        cancel,
    )
}

async fn collect<B: Benchmark + Send>(bench: &mut B) -> Vec<BenchmarkResult> {
    let stream = bench.run();
    futures::pin_mut!(stream);
    let mut frames = Vec::new();
    while let Some(item) = stream.next().await {
        frames.push(item.expect("benchmark step failed"));
    }
    frames
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrency_sweep_measures_a_real_stream_end_to_end() {
    let mock = mock_endpoint::start(8, Duration::from_millis(40), Duration::from_millis(5)).await;
    let (h, _cancel) = handle(mock.port, temp_store("sweep"));

    let mut bench = ConcurrencySweep::default();
    bench.load(h).await.expect("load");
    let mut values = ParamValues::defaults(&bench.parameters());
    values.set("concurrencies", ParamValue::IntList(vec![1, 4]));
    values.set("isls", ParamValue::IntList(vec![64]));
    values.set("osl", ParamValue::Int(8));
    values.set("warmup", ParamValue::Int(0));
    bench.configure(&values).expect("configure");

    let frames = collect(&mut bench).await;
    let last = frames.last().expect("at least one frame");
    assert_eq!(last.status, RunStatus::Completed, "{:?}", last.verdict);

    assert_eq!(frames.len(), 4, "probe, 2 cells, done");

    let table = last.table.as_ref().expect("a results table");
    assert_eq!(table.rows.len(), 2, "one row per (isl x conc) cell");
    assert_eq!(
        table
            .rows
            .iter()
            .map(|row| (row[0].text.as_str(), row[1].text.as_str()))
            .collect::<Vec<_>>(),
        [("64", "1"), ("64", "4")],
        "each row retains its configured input length and concurrency"
    );
    assert_eq!(mock.requests.load(Ordering::Relaxed), 5);

    // 2026-09-26: TTFT was measured through the mock's mid-line chunk split.
    let ttft_p50 = &table.rows[0][2].text;
    assert_ne!(ttft_p50, "—", "TTFT must be measured, not missing");
    let throughput: f64 = table.rows[0][8]
        .text
        .parse()
        .expect("throughput is numeric");
    assert!(throughput > 0.0, "throughput was {throughput}");

    assert_eq!(
        last.verdict.as_ref().map(|v| v.kind),
        Some(VerdictKind::Info),
        "a sweep measures, it does not gate"
    );
    assert!(
        last.verdict
            .as_ref()
            .unwrap()
            .reason
            .contains("no request errors"),
        "{:?}",
        last.verdict
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_sweep_against_a_dead_endpoint_fails_at_the_probe() {
    // 2026-09-26: The port is closed: the sweep must stop at the probe rather
    // than produce a table of fast empty cells.
    let dead = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = dead.local_addr().unwrap().port();
    drop(dead);

    let (h, _cancel) = handle(port, temp_store("dead"));
    let mut bench = ConcurrencySweep::default();
    bench.load(h).await.expect("load");
    let values = ParamValues::defaults(&bench.parameters());
    bench.configure(&values).expect("configure");

    let stream = bench.run();
    futures::pin_mut!(stream);
    let first = stream.next().await.expect("one item");
    let err = first.expect_err("a closed port must not look like a fast run");
    assert!(format!("{err:#}").contains("probe failed"), "{err:#}");
    assert!(
        stream.next().await.is_none(),
        "the stream ends on the error"
    );
}

/// 2026-09-26: Records a baseline on one endpoint, then gates a second run
/// against it.
///
/// The two legs run against endpoints of different speed. The gate's limit
/// is a percentage (+3%), so two runs against one endpoint would differ by
/// host scheduling jitter alone, and under `cargo test`'s parallelism that
/// can exceed the limit. Leg 2's endpoint has a tenth of leg 1's TTFT, so
/// its PASS does not depend on the clock, and the path under test is the
/// same: record to disk, read back, compare, verdict.
#[tokio::test(flavor = "multi_thread")]
async fn the_warm_gate_records_a_baseline_then_gates_against_it() {
    let mock = mock_endpoint::start(4, Duration::from_millis(200), Duration::from_millis(2)).await;
    let store = temp_store("warmgate");

    // 2026-09-26: Leg 1: no baseline exists, so the verdict reports rather
    // than passes, and the run becomes the baseline.
    let (h, _c1) = handle(mock.port, store.clone());
    let mut bench = TtftGate::new(metrale_bench::benchmarks::ttft::Mode::Warm);
    bench.load(h).await.expect("load");
    let mut values = ParamValues::defaults(&bench.parameters());
    values.set("prompt_lengths", ParamValue::IntList(vec![64]));
    values.set("repeats", ParamValue::Int(3));
    bench.configure(&values).expect("configure");

    let first = collect(&mut bench).await;
    let last = first.last().unwrap();
    assert_eq!(last.status, RunStatus::Completed);
    assert_eq!(
        last.verdict.as_ref().map(|v| v.kind),
        Some(VerdictKind::Info),
        "with nothing to compare against, this is not a PASS"
    );
    let table = last.table.as_ref().expect("a table");
    assert_eq!(table.rows.len(), 1);
    // 2026-09-26: The mock reports 40 cached prompt tokens. The warm gate
    // shows them as its cache-hit evidence; a warm leg reading 0 measured a
    // cold path.
    assert_eq!(table.rows[0][5].text, "40");
    assert_eq!(
        mock.requests.load(Ordering::Relaxed),
        6,
        "each of three warm samples requires a priming request and a measured request"
    );

    // 2026-09-26: Leg 2: the baseline is on disk, and this endpoint's TTFT is
    // a tenth of the one that recorded it, so the verdict is PASS by a wide
    // margin. Same host (127.0.0.1), so `same_box` still finds the baseline
    // comparable: a different port is not a different box.
    let fast = mock_endpoint::start(4, Duration::from_millis(20), Duration::from_millis(2)).await;
    let (h2, _c2) = handle(fast.port, store.clone());
    let mut second = TtftGate::new(metrale_bench::benchmarks::ttft::Mode::Warm);
    second.load(h2).await.expect("load");
    second.configure(&values).expect("configure");
    let frames = collect(&mut second).await;
    let verdict = frames.last().unwrap().verdict.as_ref().expect("a verdict");
    assert_eq!(
        verdict.kind,
        VerdictKind::Pass,
        "an unchanged endpoint must not regress: {}",
        verdict.reason
    );
    assert!(verdict.reason.contains("limit +3.0%"), "{}", verdict.reason);
    assert_eq!(
        fast.requests.load(Ordering::Relaxed),
        6,
        "the gating leg must also measure the warmed path"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_cold_gate_issues_one_request_per_sample_without_priming() {
    // 2026-09-26: The cold gate issues exactly one request per sample; the
    // warm gate issues two, prime then measure.
    let mock = mock_endpoint::start(2, Duration::from_millis(5), Duration::from_millis(1)).await;
    let (h, _cancel) = handle(mock.port, temp_store("coldgate"));
    let mut bench = TtftGate::new(metrale_bench::benchmarks::ttft::Mode::Cold);
    bench.load(h).await.expect("load");
    let mut values = ParamValues::defaults(&bench.parameters());
    values.set("prompt_lengths", ParamValue::IntList(vec![32]));
    values.set("repeats", ParamValue::Int(6));
    values.set("update_baseline", ParamValue::Bool(false));
    bench.configure(&values).expect("configure");

    let frames = collect(&mut bench).await;
    assert_eq!(frames.last().unwrap().status, RunStatus::Completed);
    assert_eq!(
        mock.requests.load(Ordering::Relaxed),
        6,
        "cold mode measures once per sample, with no priming request"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cancellation_stops_a_run_between_steps() {
    // 2026-09-26: Slow enough that the run is still going when it is
    // cancelled.
    let mock = mock_endpoint::start(40, Duration::from_millis(20), Duration::from_millis(30)).await;
    let (h, cancel) = handle(mock.port, temp_store("cancel"));
    let mut bench = ConcurrencySweep::default();
    bench.load(h).await.expect("load");
    let mut values = ParamValues::defaults(&bench.parameters());
    values.set("concurrencies", ParamValue::IntList(vec![1]));
    values.set("isls", ParamValue::IntList(vec![32, 64, 128, 256]));
    values.set("osl", ParamValue::Int(40));
    values.set("warmup", ParamValue::Int(0));
    bench.configure(&values).expect("configure");

    let stream = bench.run();
    futures::pin_mut!(stream);
    // 2026-09-26: Consume the probe frame, then cancel.
    let _probe = stream.next().await.expect("probe");
    cancel.store(true, Ordering::Relaxed);
    let next = stream.next().await.expect("one more item");
    let err = next.expect_err("cancellation surfaces as an error, not a clean result");
    assert!(format!("{err:#}").contains("cancelled"), "{err:#}");
    assert!(stream.next().await.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_headless_run_persists_a_record_the_history_pane_can_read() {
    // 2026-09-26: A run driven with no terminal writes the same record, in the
    // same store, that the History pane reads. In this file only this test
    // drives through the executor, so only it meets the coherence probe; the
    // mock answers coherently rather than switching the probe off.
    let mock = mock_endpoint::start_saying(
        Some("4 Paris".into()),
        8,
        Duration::from_millis(20),
        Duration::from_millis(2),
    )
    .await;
    let store = temp_store("headless");
    let executor =
        metrale_bench::BenchmarkExecutor::new(tokio::runtime::Handle::current(), store.clone());
    let descriptor = metrale_bench::registry::find("concurrency-sweep").expect("registered");
    let specs = descriptor.build().parameters();
    let values = ParamValues::from_overrides(
        &specs,
        [
            ("concurrencies", "1"),
            ("isls", "128"),
            ("warmup", "0"),
            ("osl", "8"),
        ],
    )
    .expect("overrides parse");
    let target = TargetEndpoint::local(mock.port, "mock");

    // 2026-09-26: `run_blocking` sleeps its thread, so it must not run on a
    // runtime worker.
    let (want_values, want_target) = (values.clone(), target.clone());
    let outcome = tokio::task::spawn_blocking(move || {
        metrale_bench::headless::run_blocking(
            &executor,
            metrale_bench::headless::RunRequest {
                descriptor,
                values,
                target,
                options: metrale_bench::headless::HeadlessOptions::cli("1.0.0-beta-preview"),
            },
            &mut metrale_bench::headless::SilentReporter,
            &|| false,
        )
    })
    .await
    .expect("join")
    .expect("drives");

    // 2026-09-26: Written where the History pane looks, under a
    // nanosecond-keyed name.
    let path = outcome.saved_to.as_ref().expect("saved");
    assert_eq!(
        path.parent().expect("parent"),
        store.root().join("runs").join("concurrency-sweep")
    );
    let stem = path.file_stem().expect("stem").to_str().expect("utf8");
    assert_eq!(stem, outcome.record.run_id, "the stem addresses the record");
    let digits = stem.trim_start_matches("run-");
    assert_eq!(
        digits.len(),
        19,
        "fixed width keeps sort order chronological"
    );
    assert!(digits.chars().all(|c| c.is_ascii_digit()), "{stem}");

    // 2026-09-26: Re-read through the public reader, not the in-memory value.
    let back = metrale_bench::history::load(&store, "concurrency-sweep");
    assert_eq!(back.len(), 1, "exactly one run in the directory");
    let r = &back[0];

    assert_eq!(r.schema, metrale_bench::history::SCHEMA);
    assert_eq!(r.benchmark_name, "Concurrency Sweep");
    assert_eq!(r.source, metrale_bench::RunSource::Cli);
    assert_eq!(r.metrale_version, "1.0.0-beta-preview");
    assert!(!r.is_legacy());

    // 2026-09-26: The whole configuration, defaults included, not just the
    // overrides.
    assert_eq!(r.params.len(), specs.len(), "{:?}", r.params);
    assert_eq!(r.params["osl"], "8");
    assert_eq!(r.params["prompt_mode"], "natural", "an untouched default");
    assert_eq!(r.values(&specs).expect("rehydrates"), want_values);
    assert_eq!(r.target(), want_target);

    // 2026-09-26: One row proves the single-cell override took effect; the
    // defaults run 24 cells.
    assert_eq!(r.frame.status, RunStatus::Completed);
    assert_eq!(
        r.frame.table.as_ref().expect("table").rows.len(),
        1,
        "one isl x one concurrency"
    );
    // 2026-09-26: Two coherence questions plus the single measured request.
    assert_eq!(mock.requests.load(Ordering::Relaxed), 3);
    assert_eq!(outcome.exit_code(), 0);
}
