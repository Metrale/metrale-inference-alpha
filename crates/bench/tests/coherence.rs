// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The coherence probe against a real loopback socket. The probe
//! is advisory: a wrong answer, a closed port or a wrong model name becomes a
//! warning, and the benchmark still runs.
//!
//! Owner: bench (coherence probe).
//! Invariants: none beyond the types.

// 2026-09-26: Each integration binary includes the mock separately, so the
// helpers this one does not call are dead code here only.
#[allow(dead_code)]
mod mock_endpoint;

use std::time::Duration;

use metrale_bench::coherence::{self, CoherencePolicy};
use metrale_bench::plugin::TargetEndpoint;

#[derive(Default)]
struct WarningReporter {
    warnings: Vec<String>,
}

impl metrale_bench::headless::RunReporter for WarningReporter {
    fn event(&mut self, event: &metrale_bench::PluginEvent) {
        if let metrale_bench::PluginEvent::Log(line) = event
            && line.level == metrale_bench::LogLevel::Warn
        {
            self.warnings.push(line.text.clone());
        }
    }
}

fn target(port: u16) -> TargetEndpoint {
    TargetEndpoint::local(port, "mock")
}

#[tokio::test]
async fn an_endpoint_answering_correctly_is_clean() {
    // 2026-09-26: One reply satisfies both checks.
    let mock =
        mock_endpoint::start_saying(Some("4 Paris".into()), 1, Duration::ZERO, Duration::ZERO)
            .await;
    let report = coherence::probe(&target(mock.port), Duration::from_secs(5)).await;
    assert_eq!(report.answers.len(), 2);
    assert!(report.is_clean());
    assert!(report.concern(&target(mock.port)).is_none());
}

#[tokio::test]
async fn an_endpoint_answering_nonsense_warns_and_says_what_it_said() {
    let mock = mock_endpoint::start_saying(
        Some("I am a teapot".into()),
        1,
        Duration::ZERO,
        Duration::ZERO,
    )
    .await;
    let report = coherence::probe(&target(mock.port), Duration::from_secs(5)).await;
    assert!(!report.is_clean());
    let text = report.concern(&target(mock.port)).expect("a concern");
    // 2026-09-26: The concern quotes the answer back.
    assert!(text.contains("teapot"), "{text}");
    assert!(
        text.contains("arithmetic"),
        "names the failing check: {text}"
    );
    // 2026-09-26: It does not read as a refusal: the run is still allowed.
    assert!(text.contains("still valid"), "{text}");
}

#[tokio::test]
async fn an_unreachable_endpoint_is_a_transport_error_not_a_wrong_answer() {
    // 2026-09-26: A closed port and a confused model are different
    // diagnoses.
    let report = coherence::probe(&target(1), Duration::from_secs(2)).await;
    assert!(report.transport_error.is_some());
    let text = report.concern(&target(1)).expect("a concern");
    assert!(
        !text.contains("different model"),
        "should not blame the model: {text}"
    );
}

#[tokio::test]
async fn a_failed_probe_warns_but_still_runs_the_benchmark() {
    use metrale_bench::headless::{HeadlessOptions, RunRequest, run_blocking};
    use metrale_bench::{ArtifactStore, BenchmarkExecutor, ParamValues, registry};

    let mock = mock_endpoint::start_saying(
        Some("I am a teapot".into()),
        1,
        Duration::ZERO,
        Duration::ZERO,
    )
    .await;
    let requests = mock.requests.clone();
    let dir = std::env::temp_dir().join(format!(
        "metrale-coherence-{:?}",
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).expect("scratch");

    let descriptor = registry::find("concurrency-sweep").expect("registered");
    let specs = descriptor.build().parameters();
    let executor = BenchmarkExecutor::new(
        tokio::runtime::Handle::current(),
        ArtifactStore::with_root(&dir),
    );
    let request = RunRequest {
        descriptor,
        values: ParamValues::defaults(&specs),
        target: target(mock.port),
        options: HeadlessOptions {
            poll: Duration::from_millis(10),
            save: false,
            source: metrale_bench::RunSource::Cli,
            metrale_version: "test".into(),
            coherence: CoherencePolicy::Probe,
            temp_ceilings: None,
        },
    };

    let (outcome, warnings) = tokio::task::spawn_blocking(move || {
        let mut reporter = WarningReporter::default();
        let outcome = run_blocking(&executor, request, &mut reporter, &|| false);
        (outcome, reporter.warnings)
    })
    .await
    .expect("join");
    let outcome = outcome.expect("drives");

    // 2026-09-26: An endpoint that answers oddly is a warning, so the sweep
    // still runs: more requests than the probe's two questions.
    assert!(
        requests.load(std::sync::atomic::Ordering::Relaxed) > 2,
        "the benchmark must not have been blocked by the probe"
    );
    assert_eq!(outcome.exit_code(), 0, "a warning is not a failure");
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("teapot") && warning.contains("still valid")),
        "the advisory reaches the caller: {warnings:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// 2026-09-26: `/v1/models` must be readable through chunked framing: the
/// body carries hex length prefixes and a terminating `0\r\n\r\n`, on which
/// a plain `from_str` from the first `{` fails.
#[tokio::test]
async fn the_model_list_survives_chunked_framing() {
    let mock =
        mock_endpoint::start_saying(Some("4 Paris".into()), 1, Duration::ZERO, Duration::ZERO)
            .await;
    let models = metrale_bench::http::list_models(&target(mock.port), Duration::from_secs(5))
        .await
        .expect("the list parses");
    assert_eq!(models, vec!["mock".to_string()]);
}

#[tokio::test]
async fn a_model_the_server_does_not_serve_is_reported() {
    let mock =
        mock_endpoint::start_saying(Some("4 Paris".into()), 1, Duration::ZERO, Duration::ZERO)
            .await;
    // 2026-09-26: The mock serves "mock"; ask for something else.
    let wrong = TargetEndpoint::local(mock.port, "does/not-exist");
    let report = coherence::probe(&wrong, Duration::from_secs(5)).await;
    assert!(!report.is_clean(), "a wrong model name is not clean");
    let concern = report.concern(&wrong).expect("a concern");
    assert!(concern.contains("mock"), "names what IS served: {concern}");
    assert!(concern.contains("does/not-exist"), "{concern}");
    // 2026-09-26: The questions still passed; only the model list catches
    // this.
    assert!(
        report.answers.iter().all(|a| a.passed),
        "{:?}",
        report.answers
    );
}
