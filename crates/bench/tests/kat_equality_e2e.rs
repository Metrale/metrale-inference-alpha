// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The KAT equality gate, driven through the executor against
//! the loopback mock: does the driver issue the draw in the orders it says,
//! and form the verdict it says?
//!
//! The tests are `#[ignore]`: `load()` provisions the BFCL dataset, which
//! needs python and the network. Ignored rather than skipped on a missing
//! artifact, because cargo prints `ignored` distinctly from `ok`.
//!
//! Owner: bench (integration tests).
//! Invariants: none beyond the types.
//!
//! Run them with an artifact store you can write to:
//!
//! ```text
//! HOME=/path/to/writable cargo test -p metrale-bench --test kat_equality_e2e -- --ignored
//! ```

// 2026-09-26: Each test binary uses a different subset of the shared mock's
// helpers, so what one does not call is dead code in that binary.
#[allow(dead_code)]
mod mock_endpoint;

use std::time::Duration;

use metrale_bench::{ArtifactStore, ParamValues, TargetEndpoint, VerdictKind};

/// 2026-09-26: How many samples each order issues. Small on purpose: these
/// tests prove the mechanism, not statistical power.
const CAP: &str = "3";

async fn run_against(port: u16) -> metrale_bench::RunRecord {
    run_with_orders(port, "2").await
}

async fn run_with_orders(port: u16, orders: &str) -> metrale_bench::RunRecord {
    let store = ArtifactStore::discover().expect("an artifact store");
    let executor =
        metrale_bench::BenchmarkExecutor::new(tokio::runtime::Handle::current(), store.clone());
    let descriptor = metrale_bench::registry::find("kat-equality-gate").expect("registered");
    let specs = descriptor.build().parameters();
    let values = ParamValues::from_overrides(
        &specs,
        [
            ("sample_cap", CAP),
            ("orders", orders),
            ("max_new_tokens", "32"),
        ],
    )
    .expect("overrides parse");
    let target = TargetEndpoint::local(port, "mock");
    tokio::task::spawn_blocking(move || {
        metrale_bench::headless::run_blocking(
            &executor,
            metrale_bench::headless::RunRequest {
                descriptor,
                values,
                target,
                options: metrale_bench::headless::HeadlessOptions::cli("test"),
            },
            &mut metrale_bench::headless::SilentReporter,
            &|| false,
        )
    })
    .await
    .expect("join")
    .expect("drives")
    .record
}

/// 2026-09-26: A server that answers the same thing regardless of what it
/// has served before is order-independent, and must pass.
#[tokio::test]
#[ignore = "provisions the BFCL dataset (python + network); see the module docs"]
async fn a_server_that_ignores_history_passes() {
    let mock = mock_endpoint::start_saying(
        Some("the same answer".into()),
        4,
        Duration::from_millis(1),
        Duration::from_millis(1),
    )
    .await;
    let record = run_against(mock.port).await;
    let frame = &record.frame;
    let verdict = frame.verdict.as_ref().expect("a verdict");
    assert_eq!(
        verdict.kind,
        VerdictKind::Pass,
        "an order-independent server must pass: {}",
        verdict.reason
    );
    assert_eq!(frame.metrics["samples"], 3.0);
    assert_eq!(frame.metrics["orders"], 2.0);
    assert_eq!(frame.metrics["diverged"], 0.0);
    assert_eq!(frame.metrics["unmeasured"], 0.0);
    // 2026-09-26: At least the 3x2 generations. Not an equality: the
    // executor's coherence probe also sends chat requests to the mock.
    // `each_extra_order_issues_exactly_one_more_pass_over_the_draw` pins the
    // per-order cost by a difference in which the probe cancels.
    assert!(
        mock.requests.load(std::sync::atomic::Ordering::Relaxed) >= 6,
        "3 samples x 2 orders is the floor"
    );
}

/// 2026-09-26: "Issue the draw once per order", measured as a difference so
/// the harness's own probe requests cancel out.
///
/// This catches a driver that issues one order twice, skips an order, or
/// re-issues the whole draw per order per sample. An absolute count would
/// also pin the harness's probe count.
#[tokio::test]
#[ignore = "provisions the BFCL dataset (python + network); see the module docs"]
async fn each_extra_order_issues_exactly_one_more_pass_over_the_draw() {
    let two = mock_endpoint::start_saying(
        Some("the same answer".into()),
        4,
        Duration::from_millis(1),
        Duration::from_millis(1),
    )
    .await;
    let record_two = run_with_orders(two.port, "2").await;
    let three = mock_endpoint::start_saying(
        Some("the same answer".into()),
        4,
        Duration::from_millis(1),
        Duration::from_millis(1),
    )
    .await;
    let record_three = run_with_orders(three.port, "3").await;

    let n2 = two.requests.load(std::sync::atomic::Ordering::Relaxed);
    let n3 = three.requests.load(std::sync::atomic::Ordering::Relaxed);
    let cap: usize = CAP.parse().expect("CAP is a number");
    assert_eq!(
        n3 - n2,
        cap,
        "one more order must cost exactly one more pass over the draw ({n2} -> {n3})"
    );
    assert_eq!(record_two.frame.metrics["orders"], 2.0);
    assert_eq!(record_three.frame.metrics["orders"], 3.0);
    // 2026-09-26: The third order is compared, not merely issued.
    assert_eq!(record_three.frame.metrics["identical"], cap as f64);
}

/// 2026-09-26: The control for the driver: a gate proven only against a
/// well-behaved server has been shown to say "pass", not to work.
///
/// This server's reply depends only on how many requests it has already
/// served. Its counter only grows, so every sample's reply in the second
/// order differs from its reply in the first, and every sample must be
/// reported.
#[tokio::test]
#[ignore = "provisions the BFCL dataset (python + network); see the module docs"]
async fn a_server_whose_reply_depends_on_what_it_served_before_is_caught() {
    let mock =
        mock_endpoint::start_indexed(Duration::from_millis(1), Duration::from_millis(1)).await;
    let record = run_against(mock.port).await;
    let frame = &record.frame;
    let verdict = frame.verdict.as_ref().expect("a verdict");
    assert_eq!(
        verdict.kind,
        VerdictKind::Fail,
        "a server that answers by request count is order-dependent by construction"
    );
    assert!(
        verdict.reason.contains("ORDER-DEPENDENT"),
        "the verdict must name what it found: {}",
        verdict.reason
    );
    assert_eq!(
        frame.metrics["diverged"], 3.0,
        "with 3 samples reversed, no sample keeps its position — all three differ"
    );
    assert_eq!(frame.metrics["unmeasured"], 0.0, "every request succeeded");
}
