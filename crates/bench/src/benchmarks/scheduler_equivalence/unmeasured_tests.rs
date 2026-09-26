// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Every unmeasured sample carries its cause into the record.
//!
//! Owner: bench, scheduler-equivalence gate.
//! Invariants: the failures are real responses from a loopback socket read
//! by the production client; nothing between the socket and the record is
//! stubbed.

use std::time::Duration;

use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::compare::{Leg, Pass, Reply, score, verdict};
use super::driver::request;
use super::host::Lane;
use super::report;
use super::unmeasured::{CLASSES, Cause};
use crate::http::failure::{BODY_EXCERPT, excerpt};
use crate::http::{FailureKind, RequestFailure};
use crate::plugin::TargetEndpoint;
use crate::result::{LogLevel, VerdictKind};

const GOOD: &str = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n\
    data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\
    data: {\"choices\":[],\"usage\":{\"completion_tokens\":1}}\n\
    data: [DONE]\n";

const OVERLOADED: &str = "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n\
    Content-Length: 52\r\n\r\n{\"error\":{\"message\":\"KV cache exhausted, retry\"}}  ";

/// 2026-09-25: A loopback endpoint that answers one request with `response`
/// and closes, or with `None` reads the request and never answers.
async fn endpoint(response: Option<String>) -> TargetEndpoint {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let port = listener.local_addr().expect("address").port();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut req = [0u8; 8192];
        let _ = socket.read(&mut req).await;
        match response {
            Some(r) => {
                let _ = socket.write_all(r.as_bytes()).await;
            }
            None => tokio::time::sleep(Duration::from_secs(60)).await,
        }
    });
    TargetEndpoint::local(port, "mock")
}

/// 2026-09-25: A port with no listener: connect is refused.
async fn refusing_endpoint() -> TargetEndpoint {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let port = listener.local_addr().expect("address").port();
    drop(listener);
    TargetEndpoint::local(port, "mock")
}

async fn ask(target: &TargetEndpoint, sample_id: &str, timeout: Duration) -> Reply {
    request(target, sample_id.to_string(), &json!({}), timeout).await
}

async fn answered(sample_id: &str, response: Option<String>) -> Reply {
    ask(
        &endpoint(response).await,
        sample_id,
        Duration::from_secs(10),
    )
    .await
}

fn failure(r: &Reply) -> &RequestFailure {
    r.outcome
        .as_ref()
        .expect_err("the request should have failed")
}

fn leg(pass: Pass, concurrency: usize, replies: Vec<Reply>) -> Leg {
    Leg {
        lane: Lane::SpecOff,
        pass,
        concurrency,
        replies,
    }
}

async fn good(sample_id: &str) -> Reply {
    let r = answered(sample_id, Some(GOOD.to_string())).await;
    assert!(r.outcome.is_ok(), "control request failed: {:?}", r.outcome);
    r
}

/// 2026-09-25: A C=16 cell where every leg answers both samples except the
/// async leg's `s2`, which is `failed`.
async fn cell_with_async_failure(failed: Reply) -> Vec<Leg> {
    vec![
        leg(Pass::Sync, 16, vec![good("s1").await, good("s2").await]),
        leg(Pass::Control, 16, vec![good("s1").await, good("s2").await]),
        leg(Pass::Async, 16, vec![good("s1").await, failed]),
    ]
}

#[tokio::test]
async fn path_a_a_failed_request_carries_its_cause_into_the_terminal_frame() {
    let failed = answered("s2", Some(OVERLOADED.to_string())).await;
    let legs = cell_with_async_failure(failed).await;
    let frame = report::terminal(&legs, &Default::default(), Duration::ZERO);

    // 2026-09-25: One unmeasured sample fails the gate.
    let v = frame.verdict.as_ref().expect("verdict");
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(
        v.reason
            .contains("1 of 2 samples at spec-off C=16 were unmeasured in the async pass"),
        "{}",
        v.reason
    );
    assert!(
        v.reason.contains("Causes: async http_5xx 1."),
        "{}",
        v.reason
    );
    assert_eq!(frame.metrics["unmeasured"], 1.0);
    assert_eq!(frame.metrics["unmeasured_cause_http_5xx"], 1.0);
    for class in CLASSES {
        assert!(
            frame
                .metrics
                .contains_key(&format!("unmeasured_cause_{class}"))
        );
    }

    // 2026-09-25: One warning, naming everything a diagnosis needs, in the serialised
    // frame the run record stores.
    assert_eq!(frame.log.len(), 1, "{:?}", frame.log);
    assert_eq!(frame.log[0].level, LogLevel::Warn);
    let line = &frame.log[0].text;
    for field in [
        "leg=async",
        "lane=spec-off",
        "C=16",
        "sample=\"s2\"",
        "cause=http_5xx",
        "status=503",
        "finish_reason=-",
        "elapsed=",
        "KV cache exhausted, retry",
    ] {
        assert!(line.contains(field), "{field} missing from {line}");
    }
    let stored = serde_json::to_string(&frame).expect("frame serialises");
    assert!(stored.contains("cause=http_5xx"), "{stored}");

    // 2026-09-25: The table row and the summary tile both count it by cause.
    let table = frame.table.as_ref().expect("table");
    let col = table
        .columns
        .iter()
        .position(|c| c.title == "Causes")
        .expect("causes column");
    assert_eq!(table.rows[0][col].text, "async http_5xx 1");
    let tile = frame
        .summary
        .iter()
        .find(|s| s.label == "unmeasured causes")
        .expect("causes tile");
    assert_eq!(tile.value, "async http_5xx 1");
}

#[tokio::test]
async fn path_a_a_clean_run_records_no_cause() {
    let legs = vec![
        leg(Pass::Sync, 1, vec![good("s1").await]),
        leg(Pass::Async, 1, vec![good("s1").await]),
    ];
    let frame = report::terminal(&legs, &Default::default(), Duration::ZERO);
    assert_eq!(frame.verdict.as_ref().unwrap().kind, VerdictKind::Pass);
    assert!(frame.log.is_empty(), "{:?}", frame.log);
    assert!(
        CLASSES
            .iter()
            .all(|c| frame.metrics[&format!("unmeasured_cause_{c}")] == 0.0)
    );
    let tile = frame
        .summary
        .iter()
        .find(|s| s.label == "unmeasured causes");
    assert_eq!(tile.unwrap().value, "none");
}

#[tokio::test]
async fn path_a_a_failed_reference_is_one_cause_though_two_comparisons_lose_it() {
    let legs = vec![
        leg(
            Pass::Sync,
            4,
            vec![answered("s1", Some(OVERLOADED.to_string())).await],
        ),
        leg(Pass::Control, 4, vec![good("s1").await]),
        leg(Pass::Async, 4, vec![good("s1").await]),
    ];
    let s = score(&legs);
    assert_eq!(s.cells[0].async_vs_sync.unmeasured, 1);
    assert_eq!(s.cells[0].control_vs_sync.as_ref().unwrap().unmeasured, 1);
    assert_eq!(s.cells[0].unmeasured.len(), 1);
    assert_eq!(s.cells[0].unmeasured[0].pass, Pass::Sync);
    let v = verdict(&s);
    assert!(
        v.reason.contains("Causes: sync http_5xx 1."),
        "{}",
        v.reason
    );
}

#[tokio::test]
async fn path_a_a_missing_async_leg_names_every_sample_missing() {
    let legs = vec![leg(Pass::Sync, 1, vec![good("s1").await, good("s2").await])];
    let s = score(&legs);
    let causes: Vec<String> = s.cells[0]
        .unmeasured
        .iter()
        .map(|u| u.to_string())
        .collect();
    assert_eq!(causes.len(), 2, "{causes:?}");
    assert!(
        causes
            .iter()
            .all(|c| c.contains("leg=async") && c.contains("cause=missing"))
    );
    assert!(verdict(&s).reason.contains("Causes: async missing 2."));
}

#[tokio::test]
async fn path_b_timeout_5xx_and_transport_are_three_causes() {
    let slow = ask(&endpoint(None).await, "s1", Duration::from_millis(200)).await;
    let overloaded = answered("s1", Some(OVERLOADED.to_string())).await;
    let refused = ask(&refusing_endpoint().await, "s1", Duration::from_secs(10)).await;

    assert_eq!(failure(&slow).kind, FailureKind::Timeout);
    assert_eq!(failure(&slow).message, "request exceeded 0s");
    assert!(
        slow.elapsed >= Duration::from_millis(200),
        "{:?}",
        slow.elapsed
    );
    assert_eq!(failure(&overloaded).kind, FailureKind::Status(503));
    assert_eq!(failure(&refused).kind, FailureKind::Transport);
    assert!(
        failure(&refused).message.starts_with("connecting to "),
        "{}",
        failure(&refused).message
    );

    let legs = vec![
        leg(Pass::Sync, 1, vec![good("s1").await]),
        leg(Pass::Control, 1, vec![refused]),
        leg(Pass::Async, 1, vec![slow]),
        leg(Pass::Sync, 4, vec![good("s1").await]),
        leg(Pass::Async, 4, vec![overloaded]),
    ];
    let s = score(&legs);
    let lines: Vec<String> = s
        .cells
        .iter()
        .flat_map(|c| c.unmeasured.iter().map(|u| u.to_string()))
        .collect();
    assert_eq!(lines.len(), 3, "{lines:?}");
    assert!(lines[0].contains("leg=async") && lines[0].contains("cause=timeout status=-"));
    assert!(lines[1].contains("leg=sync-control") && lines[1].contains("cause=transport status=-"));
    assert!(lines[2].contains("C=4") && lines[2].contains("cause=http_5xx status=503"));
    let m = super::compare::metrics(&s, &Default::default());
    for class in ["timeout", "http_5xx", "transport"] {
        assert_eq!(m[&format!("unmeasured_cause_{class}")], 1.0, "{class}");
    }
    assert_eq!(m["unmeasured"], 3.0);
}

#[tokio::test]
async fn path_b_a_4xx_is_not_counted_as_a_5xx() {
    let rejected = answered(
        "s1",
        Some("HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n".to_string()),
    )
    .await;
    assert_eq!(failure(&rejected).kind, FailureKind::Status(400));
    let legs = vec![
        leg(Pass::Sync, 1, vec![good("s1").await]),
        leg(Pass::Async, 1, vec![rejected]),
    ];
    let m = super::compare::metrics(&score(&legs), &Default::default());
    assert_eq!(m["unmeasured_cause_http_4xx"], 1.0);
    assert_eq!(m["unmeasured_cause_http_5xx"], 0.0);
}

#[tokio::test]
async fn path_c_undecodable_frames_are_unmeasured_as_malformed_with_the_frame() {
    let garbage = answered(
        "s2",
        Some(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {\"choices\": [oops\n"
                .to_string(),
        ),
    )
    .await;
    let f = failure(&garbage);
    assert_eq!(f.kind, FailureKind::Malformed);
    assert_eq!(f.body, "{\"choices\": [oops");
    let legs = cell_with_async_failure(garbage).await;
    let frame = report::terminal(&legs, &Default::default(), Duration::ZERO);
    assert_eq!(frame.verdict.as_ref().unwrap().kind, VerdictKind::Fail);
    assert_eq!(frame.metrics["unmeasured_cause_malformed"], 1.0);
    assert!(
        frame.log[0].text.contains("cause=malformed"),
        "{}",
        frame.log[0].text
    );
    assert!(frame.log[0].text.contains("[oops"), "{}", frame.log[0].text);
}

#[tokio::test]
async fn path_c_broken_chunk_framing_is_malformed_not_a_crash() {
    let broken = answered(
        "s1",
        Some("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\ndata: x\r\n".to_string()),
    )
    .await;
    let f = failure(&broken);
    assert_eq!(f.kind, FailureKind::Malformed);
    assert!(
        f.message.starts_with("malformed chunk size"),
        "{}",
        f.message
    );
    assert!(f.body.starts_with("zz"), "{}", f.body);
}

#[tokio::test]
async fn path_c_a_non_json_error_body_is_kept_truncated() {
    let page = format!("<html>{}</html>", "proxy says no ".repeat(60));
    let proxied = answered(
        "s1",
        Some(format!(
            "HTTP/1.1 502 Bad Gateway\r\nContent-Length: {}\r\n\r\n{page}",
            page.len()
        )),
    )
    .await;
    let f = failure(&proxied);
    assert_eq!(f.kind, FailureKind::Status(502));
    assert_eq!(f.message, "endpoint returned \"HTTP/1.1 502 Bad Gateway\"");
    assert_eq!(f.body.len(), BODY_EXCERPT);
    assert!(page.starts_with(&f.body));
}

#[test]
fn an_excerpt_never_splits_a_character_and_a_logged_body_stays_on_one_line() {
    let text = format!("{}é", "a".repeat(BODY_EXCERPT - 1));
    let cut = excerpt(text.as_bytes());
    assert_eq!(cut, "a".repeat(BODY_EXCERPT - 1));

    let u = super::unmeasured::Unmeasured {
        pass: Pass::Async,
        lane: Lane::MtpForce,
        concurrency: 4,
        sample_id: "s1".into(),
        cause: Cause::Failed {
            failure: RequestFailure::new(FailureKind::ServerError, "server reported an error")
                .with_body(b"line one\nline two")
                .with_finish_reason(Some("stop".into())),
            elapsed: Duration::from_millis(1500),
        },
    };
    let line = u.to_string();
    assert!(!line.contains('\n'), "{line}");
    assert!(line.contains("finish_reason=stop elapsed=1.500s"), "{line}");
    assert!(line.contains("body=\"line one\\nline two\""), "{line}");
    assert!(line.contains("cause=server_error"), "{line}");
}
