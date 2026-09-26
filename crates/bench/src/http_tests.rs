// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the HTTP client: response framing, SSE chunk
//! folding and error bodies. The child module `itl` uses the `sse` helper
//! here, and `stream_end` the loopback mock `endpoint_answering`.
//!
//! Owner: bench (HTTP client).
//! Invariants: none beyond the types.

use super::*;

fn sse(payload: &str) -> Value {
    serde_json::from_str(payload).unwrap()
}

async fn endpoint_answering(response: String) -> TargetEndpoint {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let port = listener.local_addr().expect("address").port();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut request = [0u8; 4096];
        let _ = socket.read(&mut request).await.expect("read request");
        socket.write_all(response.as_bytes()).await.expect("reply");
    });
    TargetEndpoint::local(port, "mock")
}

#[test]
fn chunked_body_split_mid_line_still_yields_one_intact_line() {
    // 2026-09-26: A chunk boundary in the middle of a `data:` line must still
    // yield one intact line.
    let mut r = Reader::default();
    let head = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: Chunked\r\n\r\n";
    assert!(r.push(head).unwrap().is_empty());

    let first = "data: {\"choices\":[{\"de";
    let second = "lta\":{\"content\":\"hi\"}}]}\n";
    let mut wire = Vec::new();
    wire.extend_from_slice(format!("{:x}\r\n{first}\r\n", first.len()).as_bytes());
    let lines = r.push(&wire).unwrap();
    assert!(lines.is_empty(), "no complete line yet, got {lines:?}");

    let mut wire2 = Vec::new();
    wire2.extend_from_slice(format!("{:x}\r\n{second}\r\n", second.len()).as_bytes());
    let lines = r.push(&wire2).unwrap();
    assert_eq!(lines, [r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#]);
}

#[test]
fn identity_body_is_read_straight_through() {
    let mut r = Reader::default();
    let lines = r
        .push(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: a\ndata: b\n")
        .unwrap();
    assert_eq!(lines, vec!["data: a", "data: b"]);
}

#[test]
fn a_non_200_status_is_an_error_not_an_empty_stream() {
    // 2026-09-26: A non-200 never reads as a successful empty stream. The
    // reader waits for the body; with no Content-Length and no body, as here,
    // the error is raised at EOF by `finish`.
    let mut r = Reader::default();
    let lines = r.push(b"HTTP/1.1 404 Not Found\r\n\r\n").unwrap();
    assert!(lines.is_empty(), "a failed response yields no data lines");
    let err = r.finish().unwrap_err().to_string();
    assert!(err.contains("404"), "{err}");

    let mut lookalike = Reader::default();
    lookalike
        .push(b"HTTP/1.1 2000 Not-A-Status\r\nContent-Length: 0\r\n\r\n")
        .expect_err("only the exact 200 status is successful");
}

#[tokio::test]
async fn endpoint_probe_rejects_a_status_code_lookalike() {
    let target = endpoint_answering(
        "HTTP/1.1 2000 Not-A-Status\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into(),
    )
    .await;
    let err = probe(&target, Duration::from_secs(2))
        .await
        .expect_err("only status 200 is reachable");
    assert!(err.to_string().contains("2000"), "{err}");
}

#[test]
fn content_deltas_accumulate_text_and_token_count() {
    let mut out = ChatOutcome::default();
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"content":"He"}}]}"#),
        &mut out
    ));
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"content":"llo"}}]}"#),
        &mut out
    ));
    // 2026-09-26: A role-only chunk carries no token.
    assert!(!apply_chunk(
        &sse(r#"{"choices":[{"delta":{"role":"assistant"}}]}"#),
        &mut out
    ));
    assert_eq!(out.text, "Hello");
    assert_eq!(out.completion_tokens, 2);
}

#[test]
fn a_reasoning_delta_is_a_token_and_starts_the_clock() {
    // 2026-09-26: A thinking model streams `reasoning_content` first; that
    // delta must count as a token so it starts the TTFT clock.
    let mut out = ChatOutcome::default();
    assert!(
        apply_chunk(
            &sse(r#"{"choices":[{"delta":{"reasoning_content":"Let me think"}}]}"#),
            &mut out
        ),
        "a reasoning delta must count as carried, or it cannot start the TTFT clock"
    );
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"content":"4"}}]}"#),
        &mut out
    ));
    // 2026-09-26: Reasoning stays out of `text`, which scorers parse for the
    // answer.
    assert_eq!(out.text, "4", "reasoning must not leak into the answer");
    assert_eq!(out.reasoning, "Let me think");
    assert_eq!(
        out.completion_tokens, 2,
        "both are decoded tokens -- the server's usage.completion_tokens \
         includes reasoning_tokens, and the streamed count must agree"
    );
}

#[test]
fn an_empty_reasoning_delta_carries_nothing() {
    let mut out = ChatOutcome::default();
    assert!(!apply_chunk(
        &sse(r#"{"choices":[{"delta":{"reasoning_content":""}}]}"#),
        &mut out
    ));
    assert_eq!(out.completion_tokens, 0);
}

#[test]
fn tool_call_deltas_assemble_by_index() {
    let mut out = ChatOutcome::default();
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"id":"c1","function":{"name":"get_","arguments":"{\"a\""}}]}}]}"#),
        &mut out,
    ));
    assert!(apply_chunk(
        &sse(r#"{"choices":[{"delta":{"tool_calls":[
                {"index":1,"id":"c2","function":{"name":"clock","arguments":"{}"}},
                {"index":0,"function":{"name":"weather","arguments":":1}"}}]}}]}"#),
        &mut out,
    ));
    assert_eq!(
        out.tool_calls,
        [
            ToolCall {
                id: "c1".into(),
                name: "get_weather".into(),
                arguments: r#"{"a":1}"#.into(),
            },
            ToolCall {
                id: "c2".into(),
                name: "clock".into(),
                arguments: "{}".into(),
            },
        ]
    );
}

#[test]
fn server_usage_overrides_the_streamed_delta_count() {
    let mut out = ChatOutcome::default();
    apply_chunk(&sse(r#"{"choices":[{"delta":{"content":"x"}}]}"#), &mut out);
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":37,"prompt_tokens":12,
                "prompt_tokens_details":{"cached_tokens":8}},"choices":[]}"#),
        &mut out,
    );
    assert_eq!(out.completion_tokens, 37);
    assert_eq!(out.prompt_tokens, 12);
    assert_eq!(out.cached_prompt_tokens, 8);
}

/// 2026-09-26: The server's timing extensions ride in `usage` and are kept
/// as sent (`quick-speed-bench` derives its TPOT from `server_tps`). Absent
/// extensions stay `None`, not 0.
#[test]
fn server_timing_extensions_are_captured_and_absent_ones_stay_none() {
    let mut out = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":49,"prompt_tokens":12,
                "time_to_first_token_ms":1451.2,"response_token/s":59.9},"choices":[]}"#),
        &mut out,
    );
    assert_eq!(out.server_ttft_ms, Some(1451.2));
    assert_eq!(out.server_tps, Some(59.9));

    let mut bare = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":3,"prompt_tokens":2},"choices":[]}"#),
        &mut bare,
    );
    assert_eq!(bare.server_ttft_ms, None);
    assert_eq!(bare.server_tps, None);
}

/// 2026-09-26: The accept count rides in `completion_tokens_details` and
/// keeps three cases apart, as decode-floor does: a reported count
/// (`Some(n)`), a reported zero (`Some(0)`), and no details object at all
/// (`None`), which is never turned into 0.
#[test]
fn accepted_prediction_tokens_are_captured_and_absence_stays_none() {
    let mut out = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":49,"prompt_tokens":12,
                "completion_tokens_details":{"reasoning_tokens":0,
                "accepted_prediction_tokens":31}},"choices":[]}"#),
        &mut out,
    );
    assert_eq!(out.accepted_prediction_tokens, Some(31));

    let mut zero = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":3,"prompt_tokens":2,
                "completion_tokens_details":{"accepted_prediction_tokens":0}},"choices":[]}"#),
        &mut zero,
    );
    assert_eq!(zero.accepted_prediction_tokens, Some(0));

    let mut bare = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":3,"prompt_tokens":2},"choices":[]}"#),
        &mut bare,
    );
    assert_eq!(bare.accepted_prediction_tokens, None);
}

#[test]
fn finish_reason_is_captured() {
    let mut out = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#),
        &mut out,
    );
    assert_eq!(out.finish_reason.as_deref(), Some("stop"));
}

/// 2026-09-26: A 503 with the server's JSON error body and a
/// `Content-Length`.
fn error_response(body: &str) -> Vec<u8> {
    format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

const NO_MODEL_BODY: &str = r#"{"error":{"message":"no model is loaded — Open the Library (press 4), choose a model and a recipe, and start it; then retry this request.","type":"model_not_loaded"}}"#;

#[test]
fn an_error_body_reaches_the_caller_instead_of_just_the_status_line() {
    let mut r = Reader::default();
    let err = r.push(&error_response(NO_MODEL_BODY)).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("503"), "keeps the status: {msg}");
    assert!(msg.contains("Library"), "and carries the hint: {msg}");
}

#[test]
fn an_error_body_arriving_after_its_headers_is_still_reported() {
    // 2026-09-26: Headers can land in a read of their own; the body that
    // follows must still be reported.
    let whole = error_response(NO_MODEL_BODY);
    let split = find(&whole, b"\r\n\r\n").unwrap() + 4;

    let mut r = Reader::default();
    assert!(
        r.push(&whole[..split]).is_ok(),
        "headers alone are not yet a verdict — the body is still coming"
    );
    let msg = format!("{}", r.push(&whole[split..]).unwrap_err());
    assert!(msg.contains("Library"), "hint survives the split: {msg}");
}

#[test]
fn a_chunked_error_body_is_decoded_before_it_is_parsed() {
    // 2026-09-26: A chunked error body with no Content-Length; collected
    // raw, it would not be JSON.
    let body = NO_MODEL_BODY;
    let raw = format!(
        "HTTP/1.1 503 Service Unavailable\r\ncontent-type: application/json\r\n\
         transfer-encoding: chunked\r\n\r\n{:X}\r\n{body}\r\n0\r\n\r\n",
        body.len()
    );
    let mut r = Reader::default();
    let msg = format!("{}", r.push(raw.as_bytes()).unwrap_err());
    assert!(msg.contains("503"), "{msg}");
    assert!(msg.contains("Library"), "chunked hint must survive: {msg}");
}

#[test]
fn a_chunked_error_split_across_reads_is_still_decoded() {
    let body = NO_MODEL_BODY;
    let raw = format!(
        "HTTP/1.1 503 Service Unavailable\r\ntransfer-encoding: chunked\r\n\r\n\
         {:X}\r\n{body}\r\n0\r\n\r\n",
        body.len()
    );
    let bytes = raw.as_bytes();
    // 2026-09-26: Split inside the chunk data.
    let cut = bytes.len() - 40;
    let mut r = Reader::default();
    assert!(r.push(&bytes[..cut]).is_ok(), "partial chunk: keep waiting");
    let msg = format!("{}", r.push(&bytes[cut..]).unwrap_err());
    assert!(msg.contains("Library"), "{msg}");
}

#[test]
fn an_error_without_content_length_is_reported_at_eof() {
    // 2026-09-26: No length and no chunking means the body ends at close, so
    // the error is reported at EOF.
    let mut r = Reader::default();
    let raw = format!("HTTP/1.1 500 Internal Server Error\r\n\r\n{NO_MODEL_BODY}");
    assert!(r.push(raw.as_bytes()).is_ok(), "still open, still waiting");
    let msg = format!("{}", r.finish().unwrap_err());
    assert!(msg.contains("500"), "{msg}");
    assert!(msg.contains("Library"), "{msg}");
}

#[test]
fn an_unparseable_error_body_still_reports_the_status() {
    // 2026-09-26: A body that is not an OpenAI error still reports the
    // status.
    let mut r = Reader::default();
    let raw = "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 9\r\n\r\n<html></".to_string();
    let _ = r.push(raw.as_bytes());
    let msg = format!("{}", r.finish().unwrap_err());
    assert!(msg.contains("502"), "{msg}");
}

#[test]
fn a_200_response_is_completely_unaffected_by_the_error_path() {
    let mut r = Reader::default();
    let raw = "HTTP/1.1 200 OK\r\n\r\ndata: {\"a\":1}\n";
    let lines = r.push(raw.as_bytes()).unwrap();
    assert_eq!(lines, vec!["data: {\"a\":1}".to_string()]);
    assert!(r.finish().is_ok(), "no pending error on a good response");
}

#[test]
fn a_huge_error_body_is_capped_rather_than_buffered_without_limit() {
    let mut r = Reader::default();
    let big = "x".repeat(MAX_ERROR_BODY + 4096);
    let raw = format!("HTTP/1.1 503 Service Unavailable\r\n\r\n{big}");
    // 2026-09-26: Fails at the cap rather than growing until the sender
    // stops.
    assert!(r.push(raw.as_bytes()).is_err());
    assert_eq!(r.body.len(), MAX_ERROR_BODY, "the retained body is capped");

    let mut chunked = Reader::default();
    let wire = format!(
        "HTTP/1.1 503 Service Unavailable\r\nTransfer-Encoding: chunked\r\n\r\n\
         {:X}\r\n{big}\r\n0\r\n\r\n",
        big.len()
    );
    assert!(chunked.push(wire.as_bytes()).is_err());
    assert_eq!(
        chunked.body.len(),
        MAX_ERROR_BODY,
        "chunk framing cannot bypass the same cap"
    );
}

#[test]
fn message_from_body_ignores_bodies_that_are_not_openai_shaped() {
    assert_eq!(message_from_body(""), None);
    assert_eq!(message_from_body("{"), None);
    assert_eq!(message_from_body(r#"{"error":"str"}"#), None);
    assert_eq!(message_from_body(r#"{"error":{"message":"  "}}"#), None);
    assert_eq!(
        message_from_body(r#"{"error":{"message":"boom"}}"#).as_deref(),
        Some("boom")
    );
}

#[test]
fn a_whole_chunked_error_response_yields_its_message_in_one_shot() {
    // 2026-09-26: The TUI chat pane holds the entire response, not a stream,
    // and gets the same de-chunked answer as the streaming reader.
    let body = NO_MODEL_BODY;
    let raw = format!(
        "HTTP/1.1 503 Service Unavailable\r\ntransfer-encoding: chunked\r\n\r\n\
         {:X}\r\n{body}\r\n0\r\n\r\n",
        body.len()
    );
    let msg = error_message_from_response(raw.as_bytes()).expect("a message is there");
    assert!(msg.contains("Library"), "{msg}");
}

#[test]
fn a_successful_response_has_no_error_message_to_extract() {
    let raw = "HTTP/1.1 200 OK\r\n\r\ndata: {\"a\":1}\n";
    assert_eq!(error_message_from_response(raw.as_bytes()), None);
}

#[test]
fn an_error_response_with_an_unreadable_body_yields_nothing_rather_than_junk() {
    let raw = "HTTP/1.1 502 Bad Gateway\r\n\r\n<html>nope</html>";
    assert_eq!(error_message_from_response(raw.as_bytes()), None);
}

#[tokio::test]
async fn blocking_client_reports_the_decoded_openai_error() {
    let response = format!(
        "HTTP/1.1 503 Service Unavailable\r\nTransfer-Encoding: chunked\r\n\
         Connection: close\r\n\r\n{:X}\r\n{NO_MODEL_BODY}\r\n0\r\n\r\n",
        NO_MODEL_BODY.len()
    );
    let target = endpoint_answering(response).await;
    let err = chat_blocking(
        &target,
        &serde_json::json!({"messages": []}),
        Duration::from_secs(2),
    )
    .await
    .expect_err("503 is an error");
    let message = err.to_string();
    assert!(message.contains("503"), "{message}");
    assert!(message.contains("Library"), "{message}");
    assert!(!message.contains(r#"{"error"#), "decoded detail: {message}");
}

#[path = "http_itl_tests.rs"]
mod itl;

#[path = "http_stream_end_tests.rs"]
mod stream_end;
