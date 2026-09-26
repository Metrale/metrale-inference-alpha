// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A minimal OpenAI-compatible client over raw HTTP/1.1 on
//! `tokio::net::TcpStream`, with no TLS: one streaming chat call, measured
//! ([`chat_stream`]), plus the blocking, GET, gap and failure helpers in the
//! submodules. Chunked transfer-encoding is decoded by the private `reader`
//! module before SSE lines are split, so a chunk boundary inside a `data:`
//! line does not lose a token.
//!
//! Owner: bench (HTTP client).
//! Invariants:
//! - A 200 stream that ends with neither `[DONE]` nor a `finish_reason` is an
//!   error, never a success.
//! - An in-band SSE error frame is an error, never a success.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::plugin::TargetEndpoint;

/// 2026-09-26: A tool call assembled from streamed deltas.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// 2026-09-26: The streamed argument text, concatenated and not parsed.
    pub arguments: String,
}

/// 2026-09-26: Everything one request produced, plus its timings.
#[derive(Clone, Debug, Default)]
pub struct ChatOutcome {
    pub text: String,
    /// 2026-09-26: The streamed `reasoning_content`, kept apart from `text`.
    /// Each non-empty reasoning delta still counts toward `completion_tokens`
    /// and can start the TTFT clock.
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    /// 2026-09-26: Client clock: request write → first chunk that carried a
    /// reasoning, content or tool-call delta.
    pub ttft_ms: Option<f64>,
    /// 2026-09-26: Client-clock inter-token latency, called TPOT here: first
    /// token-carrying chunk → end of stream (`[DONE]` or EOF), over
    /// `completion_tokens − 1` ([`itl_ms`]). The window ends at the final
    /// chunk, not the last token. `None` below two tokens.
    pub tpot_ms: Option<f64>,
    /// 2026-09-26: Client clock: request write → end of stream, the same end
    /// instant as `tpot_ms`.
    pub e2e_ms: f64,
    /// 2026-09-26: Gaps between socket reads that carried a token ([`gaps`]).
    pub arrival_gaps: GapSample,
    /// 2026-09-26: One per token-carrying reasoning or content delta; a
    /// `usage.completion_tokens` replaces the running count, and later deltas
    /// add to it.
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub cached_prompt_tokens: usize,
    /// 2026-09-26: `usage.time_to_first_token_ms`, on the server's clock.
    pub server_ttft_ms: Option<f64>,
    /// 2026-09-26: `usage."response_token/s"`, which this engine's server
    /// computes as `(completion_tokens − 1) / decode seconds`
    /// (`ir::Usage::decode_rate_tok_s` in the server crate).
    pub server_tps: Option<f64>,
    /// 2026-09-26: Accepted speculative draft tokens
    /// (`usage.completion_tokens_details.accepted_prediction_tokens`). `None`
    /// when the field is absent, so it differs from `Some(0)`.
    pub accepted_prediction_tokens: Option<usize>,
    /// 2026-09-26: `usage.decode_time_ms`, the numerator of
    /// [`Self::server_tpot_ms`]. `None` when absent.
    pub server_decode_time_ms: Option<f64>,
    /// 2026-09-26: `usage.total_time_ms`. `None` when absent.
    pub server_total_time_ms: Option<f64>,
}

impl ChatOutcome {
    /// 2026-09-26: Server-clock inter-token latency: `decode_time_ms /
    /// (completion_tokens − 1)`, the same [`itl_ms`] rule as
    /// [`Self::tpot_ms`]. `None` without the window or below two tokens.
    pub fn server_tpot_ms(&self) -> Option<f64> {
        itl_ms(self.server_decode_time_ms?, self.completion_tokens)
    }
}

/// 2026-09-26: POST `body` to `/v1/chat/completions` as an SSE request and
/// measure it. A request that exceeds `timeout` fails as
/// `FailureKind::Timeout`.
pub async fn chat_stream(
    target: &TargetEndpoint,
    body: &Value,
    timeout: Duration,
) -> Result<ChatOutcome> {
    tokio::time::timeout(timeout, chat_stream_inner(target, body))
        .await
        .map_err(|_| {
            RequestFailure::new(
                FailureKind::Timeout,
                format!("request exceeded {:.0}s", timeout.as_secs_f64()),
            )
        })?
}

async fn chat_stream_inner(target: &TargetEndpoint, body: &Value) -> Result<ChatOutcome> {
    let (host, port) = target.host_port()?;
    let payload = serde_json::to_string(body)?;
    let mut sock = TcpStream::connect((host.as_str(), port))
        .await
        .with_context(|| format!("connecting to {}", target.base_url))?;
    let _ = sock.set_nodelay(true);
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {host}:{port}\r\n\
         Content-Type: application/json\r\nAccept: text/event-stream\r\n\
         Connection: close\r\nContent-Length: {}\r\n\r\n{payload}",
        payload.len()
    );

    let started = Instant::now();
    sock.write_all(request.as_bytes()).await.context("write")?;

    let mut reader = Reader::default();
    let mut out = ChatOutcome::default();
    let mut first_delta: Option<Instant> = None;
    // 2026-09-26: The previous socket read that carried a token; gaps are per
    // read, not per delta.
    let mut last_arrival: Option<Instant> = None;
    // 2026-09-26: The first `data:` frame that was not JSON, kept so a stream
    // that then ends without a terminal frame reports as malformed.
    let mut undecodable: Option<String> = None;
    let mut buf = [0u8; 16 * 1024];
    loop {
        let n = sock.read(&mut buf).await.context("read")?;
        if n == 0 {
            // 2026-09-26: EOF. A non-200 response still being collected is
            // reported now with whatever body arrived.
            reader.finish()?;
            // 2026-09-26: A 200 stream that ends with neither `[DONE]` nor a
            // `finish_reason` was cut off: `Truncated`, or `Malformed` when a
            // `data:` frame did not parse.
            if out.finish_reason.is_none() {
                let message = format!(
                    "stream ended without a terminal frame (no [DONE], no finish_reason) after \
                     {} token(s): truncated, not completed",
                    out.completion_tokens
                );
                return Err(match &undecodable {
                    Some(frame) => RequestFailure::new(FailureKind::Malformed, message)
                        .with_body(frame.as_bytes()),
                    None => RequestFailure::new(FailureKind::Truncated, message),
                }
                .into());
            }
            break;
        }
        let arrived = Instant::now();
        let mut carried = false;
        let mut done = false;
        for line in reader.push(&buf[..n])? {
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                done = true;
                break;
            }
            let Ok(chunk) = serde_json::from_str::<Value>(data) else {
                undecodable.get_or_insert_with(|| data.to_string());
                continue;
            };
            // 2026-09-26: An in-band error frame fails the request even though
            // the status was 200.
            if let Some(msg) = stream_error(&chunk) {
                return Err(RequestFailure::new(
                    FailureKind::ServerError,
                    format!("server reported an error mid-stream: {msg}"),
                )
                .with_body(data.as_bytes())
                .with_finish_reason(out.finish_reason.clone())
                .into());
            }
            if apply_chunk(&chunk, &mut out) {
                first_delta.get_or_insert_with(Instant::now);
                carried = true;
            }
        }
        // 2026-09-26: Book the arrival before honouring `[DONE]`, which can
        // share a read with the last token.
        if carried {
            if let Some(prev) = last_arrival {
                out.arrival_gaps
                    .push(arrived.duration_since(prev).as_secs_f64() * 1000.0);
            }
            last_arrival = Some(arrived);
        }
        if done {
            break;
        }
    }
    // 2026-09-26: One end instant for `e2e_ms` and the ITL window, so
    // `(e2e − ttft) / (n − 1)` equals `tpot_ms`.
    let ended = Instant::now();
    out.e2e_ms = ended.duration_since(started).as_secs_f64() * 1000.0;
    out.ttft_ms = first_delta.map(|t| t.duration_since(started).as_secs_f64() * 1000.0);
    out.tpot_ms = first_delta.and_then(|f| {
        itl_ms(
            ended.duration_since(f).as_secs_f64() * 1000.0,
            out.completion_tokens,
        )
    });
    Ok(out)
}

/// 2026-09-26: The message of an in-band SSE error frame, if `chunk` is one:
/// a non-null `error` that is a string, an object (its `message`, else the
/// whole object) or any other value. The server's frame shapes are pinned in
/// its `api/tests/error_frames.rs`.
pub(crate) fn stream_error(chunk: &Value) -> Option<String> {
    let err = chunk.get("error")?;
    if err.is_null() {
        return None;
    }
    Some(match err {
        Value::String(m) => m.clone(),
        Value::Object(o) => o
            .get("message")
            .and_then(Value::as_str)
            .map_or_else(|| err.to_string(), str::to_string),
        other => other.to_string(),
    })
}

/// 2026-09-26: Fold one SSE chunk into the outcome. Returns true when it
/// carried a non-empty reasoning or content delta, or any tool-call delta.
fn apply_chunk(chunk: &Value, out: &mut ChatOutcome) -> bool {
    if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
        if let Some(c) = usage.get("completion_tokens").and_then(Value::as_u64) {
            out.completion_tokens = c as usize;
        }
        if let Some(p) = usage.get("prompt_tokens").and_then(Value::as_u64) {
            out.prompt_tokens = p as usize;
        }
        if let Some(c) = usage
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(Value::as_u64)
        {
            out.cached_prompt_tokens = c as usize;
        }
        if let Some(v) = usage.get("time_to_first_token_ms").and_then(Value::as_f64) {
            out.server_ttft_ms = Some(v);
        }
        if let Some(v) = usage.get("response_token/s").and_then(Value::as_f64) {
            out.server_tps = Some(v);
        }
        if let Some(v) = usage.get("decode_time_ms").and_then(Value::as_f64) {
            out.server_decode_time_ms = Some(v);
        }
        if let Some(v) = usage.get("total_time_ms").and_then(Value::as_f64) {
            out.server_total_time_ms = Some(v);
        }
        if let Some(v) = usage
            .get("completion_tokens_details")
            .and_then(|d| d.get("accepted_prediction_tokens"))
            .and_then(Value::as_u64)
        {
            out.accepted_prediction_tokens = Some(v as usize);
        }
    }
    let Some(choice) = chunk.get("choices").and_then(|c| c.get(0)) else {
        return false;
    };
    if let Some(r) = choice.get("finish_reason").and_then(Value::as_str) {
        out.finish_reason = Some(r.to_string());
    }
    let Some(delta) = choice.get("delta") else {
        return false;
    };
    let mut carried = false;
    // 2026-09-26: A reasoning delta is a token: it counts and it can start the
    // TTFT clock, or a thinking model's TTFT would include its whole
    // reasoning block.
    if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str)
        && !reasoning.is_empty()
    {
        out.reasoning.push_str(reasoning);
        out.completion_tokens += 1;
        carried = true;
    }
    if let Some(content) = delta.get("content").and_then(Value::as_str)
        && !content.is_empty()
    {
        out.text.push_str(content);
        out.completion_tokens += 1;
        carried = true;
    }
    if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            let idx = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            if out.tool_calls.len() <= idx {
                out.tool_calls.resize(idx + 1, ToolCall::default());
            }
            let slot = &mut out.tool_calls[idx];
            if let Some(id) = call.get("id").and_then(Value::as_str) {
                slot.id.push_str(id);
            }
            if let Some(f) = call.get("function") {
                if let Some(name) = f.get("name").and_then(Value::as_str) {
                    slot.name.push_str(name);
                }
                if let Some(args) = f.get("arguments").and_then(Value::as_str) {
                    slot.arguments.push_str(args);
                }
            }
            carried = true;
        }
    }
    carried
}

/// 2026-09-26: The trimmed `error.message` of an OpenAI-shaped error body.
/// `None` when the body is not JSON with a non-empty string there, so callers
/// can fall back to the status line.
pub fn message_from_body(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let msg = v.get("error")?.get("message")?.as_str()?.trim();
    (!msg.is_empty()).then(|| msg.to_string())
}

/// 2026-09-26: The message from a complete non-200 HTTP response (status
/// line, headers and body), decoded by the same private reader as streams,
/// chunked bodies included. `None` unless the response is a non-200 whose
/// body [`message_from_body`] can read.
pub fn error_message_from_response(raw: &[u8]) -> Option<String> {
    let mut r = Reader::default();
    // 2026-09-26: Both return the error being reported; only the decoded body
    // is needed.
    let _ = r.push(raw);
    let _ = r.finish();
    r.error_status.as_ref()?;
    message_from_body(String::from_utf8_lossy(&r.body).trim())
}

/// 2026-09-26: Cap on how much of a non-200 body to buffer before reporting
/// it.
pub(super) const MAX_ERROR_BODY: usize = 64 * 1024;

pub(super) fn is_chunked(head: &str) -> bool {
    head.lines().any(|line| {
        let line = line.to_ascii_lowercase();
        line.starts_with("transfer-encoding:") && line.contains("chunked")
    })
}

pub(super) fn status_is_success(status_line: &str) -> bool {
    status_line.split_whitespace().nth(1) == Some("200")
}

/// 2026-09-26: `Content-Length` from a header block, if declared and
/// parseable.
pub(super) fn content_length(head: &str) -> Option<usize> {
    head.lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split_once(':'))
        .and_then(|(_, v)| v.trim().parse().ok())
}

mod reader;
use reader::Reader;

mod blocking;
pub use blocking::{BlockingOutcome, chat_blocking, responses_blocking};

mod get;
pub use get::{fetch_hardware, get_json, list_models, probe};

pub mod gaps;
pub use gaps::{GapSample, GapStats, itl_ms};

pub mod failure;
pub use failure::{FailureKind, RequestFailure, classify};

pub(super) fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
#[path = "http_tests.rs"]
mod tests;
