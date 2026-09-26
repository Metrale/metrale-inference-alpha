// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Chat pane's client: `POST /v1/chat/completions` over plain
//! HTTP/1.1 to `127.0.0.1:<port>`, with the SSE deltas sent as `ChatDelta`s on
//! the std mpsc channel that `ChatState::pump` drains.
//!
//! Owner: server tui.
//! Invariants:
//! - Every return from `stream_chat` sends one terminal delta (`Done` or
//!   `Error`) as its last message, except the return after a failed send to
//!   `tx`. A dropped `stream_chat` future sends nothing.

use std::sync::mpsc::Sender;
use std::time::Instant;

use super::chat::ChatDelta;
use super::chat_thinking::ThinkingRequest;

/// 2026-09-26: A non-200 response is read until the buffer, headers included,
/// reaches this many bytes (it can overshoot by one 8 KiB read), or a read
/// returns 0 or fails.
const MAX_ERROR_BODY: usize = 64 * 1024;

/// 2026-09-26: Clocks and counts for one reply. `first_any` is the first
/// non-empty `reasoning_content` or `content` delta (the TTFT); `first_answer`
/// is the first non-empty `content` delta.
#[derive(Default)]
struct Clocks {
    first_any: Option<Instant>,
    first_answer: Option<Instant>,
    first_reasoning: Option<Instant>,
    last_reasoning: Option<Instant>,
    tokens: usize,
    reasoning_tokens: usize,
}

impl Clocks {
    fn done(&self, started: Instant) -> ChatDelta {
        let ms = |t: Instant| (t - started).as_secs_f64() * 1000.0;
        // 2026-09-26: The thinking span ends at the first answer delta, or at
        // the last reasoning delta when no answer arrived.
        let think_ms = self.first_reasoning.and_then(|start| {
            self.first_answer
                .or(self.last_reasoning)
                .map(|end| (end - start).as_secs_f64() * 1000.0)
        });
        let total = self.tokens + self.reasoning_tokens;
        // 2026-09-26: Answer and reasoning deltas together, over the time since
        // `first_any`.
        let tok_per_s = self.first_any.filter(|_| total > 0).map(|t| {
            let gen_secs = t.elapsed().as_secs_f64().max(1e-3);
            total as f64 / gen_secs
        });
        ChatDelta::Done {
            ttft_ms: self.first_any.map(ms),
            answer_ttft_ms: self.first_answer.map(ms),
            think_ms,
            tok_per_s,
            tokens: self.tokens,
            reasoning_tokens: self.reasoning_tokens,
        }
    }
}

/// 2026-09-26: The request body. `chat_template_kwargs.enable_thinking` is
/// present only when [`ThinkingRequest::enable_thinking`] returns a value, so
/// `Auto` sends no `chat_template_kwargs`.
fn request_body(messages: &[(String, String)], thinking: ThinkingRequest) -> String {
    let mut body = serde_json::json!({
        "model": "metrale-tui",
        "stream": true,
        "messages": messages
            .iter()
            .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
            .collect::<Vec<_>>(),
    });
    if let Some(enable) = thinking.enable_thinking() {
        body["chat_template_kwargs"] = serde_json::json!({ "enable_thinking": enable });
    }
    body.to_string()
}

/// 2026-09-26: POST the chat request and forward its SSE deltas on `tx`.
/// Connect, write and read errors and a non-200 status end the stream with an
/// `Error`; `[DONE]` or EOF ends it with `Done`; a failed send to `tx` ends it
/// with nothing more.
pub(super) async fn stream_chat(
    port: u16,
    messages: Vec<(String, String)>,
    thinking: ThinkingRequest,
    tx: Sender<ChatDelta>,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let started = Instant::now();
    let mut c = Clocks::default();
    let body = request_body(&messages, thinking);

    let mut stream = match tokio::net::TcpStream::connect(("127.0.0.1", port)).await {
        Ok(s) => s,
        Err(e) => {
            let _ = tx.send(ChatDelta::Error(format!("connect: {e}")));
            return;
        }
    };
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\nAccept: text/event-stream\r\n\
         Connection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    if let Err(e) = stream.write_all(req.as_bytes()).await {
        let _ = tx.send(ChatDelta::Error(format!("write: {e}")));
        return;
    }

    // 2026-09-26: The body is split on `\n` and only `data: ` lines are read,
    // so chunked-encoding size lines are skipped. A `data:` line that a chunk
    // boundary cuts in two fails to parse and is dropped.
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let mut header_done = false;
    let mut consumed = 0usize;
    loop {
        let n = match stream.read(&mut tmp).await {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                let _ = tx.send(ChatDelta::Error(format!("read: {e}")));
                return;
            }
        };
        buf.extend_from_slice(&tmp[..n]);
        if !header_done {
            if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..pos]).to_string();
                if !head.starts_with("HTTP/1.1 200") && !head.starts_with("HTTP/1.0 200") {
                    let status = head.lines().next().unwrap_or("?").to_string();
                    // 2026-09-26: Read the rest of the response, so the body's
                    // error message can follow the status line; the headers
                    // can arrive in a read of their own.
                    while buf.len() < MAX_ERROR_BODY {
                        match stream.read(&mut tmp).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                    }
                    // 2026-09-26: `error_message_from_response` takes the whole
                    // response, headers included, and de-chunks the body itself.
                    let msg = metrale_bench::http::error_message_from_response(&buf)
                        .map(|m| format!("{status} — {m}"))
                        .unwrap_or(status);
                    let _ = tx.send(ChatDelta::Error(msg));
                    return;
                }
                consumed = pos + 4;
                header_done = true;
            } else {
                continue;
            }
        }
        while let Some(nl) = buf[consumed..].iter().position(|b| *b == b'\n') {
            let line_end = consumed + nl;
            let line = String::from_utf8_lossy(&buf[consumed..line_end])
                .trim()
                .to_string();
            consumed = line_end + 1;
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                let _ = tx.send(c.done(started));
                return;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            let delta = &v["choices"][0]["delta"];
            // 2026-09-26: A delta carrying both fields is forwarded reasoning
            // first.
            if let Some(text) = nonempty(&delta["reasoning_content"]) {
                let now = Instant::now();
                c.first_any.get_or_insert(now);
                c.first_reasoning.get_or_insert(now);
                c.last_reasoning = Some(now);
                c.reasoning_tokens += 1;
                if tx.send(ChatDelta::Reasoning(text)).is_err() {
                    return;
                }
            }
            if let Some(text) = nonempty(&delta["content"]) {
                let now = Instant::now();
                c.first_any.get_or_insert(now);
                c.first_answer.get_or_insert(now);
                c.tokens += 1;
                if tx.send(ChatDelta::Token(text)).is_err() {
                    return;
                }
            }
        }
    }
    // 2026-09-26: EOF without `[DONE]` still reports the measurements.
    let _ = tx.send(c.done(started));
}

fn nonempty(v: &serde_json::Value) -> Option<String> {
    v.as_str().filter(|s| !s.is_empty()).map(str::to_string)
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// 2026-09-26: The loopback HTTP server the Chat tests run against;
/// `pub(super)` so the `chat` test modules share it.
#[cfg(test)]
#[path = "chat_fake_server.rs"]
pub(super) mod fake;

#[cfg(test)]
#[path = "chat_stream_tests.rs"]
mod stream_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn kwargs(req: ThinkingRequest) -> Option<serde_json::Value> {
        let body: serde_json::Value =
            serde_json::from_str(&request_body(&[("user".into(), "hi".into())], req))
                .expect("valid JSON");
        body.get("chat_template_kwargs").cloned()
    }

    #[test]
    fn auto_sends_no_thinking_key_at_all() {
        assert_eq!(kwargs(ThinkingRequest::Auto), None);
    }

    #[test]
    fn off_and_on_send_the_key_that_is_actually_honored() {
        assert_eq!(
            kwargs(ThinkingRequest::Off),
            Some(serde_json::json!({"enable_thinking": false}))
        );
        assert_eq!(
            kwargs(ThinkingRequest::On),
            Some(serde_json::json!({"enable_thinking": true}))
        );
    }

    #[test]
    fn the_ttft_clock_stops_on_the_first_delta_of_any_kind() {
        let started = Instant::now();
        let mut c = Clocks::default();
        let think = Instant::now();
        c.first_any = Some(think);
        c.first_reasoning = Some(think);
        c.reasoning_tokens = 200;
        std::thread::sleep(std::time::Duration::from_millis(15));
        let answer = Instant::now();
        c.first_answer = Some(answer);
        c.tokens = 10;
        let ChatDelta::Done {
            ttft_ms,
            answer_ttft_ms,
            think_ms,
            ..
        } = c.done(started)
        else {
            panic!("Done")
        };
        let (ttft, ans) = (ttft_ms.expect("ttft"), answer_ttft_ms.expect("answer"));
        assert!(ans > ttft, "the answer landed after the first token");
        assert!(think_ms.expect("thought") >= 10.0, "and it thought first");
    }

    #[test]
    fn a_reply_that_only_ever_thought_still_reports_a_span() {
        let started = Instant::now();
        let mut c = Clocks::default();
        let t = Instant::now();
        c.first_any = Some(t);
        c.first_reasoning = Some(t);
        c.last_reasoning = Some(t + std::time::Duration::from_millis(500));
        c.reasoning_tokens = 247;
        let ChatDelta::Done {
            think_ms,
            tokens,
            reasoning_tokens,
            tok_per_s,
            ..
        } = c.done(started)
        else {
            panic!("Done")
        };
        assert_eq!(tokens, 0, "no answer arrived");
        assert_eq!(reasoning_tokens, 247);
        assert!((think_ms.expect("thought") - 500.0).abs() < 1.0);
        assert!(
            tok_per_s.is_some(),
            "reasoning tokens are still decode work"
        );
    }

    #[test]
    fn a_reply_with_no_tokens_at_all_reports_no_rate() {
        let c = Clocks::default();
        let ChatDelta::Done {
            ttft_ms, tok_per_s, ..
        } = c.done(Instant::now())
        else {
            panic!("Done")
        };
        assert!(ttft_ms.is_none());
        assert!(tok_per_s.is_none(), "no division by an empty reply");
    }
}
