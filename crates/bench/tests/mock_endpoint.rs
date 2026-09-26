// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A minimal OpenAI-compatible SSE server on loopback, for
//! driving benchmarks end to end. Included as a module by the other
//! integration tests.
//!
//! It answers `GET /v1/models` with a chunked body, and every other request
//! with a chunked SSE stream whose first event is split across two chunks
//! mid-line, the framing case the client's decoder must survive.
//!
//! Owner: bench (integration tests).
//! Invariants: none beyond the types.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

pub struct MockEndpoint {
    pub port: u16,
    /// 2026-09-26: Requests other than `GET /v1/models` answered so far.
    pub requests: Arc<AtomicUsize>,
}

/// 2026-09-26: Start the mock on an ephemeral port. Each reply waits `ttft`,
/// streams `tokens` filler deltas `gap` apart, then a usage chunk and
/// `[DONE]`.
pub async fn start(tokens: usize, ttft: Duration, gap: Duration) -> MockEndpoint {
    start_saying(None, tokens, ttft, gap).await
}

/// 2026-09-26: A server whose reply (`reply-<n>`) depends only on how many
/// requests it has already served.
///
/// This is the order dependence a KAT gate must detect: the same question in
/// different positions gets different answers. It lets a test show the gate
/// detecting it, not only passing a well-behaved server.
pub async fn start_indexed(ttft: Duration, gap: Duration) -> MockEndpoint {
    start_inner(Reply::Indexed, 0, ttft, gap).await
}

/// 2026-09-26: What a completion answers with.
#[derive(Clone)]
enum Reply {
    /// 2026-09-26: `tokens` deltas of filler.
    Filler,
    /// 2026-09-26: The same text every time, as one delta.
    Fixed(String),
    /// 2026-09-26: `reply-<n>` for the nth request served, counting from 0.
    Indexed,
}

/// 2026-09-26: As [`start`], but with `Some(reply)` every completion answers
/// with that text instead of filler. All three constructors share
/// `start_inner`, so the chunk splitting is identical.
pub async fn start_saying(
    reply: Option<String>,
    tokens: usize,
    ttft: Duration,
    gap: Duration,
) -> MockEndpoint {
    let reply = match reply {
        Some(text) => Reply::Fixed(text),
        None => Reply::Filler,
    };
    start_inner(reply, tokens, ttft, gap).await
}

async fn start_inner(reply: Reply, tokens: usize, ttft: Duration, gap: Duration) -> MockEndpoint {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let requests = Arc::new(AtomicUsize::new(0));
    let counter = requests.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let counter = counter.clone();
            let reply = reply.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 64 * 1024];
                let mut request = Vec::new();
                // 2026-09-26: Read until the headers end; the mock does not
                // need the body.
                loop {
                    let Ok(n) = socket.read(&mut buf).await else {
                        return;
                    };
                    if n == 0 {
                        return;
                    }
                    request.extend_from_slice(&buf[..n]);
                    if request.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                let head = String::from_utf8_lossy(&request).to_string();
                if head.starts_with("GET /v1/models") {
                    // 2026-09-26: A chunked body, not Content-Length: a reader
                    // that parses from the first `{` to the end of the buffer
                    // trips over the trailing `0\r\n\r\n`.
                    let body = r#"{"object":"list","data":[{"id":"mock"}]}"#;
                    let _ = socket
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n                              Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                        )
                        .await;
                    let _ = write_chunk(&mut socket, body).await;
                    let _ = socket.write_all(b"0\r\n\r\n").await;
                    return;
                }
                let served = counter.fetch_add(1, Ordering::Relaxed);
                let _ = socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                          Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                    )
                    .await;
                tokio::time::sleep(ttft).await;
                // 2026-09-26: A canned reply is one delta; filler is `tokens`
                // of them.
                let deltas: Vec<String> = match &reply {
                    Reply::Fixed(text) => vec![text.clone()],
                    Reply::Indexed => vec![format!("reply-{served}")],
                    Reply::Filler => (0..tokens).map(|i| format!("t{i} ")).collect(),
                };
                let tokens = deltas.len();
                for (i, delta) in deltas.iter().enumerate() {
                    let escaped = delta.replace('\\', "\\\\").replace('"', "\\\"");
                    let payload = format!(
                        "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{escaped}\"}}}}]}}\n"
                    );
                    if i == 0 {
                        // 2026-09-26: Split the first event across two chunks,
                        // mid-line.
                        let (a, b) = payload.split_at(payload.len() / 2);
                        if write_chunk(&mut socket, a).await.is_err() {
                            return;
                        }
                        if write_chunk(&mut socket, b).await.is_err() {
                            return;
                        }
                    } else if write_chunk(&mut socket, &payload).await.is_err() {
                        return;
                    }
                    tokio::time::sleep(gap).await;
                }
                let usage = format!(
                    "data: {{\"usage\":{{\"completion_tokens\":{tokens},\"prompt_tokens\":42,\
                     \"prompt_tokens_details\":{{\"cached_tokens\":40}}}},\"choices\":[]}}\n"
                );
                let _ = write_chunk(&mut socket, &usage).await;
                let _ = write_chunk(&mut socket, "data: [DONE]\n").await;
                let _ = socket.write_all(b"0\r\n\r\n").await;
                let _ = socket.shutdown().await;
            });
        }
    });
    MockEndpoint { port, requests }
}

async fn write_chunk(socket: &mut tokio::net::TcpStream, text: &str) -> std::io::Result<()> {
    socket
        .write_all(format!("{:x}\r\n{text}\r\n", text.len()).as_bytes())
        .await
}
