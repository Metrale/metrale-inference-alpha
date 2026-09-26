// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Non-streaming requests ([`chat_blocking`],
//! [`responses_blocking`]) and the chunked-body decoder they use.
//!
//! Owner: bench (HTTP client).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: What a non-streaming chat-completions or Responses request
/// returned.
#[derive(Debug, Default, Clone)]
pub struct BlockingOutcome {
    /// 2026-09-26: One entry per `choices[]`, in order; a Responses reply is
    /// one entry.
    pub choices: Vec<String>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// 2026-09-26: POST to `/v1/chat/completions` and read the whole JSON
/// response. Errors on a timeout, a non-200 (with the server's message when
/// readable) and a body that is not JSON.
pub async fn chat_blocking(
    target: &TargetEndpoint,
    body: &Value,
    timeout: Duration,
) -> Result<BlockingOutcome> {
    tokio::time::timeout(timeout, chat_blocking_inner(target, body))
        .await
        .map_err(|_| anyhow!("request exceeded {:.0}s", timeout.as_secs_f64()))?
}

async fn chat_blocking_inner(target: &TargetEndpoint, body: &Value) -> Result<BlockingOutcome> {
    post_json(target, "/v1/chat/completions", body).await
}

/// 2026-09-26: The same as [`chat_blocking`], against the Responses API
/// (`/v1/responses`).
pub async fn responses_blocking(
    target: &TargetEndpoint,
    body: &Value,
    timeout: Duration,
) -> Result<BlockingOutcome> {
    tokio::time::timeout(timeout, post_json(target, "/v1/responses", body))
        .await
        .map_err(|_| anyhow!("request exceeded {:.0}s", timeout.as_secs_f64()))?
}

async fn post_json(target: &TargetEndpoint, path: &str, body: &Value) -> Result<BlockingOutcome> {
    let (host, port) = target.host_port()?;
    let payload = serde_json::to_string(body)?;
    let mut sock = TcpStream::connect((host.as_str(), port))
        .await
        .with_context(|| format!("connecting to {}", target.base_url))?;
    let _ = sock.set_nodelay(true);
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\n\
         Content-Type: application/json\r\nAccept: application/json\r\n\
         Connection: close\r\nContent-Length: {}\r\n\r\n{payload}",
        payload.len()
    );
    sock.write_all(request.as_bytes()).await.context("write")?;

    // 2026-09-26: Read to EOF: the request sends `Connection: close`, so the
    // server ends the response by closing.
    let mut raw = Vec::new();
    let mut buf = [0u8; 16 * 1024];
    loop {
        let n = sock.read(&mut buf).await.context("read")?;
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&buf[..n]);
    }

    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .context("no header/body split in the response")?;
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let body_bytes = &raw[split + 4..];

    let status = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .unwrap_or("?")
        .to_string();
    if status != "200" {
        let detail = error_message_from_response(&raw).unwrap_or_else(|| {
            let text = if head
                .to_ascii_lowercase()
                .contains("transfer-encoding: chunked")
            {
                dechunk(body_bytes)
            } else {
                String::from_utf8_lossy(body_bytes).to_string()
            };
            text.chars().take(200).collect()
        });
        bail!("endpoint returned HTTP {status}: {detail}");
    }

    let text = if head.to_lowercase().contains("transfer-encoding: chunked") {
        dechunk(body_bytes)
    } else {
        String::from_utf8_lossy(body_bytes).to_string()
    };
    let v: Value = serde_json::from_str(text.trim()).with_context(|| {
        format!(
            "response was not JSON: {}",
            text.chars().take(200).collect::<String>()
        )
    })?;

    // 2026-09-26: Chat-completions puts the reply in
    // `choices[].message.content`, Responses in `output[]`; a body with
    // `output` and no `choices` is read as Responses.
    if v.get("choices").is_none() && v.get("output").is_some() {
        let mut texts = Vec::new();
        if let Some(items) = v["output"].as_array() {
            for item in items {
                // 2026-09-26: Both `content[].text` (the answer) and
                // `summary[].text` (the reasoning) are collected, so a reply
                // that arrives only as reasoning is still read.
                for key in ["content", "summary"] {
                    if let Some(parts) = item[key].as_array() {
                        for part in parts {
                            if let Some(s) = part["text"].as_str() {
                                texts.push(s.to_string());
                            }
                        }
                    }
                }
            }
        }
        return Ok(BlockingOutcome {
            choices: vec![texts.join("")],
            prompt_tokens: v["usage"]["input_tokens"].as_u64().unwrap_or(0) as usize,
            completion_tokens: v["usage"]["output_tokens"].as_u64().unwrap_or(0) as usize,
        });
    }
    let choices = v["choices"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|c| {
                    c["message"]["content"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string()
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(BlockingOutcome {
        choices,
        prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize,
        completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize,
    })
}

/// 2026-09-26: Minimal chunked-transfer decoder: `<hex-len>\r\n<data>\r\n`
/// until a zero-length chunk. A size line that does not parse as hex (for
/// example one with an extension) ends the body; trailers are ignored.
fn dechunk(mut b: &[u8]) -> String {
    let mut out = Vec::new();
    loop {
        let Some(nl) = b.windows(2).position(|w| w == b"\r\n") else {
            break;
        };
        let len = usize::from_str_radix(String::from_utf8_lossy(&b[..nl]).trim(), 16).unwrap_or(0);
        if len == 0 {
            break;
        }
        let start = nl + 2;
        let end = (start + len).min(b.len());
        out.extend_from_slice(&b[start..end]);
        b = &b[(end + 2).min(b.len())..];
    }
    String::from_utf8_lossy(&out).to_string()
}
