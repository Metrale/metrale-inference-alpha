// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Plain GET requests: `/v1/models` (`list_models`, `probe`),
//! `get_json` and `fetch_hardware`.
//!
//! Owner: bench (HTTP client).
//! Invariants:
//! - Every request sends `Connection: close` and is bounded by the caller's
//!   `timeout`.
//! - Every read loop stops at EOF or once past a byte bound (512 B for
//!   `probe`, 64 KiB for `/hardware`, 1 MiB otherwise).

use super::*;

pub async fn list_models(target: &TargetEndpoint, timeout: Duration) -> Result<Vec<String>> {
    let body = get_models(target, timeout).await?;
    let start = body
        .find('{')
        .context("no JSON in the /v1/models response")?;
    // 2026-09-26: Parse the first value and ignore whatever follows. A
    // chunked reply carries hex length prefixes and a terminating
    // `0\r\n\r\n`, on which a plain `from_str` fails.
    let doc: serde_json::Value = serde_json::Deserializer::from_str(&body[start..])
        .into_iter()
        .next()
        .context("/v1/models returned an empty body")?
        .context("/v1/models did not return JSON")?;
    Ok(doc
        .get("data")
        .and_then(|d| d.as_array())
        .map(|rows| {
            rows.iter()
                .filter_map(|r| r.get("id").and_then(|i| i.as_str()))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default())
}

/// 2026-09-26: Reachability check: `GET /v1/models` must answer 200 within
/// `timeout`. Every benchmark driver except vision and video calls it.
pub async fn probe(target: &TargetEndpoint, timeout: Duration) -> Result<()> {
    let (host, port) = target.host_port()?;
    let fut = async {
        let mut sock = TcpStream::connect((host.as_str(), port)).await?;
        let req =
            format!("GET /v1/models HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
        sock.write_all(req.as_bytes()).await?;
        let mut head = Vec::new();
        let mut buf = [0u8; 1024];
        while head.len() < 512 {
            let n = sock.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            head.extend_from_slice(&buf[..n]);
        }
        anyhow::Ok(String::from_utf8_lossy(&head).into_owned())
    };
    let head = tokio::time::timeout(timeout, fut)
        .await
        .map_err(|_| anyhow!("{} did not answer within {:?}", target.base_url, timeout))?
        .with_context(|| format!("probing {}", target.base_url))?;
    let status = head.lines().next().unwrap_or_default();
    if !status_is_success(status) {
        bail!("{} /v1/models returned {status:?}", target.base_url);
    }
    Ok(())
}

/// 2026-09-26: The whole `/v1/models` response, headers included.
async fn get_models(target: &TargetEndpoint, timeout: Duration) -> Result<String> {
    let (host, port) = target.host_port()?;
    let fut = async {
        let mut sock = TcpStream::connect((host.as_str(), port)).await?;
        let req =
            format!("GET /v1/models HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
        sock.write_all(req.as_bytes()).await?;
        let mut body = Vec::new();
        let mut buf = [0u8; 4096];
        // 2026-09-26: Read to EOF, bounded at 1 MiB: a short fixed read would
        // cut a long model list into invalid JSON.
        loop {
            let n = sock.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&buf[..n]);
            if body.len() > 1 << 20 {
                break;
            }
        }
        anyhow::Ok(String::from_utf8_lossy(&body).into_owned())
    };
    tokio::time::timeout(timeout, fut)
        .await
        .map_err(|_| anyhow!("{} did not answer within {:?}", target.base_url, timeout))?
        .with_context(|| format!("reading models from {}", target.base_url))
}

/// 2026-09-26: `GET <path>` and parse the first JSON value of the reply (a
/// chunked reply carries framing after the document). A status other than
/// 200 is an error that quotes the document.
pub async fn get_json(
    target: &TargetEndpoint,
    path: &str,
    timeout: Duration,
) -> Result<serde_json::Value> {
    let (host, port) = target.host_port()?;
    let fut = async {
        let mut sock = TcpStream::connect((host.as_str(), port)).await?;
        let req =
            format!("GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
        sock.write_all(req.as_bytes()).await?;
        let mut body = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let n = sock.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&buf[..n]);
            if body.len() > 1 << 20 {
                break;
            }
        }
        anyhow::Ok(String::from_utf8_lossy(&body).into_owned())
    };
    let raw = tokio::time::timeout(timeout, fut)
        .await
        .map_err(|_| {
            anyhow!(
                "{} did not answer {path} within {:?}",
                target.base_url,
                timeout
            )
        })?
        .with_context(|| format!("reading {path} from {}", target.base_url))?;
    let status = raw
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    let start = raw
        .find('{')
        .with_context(|| format!("{path} returned no JSON (status {status})"))?;
    let doc: serde_json::Value = serde_json::Deserializer::from_str(&raw[start..])
        .into_iter()
        .next()
        .with_context(|| format!("{path} returned an empty body"))?
        .with_context(|| format!("{path} did not return JSON"))?;
    if status != 200 {
        anyhow::bail!("{path} answered {status}: {doc}");
    }
    Ok(doc)
}

/// 2026-09-26: `GET /hardware`: the serving box's hardware fingerprint.
///
/// Fetched from the endpoint rather than probed locally because a benchmark
/// number belongs to the box that did the inference, which is not
/// necessarily the box running the benchmark CLI. Any failure (bad target,
/// connect, timeout, no JSON, wrong shape) returns
/// [`crate::hardware::Hardware::unknown`], so a server without the endpoint
/// does not make a run unrecordable.
pub async fn fetch_hardware(
    target: &TargetEndpoint,
    timeout: Duration,
) -> crate::hardware::Hardware {
    let (host, port) = match target.host_port() {
        Ok(hp) => hp,
        Err(_) => return crate::hardware::Hardware::unknown(),
    };
    let fut = async {
        let mut sock = TcpStream::connect((host.as_str(), port)).await?;
        let req =
            format!("GET /hardware HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
        sock.write_all(req.as_bytes()).await?;
        let mut body = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let n = sock.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&buf[..n]);
            if body.len() > 64 * 1024 {
                break;
            }
        }
        anyhow::Ok(String::from_utf8_lossy(&body).into_owned())
    };
    let Ok(raw) = tokio::time::timeout(timeout, fut).await else {
        return crate::hardware::Hardware::unknown();
    };
    let Ok(raw) = raw else {
        return crate::hardware::Hardware::unknown();
    };
    // 2026-09-26: Chunked framing: parse the first JSON value and ignore the
    // rest, as `list_models` does.
    let Some(start) = raw.find('{') else {
        return crate::hardware::Hardware::unknown();
    };
    serde_json::Deserializer::from_str(&raw[start..])
        .into_iter()
        .next()
        .and_then(|r| r.ok())
        .and_then(|doc: serde_json::Value| serde_json::from_value(doc).ok())
        .unwrap_or_else(crate::hardware::Hardware::unknown)
}
