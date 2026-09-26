// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Server-side fetching of media URLs, off unless
//! `--vision-allow-remote-images` is set. When on, any client can make the
//! server send HTTP requests to a URL of its choosing, so every fetch is
//! bounded:
//!
//! - Address: loopback, private, link-local, unique-local, broadcast,
//!   documentation, unspecified and carrier-grade NAT destinations are
//!   refused unless `--vision-remote-image-allow-private` is set.
//! - Size: `--vision-remote-image-max-mb`, counted while reading.
//!   `Content-Length` is not consulted.
//! - Time: `--vision-remote-image-timeout-s` per HTTP request.
//! - Redirects: at most `MAX_REDIRECTS`, each hop's address checked again.
//! - Type: the response must declare an `image/*` content type.
//!
//! It runs inside `prepare_chat_prompt`, which both callers run on the
//! blocking pool, so the HTTP client is a blocking one.
//!
//! Owner: server (chat API).
//! Invariants:
//! - With `enabled: false`, no request leaves the server from here.
//! - Every URL, redirect targets included, passes `check_host` before it is
//!   requested.

use std::io::Read;
use std::net::IpAddr;

/// 2026-09-26: Remote-fetch policy, built from the `--vision-*` flags in
/// `serve_load.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteImagePolicy {
    pub enabled: bool,
    pub max_bytes: usize,
    pub timeout_secs: u64,
    /// 2026-09-26: Skip the address check (`is_blocked_address`). A separate
    /// grant from `enabled`.
    pub allow_private: bool,
}

impl Default for RemoteImagePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_bytes: 20 * 1024 * 1024,
            timeout_secs: 10,
            allow_private: false,
        }
    }
}

/// 2026-09-26: Redirects followed before giving up.
const MAX_REDIRECTS: u32 = 4;

/// 2026-09-26: True for an address a client must not make the server reach:
/// loopback, private, link-local, broadcast, documentation, unspecified,
/// carrier-grade NAT (IPv4); loopback, unspecified, unique-local, link-local
/// (IPv6). An IPv4-mapped IPv6 address is judged as IPv4.
pub fn is_blocked_address(ip: IpAddr) -> bool {
    // 2026-09-26: `::ffff:127.0.0.1` is loopback, but none of the IPv6
    // predicates below say so; map it to IPv4 first.
    let ip = match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(ip, IpAddr::V4),
        v4 => v4,
    };
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                // 2026-09-26: 100.64.0.0/10, carrier-grade NAT, which
                // `is_private` does not cover.
                || (v4.octets()[0] == 100 && (64..128).contains(&v4.octets()[1]))
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                // 2026-09-26: fc00::/7 unique-local and fe80::/10 link-local,
                // matched by prefix.
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                || (v6.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

/// 2026-09-26: Refuse `host` if it is, or resolves to, any blocked address,
/// or resolves to none. Skipped when `allow_private` is set.
fn check_host(host: &str, port: u16, policy: &RemoteImagePolicy) -> Result<(), String> {
    if policy.allow_private {
        return Ok(());
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        return if is_blocked_address(ip) {
            Err(format!("{ip} is a loopback/private/link-local address"))
        } else {
            Ok(())
        };
    }
    use std::net::ToSocketAddrs;
    let addrs = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("could not resolve {host}: {e}"))?;
    let mut saw = false;
    for a in addrs {
        saw = true;
        if is_blocked_address(a.ip()) {
            return Err(format!(
                "{host} resolves to {}, a loopback/private/link-local address",
                a.ip()
            ));
        }
    }
    if saw {
        Ok(())
    } else {
        Err(format!("{host} resolved to no addresses"))
    }
}

/// 2026-09-26: Split an http(s) URL into (scheme, host, port). Userinfo is
/// dropped; an unparseable port gives the scheme's default.
fn split_url(url: &str) -> Result<(&str, &str, u16), String> {
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| "not an absolute http(s) URL".to_string())?;
    if scheme != "http" && scheme != "https" {
        return Err(format!("scheme {scheme:?} is not http or https"));
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    // 2026-09-26: `http://metadata@evil/` points at `evil`, so the host is
    // what follows the last `@`.
    let authority = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let default_port = if scheme == "https" { 443 } else { 80 };
    let (host, port) = match authority.rfind(':') {
        // 2026-09-26: A `:` inside a bracketed IPv6 literal is not a port
        // separator.
        Some(i) if !authority[i..].contains(']') => (
            &authority[..i],
            authority[i + 1..].parse().unwrap_or(default_port),
        ),
        _ => (authority, default_port),
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if host.is_empty() {
        return Err("URL has no host".to_string());
    }
    Ok((scheme, host, port))
}

/// 2026-09-26: Fetch `url` and return it as a `data:<mime>;base64,` URI.
/// `Err` carries the reason; `msg_entry` turns it into a 400.
pub fn fetch_as_data_uri(url: &str, policy: &RemoteImagePolicy) -> Result<String, String> {
    if !policy.enabled {
        return Err("remote image fetching is disabled".to_string());
    }

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(policy.timeout_secs)))
        // 2026-09-26: Redirects are followed by the loop below, so each hop's
        // address is checked.
        .max_redirects(0)
        .build()
        .into();

    let mut current = url.to_string();
    for _ in 0..=MAX_REDIRECTS {
        let (_scheme, host, port) = split_url(&current)?;
        check_host(host, port, policy)?;

        let resp = agent
            .get(&current)
            .call()
            .map_err(|e| format!("fetch failed: {e}"))?;
        let status = resp.status().as_u16();

        if (300..400).contains(&status) {
            let loc = resp
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| format!("HTTP {status} with no Location header"))?;
            // 2026-09-26: A relative `Location` is refused, not resolved.
            if !loc.starts_with("http://") && !loc.starts_with("https://") {
                return Err(format!("relative redirect to {loc:?} is not followed"));
            }
            current = loc.to_string();
            continue;
        }
        if status != 200 {
            return Err(format!("HTTP {status}"));
        }

        let ctype = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let mime = ctype.split(';').next().unwrap_or("").trim().to_lowercase();
        if !mime.starts_with("image/") {
            return Err(format!("content-type {mime:?} is not an image"));
        }

        // 2026-09-26: One byte past the cap is read so an oversized body is
        // refused rather than truncated.
        let mut buf = Vec::new();
        let cap = policy.max_bytes;
        resp.into_body()
            .into_reader()
            .take(cap as u64 + 1)
            .read_to_end(&mut buf)
            .map_err(|e| format!("read failed: {e}"))?;
        if buf.len() > cap {
            return Err(format!(
                "image exceeds the {cap}-byte cap (--vision-remote-image-max-mb)"
            ));
        }
        if buf.is_empty() {
            return Err("fetched an empty body".to_string());
        }

        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&buf);
        return Ok(format!("data:{mime};base64,{b64}"));
    }
    Err(format!("more than {MAX_REDIRECTS} redirects"))
}

#[cfg(test)]
#[path = "remote_image_tests.rs"]
mod tests;
