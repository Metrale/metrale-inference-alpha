// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Which endpoint URLs name the same host, for the TTFT gates'
//! per-host baseline.
//!
//! Owner: bench, ttft.
//! Invariants: none beyond the types.

/// 2026-09-26: Do two endpoint URLs name the same host?
///
/// Compares the host only, case-insensitively; the port is ignored because a
/// self-started run serves on an OS-assigned port, so each run of the same box
/// has a different one. `localhost`, `127.0.0.1` and `[::1]` are one host.
pub(super) fn same_box(a: &str, b: &str) -> bool {
    fn host(url: &str) -> String {
        let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
        let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
        let hostport = authority
            .rsplit_once('@')
            .map_or(authority, |(_, hostport)| hostport);
        let h = if let Some(bracketed) = hostport.strip_prefix('[') {
            bracketed
                .split_once(']')
                .map_or(hostport, |(address, _)| address)
        } else if hostport.matches(':').count() == 1 {
            hostport.split_once(':').map_or(hostport, |(host, _)| host)
        } else {
            // 2026-09-26: An unbracketed IPv6 literal has no unambiguous port
            // delimiter, so the whole address is the host; splitting at the
            // last `:` would make `fe80::1` and `fe80::2` equal.
            hostport
        };
        match h {
            "localhost" | "127.0.0.1" | "::1" => "localhost".to_string(),
            other => other.to_ascii_lowercase(),
        }
    }
    host(a) == host(b)
}

#[cfg(test)]
mod tests {
    use super::same_box;

    #[test]
    fn an_ephemeral_port_is_still_the_same_box() {
        assert!(same_box("http://127.0.0.1:8888", "http://127.0.0.1:33033"));
        assert!(same_box("http://localhost:8888", "http://127.0.0.1:41999"));
        assert!(same_box(
            "https://Example.COM:8888/a",
            "http://example.com:41999/b"
        ));
    }

    #[test]
    fn another_machine_is_never_the_same_box() {
        assert!(!same_box("http://10.10.10.3:8888", "http://127.0.0.1:8888"));
        assert!(!same_box(
            "http://10.10.10.1:8888",
            "http://10.10.10.2:8888"
        ));
    }

    #[test]
    fn an_ipv6_literal_keeps_its_address() {
        assert!(same_box("http://[::1]:8888", "http://localhost:9"));
        assert!(same_box("http://[::1]", "http://127.0.0.1:9"));
        assert!(!same_box("http://[fe80::1]:8888", "http://[fe80::2]:8888"));
        assert!(!same_box("http://fe80::1", "http://fe80::2"));
        assert!(!same_box("http://[fe80::1]", "http://[fe80::2]"));
    }
}
