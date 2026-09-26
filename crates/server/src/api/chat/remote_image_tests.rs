// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for remote image fetching. The address checks are tested
//! directly; the transport is tested against a one-shot listener on loopback,
//! which `allow_private: true` lets the fetch reach.
//!
//! Owner: server (chat API) tests.
//! Invariants: none beyond the types.

use super::*;
use std::io::Write;
use std::net::{Ipv4Addr, Ipv6Addr, TcpListener};

fn enabled() -> RemoteImagePolicy {
    RemoteImagePolicy {
        enabled: true,
        allow_private: true,
        ..Default::default()
    }
}

#[test]
fn the_default_policy_fetches_nothing() {
    let p = RemoteImagePolicy::default();
    assert!(!p.enabled, "remote fetching must be OFF unless asked for");
    assert!(
        !p.allow_private,
        "private destinations must be a second grant"
    );
    let err = fetch_as_data_uri("http://example.com/a.png", &p).unwrap_err();
    assert!(err.contains("disabled"), "{err}");
}

/// 2026-09-26: Disabled must return before any network work, not fail after
/// it.
#[test]
fn disabled_refuses_before_resolving_anything() {
    let p = RemoteImagePolicy::default();
    let t = std::time::Instant::now();
    let err = fetch_as_data_uri("http://127.0.0.1:1/nope.png", &p).unwrap_err();
    assert!(err.contains("disabled"), "{err}");
    assert!(t.elapsed().as_millis() < 200, "it tried to connect");
}

#[test]
fn loopback_private_and_link_local_are_blocked() {
    for ip in [
        "127.0.0.1",
        "10.0.0.5",
        "192.168.1.10",
        "172.16.0.1",
        "0.0.0.0",
        "169.254.169.254",
        // 2026-09-26: Carrier-grade NAT, not covered by
        // `Ipv4Addr::is_private`.
        "100.64.0.1",
    ] {
        let a: IpAddr = ip.parse().unwrap();
        assert!(is_blocked_address(a), "{ip} should be blocked");
    }
}

#[test]
fn ipv6_loopback_unique_local_and_link_local_are_blocked() {
    for ip in ["::1", "fc00::1", "fd12:3456::1", "fe80::1", "::"] {
        let a: IpAddr = ip.parse().unwrap();
        assert!(is_blocked_address(a), "{ip} should be blocked");
    }
}

/// 2026-09-26: The IPv6 predicates alone would let `::ffff:127.0.0.1`
/// through; the classifier maps it to IPv4 first.
#[test]
fn ipv4_mapped_loopback_does_not_slip_past_the_v6_arm() {
    let mapped: IpAddr = "::ffff:127.0.0.1".parse().unwrap();
    assert!(is_blocked_address(mapped));
    let mapped_private: IpAddr = IpAddr::V6(Ipv4Addr::new(10, 0, 0, 1).to_ipv6_mapped());
    assert!(is_blocked_address(mapped_private));
}

#[test]
fn ordinary_public_addresses_are_allowed() {
    for ip in ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700::1111"] {
        let a: IpAddr = ip.parse().unwrap();
        assert!(!is_blocked_address(a), "{ip} should be allowed");
    }
    assert!(!is_blocked_address(IpAddr::V6(Ipv6Addr::new(
        0x2001, 0xdb9, 0, 0, 0, 0, 0, 1
    ))));
}

#[test]
fn a_literal_blocked_ip_in_the_url_is_refused_when_private_is_not_granted() {
    let p = RemoteImagePolicy {
        enabled: true,
        ..Default::default()
    };
    let err = fetch_as_data_uri("http://169.254.169.254/latest/meta-data/", &p).unwrap_err();
    assert!(err.contains("link-local"), "{err}");
}

#[test]
fn userinfo_does_not_masquerade_as_the_host() {
    let (_, host, port) = split_url("http://169.254.169.254@example.com/a.png").unwrap();
    assert_eq!(host, "example.com");
    assert_eq!(port, 80);
}

#[test]
fn ports_schemes_and_ipv6_literals_parse() {
    assert_eq!(
        split_url("https://h.test/a.png").unwrap(),
        ("https", "h.test", 443)
    );
    assert_eq!(
        split_url("http://h.test:8080/a").unwrap(),
        ("http", "h.test", 8080)
    );
    let (_, host, port) = split_url("http://[::1]:9000/a.png").unwrap();
    assert_eq!((host, port), ("::1", 9000));
}

#[test]
fn non_http_schemes_are_refused() {
    for u in [
        "file:///etc/passwd",
        "gopher://h.test/",
        "ftp://h.test/a.png",
        "not-a-url",
    ] {
        assert!(split_url(u).is_err(), "{u} should not parse as fetchable");
    }
}

/// 2026-09-26: Serve one canned response on loopback and return its URL.
fn one_shot(response: Vec<u8>) -> String {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut s, _)) = l.accept() {
            use std::io::Read as _;
            let mut scratch = [0u8; 2048];
            let _ = s.read(&mut scratch);
            let _ = s.write_all(&response);
            let _ = s.flush();
        }
    });
    format!("http://127.0.0.1:{port}/image.png")
}

fn http_response(ctype: &str, body: &[u8]) -> Vec<u8> {
    let mut v = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    v.extend_from_slice(body);
    v
}

#[test]
fn a_fetched_image_comes_back_as_a_decodable_data_uri() {
    let png = b"\x89PNG\r\n\x1a\nnot-a-real-png-but-bytes-are-bytes";
    let url = one_shot(http_response("image/png", png));
    let uri = fetch_as_data_uri(&url, &enabled()).expect("fetch");
    assert!(uri.starts_with("data:image/png;base64,"), "{uri}");

    use base64::Engine as _;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(uri.strip_prefix("data:image/png;base64,").unwrap())
        .expect("round-trips through base64");
    assert_eq!(decoded, png, "the bytes served are the bytes handed on");
}

#[test]
fn a_charset_parameter_does_not_defeat_the_image_check() {
    let url = one_shot(http_response("image/jpeg; charset=binary", b"jpegbytes"));
    let uri = fetch_as_data_uri(&url, &enabled()).expect("fetch");
    assert!(uri.starts_with("data:image/jpeg;base64,"), "{uri}");
}

#[test]
fn a_non_image_content_type_is_refused() {
    let url = one_shot(http_response("text/html", b"<html>nope</html>"));
    let err = fetch_as_data_uri(&url, &enabled()).unwrap_err();
    assert!(err.contains("not an image"), "{err}");
}

/// 2026-09-26: With no `Content-Length`, the body runs until the peer closes,
/// so only the read cap bounds it.
#[test]
fn the_size_cap_stops_an_unbounded_body() {
    let mut resp =
        b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nConnection: close\r\n\r\n".to_vec();
    resp.extend_from_slice(&vec![b'x'; 64 * 1024]);
    let url = one_shot(resp);
    let p = RemoteImagePolicy {
        max_bytes: 1024,
        ..enabled()
    };
    let err = fetch_as_data_uri(&url, &p).unwrap_err();
    assert!(err.contains("exceeds"), "{err}");
}

/// 2026-09-26: A body of exactly `max_bytes` is accepted; the check is `>`.
#[test]
fn a_body_exactly_at_the_cap_is_accepted() {
    let body = vec![b'x'; 1024];
    let url = one_shot(http_response("image/png", &body));
    let p = RemoteImagePolicy {
        max_bytes: 1024,
        ..enabled()
    };
    assert!(fetch_as_data_uri(&url, &p).is_ok());
}

/// 2026-09-26: An understated `Content-Length` cannot cause an over-read: the
/// client stops at the declared length and the fetch returns those bytes.
#[test]
fn an_understated_content_length_truncates_rather_than_overruns() {
    let mut resp =
        b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 4\r\nConnection: close\r\n\r\n"
            .to_vec();
    resp.extend_from_slice(&vec![b'x'; 4096]);
    let url = one_shot(resp);
    let p = RemoteImagePolicy {
        max_bytes: 1024,
        ..enabled()
    };
    let uri = fetch_as_data_uri(&url, &p).expect("declared length is honoured");
    use base64::Engine as _;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(uri.strip_prefix("data:image/png;base64,").unwrap())
        .unwrap();
    assert_eq!(decoded.len(), 4, "read stopped at the declared length");
}

#[test]
fn an_empty_body_is_an_error_not_an_empty_image() {
    let url = one_shot(http_response("image/png", b""));
    let err = fetch_as_data_uri(&url, &enabled()).unwrap_err();
    assert!(err.contains("empty"), "{err}");
}

#[test]
fn a_non_200_status_is_reported_with_its_code() {
    let url = one_shot(
        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
    );
    let err = fetch_as_data_uri(&url, &enabled()).unwrap_err();
    assert!(err.contains("404"), "{err}");
}

/// 2026-09-26: The loopback test server is itself blocked under this policy,
/// so the fetch is refused at hop zero; the redirect is never followed.
#[test]
fn a_redirect_to_a_blocked_address_is_refused_at_the_second_hop() {
    let url = one_shot(
        b"HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest/meta-data/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            .to_vec(),
    );
    let p = RemoteImagePolicy {
        enabled: true,
        allow_private: false,
        ..Default::default()
    };
    let err = fetch_as_data_uri(&url, &p).unwrap_err();
    assert!(
        err.contains("loopback") || err.contains("link-local"),
        "{err}"
    );
}

#[test]
fn a_relative_redirect_is_refused_rather_than_guessed_at() {
    let url = one_shot(
        b"HTTP/1.1 302 Found\r\nLocation: /elsewhere.png\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            .to_vec(),
    );
    let err = fetch_as_data_uri(&url, &enabled()).unwrap_err();
    assert!(err.contains("relative redirect"), "{err}");
}
