// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the redaction rules, the log budget and the fence width.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;

fn ctx() -> RedactCtx {
    RedactCtx {
        home: Some("/home/claude".into()),
        user: Some("claude".into()),
        host: Some("dgx1".into()),
    }
}

fn no_identity() -> RedactCtx {
    RedactCtx {
        home: None,
        user: None,
        host: None,
    }
}

#[test]
fn github_token_shapes_are_scrubbed() {
    // 2026-09-26: The five `gh?_` prefixes in `SHAPES`, which share a tail
    // rule.
    for prefix in ["ghp", "gho", "ghu", "ghs", "ghr"] {
        let line = format!("using {prefix}_{} for auth", "A1b2C3d4".repeat(5));
        let out = redact_line(&line, &no_identity());
        assert!(out.contains(REDACTED), "{prefix}: {out}");
        assert!(!out.contains("A1b2C3d4"), "{prefix} leaked: {out}");
    }
}

#[test]
fn fine_grained_pat_and_hf_and_sk_and_akia_are_scrubbed() {
    let cases = [
        format!("github_pat_{}", "11ABCDEFG0_abcdefghijklmnop".repeat(2)),
        format!("hf_{}", "aBcDeFgHiJ".repeat(3)),
        format!("sk-proj-{}", "abcdef0123".repeat(3)),
        format!("AKIA{}", "IOSFODNN7EXAMPLE"),
    ];
    for secret in cases {
        let out = redact_line(&format!("key: {secret}"), &no_identity());
        assert!(out.contains(REDACTED), "{secret}: {out}");
        assert!(!out.contains(&secret), "leaked: {out}");
    }
}

#[test]
fn short_lookalikes_survive() {
    // 2026-09-26: A tail shorter than `min_tail` is not redacted.
    let out = redact_line("saved ghp_cache.bin and hf_dataset", &no_identity());
    assert!(!out.contains(REDACTED), "{out}");
}

#[test]
fn sk_inside_a_word_survives() {
    // 2026-09-26: This contains `sk-` and a tail longer than the minimum; the
    // alphanumeric char before the prefix keeps it.
    let out = redact_line(
        "a risk-assessment-driven-decision-making log",
        &no_identity(),
    );
    assert!(!out.contains(REDACTED), "{out}");
}

#[test]
fn authorization_header_value_is_scrubbed_to_eol() {
    let out = redact_line(
        "request headers: Authorization: Bearer abc.def.ghi",
        &no_identity(),
    );
    assert!(out.contains("Authorization:"), "{out}");
    assert!(!out.contains("abc.def.ghi"), "{out}");
    assert!(out.contains(REDACTED), "{out}");
}

#[test]
fn bearer_and_kv_values_are_scrubbed() {
    for line in [
        "sent bearer sOmEtOkEn to the hub",
        "url?access_token=deadbeef123&x=1",
        "config secret=hunter2 loaded",
        "api_key=abc123, retrying",
        "password=pa55w0rd!",
    ] {
        let out = redact_line(line, &no_identity());
        assert!(out.contains(REDACTED), "{line} -> {out}");
        for leaked in ["sOmEtOkEn", "deadbeef123", "hunter2", "abc123", "pa55w0rd"] {
            assert!(!out.contains(leaked), "{line} -> {out}");
        }
    }
}

#[test]
fn kv_keeps_the_key_and_the_delimited_tail() {
    // 2026-09-26: The key and the text after the delimiter are kept.
    let out = redact_line("GET /x?access_token=deadbeef123&page=2", &no_identity());
    assert_eq!(out, format!("GET /x?access_token={REDACTED}&page=2"));
}

#[test]
fn home_user_and_host_are_substituted() {
    let out = redact_line(
        "claude@dgx1 wrote /home/claude/.cache/metrale/logs/x.log",
        &ctx(),
    );
    assert_eq!(out, "«user»@«host» wrote ~/.cache/metrale/logs/x.log");
}

#[test]
fn username_inside_a_longer_word_survives() {
    // 2026-09-26: The username inside a longer word is not replaced.
    let out = redact_line("claudeadjacentword stays", &ctx());
    assert!(out.contains("claudeadjacentword"), "{out}");
}

#[test]
fn fabric_ips_are_scrubbed_loopback_and_unspecified_survive() {
    let out = redact_line(
        "bind 0.0.0.0:8000, peer 10.10.10.2, local 127.0.0.1, v6 ::1",
        &no_identity(),
    );
    assert!(out.contains("0.0.0.0:8000"), "{out}");
    assert!(out.contains("127.0.0.1"), "{out}");
    assert!(out.contains("::1"), "{out}");
    assert!(!out.contains("10.10.10.2"), "{out}");
    assert!(out.contains(REDACTED_IP), "{out}");
}

#[test]
fn ip_with_port_loses_only_the_address() {
    let out = redact_line("connecting to 10.10.10.1:8888", &no_identity());
    assert_eq!(out, format!("connecting to {REDACTED_IP}:8888"));
}

#[test]
fn versions_and_timestamps_are_not_ips() {
    // 2026-09-26: 3-part versions and hh:mm:ss.f timestamps do not parse as
    // addresses.
    let line = "v3.3.0 at 12:34:56.789 loaded 1.0.0-beta";
    assert_eq!(redact_line(line, &no_identity()), line);
}

#[test]
fn ipv6_literal_is_scrubbed() {
    let out = redact_line("peer [2001:db8::7] answered", &no_identity());
    assert!(!out.contains("2001:db8::7"), "{out}");
    assert!(out.contains(REDACTED_IP), "{out}");
}

#[test]
fn trailing_punctuation_does_not_hide_an_ip() {
    let out = redact_line("reached 10.10.10.1.", &no_identity());
    assert_eq!(out, format!("reached {REDACTED_IP}."));
}

fn lines(n: usize, len: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("{i:04}{}", "x".repeat(len.saturating_sub(4))))
        .collect()
}

#[test]
fn everything_fits_when_under_budget() {
    let l = lines(10, 20);
    let t = trim_to_budget(&l, 10_000, Some("/tee"));
    assert_eq!(t.included, 10);
    assert_eq!(t.total, 10);
    assert_eq!(t.text, l.join("\n"));
    assert!(
        !t.text.contains("omitted"),
        "no marker when nothing was dropped"
    );
}

#[test]
fn trim_is_oldest_first_and_the_marker_names_the_count() {
    let l = lines(100, 50);
    let t = trim_to_budget(&l, 1_000, Some("/tee/path.log"));
    assert!(t.included < 100);
    assert_eq!(t.total, 100);
    // 2026-09-26: The newest line is kept and the oldest dropped.
    assert!(t.text.contains("0099"), "{}", t.text);
    assert!(!t.text.contains("0000"), "{}", t.text);
    assert!(
        t.text
            .starts_with(&omission_marker(100 - t.included, Some("/tee/path.log")))
    );
    assert!(t.text.contains("/tee/path.log"));
}

#[test]
fn trimmed_text_never_exceeds_the_budget() {
    for budget in [200, 1_000, 5_000] {
        let t = trim_to_budget(&lines(500, 37), budget, Some("/t"));
        assert!(
            t.text.chars().count() <= budget,
            "budget {budget}: {}",
            t.text.chars().count()
        );
    }
}

#[test]
fn exact_boundary_line_is_kept() {
    // 2026-09-26: A line of budget-1 chars plus its newline allowance fits.
    let l = vec!["x".repeat(99)];
    let t = trim_to_budget(&l, 100, None);
    assert_eq!(t.included, 1);
}

#[test]
fn fence_grows_past_the_longest_embedded_run() {
    assert_eq!(fence_for("no backticks"), "```");
    assert_eq!(fence_for("inline `code` here"), "```");
    assert_eq!(fence_for("a ``` fence inside"), "````");
    assert_eq!(fence_for("````` five"), "``````");
}

#[test]
fn redaction_is_idempotent_on_already_clean_text() {
    let line = "INFO metrale_model_layers loaded 85 shards in 42s";
    assert_eq!(redact_line(line, &no_identity()), line);
}
