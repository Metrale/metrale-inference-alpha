// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Who owns a changed path, per `.github/CODEOWNERS`, for the telemetry table.
//!
//! Owner: bench gate.
//! Invariants:
//! - A pattern [`is_supported`] refuses (`**`, `?`, `[`, `]`, `\`) matches no path, so an
//!   unsupported rule names nobody rather than the wrong owner.
//! - No gate verdict reads these owners; only `telemetry.rs` calls this module.
//!
//! `every_pattern_in_the_real_file_is_supported` fails if the committed file uses a pattern
//! this does not implement.

use std::path::Path;

/// 2026-09-26: One CODEOWNERS rule, in file order.
#[derive(Debug, Clone)]
pub struct Rule {
    pub pattern: String,
    pub owners: Vec<String>,
}

/// 2026-09-26: Parse `.github/CODEOWNERS`; a missing or unreadable file yields no rules.
pub fn load(root: &Path) -> Vec<Rule> {
    let Ok(text) = std::fs::read_to_string(root.join(".github/CODEOWNERS")) else {
        return Vec::new();
    };
    parse(&text)
}

pub fn parse(text: &str) -> Vec<Rule> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .filter_map(|l| {
            let mut parts = l.split_whitespace();
            let pattern = parts.next()?.to_string();
            let owners: Vec<String> = parts
                .filter(|o| o.starts_with('@') || o.contains('@'))
                .map(str::to_string)
                .collect();
            // 2026-09-26: A pattern with no owners clears ownership for its paths, so it is
            // kept as a rule with an empty owner list for `owners_of` to match last.
            Some(Rule { pattern, owners })
        })
        .collect()
}

/// 2026-09-26: Owners of `path` from the last matching rule, or empty when nothing matches, so
/// a specific rule placed after a broad one overrides it.
pub fn owners_of<'a>(rules: &'a [Rule], path: &str) -> &'a [String] {
    rules
        .iter()
        .rfind(|r| matches(&r.pattern, path))
        .map(|r| r.owners.as_slice())
        .unwrap_or(&[])
}

/// 2026-09-26: Every owner of any of `paths`, deduplicated and sorted.
pub fn owners_for_paths(rules: &[Rule], paths: &[String]) -> Vec<String> {
    let mut out: Vec<String> = paths
        .iter()
        .flat_map(|p| owners_of(rules, p).iter().cloned())
        .collect();
    out.sort();
    out.dedup();
    out
}

/// 2026-09-26: Whether this module implements `pattern`: false for `**`, `?`, character
/// classes and `\` escapes, none of which the committed CODEOWNERS uses
/// (`every_pattern_in_the_real_file_is_supported`).
pub fn is_supported(pattern: &str) -> bool {
    !pattern.contains("**")
        && !pattern.contains('?')
        && !pattern.contains('[')
        && !pattern.contains(']')
        && !pattern.contains('\\')
}

/// 2026-09-26: Whether a CODEOWNERS pattern matches a repo-relative path, for the subset in
/// [`is_supported`]:
/// - a pattern with a leading `/`, or a `/` anywhere but the end, is anchored to the root and
///   matches that path or anything under it;
/// - an unanchored pattern matches that basename at any depth (`Cargo.toml` covers
///   `crates/server/Cargo.toml`), and a trailing-`/` or star-free one also matches a
///   directory of that name at any depth.
fn matches(pattern: &str, path: &str) -> bool {
    if !is_supported(pattern) {
        return false;
    }
    let (body, dir_only) = match pattern.strip_suffix('/') {
        Some(p) => (p, true),
        None => (pattern, false),
    };
    let anchored = pattern.starts_with('/') || body.contains('/');
    let body = body.strip_prefix('/').unwrap_or(body);
    if body.is_empty() {
        return false;
    }

    let under = |base: &str| path.starts_with(&format!("{base}/"));

    if anchored {
        return glob_segment(body, path) || under(body);
    }
    if path.rsplit('/').next().is_some_and(|n| star(body, n)) {
        return true;
    }
    if dir_only || !body.contains('*') {
        return path
            .split('/')
            .any(|segment| star(body, segment) && path != segment);
    }
    false
}

/// 2026-09-26: Segment-by-segment match with the same number of segments, so `*` never
/// crosses a `/`.
fn glob_segment(pattern: &str, path: &str) -> bool {
    let pat: Vec<&str> = pattern.split('/').collect();
    let seg: Vec<&str> = path.split('/').collect();
    if pat.len() != seg.len() {
        return false;
    }
    pat.iter().zip(&seg).all(|(p, s)| star(p, s))
}

/// 2026-09-26: Whether `s` matches `pattern`, where `*` matches any run of characters.
fn star(pattern: &str, s: &str) -> bool {
    let text: Vec<char> = s.chars().collect();
    let mut matched = vec![false; text.len() + 1];
    matched[0] = true;
    for token in pattern.chars() {
        let mut next = vec![false; text.len() + 1];
        if token == '*' {
            next[0] = matched[0];
            for index in 1..=text.len() {
                next[index] = matched[index] || next[index - 1];
            }
        } else {
            for index in 1..=text.len() {
                next[index] = matched[index - 1] && token == text[index - 1];
            }
        }
        matched = next;
    }
    matched[text.len()]
}

#[cfg(test)]
#[path = "codeowners_tests.rs"]
mod codeowners_tests;
