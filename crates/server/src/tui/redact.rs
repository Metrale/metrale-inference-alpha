// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Best-effort scrubbing and size budgeting for the log tail a GitHub issue report attaches.
//!
//! Owner: server tui.
//! Invariants:
//! - Every function here is pure except `RedactCtx::from_env`, which reads the
//!   environment and `/proc/sys/kernel/hostname`.
//!
//! Redacted: the credential shapes in `SHAPES`, values after
//! `authorization:`, `bearer `, `token=`, `secret=`, `api_key=`, `apikey=` and
//! `password=`, the home directory, username and hostname, and non-loopback IP
//! literals. Text with no recognizable shape (prompts, model names, error
//! messages) is not redacted. A string that only looks like a key is redacted
//! too.

/// 2026-09-26: What a recognized secret is replaced with.
pub const REDACTED: &str = "«redacted»";

/// 2026-09-26: What a non-loopback IP literal is replaced with.
pub const REDACTED_IP: &str = "«ip»";

/// 2026-09-26: The issue-body character limit that `report` refuses to exceed.
pub const GITHUB_BODY_LIMIT: usize = 65_536;

/// 2026-09-26: The budget `report` fills the body to, kept below
/// `GITHUB_BODY_LIMIT` as headroom.
pub const BODY_BUDGET: usize = 60_000;

/// 2026-09-26: The identity strings to scrub, resolved once by the caller
/// (`help_state`) so the scrubbing itself is pure.
pub struct RedactCtx {
    pub home: Option<String>,
    pub user: Option<String>,
    pub host: Option<String>,
}

impl RedactCtx {
    /// 2026-09-26: Resolve from the live environment. Values of one character
    /// or less, after trimming, are dropped.
    pub fn from_env() -> Self {
        let keep = |s: String| {
            let t = s.trim().to_string();
            (t.len() > 1).then_some(t)
        };
        Self {
            home: std::env::var("HOME").ok().and_then(keep),
            user: std::env::var("USER").ok().and_then(keep),
            host: std::fs::read_to_string("/proc/sys/kernel/hostname")
                .ok()
                .or_else(|| std::env::var("HOSTNAME").ok())
                .and_then(keep),
        }
    }
}

/// 2026-09-26: Scrub one log line: credentials, then identity, then IPs.
/// Credentials go first so that replacing an identity substring cannot break
/// up a token before it is recognized.
pub fn redact_line(line: &str, ctx: &RedactCtx) -> String {
    let s = redact_credentials(line);
    let s = redact_identity(&s, ctx);
    redact_ips(&s)
}

/// 2026-09-26: A recognizable credential prefix and what its tail looks like.
struct Shape {
    prefix: &'static str,
    tail: fn(char) -> bool,
    min_tail: usize,
}

fn alnum(c: char) -> bool {
    c.is_ascii_alphanumeric()
}
fn alnum_underscore(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}
fn keyish(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}
fn upper_alnum(c: char) -> bool {
    c.is_ascii_uppercase() || c.is_ascii_digit()
}

/// 2026-09-26: The credential shapes this build recognizes.
const SHAPES: [Shape; 9] = [
    Shape {
        prefix: "ghp_",
        tail: alnum,
        min_tail: 36,
    },
    Shape {
        prefix: "gho_",
        tail: alnum,
        min_tail: 36,
    },
    Shape {
        prefix: "ghu_",
        tail: alnum,
        min_tail: 36,
    },
    Shape {
        prefix: "ghs_",
        tail: alnum,
        min_tail: 36,
    },
    Shape {
        prefix: "ghr_",
        tail: alnum,
        min_tail: 36,
    },
    Shape {
        prefix: "github_pat_",
        tail: alnum_underscore,
        min_tail: 22,
    },
    Shape {
        prefix: "hf_",
        tail: alnum,
        min_tail: 30,
    },
    Shape {
        prefix: "sk-",
        tail: keyish,
        min_tail: 20,
    },
    Shape {
        prefix: "AKIA",
        tail: upper_alnum,
        min_tail: 16,
    },
];

fn redact_credentials(line: &str) -> String {
    let mut s = line.to_string();
    for sh in &SHAPES {
        s = redact_shape(&s, sh);
    }
    // 2026-09-26: Header and key=value forms, where only the surrounding
    // syntax marks the secret. The match is a case-insensitive substring, so
    // `token=` also covers `access_token=` and `refresh_token=`.
    s = redact_after(&s, "authorization:", true);
    for key in [
        "bearer ",
        "token=",
        "secret=",
        "api_key=",
        "apikey=",
        "password=",
    ] {
        s = redact_after(&s, key, false);
    }
    s
}

fn redact_shape(s: &str, sh: &Shape) -> String {
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while let Some(off) = s[i..].find(sh.prefix) {
        let start = i + off;
        // 2026-09-26: The char before the prefix must not be alphanumeric:
        // "risk-…" is prose, not an `sk-` key mid-word.
        let bounded = s[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_ascii_alphanumeric());
        let after = start + sh.prefix.len();
        // 2026-09-26: Every tail predicate accepts ASCII only, so the char
        // count is the byte count.
        let run = s[after..].chars().take_while(|&c| (sh.tail)(c)).count();
        if bounded && run >= sh.min_tail {
            out.push_str(&s[i..start]);
            out.push_str(REDACTED);
            i = after + run;
        } else {
            out.push_str(&s[i..after]);
            i = after;
        }
    }
    out.push_str(&s[i..]);
    out
}

/// 2026-09-26: ASCII-case-insensitive substring search, from byte `from`. Not
/// `to_lowercase()` + `find`: non-ASCII lowercasing can change byte lengths,
/// and the returned index must be valid in the original string.
fn find_ci(hay: &str, needle: &str, from: usize) -> Option<usize> {
    let h = hay.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || from + n.len() > h.len() {
        return None;
    }
    (from..=h.len() - n.len()).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

/// 2026-09-26: Redact what follows `key`: the rest of the line when `to_eol`,
/// else the next token. The key itself is kept.
fn redact_after(s: &str, key: &str, to_eol: bool) -> String {
    const DELIMS: &[char] = &[' ', '\t', '"', '\'', '&', ',', ';', ')', ']', '}'];
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while let Some(pos) = find_ci(s, key, i) {
        let vstart = pos + key.len();
        out.push_str(&s[i..vstart]);
        if to_eol {
            if s[vstart..].trim().is_empty() {
                return out + &s[vstart..];
            }
            out.push(' ');
            out.push_str(REDACTED);
            return out;
        }
        let rest = &s[vstart..];
        let skip: usize = rest
            .chars()
            .take_while(|&c| c == ' ')
            .map(char::len_utf8)
            .sum();
        let val: usize = rest[skip..]
            .chars()
            .take_while(|c| !DELIMS.contains(c))
            .map(char::len_utf8)
            .sum();
        if val == 0 {
            i = vstart;
            continue;
        }
        out.push_str(&rest[..skip]);
        out.push_str(REDACTED);
        i = vstart + skip + val;
    }
    out.push_str(&s[i..]);
    out
}

fn redact_identity(s: &str, ctx: &RedactCtx) -> String {
    let mut s = s.to_string();
    // 2026-09-26: Home before username: `$HOME` usually contains the
    // username, and replacing the name first would leave `/home/«user»`
    // fragments the home pass no longer matches.
    if let Some(h) = &ctx.home {
        s = s.replace(h.as_str(), "~");
    }
    if let Some(u) = &ctx.user {
        s = s.replace(&format!("/home/{u}"), "~");
        s = replace_word(&s, u, "«user»");
    }
    if let Some(h) = &ctx.host {
        s = replace_word(&s, h, "«host»");
    }
    s
}

/// 2026-09-26: Whole-word replacement: `word` is replaced only where the
/// characters on both sides are not ASCII alphanumeric or `_`.
fn replace_word(s: &str, word: &str, with: &str) -> String {
    let boundary = |c: Option<char>| c.is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while let Some(off) = s[i..].find(word) {
        let start = i + off;
        let end = start + word.len();
        if boundary(s[..start].chars().next_back()) && boundary(s[end..].chars().next()) {
            out.push_str(&s[i..start]);
            out.push_str(with);
        } else {
            out.push_str(&s[i..end]);
        }
        i = end;
    }
    out.push_str(&s[i..]);
    out
}

/// 2026-09-26: Replace non-loopback IP literals. Tokens are maximal runs of
/// hex digits, `:` and `.`, parsed with `std::net`, so `12:34:56` timestamps
/// and `3.3.0` versions are left alone.
fn redact_ips(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut token = String::new();
    for c in s.chars() {
        if c.is_ascii_hexdigit() || c == ':' || c == '.' {
            token.push(c);
        } else {
            push_ip_token(&mut out, &token);
            token.clear();
            out.push(c);
        }
    }
    push_ip_token(&mut out, &token);
    out
}

fn push_ip_token(out: &mut String, token: &str) {
    if token.is_empty() {
        return;
    }
    let core = token.trim_end_matches(['.', ':']);
    let suffix = &token[core.len()..];
    if let Ok(ip) = core.parse::<std::net::IpAddr>() {
        if keep_ip(&ip) {
            out.push_str(token);
        } else {
            out.push_str(REDACTED_IP);
            out.push_str(suffix);
        }
        return;
    }
    // 2026-09-26: An IPv4 with a port (`10.10.10.1:8000`) does not parse
    // whole; only the address is replaced.
    if let Some((head, tail)) = core.split_once(':')
        && let Ok(ip) = head.parse::<std::net::Ipv4Addr>()
    {
        if keep_ip(&std::net::IpAddr::V4(ip)) {
            out.push_str(token);
        } else {
            out.push_str(REDACTED_IP);
            out.push(':');
            out.push_str(tail);
            out.push_str(suffix);
        }
        return;
    }
    out.push_str(token);
}

/// 2026-09-26: Loopback and unspecified addresses are kept: they say how the
/// server was bound, not which machine it is.
fn keep_ip(ip: &std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => v4.is_loopback() || v4.is_unspecified(),
        std::net::IpAddr::V6(v6) => v6.is_loopback() || v6.is_unspecified(),
    }
}

/// 2026-09-26: A log tail trimmed to fit a character budget.
pub struct TrimmedLog {
    pub text: String,
    pub included: usize,
    pub total: usize,
}

/// 2026-09-26: The first line of a trimmed log, naming how many lines were
/// dropped and where the full log is.
pub fn omission_marker(omitted: usize, tee_path: Option<&str>) -> String {
    format!(
        "— {omitted} earlier lines omitted to fit GitHub's 65,536-character limit; full log: {} —",
        tee_path.unwrap_or("(tee file unavailable)")
    )
}

/// 2026-09-26: Keep the newest lines that fit `budget` characters, counting
/// one `\n` per line, behind an omission marker when any line is dropped.
pub fn trim_to_budget(lines: &[String], budget: usize, tee_path: Option<&str>) -> TrimmedLog {
    let total = lines.len();
    let full: usize = lines.iter().map(|l| l.chars().count() + 1).sum();
    if full <= budget {
        return TrimmedLog {
            text: lines.join("\n"),
            included: total,
            total,
        };
    }
    // 2026-09-26: Reserve marker room for the worst-case omission count; the
    // final count is smaller or equal, so its marker is no longer.
    let mut used = omission_marker(total, tee_path).chars().count() + 1;
    let mut keep = 0;
    for l in lines.iter().rev() {
        let c = l.chars().count() + 1;
        if used + c > budget {
            break;
        }
        used += c;
        keep += 1;
    }
    let mut text = omission_marker(total - keep, tee_path);
    for l in &lines[total - keep..] {
        text.push('\n');
        text.push_str(l);
    }
    TrimmedLog {
        text,
        included: keep,
        total,
    }
}

/// 2026-09-26: A backtick fence one longer than the longest backtick run in
/// `content`, and at least three, so the content cannot close it.
pub fn fence_for(content: &str) -> String {
    let mut longest = 0usize;
    let mut run = 0usize;
    for c in content.chars() {
        if c == '`' {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    "`".repeat((longest + 1).max(3))
}

#[cfg(test)]
#[path = "redact_tests.rs"]
mod tests;
