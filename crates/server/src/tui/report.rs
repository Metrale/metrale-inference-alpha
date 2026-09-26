// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Issue reporting's GitHub protocol: response parsing, failure
//! mapping and body assembly. Apart from `target()`, which reads two env
//! variables, nothing here does I/O; the HTTPS calls sit behind
//! [`report_http`](super::report_http)'s `Http` seam, so the failure table is
//! tested without a network.
//!
//! Authorization is GitHub's OAuth device flow. The TUI is a public client, so
//! it carries a `client_id` and no client secret. The access and refresh
//! tokens are [`SecretString`]s held in `help_state`'s `Auth` in process
//! memory and handed to the submit worker; they are not written to disk and
//! the issue body is composed without them (CWE-522, CWE-256).
//!
//! Owner: server tui.
//! Invariants: `SecretString` implements neither `Debug`, `Display` nor
//! `Serialize`, so a token cannot be formatted or serialised.

use std::time::Duration;

/// 2026-09-26: The official build's reporter identity. The client_id is
/// public (it is sent in every device-flow POST); a fork overrides both with
/// `METRALE_REPORT_CLIENT_ID` and `METRALE_REPORT_REPO` (`target()`).
pub const OFFICIAL_CLIENT_ID: &str = "Iv23liAv6nlb4RaYaJSp";
pub const OFFICIAL_REPO: &str = "Metrale/metrale-inference-alpha";

/// 2026-09-26: Hidden marker appended to every report body. The issue
/// payload carries no labels (`run_submit`), so the marker is how a report
/// is recognised.
pub const MARKER: &str = "<!-- metrale-tui-report -->";

pub const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
pub const TOKEN_URL: &str = "https://github.com/login/oauth/access_token";

pub fn issues_url(repo: &str) -> String {
    format!("https://api.github.com/repos/{repo}/issues")
}

/// 2026-09-26: What the user is told when the transport (DNS, TLS, refused
/// connection) failed. The device flow, the refresh and the issue POST all
/// use this one constant.
pub const NETWORK_FAILED: &str = "could not reach github.com — check network and retry (s)";

pub const NOT_CONFIGURED: &str = "issue reporting is not configured for this build (set METRALE_REPORT_CLIENT_ID / METRALE_REPORT_REPO)";

/// 2026-09-26: An in-memory secret. It implements neither `Debug` nor
/// `Display` nor `Serialize`, so a `{:?}` on it does not compile, which keeps
/// a token out of `tracing` events and format strings (CWE-532). The bytes
/// are zeroed on drop; earlier copies left by moves are not.
pub struct SecretString(String);

impl SecretString {
    pub fn new(s: String) -> Self {
        Self(s)
    }
    /// 2026-09-26: The only way to read the value, named so call sites can
    /// be found by grep.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl Clone for SecretString {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Drop for SecretString {
    fn drop(&mut self) {
        // 2026-09-26: NUL bytes are valid UTF-8, so overwriting in place keeps
        // the `String` valid.
        unsafe { self.0.as_mut_vec() }.fill(0);
    }
}

pub struct Target {
    pub client_id: String,
    pub repo: String,
}

/// 2026-09-26: Where this process's reports go: the env overrides, else the
/// compiled-in identity.
pub fn target() -> Result<Target, &'static str> {
    target_from(
        std::env::var("METRALE_REPORT_CLIENT_ID").ok(),
        std::env::var("METRALE_REPORT_REPO").ok(),
    )
}

/// 2026-09-26: A blank client_id or repo, whether from an override or from
/// the constants, is refused with [`NOT_CONFIGURED`] before any request is
/// made.
pub fn target_from(id: Option<String>, repo: Option<String>) -> Result<Target, &'static str> {
    let client_id = id.unwrap_or_else(|| OFFICIAL_CLIENT_ID.to_string());
    let repo = repo.unwrap_or_else(|| OFFICIAL_REPO.to_string());
    if client_id.trim().is_empty() || repo.trim().is_empty() {
        return Err(NOT_CONFIGURED);
    }
    Ok(Target { client_id, repo })
}

/// 2026-09-26: What the worker threads send to the render thread. No variant
/// carries the device code, which stays inside the device-flow worker; the
/// user code that is displayed is a different string.
pub enum ReportEvent {
    CodeReady {
        user_code: String,
        verification_uri: String,
        expires_in: Duration,
    },
    Authorized {
        access: SecretString,
        refresh: Option<SecretString>,
    },
    AuthFailed {
        message: String,
    },
    Created {
        number: u64,
        url: String,
    },
    SubmitFailed {
        message: String,
        /// 2026-09-26: The stored tokens were refused (a 401 with no refresh
        /// token, a failed refresh, or a 401 after the refresh); the state
        /// drops them so the next attempt re-authorizes.
        drop_auth: bool,
    },
}

/// 2026-09-26: The device-code grant. `run_device_flow` sends
/// `device_code` only to `TOKEN_URL`.
pub struct DeviceGrant {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub expires_in: Duration,
    pub interval: Duration,
}

fn json(body: &str) -> Option<serde_json::Value> {
    serde_json::from_str(body).ok()
}

fn str_field(v: &serde_json::Value, k: &str) -> Option<String> {
    v.get(k)?.as_str().map(str::to_string)
}

/// 2026-09-26: Parse `POST /login/device/code`. Every field is required, so
/// a response without `interval` is an error rather than a guessed cadence.
pub fn parse_device_code(status: u16, body: &str) -> Result<DeviceGrant, String> {
    let Some(v) = json(body) else {
        return Err(format!(
            "unexpected response from github.com (HTTP {status})"
        ));
    };
    if let Some(err) = str_field(&v, "error") {
        return Err(map_oauth_error(&err, &v));
    }
    let grant = (|| {
        Some(DeviceGrant {
            device_code: str_field(&v, "device_code")?,
            user_code: str_field(&v, "user_code")?,
            verification_uri: str_field(&v, "verification_uri")?,
            expires_in: Duration::from_secs(v.get("expires_in")?.as_u64()?),
            interval: Duration::from_secs(v.get("interval")?.as_u64()?),
        })
    })();
    let Some(grant) = grant else {
        return Err(format!(
            "github.com answered without the device-code fields (HTTP {status})"
        ));
    };
    // 2026-09-26: The URI is shown to the user as the place to type the code,
    // so anything outside `https://github.com/` is refused: a displayed URL an
    // attacker could influence would be a phishing vector.
    if !grant.verification_uri.starts_with("https://github.com/") {
        return Err("unexpected verification URL in GitHub's response — refusing".to_string());
    }
    Ok(grant)
}

/// 2026-09-26: The outcome of one poll of `POST /login/oauth/access_token`.
pub enum PollOutcome {
    Authorized {
        access: SecretString,
        refresh: Option<SecretString>,
    },
    /// 2026-09-26: Keep polling at the current interval.
    Pending,
    /// 2026-09-26: Keep polling; `run_device_flow` adds 5 seconds to the
    /// interval.
    SlowDown,
    /// 2026-09-26: The code expired (`expired_token`).
    Expired,
    /// 2026-09-26: The user declined on github.com (`access_denied`).
    Denied,
    /// 2026-09-26: Any other outcome; it ends the flow with this message.
    Fatal(String),
}

pub fn parse_poll(status: u16, body: &str) -> PollOutcome {
    let Some(v) = json(body) else {
        return PollOutcome::Fatal(format!(
            "unexpected response from github.com while polling (HTTP {status})"
        ));
    };
    if let Some(access) = str_field(&v, "access_token") {
        return PollOutcome::Authorized {
            access: SecretString::new(access),
            refresh: str_field(&v, "refresh_token").map(SecretString::new),
        };
    }
    match str_field(&v, "error").as_deref() {
        Some("authorization_pending") => PollOutcome::Pending,
        Some("slow_down") => PollOutcome::SlowDown,
        Some("expired_token") => PollOutcome::Expired,
        Some("access_denied") => PollOutcome::Denied,
        Some(err) => PollOutcome::Fatal(map_oauth_error(err, &v)),
        None => PollOutcome::Fatal(format!(
            "github.com answered the poll without a token or an error (HTTP {status})"
        )),
    }
}

/// 2026-09-26: The refresh grant answers in the poll's shape. Anything but
/// a token pair is `None`, which the caller treats as a dead refresh token.
pub fn parse_refresh(status: u16, body: &str) -> Option<(SecretString, Option<SecretString>)> {
    match parse_poll(status, body) {
        PollOutcome::Authorized { access, refresh } => Some((access, refresh)),
        _ => None,
    }
}

fn map_oauth_error(err: &str, v: &serde_json::Value) -> String {
    match err {
        "device_flow_disabled" => {
            "this build's GitHub App is misconfigured (device flow disabled) — report to the maintainers"
                .to_string()
        }
        "unsupported_grant_type" | "incorrect_client_credentials" | "incorrect_device_code" => {
            format!("GitHub refused the authorization request ({err})")
        }
        other => {
            let detail = str_field(v, "error_description").unwrap_or_default();
            if detail.is_empty() {
                format!("GitHub authorization failed ({other})")
            } else {
                format!("GitHub authorization failed: {detail}")
            }
        }
    }
}

/// 2026-09-26: Parse `POST /repos/{owner}/{repo}/issues`. `Ok` means a 201
/// and nothing else: the composer is cleared only on the `Created` event this
/// becomes, so a lenient parse would lose the user's draft.
pub fn parse_issue_created(status: u16, body: &str) -> Result<(u64, String), IssueFailure> {
    if status == 201 {
        let v = json(body);
        let number = v.as_ref().and_then(|v| v.get("number")?.as_u64());
        let url = v.as_ref().and_then(|v| str_field(v, "html_url"));
        return match (number, url) {
            (Some(n), Some(u)) => Ok((n, u)),
            // 2026-09-26: A 201 whose body cannot be read still created the
            // issue; reporting failure would invite a duplicate submission.
            _ => Ok((0, String::new())),
        };
    }
    Err(IssueFailure {
        status,
        message: github_message(body),
    })
}

#[derive(Debug)]
pub struct IssueFailure {
    pub status: u16,
    pub message: String,
}

fn github_message(body: &str) -> String {
    json(body)
        .and_then(|v| {
            let msg = str_field(&v, "message")?;
            let detail = v
                .get("errors")
                .and_then(|e| e.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|e| e.get("message").and_then(|m| m.as_str()))
                        .collect::<Vec<_>>()
                        .join("; ")
                })
                .unwrap_or_default();
            Some(if detail.is_empty() {
                msg
            } else {
                format!("{msg}: {detail}")
            })
        })
        .unwrap_or_default()
}

/// 2026-09-26: A non-201 outcome as words the user can act on. A 401 never
/// reaches it: `post_issue` returns it to `run_submit`, which refreshes.
pub fn describe_issue_failure(f: &IssueFailure, repo: &str, retry_after: Option<u64>) -> String {
    match f.status {
        403 | 429 => match retry_after {
            Some(n) => format!("GitHub is rate-limiting — retry in {n}s"),
            None if f.message.to_lowercase().contains("rate limit") => {
                "GitHub is rate-limiting — retry shortly".to_string()
            }
            None => format!(
                "GitHub refused the request (403): {}",
                or_unstated(&f.message)
            ),
        },
        404 => format!(
            "GitHub answered 404 for {repo} — the repository may not exist, or the reporter app is not installed on it"
        ),
        410 => {
            format!("the issue tracker at {repo} is archived or disabled — it cannot accept issues")
        }
        422 => format!("GitHub rejected the issue: {}", or_unstated(&f.message)),
        s if (500..600).contains(&s) => format!("GitHub returned {s} — try again shortly"),
        s => format!(
            "GitHub returned an unexpected {s}: {}",
            or_unstated(&f.message)
        ),
    }
}

fn or_unstated(msg: &str) -> &str {
    if msg.is_empty() {
        "(no detail given)"
    } else {
        msg
    }
}

/// 2026-09-26: The final issue body, plus the numbers the preview states
/// about it.
#[derive(Clone, Debug)]
pub struct Composed {
    pub body: String,
    pub chars: usize,
    pub logs_included: usize,
    pub logs_total: usize,
}

/// 2026-09-26: The `## Environment` line. A build without
/// `METRALE_BUILD_COMMIT` says `commit unknown` rather than omitting it.
pub fn env_line(model: &str, engine_ready: bool) -> String {
    format!(
        "Metrale Engine {} · {} · {}/{} · model: {} · engine ready: {engine_ready}",
        crate::cli::METRALE_VERSION,
        option_env!("METRALE_BUILD_COMMIT").unwrap_or("commit unknown"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        if model.is_empty() { "none" } else { model },
    )
}

/// 2026-09-26: Assemble the body that will be posted. `help_state::compose`
/// calls it, and the `Composed` it returns is what the preview shows and what
/// `proceed` posts. `logs` must already be redacted; this function only
/// budgets them.
///
/// The user's text is never truncated: if it does not fit, this returns an
/// error. Only the log tail is trimmed.
pub fn compose_body(
    user_text: &str,
    env: &str,
    logs: Option<&[String]>,
    tee_path: Option<&str>,
) -> Result<Composed, String> {
    use super::redact::{BODY_BUDGET, GITHUB_BODY_LIMIT, fence_for, trim_to_budget};
    let head = format!("{}\n\n## Environment\n\n{env}\n", user_text.trim_end());
    let tail = format!("\n{MARKER}\n");
    // 2026-09-26: A flat reserve for the log section's heading and fences.
    // The `GITHUB_BODY_LIMIT` check below still refuses a longer body.
    const SECTION_RESERVE: usize = 200;
    let fixed = head.chars().count() + tail.chars().count();
    if fixed + if logs.is_some() { SECTION_RESERVE } else { 0 } > BODY_BUDGET {
        return Err(format!(
            "the report text is {fixed} characters; GitHub's limit is {GITHUB_BODY_LIMIT} and the dashboard reserves headroom — trim it below {BODY_BUDGET}",
        ));
    }
    let (section, logs_included, logs_total) = match logs {
        None => (String::new(), 0, 0),
        Some(lines) => {
            let trimmed = trim_to_budget(lines, BODY_BUDGET - fixed - SECTION_RESERVE, tee_path);
            let fence = fence_for(&trimmed.text);
            let section = format!(
                "\n## Server log (last {} of {} lines, redacted best-effort)\n\n{fence}text\n{}\n{fence}\n",
                trimmed.included, trimmed.total, trimmed.text
            );
            (section, trimmed.included, trimmed.total)
        }
    };
    let body = format!("{head}{section}{tail}");
    let chars = body.chars().count();
    if chars > GITHUB_BODY_LIMIT {
        // 2026-09-26: Refused here rather than as a 422 after the user
        // pressed send.
        return Err(format!(
            "assembled body is {chars} characters — over GitHub's limit"
        ));
    }
    Ok(Composed {
        body,
        chars,
        logs_included,
        logs_total,
    })
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
