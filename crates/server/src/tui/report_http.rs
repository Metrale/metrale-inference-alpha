// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The I/O half of issue reporting: the `Http` transport seam,
//! the device-flow and submit runners, and their thread spawns. The protocol
//! parsing is in [`report`](super::report).
//!
//! Both flows run blocking `ureq` on a thread named `metrale-report` and
//! answer over a `std::sync::mpsc` channel that the tick drains
//! (`.github/workflows/tui-threading.yml` rejects `block_on` under `tui/`).
//! The device flow sends more than one event, so it spawns its own thread
//! instead of using `worker::spawn`, and sends a failure event itself when
//! the spawn fails. Every URL is `https://`; the agent is built with no
//! option that turns off certificate verification (CWE-319).
//!
//! Owner: server tui.
//! Invariants: a spawn failure in `LiveWorkers` sends a failure event, so the
//! returned receiver never waits on a thread that does not exist.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use super::report::{
    DEVICE_CODE_URL, NETWORK_FAILED, PollOutcome, ReportEvent, SecretString, TOKEN_URL,
    describe_issue_failure, issues_url, parse_device_code, parse_issue_created, parse_poll,
    parse_refresh,
};

/// 2026-09-26: One HTTP exchange, reduced to what the parsers need. `Err` is
/// a transport failure (DNS, TLS, refused); any HTTP status is `Ok`, because
/// GitHub's error bodies carry the message the user is shown.
pub struct HttpReply {
    pub status: u16,
    pub body: String,
    pub retry_after: Option<u64>,
}

pub type HttpResult = Result<HttpReply, String>;

/// 2026-09-26: The transport seam (SBIO): the flow runners take `&dyn Http`,
/// the tests pass a scripted fake, and [`Live`] is the only implementation
/// that opens a socket.
pub trait Http {
    /// 2026-09-26: Form-encoded POST to a github.com OAuth endpoint, which
    /// issues tokens and so takes none.
    fn post_form(&self, url: &str, form: &[(&str, &str)]) -> HttpResult;
    /// 2026-09-26: Authenticated JSON POST to the API. The token is its own
    /// argument, and [`Live`] puts it only in the `Authorization` header.
    fn post_json(&self, url: &str, token: &str, json: &serde_json::Value) -> HttpResult;
}

pub struct Live {
    agent: ureq::Agent,
}

impl Default for Live {
    fn default() -> Self {
        Self::new()
    }
}

impl Live {
    pub fn new() -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            // 2026-09-26: A non-2xx status comes back as a reply, not an
            // error: the failure table reads GitHub's error bodies.
            .http_status_as_error(false)
            // 2026-09-26: Bounded, so a stalled POST ends in the failure path
            // instead of leaving the Submitting spinner up. The body is at
            // most `GITHUB_BODY_LIMIT` characters.
            .timeout_global(Some(Duration::from_secs(30)))
            .build()
            .into();
        Self { agent }
    }

    fn reply(res: ureq::http::Response<ureq::Body>) -> HttpResult {
        let status = res.status().as_u16();
        let retry_after = res
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok());
        let mut res = res;
        let body = res.body_mut().read_to_string().map_err(|e| e.to_string())?;
        Ok(HttpReply {
            status,
            body,
            retry_after,
        })
    }
}

const AGENT_HEADER: &str = crate::identity::USER_AGENT;

impl Http for Live {
    fn post_form(&self, url: &str, form: &[(&str, &str)]) -> HttpResult {
        let res = self
            .agent
            .post(url)
            .header("Accept", "application/json")
            .header("User-Agent", AGENT_HEADER)
            .send_form(form.iter().copied())
            .map_err(|e| e.to_string())?;
        Self::reply(res)
    }

    fn post_json(&self, url: &str, token: &str, json: &serde_json::Value) -> HttpResult {
        let res = self
            .agent
            .post(url)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", AGENT_HEADER)
            .header("Authorization", &format!("Bearer {token}"))
            .send_json(json)
            .map_err(|e| e.to_string())?;
        Self::reply(res)
    }
}

/// 2026-09-26: Run the whole grant: request a code, send it to the UI, and
/// poll until a verdict. `wait` sleeps and returns whether to keep going;
/// `false` means the user cancelled, and the flow ends without an event
/// because Esc already dropped the receiver (`help_keys.rs`).
///
/// The device code is used only in the token poll: it is not sent to the UI,
/// displayed or logged.
pub fn run_device_flow(
    http: &dyn Http,
    client_id: &str,
    wait: &mut dyn FnMut(Duration) -> bool,
    tx: &Sender<ReportEvent>,
) {
    let fail = |tx: &Sender<ReportEvent>, message: String| {
        let _ = tx.send(ReportEvent::AuthFailed { message });
    };
    let grant = match http.post_form(DEVICE_CODE_URL, &[("client_id", client_id)]) {
        Ok(reply) => match parse_device_code(reply.status, &reply.body) {
            Ok(g) => g,
            Err(message) => return fail(tx, message),
        },
        Err(e) => {
            // 2026-09-26: The transport detail goes to the log; the user sees
            // `NETWORK_FAILED`.
            tracing::warn!("device-code request failed: {e}");
            return fail(tx, NETWORK_FAILED.to_string());
        }
    };
    let _ = tx.send(ReportEvent::CodeReady {
        user_code: grant.user_code.clone(),
        verification_uri: grant.verification_uri.clone(),
        expires_in: grant.expires_in,
    });
    let mut interval = grant.interval.max(Duration::from_secs(1));
    loop {
        if !wait(interval) {
            return;
        }
        let reply = match http.post_form(
            TOKEN_URL,
            &[
                ("client_id", client_id),
                ("device_code", &grant.device_code),
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ],
        ) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("device-flow poll failed: {e}");
                return fail(tx, NETWORK_FAILED.to_string());
            }
        };
        match parse_poll(reply.status, &reply.body) {
            PollOutcome::Pending => {}
            PollOutcome::SlowDown => interval += Duration::from_secs(5),
            PollOutcome::Authorized { access, refresh } => {
                let _ = tx.send(ReportEvent::Authorized { access, refresh });
                return;
            }
            PollOutcome::Expired => {
                return fail(
                    tx,
                    "the code expired before it was entered — s requests a fresh one".to_string(),
                );
            }
            PollOutcome::Denied => {
                return fail(
                    tx,
                    "authorization was declined on github.com — nothing was sent".to_string(),
                );
            }
            PollOutcome::Fatal(message) => return fail(tx, message),
        }
    }
}

/// 2026-09-26: Everything one submission needs.
pub struct SubmitJob {
    pub client_id: String,
    pub repo: String,
    pub access: SecretString,
    pub refresh: Option<SecretString>,
    pub title: String,
    pub body: String,
}

/// 2026-09-26: POST the issue. On a 401, refresh the token once (the refresh
/// form carries no client secret) and retry once. Every path ends in one
/// terminal event; a successful refresh first sends `Authorized` so the state
/// keeps the rotated tokens.
pub fn run_submit(http: &dyn Http, job: SubmitJob, tx: &Sender<ReportEvent>) {
    let url = issues_url(&job.repo);
    let payload = serde_json::json!({ "title": job.title, "body": job.body });
    // 2026-09-26: Only title and body are sent; the body's `MARKER` is what
    // identifies a report.
    if post_issue(http, &url, job.access.expose(), &payload, &job.repo, tx).is_none() {
        return;
    }
    // 2026-09-26: The first POST answered 401, the only status that is
    // retried, and only with a refreshed token.
    let Some(refresh) = job.refresh.as_ref() else {
        return drop_auth_failed(tx);
    };
    let refreshed = http.post_form(
        TOKEN_URL,
        &[
            ("client_id", &job.client_id),
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh.expose()),
        ],
    );
    let (access, new_refresh) = match refreshed {
        Ok(reply) => match parse_refresh(reply.status, &reply.body) {
            Some(pair) => pair,
            None => return drop_auth_failed(tx),
        },
        Err(e) => {
            tracing::warn!("token refresh failed: {e}");
            let _ = tx.send(ReportEvent::SubmitFailed {
                message: NETWORK_FAILED.to_string(),
                drop_auth: false,
            });
            return;
        }
    };
    let _ = tx.send(ReportEvent::Authorized {
        access: access.clone(),
        refresh: new_refresh,
    });
    if post_issue(http, &url, access.expose(), &payload, &job.repo, tx) == Some(401) {
        drop_auth_failed(tx);
    }
}

/// 2026-09-26: One POST. Returns `Some(401)` when the caller still owes the
/// user an answer, and `None` when a terminal event was already sent.
fn post_issue(
    http: &dyn Http,
    url: &str,
    token: &str,
    payload: &serde_json::Value,
    repo: &str,
    tx: &Sender<ReportEvent>,
) -> Option<u16> {
    let reply = match http.post_json(url, token, payload) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("issue POST failed: {e}");
            let _ = tx.send(ReportEvent::SubmitFailed {
                message: NETWORK_FAILED.to_string(),
                drop_auth: false,
            });
            return None;
        }
    };
    match parse_issue_created(reply.status, &reply.body) {
        Ok((number, url)) => {
            let _ = tx.send(ReportEvent::Created { number, url });
            None
        }
        Err(f) if f.status == 401 => Some(401),
        Err(f) => {
            let _ = tx.send(ReportEvent::SubmitFailed {
                message: describe_issue_failure(&f, repo, reply.retry_after),
                drop_auth: false,
            });
            None
        }
    }
}

fn drop_auth_failed(tx: &Sender<ReportEvent>) {
    let _ = tx.send(ReportEvent::SubmitFailed {
        message: "GitHub no longer accepts this authorization — s re-authorizes".to_string(),
        drop_auth: true,
    });
}

/// 2026-09-26: The worker-thread boundary `help_state` talks to, injectable
/// so its flow is tested with no thread and no socket.
pub trait Workers {
    fn device_flow(&self, client_id: String, cancel: Arc<AtomicBool>) -> Receiver<ReportEvent>;
    fn submit(&self, job: SubmitJob) -> Receiver<ReportEvent>;
}

pub struct LiveWorkers;

impl Workers for LiveWorkers {
    fn device_flow(&self, client_id: String, cancel: Arc<AtomicBool>) -> Receiver<ReportEvent> {
        let (tx, rx) = channel();
        let spawned = std::thread::Builder::new()
            .name("metrale-report".into())
            .spawn({
                let tx = tx.clone();
                move || {
                    // 2026-09-26: Sleep in 200 ms slices, so a cancel during
                    // the wait between polls takes effect within one slice
                    // rather than at the end of the poll interval. A POST
                    // already in flight is not interrupted; it ends at the
                    // agent's 30 s timeout at most.
                    let mut wait = |d: Duration| {
                        let deadline = std::time::Instant::now() + d;
                        while std::time::Instant::now() < deadline {
                            if cancel.load(Ordering::Relaxed) {
                                return false;
                            }
                            std::thread::sleep(Duration::from_millis(200));
                        }
                        !cancel.load(Ordering::Relaxed)
                    };
                    run_device_flow(&Live::new(), &client_id, &mut wait, &tx);
                }
            });
        if let Err(e) = spawned {
            // 2026-09-26: Answer even when the thread did not start, so the
            // UI is not left polling a receiver that never resolves.
            let _ = tx.send(ReportEvent::AuthFailed {
                message: format!("could not start the report worker: {e}"),
            });
        }
        rx
    }

    fn submit(&self, job: SubmitJob) -> Receiver<ReportEvent> {
        let (tx, rx) = channel();
        let spawned = std::thread::Builder::new()
            .name("metrale-report".into())
            .spawn({
                let tx = tx.clone();
                move || run_submit(&Live::new(), job, &tx)
            });
        if let Err(e) = spawned {
            let _ = tx.send(ReportEvent::SubmitFailed {
                message: format!("could not start the report worker: {e}"),
                drop_auth: false,
            });
        }
        rx
    }
}

#[cfg(test)]
#[path = "report_http_tests.rs"]
mod tests;
