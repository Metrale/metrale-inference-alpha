// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `run_device_flow` and `run_submit` driven end to end against a
//! scripted transport, with no socket, thread or sleep. `LiveWorkers` calls
//! the same two functions with the `Live` transport.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use std::cell::RefCell;
use std::sync::mpsc::channel;
use std::time::Duration;

use super::*;
use crate::tui::report::{OFFICIAL_CLIENT_ID, ReportEvent, SecretString};

/// 2026-09-26: A transport that replays a script and records every request.
struct Fake {
    replies: RefCell<Vec<HttpResult>>,
    /// 2026-09-26: (url, serialized payload or form, token if any)
    seen: RefCell<Vec<(String, String, Option<String>)>>,
}

impl Fake {
    fn new(replies: Vec<HttpResult>) -> Self {
        Self {
            replies: RefCell::new(replies),
            seen: RefCell::new(Vec::new()),
        }
    }
    fn next(&self) -> HttpResult {
        let mut r = self.replies.borrow_mut();
        assert!(
            !r.is_empty(),
            "flow made more requests than the script expected"
        );
        r.remove(0)
    }
}

impl Http for Fake {
    fn post_form(&self, url: &str, form: &[(&str, &str)]) -> HttpResult {
        let flat = form
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");
        self.seen.borrow_mut().push((url.to_string(), flat, None));
        self.next()
    }
    fn post_json(&self, url: &str, token: &str, json: &serde_json::Value) -> HttpResult {
        self.seen
            .borrow_mut()
            .push((url.to_string(), json.to_string(), Some(token.to_string())));
        self.next()
    }
}

fn ok(status: u16, body: &str) -> HttpResult {
    Ok(HttpReply {
        status,
        body: body.to_string(),
        retry_after: None,
    })
}

const GRANT: &str = r#"{"device_code":"dc_SECRET","user_code":"WDJB-MJHT","verification_uri":"https://github.com/login/device","expires_in":899,"interval":5}"#;
const PENDING: &str = r#"{"error":"authorization_pending"}"#;
const TOKENS: &str = r#"{"access_token":"ghu_NEW","refresh_token":"ghr_NEW","expires_in":28800}"#;

/// 2026-09-26: Run the device flow with an instrumented `wait` that never
/// sleeps.
fn drive_device_flow(
    fake: &Fake,
    cancel_after: Option<usize>,
) -> (Vec<ReportEvent>, Vec<Duration>) {
    let (tx, rx) = channel();
    let waits = RefCell::new(Vec::new());
    let mut wait = |d: Duration| {
        waits.borrow_mut().push(d);
        cancel_after.is_none_or(|n| waits.borrow().len() < n)
    };
    run_device_flow(fake, OFFICIAL_CLIENT_ID, &mut wait, &tx);
    drop(tx);
    (rx.try_iter().collect(), waits.into_inner())
}

#[test]
fn happy_grant_sends_code_then_tokens() {
    let fake = Fake::new(vec![ok(200, GRANT), ok(200, PENDING), ok(200, TOKENS)]);
    let (events, waits) = drive_device_flow(&fake, None);
    assert_eq!(
        waits,
        vec![Duration::from_secs(5); 2],
        "polls at GitHub's interval"
    );
    let [
        ReportEvent::CodeReady {
            user_code,
            verification_uri,
            expires_in,
        },
        ReportEvent::Authorized { access, refresh },
    ] = events.as_slice()
    else {
        panic!(
            "expected CodeReady then Authorized, got {} events",
            events.len()
        );
    };
    assert_eq!(user_code, "WDJB-MJHT");
    assert_eq!(verification_uri, "https://github.com/login/device");
    assert_eq!(expires_in.as_secs(), 899);
    assert_eq!(access.expose(), "ghu_NEW");
    assert_eq!(refresh.as_ref().expect("refresh").expose(), "ghr_NEW");
    // 2026-09-26: The device code is sent in the token poll.
    let seen = fake.seen.borrow();
    assert!(seen[1].1.contains("device_code=dc_SECRET"));
}

#[test]
fn slow_down_stretches_the_interval_by_five_seconds() {
    let fake = Fake::new(vec![
        ok(200, GRANT),
        ok(200, r#"{"error":"slow_down","interval":10}"#),
        ok(200, PENDING),
        ok(200, TOKENS),
    ]);
    let (_, waits) = drive_device_flow(&fake, None);
    assert_eq!(
        waits,
        vec![
            Duration::from_secs(5),
            Duration::from_secs(10),
            Duration::from_secs(10)
        ],
        "the +5s backoff must persist, not apply once"
    );
}

#[test]
fn expired_and_denied_end_with_their_own_words() {
    for (body, needle) in [
        (r#"{"error":"expired_token"}"#, "expired"),
        (r#"{"error":"access_denied"}"#, "declined"),
    ] {
        let fake = Fake::new(vec![ok(200, GRANT), ok(200, body)]);
        let (events, _) = drive_device_flow(&fake, None);
        let Some(ReportEvent::AuthFailed { message }) = events.last() else {
            panic!("expected AuthFailed");
        };
        assert!(message.contains(needle), "{body}: {message}");
    }
}

#[test]
fn cancel_ends_the_flow_in_silence() {
    // 2026-09-26: After a cancel the flow sends no verdict; Esc has already
    // returned the user to the composer.
    let fake = Fake::new(vec![ok(200, GRANT)]);
    let (events, waits) = drive_device_flow(&fake, Some(1));
    assert_eq!(events.len(), 1, "only CodeReady; no verdict after cancel");
    assert_eq!(waits.len(), 1);
    assert_eq!(
        fake.replies.borrow().len(),
        0,
        "no poll after the cancelled wait"
    );
}

#[test]
fn transport_failure_maps_to_the_network_message() {
    let fake = Fake::new(vec![Err("dns failure".into())]);
    let (events, _) = drive_device_flow(&fake, None);
    let [ReportEvent::AuthFailed { message }] = events.as_slice() else {
        panic!("expected AuthFailed only");
    };
    assert_eq!(message, crate::tui::report::NETWORK_FAILED);
    assert!(
        !message.contains("dns failure"),
        "transport detail goes to the log, not the toast"
    );
}

fn job() -> SubmitJob {
    SubmitJob {
        client_id: "Iv1TEST".into(),
        repo: "owner/metrale".into(),
        access: SecretString::new("ghu_OLD".into()),
        refresh: Some(SecretString::new("ghr_OLD".into())),
        title: "it broke".into(),
        body: "details <!-- metrale-tui-report -->".into(),
    }
}

fn drive_submit(fake: &Fake, job: SubmitJob) -> Vec<ReportEvent> {
    let (tx, rx) = channel();
    run_submit(fake, job, &tx);
    drop(tx);
    rx.try_iter().collect()
}

#[test]
fn created_on_201_and_the_token_stays_out_of_the_payload() {
    let fake = Fake::new(vec![ok(
        201,
        r#"{"number":214,"html_url":"https://github.com/o/r/issues/214"}"#,
    )]);
    let events = drive_submit(&fake, job());
    let [ReportEvent::Created { number: 214, url }] = events.as_slice() else {
        panic!("expected Created");
    };
    assert_eq!(url, "https://github.com/o/r/issues/214");
    let seen = fake.seen.borrow();
    let (url, payload, token) = &seen[0];
    assert_eq!(url, "https://api.github.com/repos/owner/metrale/issues");
    // 2026-09-26: The token is passed to the transport for the
    // `Authorization` header and is not in the JSON body.
    assert_eq!(token.as_deref(), Some("ghu_OLD"));
    assert!(!payload.contains("ghu_OLD"), "{payload}");
    assert!(
        !payload.contains("labels"),
        "labels are silently dropped for non-push users"
    );
    assert!(payload.contains("it broke"));
}

#[test]
fn a_401_refreshes_once_and_retries_with_the_new_token() {
    let fake = Fake::new(vec![
        ok(401, r#"{"message":"Bad credentials"}"#),
        ok(200, TOKENS),
        ok(201, r#"{"number":215,"html_url":"https://x"}"#),
    ]);
    let events = drive_submit(&fake, job());
    let [
        ReportEvent::Authorized { access, .. },
        ReportEvent::Created { number: 215, .. },
    ] = events.as_slice()
    else {
        panic!("expected Authorized (rotated tokens) then Created");
    };
    assert_eq!(access.expose(), "ghu_NEW");
    let seen = fake.seen.borrow();
    assert_eq!(seen.len(), 3);
    assert!(seen[1].1.contains("grant_type=refresh_token"));
    assert!(seen[1].1.contains("refresh_token=ghr_OLD"));
    assert_eq!(
        seen[2].2.as_deref(),
        Some("ghu_NEW"),
        "retry must use the rotated token"
    );
}

#[test]
fn refresh_failure_drops_the_auth_and_says_so() {
    for script in [
        // 2026-09-26: The refresh endpoint answers with an OAuth error.
        vec![ok(401, "{}"), ok(200, r#"{"error":"bad_refresh_token"}"#)],
        // 2026-09-26: The refresh succeeds but the retry still gets 401.
        vec![ok(401, "{}"), ok(200, TOKENS), ok(401, "{}")],
    ] {
        let fake = Fake::new(script);
        let events = drive_submit(&fake, job());
        let Some(ReportEvent::SubmitFailed { message, drop_auth }) = events.last() else {
            panic!("expected SubmitFailed last");
        };
        assert!(*drop_auth, "dead tokens must be dropped, not replayed");
        assert!(message.contains("re-authorizes"), "{message}");
    }
}

#[test]
fn a_401_with_no_refresh_token_is_terminal() {
    let fake = Fake::new(vec![ok(401, "{}")]);
    let mut j = job();
    j.refresh = None;
    let events = drive_submit(&fake, j);
    let [
        ReportEvent::SubmitFailed {
            drop_auth: true, ..
        },
    ] = events.as_slice()
    else {
        panic!("expected one SubmitFailed with drop_auth");
    };
}

#[test]
fn rate_limit_and_archive_and_validation_reach_the_user_verbatim() {
    let rows: [(HttpResult, &str); 3] = [
        (
            Ok(HttpReply {
                status: 403,
                body: "{}".into(),
                retry_after: Some(42),
            }),
            "retry in 42s",
        ),
        (ok(410, "{}"), "archived or disabled"),
        (
            ok(422, r#"{"message":"Validation Failed"}"#),
            "Validation Failed",
        ),
    ];
    for (reply, needle) in rows {
        let fake = Fake::new(vec![reply]);
        let events = drive_submit(&fake, job());
        let [
            ReportEvent::SubmitFailed {
                message,
                drop_auth: false,
            },
        ] = events.as_slice()
        else {
            panic!("expected one SubmitFailed");
        };
        assert!(message.contains(needle), "{message}");
    }
}

#[test]
fn transport_failure_keeps_auth_and_names_the_network() {
    let fake = Fake::new(vec![Err("connection refused".into())]);
    let events = drive_submit(&fake, job());
    let [
        ReportEvent::SubmitFailed {
            message,
            drop_auth: false,
        },
    ] = events.as_slice()
    else {
        panic!("expected one SubmitFailed");
    };
    assert_eq!(message, crate::tui::report::NETWORK_FAILED);
}

#[test]
fn spawn_failure_still_answers() {
    // 2026-09-26: `LiveWorkers` sends an event when the thread cannot be
    // spawned. A failed spawn cannot be forced portably in a test, so this
    // checks the weaker half: a send to a dropped receiver returns an error
    // rather than panicking.
    let (tx, rx) = channel::<ReportEvent>();
    drop(rx);
    assert!(
        tx.send(ReportEvent::AuthFailed {
            message: "x".into()
        })
        .is_err()
    );
}
