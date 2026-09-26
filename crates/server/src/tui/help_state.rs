// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Help section state: the Guide/Report sub-tabs, the issue composer and the report phase machine.
//!
//! Owner: server tui.
//! Invariants:
//! - The draft title and body are reset only on `ReportEvent::Created`; every
//!   failure path keeps them.
//! - With `attach_logs` set, `review` opens the Preview, and `proceed` is
//!   reached otherwise only from the Preview keys, so a body with logs is
//!   shown before it is posted.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use super::report::{Composed, ReportEvent, SecretString, Target};
use super::report_http::{LiveWorkers, SubmitJob, Workers};

#[derive(Clone, Copy, PartialEq)]
pub enum HelpSub {
    Guide,
    Report,
}

#[derive(Clone, Copy, PartialEq)]
pub enum ComposerField {
    Title,
    Body,
    Attach,
}

pub enum ReportPhase {
    Compose,
    Preview,
    /// 2026-09-26: Device code requested, not yet received.
    RequestingCode,
    /// 2026-09-26: The user has a code to enter at `verification_uri`; a worker thread polls for the grant.
    WaitingAuth {
        user_code: String,
        verification_uri: String,
        expires_at: Instant,
    },
    Submitting,
    Done {
        number: u64,
        url: String,
    },
    Failed {
        message: String,
    },
}

/// 2026-09-26: The device-flow tokens, held in memory only; `SecretString` zeroes them on drop.
pub(super) struct Auth {
    pub(super) access: SecretString,
    pub(super) refresh: Option<SecretString>,
}

/// 2026-09-26: A composed submission waiting on auth or a retry.
///
/// Dropped on `Created` and by `back_to_compose`; kept through a failure, so
/// `s` in the Failed phase resubmits the same body.
pub(super) struct Pending {
    pub(super) title: String,
    pub(super) body: String,
    pub(super) target: Target,
}

/// 2026-09-26: What the composer needs from `App` to build a body.
pub struct ReportCtx {
    pub model: String,
    pub engine_ready: bool,
    pub tee: Option<&'static str>,
}

pub struct HelpState {
    pub sub: HelpSub,
    pub phase: ReportPhase,
    pub title: String,
    pub title_editing: bool,
    pub body: tui_textarea::TextArea<'static>,
    pub(super) body_editing: bool,
    pub field: ComposerField,
    /// 2026-09-26: On by default; see the module invariants for the preview it forces.
    pub attach_logs: bool,
    pub preview: Option<Composed>,
    pub preview_scroll: usize,
    /// 2026-09-26: Set by `render/help_tab.rs`; the key handlers clamp `preview_scroll` to it.
    pub preview_scroll_max: std::cell::Cell<usize>,
    pub(super) auth: Option<Auth>,
    pub(super) pending: Option<Pending>,
    pub(super) rx: Option<Receiver<ReportEvent>>,
    pub(super) cancel: Option<Arc<AtomicBool>>,
    pub(super) workers: Box<dyn Workers + Send>,
    pub(super) last_message: Option<(String, bool)>,
}

fn fresh_body() -> tui_textarea::TextArea<'static> {
    let mut body = tui_textarea::TextArea::default();
    body.set_cursor_line_style(ratatui::style::Style::default());
    body.set_cursor_style(ratatui::style::Style::default());
    body.set_placeholder_text("What happened, what you expected, steps to reproduce…");
    body
}

impl Default for HelpState {
    fn default() -> Self {
        Self {
            sub: HelpSub::Guide,
            phase: ReportPhase::Compose,
            title: String::new(),
            title_editing: false,
            body: fresh_body(),
            body_editing: false,
            field: ComposerField::Title,
            attach_logs: true,
            preview: None,
            preview_scroll: 0,
            preview_scroll_max: std::cell::Cell::new(0),
            auth: None,
            pending: None,
            rx: None,
            cancel: None,
            workers: Box::new(LiveWorkers),
            last_message: None,
        }
    }
}

impl HelpState {
    /// 2026-09-26: True while the title or body field owns the keyboard; read by `App::in_input`.
    pub fn is_editing(&self) -> bool {
        self.title_editing || self.body_editing
    }

    /// 2026-09-26: The title or body holds non-blank text; the quit prompt names it.
    pub fn has_draft(&self) -> bool {
        !self.title.trim().is_empty() || self.body.lines().iter().any(|l| !l.trim().is_empty())
    }

    /// 2026-09-26: An authorization or submission is in flight.
    pub fn report_in_flight(&self) -> bool {
        matches!(
            self.phase,
            ReportPhase::RequestingCode | ReportPhase::WaitingAuth { .. } | ReportPhase::Submitting
        )
    }

    /// 2026-09-26: The pending toast text and its error flag, taken by the event loop.
    pub fn take_message(&mut self) -> Option<(String, bool)> {
        self.last_message.take()
    }

    pub(super) fn say(&mut self, text: impl Into<String>, error: bool) {
        self.last_message = Some((text.into(), error));
    }

    /// 2026-09-26: Enter the Failed phase, raise an error toast and log the message at warn.
    pub(super) fn fail(&mut self, message: String) {
        tracing::warn!(target: "metrale_tui", "issue report: {message}");
        self.say(message.clone(), true);
        self.phase = ReportPhase::Failed { message };
    }

    pub(super) fn body_text(&self) -> String {
        self.body.lines().join("\n")
    }

    pub(super) fn set_body_editing(&mut self, editing: bool) {
        self.body_editing = editing;
        // 2026-09-26: The cursor is drawn reversed only while the body is being edited.
        self.body.set_cursor_style(if editing {
            ratatui::style::Style::default().add_modifier(ratatui::style::Modifier::REVERSED)
        } else {
            ratatui::style::Style::default()
        });
    }

    /// 2026-09-26: Back to the composer, dropping the preview and `pending`.
    ///
    /// A kept `pending` would let a later `s` submit a body that no longer
    /// matches the edited draft.
    pub(super) fn back_to_compose(&mut self) {
        self.phase = ReportPhase::Compose;
        self.preview = None;
        self.pending = None;
    }

    pub(super) fn review(&mut self, ctx: &ReportCtx) {
        if self.title.trim().is_empty() {
            self.say("a title is required — k to the Title row, ⏎ to edit", true);
            return;
        }
        match self.compose(ctx) {
            Err(e) => self.say(e, true),
            Ok(c) => {
                if self.attach_logs {
                    self.preview = Some(c);
                    self.preview_scroll = 0;
                    self.phase = ReportPhase::Preview;
                } else {
                    // 2026-09-26: Without logs the body is the typed text, the environment
                    // line and the report marker, so it is submitted without a preview.
                    self.proceed(c);
                }
            }
        }
    }

    /// 2026-09-26: Build the body a submit would post; the preview and the POST both use it.
    pub(super) fn compose(&self, ctx: &ReportCtx) -> Result<Composed, String> {
        let env = super::report::env_line(&ctx.model, ctx.engine_ready);
        let logs: Option<Vec<String>> = self.attach_logs.then(|| {
            let redact_ctx = super::redact::RedactCtx::from_env();
            super::log_ring::tail(10_000)
                .into_iter()
                .map(|l| {
                    super::redact::redact_line(
                        &format!("{:>5} {} {}", l.level, l.target, l.message),
                        &redact_ctx,
                    )
                })
                .collect()
        });
        super::report::compose_body(&self.body_text(), &env, logs.as_deref(), ctx.tee)
    }

    pub(super) fn proceed(&mut self, composed: Composed) {
        match super::report::target() {
            Err(m) => self.fail(m.to_string()),
            Ok(target) => {
                self.pending = Some(Pending {
                    title: self.title.trim().to_string(),
                    body: composed.body,
                    target,
                });
                self.auth_or_submit();
            }
        }
    }

    pub(super) fn auth_or_submit(&mut self) {
        let Some(p) = &self.pending else { return };
        if let Some(auth) = &self.auth {
            self.rx = Some(self.workers.submit(SubmitJob {
                client_id: p.target.client_id.clone(),
                repo: p.target.repo.clone(),
                access: auth.access.clone(),
                refresh: auth.refresh.clone(),
                title: p.title.clone(),
                body: p.body.clone(),
            }));
            self.phase = ReportPhase::Submitting;
        } else {
            let cancel = Arc::new(AtomicBool::new(false));
            self.rx = Some(
                self.workers
                    .device_flow(p.target.client_id.clone(), cancel.clone()),
            );
            self.cancel = Some(cancel);
            self.phase = ReportPhase::RequestingCode;
        }
    }

    pub fn pump(&mut self) {
        let Some(rx) = &self.rx else { return };
        let mut events: Vec<ReportEvent> = rx.try_iter().collect();
        // 2026-09-26: `try_iter` cannot tell "nothing yet" from a dead worker, so an
        // empty drain while in flight probes with `try_recv` for `Disconnected`.
        let mut disconnected = false;
        if events.is_empty() && self.report_in_flight() {
            match rx.try_recv() {
                Ok(e) => events.push(e),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => disconnected = true,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if disconnected {
            self.rx = None;
            self.cancel = None;
            self.fail("the report worker stopped unexpectedly — s retries".to_string());
            return;
        }
        for e in events {
            self.apply(e);
        }
    }

    fn apply(&mut self, event: ReportEvent) {
        match event {
            ReportEvent::CodeReady {
                user_code,
                verification_uri,
                expires_in,
            } => {
                self.phase = ReportPhase::WaitingAuth {
                    user_code,
                    verification_uri,
                    expires_at: Instant::now() + expires_in,
                };
            }
            ReportEvent::Authorized { access, refresh } => {
                self.auth = Some(Auth { access, refresh });
                // 2026-09-26: After the device flow, submit what waited. During a submit
                // (a refresh rotation) the result follows on the same channel, and
                // submitting again would post the issue twice.
                if matches!(
                    self.phase,
                    ReportPhase::RequestingCode | ReportPhase::WaitingAuth { .. }
                ) {
                    self.cancel = None;
                    self.auth_or_submit();
                }
            }
            ReportEvent::AuthFailed { message } => {
                self.rx = None;
                self.cancel = None;
                self.fail(message);
            }
            ReportEvent::Created { number, url } => {
                self.rx = None;
                self.title.clear();
                self.body = fresh_body();
                self.title_editing = false;
                self.body_editing = false;
                self.field = ComposerField::Title;
                self.preview = None;
                self.pending = None;
                self.say(format!("issue #{number} opened"), false);
                self.phase = ReportPhase::Done { number, url };
            }
            ReportEvent::SubmitFailed { message, drop_auth } => {
                self.rx = None;
                if drop_auth {
                    self.auth = None;
                }
                self.fail(message);
            }
        }
    }
}

#[cfg(test)]
#[path = "help_state_tests.rs"]
mod tests;
