// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Chat tab state: the transcript, the view preferences, and the reducer that turns
//! streamed deltas into it. The HTTP/SSE half is `chat_stream`.
//!
//! The request runs on the runtime handle given to `set_runtime`; its deltas
//! reach the TUI thread over a std mpsc channel that `pump` drains.
//!
//! Owner: server tui.
//! Invariants:
//! - `streaming` is set only by `send` and cleared by `settle` or `reset`.

use std::sync::mpsc::{Receiver, channel};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::chat_thinking::{Reasoning, ThinkingRequest, ThinkingView};

#[derive(Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Model,
}

pub struct ChatMessage {
    pub role: Role,
    pub text: String,
    /// 2026-09-26: The reasoning trace, if the model emitted one; empty otherwise.
    pub reasoning: Reasoning,
    /// 2026-09-26: Time to the first non-empty reasoning or answer delta
    /// (`chat_stream::Clocks::first_any`); role-only and tool-call deltas do not count.
    pub ttft_ms: Option<f64>,
    /// 2026-09-26: Time to the first answer delta; later than `ttft_ms` when the model thinks first.
    pub answer_ttft_ms: Option<f64>,
    pub tok_per_s: Option<f64>,
    pub tokens: usize,
    /// 2026-09-26: The reply has finished, however it ended. Tells "no answer arrived" apart from
    /// "the answer has not started yet", which both leave `text` empty.
    pub done: bool,
}

impl ChatMessage {
    pub fn new(role: Role, text: String) -> Self {
        Self {
            role,
            text,
            reasoning: Reasoning::default(),
            ttft_ms: None,
            answer_ttft_ms: None,
            tok_per_s: None,
            tokens: 0,
            done: false,
        }
    }

    /// 2026-09-26: A finished model reply with no answer text.
    pub fn is_answerless(&self) -> bool {
        self.role == Role::Model && self.done && self.text.is_empty()
    }
}

pub enum ChatDelta {
    Token(String),
    Reasoning(String),
    Done {
        ttft_ms: Option<f64>,
        answer_ttft_ms: Option<f64>,
        think_ms: Option<f64>,
        tok_per_s: Option<f64>,
        tokens: usize,
        reasoning_tokens: usize,
    },
    Error(String),
}

impl ChatDelta {
    /// 2026-09-26: A terminal delta carrying no measurement, which a cancelled stream reports.
    /// Not an `Error`, which would replace the partial reply on screen.
    pub(super) fn cancelled() -> Self {
        Self::Done {
            ttft_ms: None,
            answer_ttft_ms: None,
            think_ms: None,
            tok_per_s: None,
            tokens: 0,
            reasoning_tokens: 0,
        }
    }
}

#[derive(Default)]
pub struct ChatState {
    pub transcript: Vec<ChatMessage>,
    pub input: String,
    pub streaming: bool,
    /// 2026-09-26: What the next message asks the model to do about thinking. It persists until
    /// changed, including across `reset`.
    pub think_req: ThinkingRequest,
    /// 2026-09-26: How a reasoning trace is drawn. It persists until changed, including across `reset`.
    pub think_view: ThinkingView,
    /// 2026-09-26: Whether the last finished reply that produced anything thought, so `Auto` can
    /// report what the model did. Cleared when the request changes and by `reset`.
    pub observed_thinking: Option<bool>,
    /// 2026-09-26: Transcript viewport, in wrapped display rows above the bottom. `None` follows the
    /// streaming tip; `Some(n)` stays n rows up.
    pub scroll: Option<usize>,
    rx: Option<Receiver<ChatDelta>>,
    cancel: Option<tokio::sync::oneshot::Sender<()>>,
    runtime: Option<tokio::runtime::Handle>,
}

impl ChatState {
    pub fn set_runtime(&mut self, handle: tokio::runtime::Handle) {
        self.runtime = Some(handle);
    }

    /// 2026-09-26: Scroll the transcript by `rows` (positive = back toward older turns).
    /// Landing at or past the bottom restores follow (`None`).
    pub fn scroll_by(&mut self, rows: i32) {
        let cur = self.scroll.unwrap_or(0) as i32;
        let next = cur + rows;
        self.scroll = if next <= 0 { None } else { Some(next as usize) };
    }

    /// 2026-09-26: Snap back to the live tip.
    pub fn follow(&mut self) {
        self.scroll = None;
    }

    /// 2026-09-26: Cycle the thinking request: Auto → Off → On → Auto.
    pub fn cycle_request(&mut self) -> ThinkingRequest {
        self.think_req = self.think_req.next();
        // 2026-09-26: The observation described the previous request.
        self.observed_thinking = None;
        self.think_req
    }

    /// 2026-09-26: Cycle how the reasoning block is drawn. Display only; the request is unchanged.
    pub fn cycle_view(&mut self) -> ThinkingView {
        self.think_view = self.think_view.next();
        self.think_view
    }

    /// 2026-09-26: Chat keys that work whether or not the input box has focus.
    ///
    /// `typing` is true when a bare letter belongs to the input buffer, so
    /// only the chorded forms count there. Returns the toast text, if any.
    pub fn on_view_key(&mut self, key: KeyEvent, typing: bool) -> Option<String> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            // 2026-09-26: Alt first: `Alt+t` still reports `Char('t')`, so the plain `t` arm would take it.
            KeyCode::Char('t') if alt => {
                Some(format!("chat: reasoning {}", self.cycle_view().label()))
            }
            KeyCode::Char('T') if !typing => {
                Some(format!("chat: reasoning {}", self.cycle_view().label()))
            }
            KeyCode::Char('t') if ctrl || !typing => {
                let chip = self.cycle_request().chip(None, true);
                Some(format!("chat: {chip} — applies to the next message"))
            }
            _ => None,
        }
    }

    /// 2026-09-26: Transcript navigation when the pane, not the input box, has focus.
    pub fn on_content_key(&mut self, key: KeyEvent) -> Option<String> {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.scroll_by(1),
            KeyCode::Down | KeyCode::Char('j') => self.scroll_by(-1),
            KeyCode::PageUp => self.scroll_by(10),
            KeyCode::PageDown => self.scroll_by(-10),
            KeyCode::Char('G') | KeyCode::End => self.follow(),
            _ => return self.on_view_key(key, false),
        }
        None
    }

    /// 2026-09-26: Send the current input as a user message and start streaming a reply.
    /// Does nothing for blank input or while a reply is streaming.
    pub fn send(&mut self, port: u16) {
        let prompt = self.input.trim().to_string();
        if prompt.is_empty() || self.streaming {
            return;
        }
        self.input.clear();
        // 2026-09-26: Sending resumes following.
        self.follow();
        self.transcript.push(ChatMessage::new(Role::User, prompt));
        self.transcript
            .push(ChatMessage::new(Role::Model, String::new()));
        let Some(rt) = self.runtime.clone() else {
            if let Some(last) = self.transcript.last_mut() {
                last.text = "(chat unavailable: no runtime handle)".into();
                last.done = true;
            }
            return;
        };
        let (tx, rx) = channel();
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
        self.rx = Some(rx);
        self.cancel = Some(cancel_tx);
        self.streaming = true;
        let thinking = self.think_req;
        // 2026-09-26: History for multi-turn, excluding only the empty placeholder `send` just pushed
        // (the last element). An earlier empty model turn, such as a cancelled reply, is kept, so
        // user and assistant turns keep alternating.
        let history = match self.transcript.last() {
            Some(m) if m.role == Role::Model && m.text.is_empty() => {
                &self.transcript[..self.transcript.len() - 1]
            }
            _ => &self.transcript[..],
        };
        let messages: Vec<(String, String)> = history
            .iter()
            .map(|m| {
                (
                    match m.role {
                        Role::User => "user".to_string(),
                        Role::Model => "assistant".to_string(),
                    },
                    m.text.clone(),
                )
            })
            .collect();
        rt.spawn(async move {
            tokio::select! {
                _ = super::chat_stream::stream_chat(port, messages, thinking, tx.clone()) => {}
                _ = cancel_rx => {
                    let _ = tx.send(ChatDelta::cancelled());
                }
            }
        });
    }

    pub fn cancel(&mut self) {
        if let Some(c) = self.cancel.take() {
            let _ = c.send(());
        }
    }

    /// 2026-09-26: Clear the conversation and start a fresh session.
    ///
    /// Client state only: `send` re-sends the full transcript on every request,
    /// so clearing it is the new session. `rx` is dropped as well as cancelled,
    /// because cancellation is async and a delta already in the channel would
    /// otherwise land in the fresh transcript. The typed input and the thinking
    /// request and view are kept.
    pub fn reset(&mut self) {
        self.cancel();
        self.rx = None;
        self.streaming = false;
        self.transcript.clear();
        self.observed_thinking = None;
        self.scroll = None;
    }

    /// 2026-09-26: End the stream, whatever ended it.
    fn settle(&mut self) {
        if let Some(m) = self.transcript.last_mut() {
            m.done = true;
            // 2026-09-26: Record whether the reply thought, read off what arrived. A reply that produced
            // nothing records no observation.
            if m.tokens > 0 || !m.reasoning.is_empty() {
                self.observed_thinking = Some(!m.reasoning.is_empty());
            }
        }
        self.streaming = false;
        self.rx = None;
        self.cancel = None;
    }

    /// 2026-09-26: Drain pending deltas into the transcript. Called on every pass of the event loop.
    pub fn pump(&mut self) {
        let Some(rx) = &self.rx else { return };
        let mut deltas: Vec<ChatDelta> = rx.try_iter().collect();
        // 2026-09-26: `try_iter` stops on Empty and Disconnected alike, so a sender that died without a
        // terminal delta looks like "nothing yet" and would leave `streaming` set, which makes `send`
        // refuse. One `try_recv` tells them apart, and its value is kept: a delta can arrive between
        // `try_iter` ending and this call.
        let mut disconnected = false;
        if deltas.is_empty() && self.streaming {
            match rx.try_recv() {
                Ok(d) => deltas.push(d),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => disconnected = true,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if disconnected {
            if let Some(m) = self.transcript.last_mut()
                && m.text.is_empty()
                && m.reasoning.is_empty()
            {
                m.text = "(the reply ended without finishing)".into();
            }
            self.settle();
            return;
        }
        for d in deltas {
            match d {
                ChatDelta::Reasoning(t) => {
                    if let Some(m) = self.transcript.last_mut() {
                        m.reasoning.begin();
                        m.reasoning.text.push_str(&t);
                        m.reasoning.tokens += 1;
                    }
                }
                ChatDelta::Token(t) => {
                    if let Some(m) = self.transcript.last_mut() {
                        if m.text.is_empty() {
                            // 2026-09-26: The answer has started: stop the thinking clock.
                            m.reasoning.seal();
                        }
                        m.text.push_str(&t);
                        m.tokens += 1;
                    }
                }
                ChatDelta::Done {
                    ttft_ms,
                    answer_ttft_ms,
                    think_ms,
                    tok_per_s,
                    tokens,
                    reasoning_tokens,
                } => {
                    if let Some(m) = self.transcript.last_mut() {
                        m.ttft_ms = ttft_ms;
                        m.answer_ttft_ms = answer_ttft_ms;
                        m.tok_per_s = tok_per_s;
                        // 2026-09-26: Only a real measurement replaces the sealed estimate: a cancel
                        // reports `None`, which would put the thinking clock back on live time.
                        if think_ms.is_some() {
                            m.reasoning.think_ms = think_ms;
                        }
                        if tokens > 0 {
                            m.tokens = tokens;
                        }
                        if reasoning_tokens > 0 {
                            m.reasoning.tokens = reasoning_tokens;
                        }
                    }
                    self.settle();
                    return;
                }
                ChatDelta::Error(e) => {
                    if let Some(m) = self.transcript.last_mut() {
                        m.text = format!("(error: {e})");
                    }
                    self.settle();
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "chat_tests.rs"]
mod pump_tests;

#[cfg(test)]
#[path = "chat_more_tests.rs"]
mod turn_tests;

#[cfg(test)]
#[path = "chat_history_tests.rs"]
mod chat_history_tests;
