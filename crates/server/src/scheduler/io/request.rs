// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`RequestIo`]: the scheduler's inbox and every client-bound
//! frame.
//!
//! Arrivals come back from [`RequestIo::recv`] as a plain [`Arrivals`] batch
//! the core folds into its own queue (`PendingQueue::absorb`); token, finish
//! and error frames leave through [`RequestIo::emit`], [`RequestIo::finish`]
//! and [`RequestIo::error`].
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use metrale_scheduler::WaitPolicy;
use parking_lot::{Condvar, Mutex};

use crate::api::{InferenceRequest, InferenceResponse, StreamEvent};
use crate::scheduler::mod_helpers::{bounded_stream_send, spawn_terminal_send};
use crate::scheduler::types::ResponseSink;
use crate::scheduler::{LoraAck, LoraRotation};

/// 2026-09-25: What arrived since the previous `recv`.
#[derive(Default)]
pub struct Arrivals {
    pub requests: Vec<InferenceRequest>,
    pub rotations: Vec<LoraRotation>,
    /// 2026-09-25: The request channel has closed: no more requests will
    /// arrive.
    pub closed: bool,
}

impl Arrivals {
    fn is_empty(&self) -> bool {
        self.requests.is_empty() && self.rotations.is_empty()
    }
}

/// 2026-09-25: The terminal frame of a finished sequence. The streaming and
/// blocking arms read different fields; the router picks by sink.
pub struct FinishFrame<'a> {
    pub finish_reason: &'a str,
    pub output_tokens: &'a [u32],
    pub time_to_first_token_ms: f64,
    pub decode_time_ms: f64,
    pub reasoning_tokens: u32,
    pub cached_prompt_tokens: u32,
    pub accepted_prediction_tokens: usize,
    pub guard_stop: Option<&'static str>,
    /// 2026-09-25: Taken by the blocking arm, left alone by the streaming one.
    pub logprobs: &'a mut Vec<crate::api::TokenLogprobs>,
    pub prompt_logprobs: &'a mut Vec<metrale_model_engine::traits::PromptTokenLogprob>,
}

pub trait RequestIo: Send + Sync {
    /// 2026-09-25: Everything queued since the last call, waiting per `policy`.
    fn recv(&self, policy: WaitPolicy) -> Arrivals;
    /// 2026-09-25: Answer a LoRA control command.
    fn lora_ack(&self, ack: LoraRotationAck, res: Result<LoraAck, String>);
    /// 2026-09-25: A mid-stream event. `true` when it was queued (or the
    /// sink is not a stream); `false` when it was not delivered: the
    /// receiver dropped, or the channel stayed full past the send deadline.
    fn emit(&self, sink: &ResponseSink, event: StreamEvent, what: &str) -> bool;
    /// 2026-09-25: The terminal frame of a finished sequence.
    fn finish(&self, sink: &mut ResponseSink, frame: FinishFrame<'_>);
    /// 2026-09-25: The terminal error frame.
    fn error(&self, sink: &mut ResponseSink, msg: &str, what: &'static str);
    /// 2026-09-25: Whether the request's cancel flag is set; `None` never
    /// cancels.
    fn is_cancelled(&self, flag: Option<&Arc<AtomicBool>>) -> bool;
}

/// 2026-09-25: The ack half of a [`LoraRotation`].
pub type LoraRotationAck = tokio::sync::oneshot::Sender<Result<LoraAck, String>>;

/// 2026-09-25: The serving inbox: two forwarder threads move the tokio
/// channels into a condvar-signalled queue the scheduler thread drains.
pub struct TokioRequestIo {
    queue: Mutex<Arrivals>,
    cv: Condvar,
}

impl TokioRequestIo {
    /// 2026-09-25: Forward `request_rx` and `rotation_rx` into the inbox.
    pub fn new(
        request_rx: tokio::sync::mpsc::Receiver<InferenceRequest>,
        rotation_rx: tokio::sync::mpsc::Receiver<LoraRotation>,
    ) -> Arc<Self> {
        let io = Arc::new(Self {
            queue: Mutex::new(Arrivals::default()),
            cv: Condvar::new(),
        });
        let p = Arc::clone(&io);
        std::thread::spawn(move || {
            let mut rx = request_rx;
            while let Some(req) = rx.blocking_recv() {
                p.queue.lock().requests.push(req);
                p.cv.notify_one();
            }
            p.queue.lock().closed = true;
            p.cv.notify_one();
        });
        let pr = Arc::clone(&io);
        std::thread::spawn(move || {
            let mut rx = rotation_rx;
            while let Some(rot) = rx.blocking_recv() {
                pr.queue.lock().rotations.push(rot);
                pr.cv.notify_one();
            }
        });
        io
    }

    /// 2026-09-25: An inbox nothing feeds, already closed.
    pub fn closed() -> Self {
        Self {
            queue: Mutex::new(Arrivals {
                closed: true,
                ..Arrivals::default()
            }),
            cv: Condvar::new(),
        }
    }
}

impl RequestIo for TokioRequestIo {
    fn recv(&self, policy: WaitPolicy) -> Arrivals {
        let mut g = self.queue.lock();
        match policy {
            WaitPolicy::NoWait => {}
            WaitPolicy::Bounded(d) => {
                if g.is_empty() && !g.closed {
                    let _ = self.cv.wait_for(&mut g, d);
                }
            }
            WaitPolicy::Block => {
                while g.is_empty() && !g.closed {
                    self.cv.wait(&mut g);
                }
            }
        }
        Arrivals {
            requests: std::mem::take(&mut g.requests),
            rotations: std::mem::take(&mut g.rotations),
            closed: g.closed,
        }
    }

    fn lora_ack(&self, ack: LoraRotationAck, res: Result<LoraAck, String>) {
        let _ = ack.send(res);
    }

    fn emit(&self, sink: &ResponseSink, event: StreamEvent, what: &str) -> bool {
        let ResponseSink::Streaming(tx) = sink else {
            return true;
        };
        bounded_stream_send(tx, event, what)
    }

    fn finish(&self, sink: &mut ResponseSink, f: FinishFrame<'_>) {
        match sink {
            ResponseSink::Streaming(tx) => {
                // 2026-09-25: The terminal frame may be sent off the
                // scheduler thread: nothing follows Done on this channel
                // and every earlier event is already queued (see
                // `spawn_terminal_send`).
                spawn_terminal_send(
                    tx,
                    StreamEvent::Done {
                        finish_reason: f.finish_reason.to_string(),
                        prompt_tokens: 0, // 2026-09-25: the API layer counts the prompt.
                        completion_tokens: f.output_tokens.len(),
                        time_to_first_token_ms: f.time_to_first_token_ms,
                        decode_time_ms: f.decode_time_ms,
                        reasoning_tokens: f.reasoning_tokens,
                        cached_prompt_tokens: f.cached_prompt_tokens,
                        accepted_prediction_tokens: f.accepted_prediction_tokens,
                        guard_stop: f.guard_stop,
                    },
                    "done frame",
                );
            }
            ResponseSink::Blocking(tx) => {
                if let Some(tx) = tx.take()
                    && tx
                        .send(Ok(InferenceResponse {
                            output_tokens: f.output_tokens.to_vec(),
                            finish_reason: f.finish_reason.to_string(),
                            time_to_first_token_ms: f.time_to_first_token_ms,
                            decode_time_ms: f.decode_time_ms,
                            logprobs: std::mem::take(f.logprobs),
                            reasoning_tokens: f.reasoning_tokens,
                            cached_prompt_tokens: f.cached_prompt_tokens,
                            accepted_prediction_tokens: f.accepted_prediction_tokens,
                            prompt_logprobs: std::mem::take(f.prompt_logprobs)
                                .into_iter()
                                .map(|p| crate::api::TokenLogprobs {
                                    token_id: p.token_id,
                                    logprob: p.logprob,
                                    top: p.top,
                                })
                                .collect(),
                        }))
                        .is_err()
                {
                    tracing::warn!(
                        "finish_sequence: blocking response send failed (receiver dropped)"
                    );
                }
            }
        }
    }

    fn error(&self, sink: &mut ResponseSink, msg: &str, what: &'static str) {
        match sink {
            ResponseSink::Streaming(tx) => {
                spawn_terminal_send(tx, StreamEvent::Error(msg.to_string()), what);
            }
            ResponseSink::Blocking(tx) => {
                if let Some(tx) = tx.take()
                    && tx.send(Err(anyhow::anyhow!("{msg}"))).is_err()
                {
                    tracing::warn!("blocking Error send failed (receiver dropped) ({what})");
                }
            }
        }
    }

    fn is_cancelled(&self, flag: Option<&Arc<AtomicBool>>) -> bool {
        flag.is_some_and(|f| f.load(std::sync::atomic::Ordering::Acquire))
    }
}
