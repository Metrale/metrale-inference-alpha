// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `stream_terminal::terminated`.
//!
//! Owner: server streaming API.
//! Invariants: none beyond the types.

use futures::StreamExt;

use super::{ENDED_WITHOUT_RESULT, terminated};
use crate::api::inference_types::StreamEvent;

fn done() -> StreamEvent {
    StreamEvent::Done {
        finish_reason: "stop".into(),
        prompt_tokens: 1,
        completion_tokens: 1,
        time_to_first_token_ms: 0.0,
        decode_time_ms: 0.0,
        reasoning_tokens: 0,
        cached_prompt_tokens: 0,
        accepted_prediction_tokens: 0,
        guard_stop: None,
    }
}

/// 2026-09-26: Run `events` through `terminated` and label each output event.
async fn labels(events: Vec<StreamEvent>) -> Vec<String> {
    terminated(futures::stream::iter(events))
        .map(|e| match e {
            StreamEvent::Token(t) => format!("token {t}"),
            StreamEvent::Done { .. } => "done".into(),
            StreamEvent::Error(m) => format!("error {m}"),
            _ => "other".into(),
        })
        .collect()
        .await
}

/// 2026-09-26: Zero or more tokens and no terminal event: the output ends in the added
/// error.
#[tokio::test]
async fn a_stream_dropped_without_a_terminal_event_ends_in_an_error() {
    let silent = format!("error {ENDED_WITHOUT_RESULT}");
    assert_eq!(labels(vec![]).await, vec![silent.clone()]);
    assert_eq!(
        labels(vec![StreamEvent::Token(7)]).await,
        vec!["token 7".to_string(), silent]
    );
}

/// 2026-09-26: Negative controls: a stream that has a terminal event gets nothing added,
/// and whatever follows a terminal event is forwarded unchanged.
#[tokio::test]
async fn a_terminated_stream_is_passed_through_unchanged() {
    assert_eq!(
        labels(vec![StreamEvent::Token(1), done()]).await,
        vec!["token 1", "done"]
    );
    assert_eq!(
        labels(vec![StreamEvent::Error("prefill_chunk failed: x".into())]).await,
        vec!["error prefill_chunk failed: x"]
    );
    assert_eq!(
        labels(vec![done(), StreamEvent::Token(2)]).await,
        vec!["done", "token 2"]
    );
}

/// 2026-09-26: The chat and completions streams read the scheduler channel through
/// `terminated`. The test reads their source text, because the property is about their
/// call sites.
#[test]
fn every_scheduler_stream_surface_reads_through_terminated() {
    const SURFACES: &[(&str, &str)] = &[
        ("chat_stream/mod.rs", include_str!("chat_stream/mod.rs")),
        ("completions.rs", include_str!("completions.rs")),
    ];
    for (name, src) in SURFACES {
        assert!(
            src.contains("terminated(ReceiverStream::new(token_rx))"),
            "{name} reads the scheduler channel without stream_terminal::terminated: \
             a dropped sender would end its SSE body in silence under HTTP 200"
        );
        assert!(
            !src.contains("ReceiverStream::new(token_rx).flat_map"),
            "{name} still maps the raw receiver"
        );
    }
}
