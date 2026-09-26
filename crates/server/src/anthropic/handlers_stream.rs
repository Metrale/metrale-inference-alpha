// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The streaming response direction: the pipeline's delta stream
//! → Anthropic SSE.
//!
//! Owner: server (Anthropic adapter).
//! Invariants: none beyond the types.

use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use futures::StreamExt;

use super::translator::*;

/// 2026-09-26: Encode the pipeline's delta stream as Anthropic SSE events.
/// The events for one `ir::StreamDelta` are sent before the next delta is
/// read.
///
/// `dump` (the `--dump` seq and writer) records the events as they are sent
/// and writes them as one `stream: true` entry when the stream ends, or when
/// the client hangs up.
pub(super) fn anthropic_sse_from_deltas(
    deltas: crate::ir::DeltaStream,
    req_model: String,
    dump: Option<(u64, crate::request_dumper::DumpHandle)>,
) -> Response {
    // 2026-09-26: Same capacity as the token channel in
    // `api/chat_stream/mod.rs`.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(1024);

    tokio::spawn(async move {
        let mut translator = AnthropicTranslator::new(req_model);
        let mut pending: Vec<SseEvent> = Vec::new();
        let mut deltas = deltas;
        let mut captured: Option<Vec<serde_json::Value>> = dump.as_ref().map(|_| Vec::new());

        // 2026-09-26: Send and clear `pending`, recording each event when
        // `--dump` is on. `false` when the receiver has gone.
        async fn flush(
            tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
            pending: &mut Vec<SseEvent>,
            captured: &mut Option<Vec<serde_json::Value>>,
        ) -> bool {
            for ev in pending.drain(..) {
                if let Some(events) = captured {
                    events.push(serde_json::json!({"event": ev.event, "data": ev.data}));
                }
                if tx.send(Ok(ev.to_axum_event())).await.is_err() {
                    return false;
                }
            }
            true
        }

        let mut aborted = false;
        while let Some(delta) = deltas.next().await {
            translator.on_delta(&delta, &mut pending);
            if !flush(&tx, &mut pending, &mut captured).await {
                aborted = true;
                break;
            }
        }

        if !aborted {
            // 2026-09-26: Closes the message when no Finish or Error delta
            // did.
            translator.finalize(&mut pending);
            if !flush(&tx, &mut pending, &mut captured).await {
                tracing::warn!("anthropic stream: final flush failed (receiver dropped)");
            }
        }

        if let (Some((seq, writer)), Some(events)) = (dump, captured) {
            writer.dump_response("/v1/messages", seq, &events, true);
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
