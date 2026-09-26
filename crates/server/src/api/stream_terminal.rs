// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A terminal event for every streaming response. [`terminated`] wraps the
//! scheduler's event channel; when the channel closes before any `Done` or `Error` (the
//! sender was dropped), it adds one `StreamEvent::Error` carrying
//! [`ENDED_WITHOUT_RESULT`]. The chat and completions streams read the channel through it.
//!
//! Owner: server streaming API.
//! Invariants:
//! - When the input stream ends, the output has passed at least one `Done` or `Error`.
//! - The output is the input, plus one trailing `Error` when the input ends without a
//!   `Done` or `Error`.

use futures::{Stream, StreamExt};

use super::inference_types::StreamEvent;

/// 2026-09-26: The message of the added terminal event.
pub(crate) const ENDED_WITHOUT_RESULT: &str = "inference ended without a result: the server dropped this request's stream \
     before it finished or reported an error";

/// 2026-09-26: `events`, plus one `StreamEvent::Error` if it ends without `Done`/`Error`.
pub(crate) fn terminated<S>(events: S) -> impl Stream<Item = StreamEvent>
where
    S: Stream<Item = StreamEvent> + Unpin,
{
    // 2026-09-26: State: (events, a terminal event has passed, the added one was sent).
    // Events after a terminal one are forwarded unchanged.
    futures::stream::unfold(
        (events, false, false),
        |(mut events, seen_terminal, synthesized)| async move {
            if synthesized {
                return None;
            }
            match events.next().await {
                Some(event) => {
                    let terminal =
                        matches!(event, StreamEvent::Done { .. } | StreamEvent::Error(_));
                    Some((event, (events, seen_terminal || terminal, false)))
                }
                None if seen_terminal => None,
                None => Some((
                    StreamEvent::Error(ENDED_WITHOUT_RESULT.to_string()),
                    (events, true, true),
                )),
            }
        },
    )
}

#[cfg(test)]
#[path = "stream_terminal_tests.rs"]
mod tests;
