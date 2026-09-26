// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Bounded sends from the scheduler thread into per-request stream channels.
//!
//! `io/tests.rs` (`no_scheduler_business_file_does_its_own_io`) lists this file
//! among the few where scheduler code may send on a channel directly.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

/// 2026-09-25: Deadline for a scheduler-thread send into a full stream channel:
/// `METRALE_STREAM_SEND_DEADLINE_MS`, or 5000 ms when it is unset or not an
/// integer. A default exists because an unbounded send would never time out.
fn stream_send_deadline() -> std::time::Duration {
    // 2026-09-25: Read once per process. It configures the transport, not the
    // model, and its two readers (`bounded_stream_send`,
    // `spawn_terminal_send`) take no scheduler context.
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    std::time::Duration::from_millis(*MS.get_or_init(|| {
        std::env::var("METRALE_STREAM_SEND_DEADLINE_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5000)
    }))
}

/// 2026-09-25: Bounded send from the scheduler thread into a per-request stream
/// channel.
///
/// The scheduler thread also drives every GPU step, so a send that blocks
/// until a stalled client reads would stall every sequence. The API's stream
/// channels hold 1024 events (`api/chat_stream`, `api/completions`).
///
/// Tries once; on a full channel, retries every 1 ms until
/// [`stream_send_deadline`]. Returns `true` when the event was queued, and
/// `false` when the receiver is gone or the deadline passed; a caller cannot
/// tell the two apart.
pub(in crate::scheduler) fn bounded_stream_send(
    tx: &tokio::sync::mpsc::Sender<crate::api::inference_types::StreamEvent>,
    event: crate::api::inference_types::StreamEvent,
    what: &str,
) -> bool {
    use tokio::sync::mpsc::error::TrySendError;
    let mut event = match tx.try_send(event) {
        Ok(()) => return true,
        Err(TrySendError::Closed(_)) => {
            // 2026-09-25: The client hung up mid-stream. Logged at info so a
            // stream that ends this way is visible in the server log.
            tracing::info!("stream receiver dropped — client went away ({what})");
            return false;
        }
        Err(TrySendError::Full(ev)) => ev,
    };
    let deadline = std::time::Instant::now() + stream_send_deadline();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(1));
        match tx.try_send(event) {
            Ok(()) => return true,
            Err(TrySendError::Closed(_)) => {
                // 2026-09-25: The client hung up while the channel was full.
                tracing::info!(
                    "stream receiver dropped during backpressure — client went away ({what})"
                );
                return false;
            }
            Err(TrySendError::Full(ev)) => {
                if std::time::Instant::now() >= deadline {
                    tracing::warn!(
                        "stream consumer stalled past {:?} with a full channel — abandoning ({what})",
                        stream_send_deadline()
                    );
                    return false;
                }
                event = ev;
            }
        }
    }
}

/// 2026-09-25: Tokio runtime handle captured at serve startup, used by
/// [`spawn_terminal_send`] on the scheduler thread, which is not a runtime
/// worker. When unset (tests), that function falls back to the bounded
/// synchronous send.
static RUNTIME_HANDLE: std::sync::OnceLock<tokio::runtime::Handle> = std::sync::OnceLock::new();

/// 2026-09-25: Capture the current tokio runtime handle. Call from inside the
/// runtime before spawning the scheduler thread. Later calls keep the first
/// handle.
pub fn capture_runtime_handle() {
    let _ = RUNTIME_HANDLE.set(tokio::runtime::Handle::current());
}

/// 2026-09-25: Fire-and-forget send for a terminal stream event (Done or Error),
/// the last event on a sequence's channel.
///
/// Use it only for that last event. The channel is FIFO and nothing follows
/// the last event, so the spawned send arrives after every event queued
/// before it, whenever the task runs. A spawned mid-stream send could be
/// overtaken by the next event's `try_send`.
///
/// The spawned task still applies [`stream_send_deadline`] via
/// `tokio::time::timeout`, so a stalled consumer cannot keep the task alive;
/// on timeout the frame is dropped. Without a captured runtime handle this
/// uses the synchronous [`bounded_stream_send`].
pub(in crate::scheduler) fn spawn_terminal_send(
    tx: &tokio::sync::mpsc::Sender<crate::api::inference_types::StreamEvent>,
    event: crate::api::inference_types::StreamEvent,
    what: &'static str,
) {
    if let Some(h) = RUNTIME_HANDLE.get() {
        let tx = tx.clone();
        let deadline = stream_send_deadline();
        h.spawn(async move {
            match tokio::time::timeout(deadline, tx.send(event)).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) => tracing::debug!("terminal send: receiver dropped ({what})"),
                Err(_) => tracing::warn!(
                    "terminal send: consumer stalled past {deadline:?}, frame dropped ({what})"
                ),
            }
        });
    } else if !bounded_stream_send(tx, event, what) {
        tracing::debug!("terminal send failed synchronously ({what})");
    }
}
