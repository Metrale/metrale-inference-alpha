// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `GET /v1/events`, the telemetry stream: Server-Sent Events whose `data:` is
//! one JSON snapshot line (`metrale_telemetry::export::events::snapshot_line`), one per
//! `DEVICE_SAMPLE_PERIOD` tick. The stream keeps no queue: each frame reads
//! `Telemetry::snapshot` when it is produced. With `--telemetry off` the route answers
//! 404 and names the flag that turns it on.
//!
//! Owner: server API (telemetry).
//! Invariants:
//! - Frames carry consecutive `seq` values starting at 0.

use std::convert::Infallible;

use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use metrale_telemetry::export::events::snapshot_line;
use metrale_telemetry::level::DEVICE_SAMPLE_PERIOD;
use metrale_telemetry::{Level, Telemetry, global};

use super::compact::openai_error_response;

pub async fn events() -> Response {
    events_for(global()).await
}

/// 2026-09-26: The `/v1/events` response for telemetry `t`.
pub(crate) async fn events_for(t: &'static Telemetry) -> Response {
    if t.level() == Level::Off {
        return openai_error_response(
            StatusCode::NOT_FOUND,
            "telemetry is off: start the server with --telemetry basic (or kernel) to stream \
             /v1/events"
                .to_string(),
        );
    }
    let ticks = tokio::time::interval(DEVICE_SAMPLE_PERIOD);
    let stream = futures::stream::unfold((ticks, 0u64), move |(mut ticks, seq)| async move {
        ticks.tick().await;
        let line = snapshot_line(seq, &t.snapshot());
        Some((
            Ok::<_, Infallible>(Event::default().data(line)),
            (ticks, seq + 1),
        ))
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use metrale_telemetry::clock::MonotonicClock;
    use metrale_telemetry::{Level, Telemetry, TelemetryConfig};

    use super::*;

    fn at(level: Level) -> &'static Telemetry {
        static CLOCK: MonotonicClock = MonotonicClock;
        let t: &'static Telemetry = Box::leak(Box::new(Telemetry::new(&CLOCK)));
        t.configure(&TelemetryConfig::serving(level, 0));
        t
    }

    #[tokio::test]
    async fn off_answers_404_naming_the_flag() {
        let r = events_for(at(Level::Off)).await;
        assert_eq!(r.status(), StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(r.into_body(), 1 << 16).await.unwrap();
        assert!(String::from_utf8_lossy(&body).contains("--telemetry basic"));
    }

    #[tokio::test]
    async fn basic_streams_sse_frames_of_one_json_snapshot_each() {
        let t = at(Level::Basic);
        t.tokens(9);
        let r = events_for(t).await;
        assert_eq!(r.status(), StatusCode::OK);
        assert_eq!(
            r.headers()[axum::http::header::CONTENT_TYPE],
            "text/event-stream"
        );
        let mut frames = r.into_body().into_data_stream();
        for want_seq in 0..2 {
            let chunk = frames.next().await.unwrap().unwrap();
            let text = String::from_utf8(chunk.to_vec()).unwrap();
            let json = text
                .strip_prefix("data: ")
                .and_then(|s| s.strip_suffix("\n\n"))
                .unwrap_or_else(|| panic!("not one SSE data frame: {text:?}"));
            let v: serde_json::Value = serde_json::from_str(json).unwrap();
            assert_eq!(v["seq"], want_seq);
            assert_eq!(v["level"], "basic");
            assert_eq!(v["energy"]["tokens_total"], 9);
        }
    }
}
