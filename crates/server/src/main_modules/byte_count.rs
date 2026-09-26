// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: HTTP byte counters `HTTP_BYTES_IN` and `HTTP_BYTES_OUT`, which
//! the dashboard polls. Request bytes come from `Content-Length` when it is
//! declared; response bytes are counted as body frames are polled, so
//! streamed responses count too.
//!
//! Owner: server (HTTP layer).
//! Invariants: none beyond the types.

use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::Body;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::Response;

use crate::metrics::{HTTP_BYTES_IN, HTTP_BYTES_OUT};

struct CountedBody {
    inner: Body,
}

impl http_body::Body for CountedBody {
    type Data = axum::body::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        // 2026-09-26: `axum::body::Body` is `Unpin`, so `Pin::new` suffices.
        let this = self.get_mut();
        let polled = Pin::new(&mut this.inner).poll_frame(cx);
        if let Poll::Ready(Some(Ok(frame))) = &polled
            && let Some(data) = frame.data_ref()
        {
            HTTP_BYTES_OUT.inc_by(data.len() as u64);
        }
        polled
    }
}

/// 2026-09-26: Count the request's declared bytes and wrap the response body
/// to count what is sent.
pub(crate) async fn byte_count_middleware(req: Request<Body>, next: Next) -> Response {
    if let Some(len) = req
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
    {
        HTTP_BYTES_IN.inc_by(len);
    }
    let resp = next.run(req).await;
    let (parts, body) = resp.into_parts();
    Response::from_parts(parts, Body::new(CountedBody { inner: body }))
}
