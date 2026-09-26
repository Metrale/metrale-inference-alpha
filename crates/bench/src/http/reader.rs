// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Incremental response reader for the streaming chat client:
//! chunked or raw framing, and error-body collection.
//!
//! Owner: bench (HTTP client).
//! Invariants:
//! - An error body is kept to at most `MAX_ERROR_BODY` bytes.
//! - `push` returns only whole lines; a partial line or chunk stays buffered.

use anyhow::Result;

use super::failure::{FailureKind, RequestFailure, status_kind};
use super::{
    MAX_ERROR_BODY, content_length, find, is_chunked, message_from_body, status_is_success,
};

/// 2026-09-26: Incremental HTTP response reader: status/header parse, chunked
/// decode, then complete lines out.
#[derive(Default)]
pub(super) struct Reader {
    raw: Vec<u8>,
    header_end: Option<usize>,
    chunked: bool,
    pub(super) body: Vec<u8>,
    consumed: usize,
    /// 2026-09-26: Set once a non-200 status line is seen. The reader then
    /// collects the body instead of failing at once (see `push`).
    pub(super) error_status: Option<String>,
    error_len: Option<usize>,
    /// 2026-09-26: The terminal zero-length chunk has been consumed.
    chunks_done: bool,
}

impl Reader {
    /// 2026-09-26: Feed socket bytes, get back the body lines they completed.
    pub(super) fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>> {
        self.raw.extend_from_slice(bytes);
        // 2026-09-26: Already known to be an error: keep collecting its body.
        if self.error_status.is_some() {
            self.collect_error()?;
            return Ok(Vec::new());
        }
        if self.header_end.is_none() {
            let Some(pos) = find(&self.raw, b"\r\n\r\n") else {
                return Ok(Vec::new());
            };
            let head = String::from_utf8_lossy(&self.raw[..pos]).into_owned();
            let status = head.lines().next().unwrap_or_default().to_string();
            if !status_is_success(&status) {
                // 2026-09-26: Do not fail yet. The body carries the server's
                // own explanation, and it may not have arrived: headers can
                // land in a read of their own.
                self.error_status = Some(status);
                self.error_len = content_length(&head);
                // 2026-09-26: An error body can be chunked like any other,
                // so the framing is decoded before the body is parsed as
                // JSON.
                self.chunked = is_chunked(&head);
                self.header_end = Some(pos + 4);
                self.consumed = pos + 4;
                self.collect_error()?;
                return Ok(Vec::new());
            }
            self.chunked = is_chunked(&head);
            self.header_end = Some(pos + 4);
            self.consumed = pos + 4;
        }
        if self.chunked {
            self.decode_chunks()?;
        } else {
            self.body.extend_from_slice(&self.raw[self.consumed..]);
            self.consumed = self.raw.len();
        }
        Ok(self.take_lines())
    }

    /// 2026-09-26: Accumulate an error body, decoding its framing, and fail
    /// once complete.
    ///
    /// Complete means the terminal chunk has arrived, the declared
    /// `Content-Length` is reached, or `MAX_ERROR_BODY` is reached. A body
    /// that reaches none of those is reported at EOF by [`Reader::finish`].
    fn collect_error(&mut self) -> Result<()> {
        if self.chunked {
            // 2026-09-26: A malformed chunk header in an error body reports
            // the status already held, not a framing error.
            if self.decode_chunks().is_err() {
                return self.fail();
            }
        } else {
            let incoming = &self.raw[self.consumed..];
            let keep = (MAX_ERROR_BODY - self.body.len().min(MAX_ERROR_BODY)).min(incoming.len());
            self.body.extend_from_slice(&incoming[..keep]);
            self.consumed = self.raw.len();
        }
        let have = self.body.len();
        let done =
            self.chunks_done || self.error_len.is_some_and(|n| have >= n) || have >= MAX_ERROR_BODY;
        if done { self.fail() } else { Ok(()) }
    }

    /// 2026-09-26: Report a pending error with whatever body arrived. Called
    /// at EOF.
    pub(super) fn finish(&self) -> Result<()> {
        if self.error_status.is_some() {
            self.fail()?;
        }
        Ok(())
    }

    fn fail(&self) -> Result<()> {
        let status = self.error_status.clone().unwrap_or_default();
        let text = String::from_utf8_lossy(&self.body);
        let message = match message_from_body(text.trim()) {
            Some(m) => format!("endpoint returned {status:?}: {m}"),
            None => format!("endpoint returned {status:?}"),
        };
        Err(RequestFailure::new(status_kind(&status), message)
            .with_body(&self.body)
            .into())
    }

    /// 2026-09-26: Pull every whole chunk currently buffered into `body`. A
    /// partial chunk stays in `raw` until the rest arrives.
    fn decode_chunks(&mut self) -> Result<()> {
        loop {
            let rest = &self.raw[self.consumed..];
            let Some(nl) = find(rest, b"\r\n") else {
                return Ok(());
            };
            let header = std::str::from_utf8(&rest[..nl]).unwrap_or("");
            // 2026-09-26: A chunk extension (`;name=value`) may follow the
            // size.
            let size_hex = header.split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(size_hex, 16).map_err(|_| {
                RequestFailure::new(
                    FailureKind::Malformed,
                    format!("malformed chunk size {size_hex:?}"),
                )
                .with_body(rest)
            })?;
            let start = nl + 2;
            let end = start + size;
            // 2026-09-26: +2 for the CRLF that terminates the chunk data.
            if rest.len() < end + 2 {
                return Ok(());
            }
            let data = &rest[start..end];
            let keep = if self.error_status.is_some() {
                (MAX_ERROR_BODY - self.body.len().min(MAX_ERROR_BODY)).min(data.len())
            } else {
                data.len()
            };
            self.body.extend_from_slice(&data[..keep]);
            self.consumed += end + 2;
            if size == 0 {
                self.chunks_done = true;
                return Ok(());
            }
        }
    }

    fn take_lines(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        let mut start = 0;
        while let Some(nl) = find(&self.body[start..], b"\n") {
            let end = start + nl;
            out.push(
                String::from_utf8_lossy(&self.body[start..end])
                    .trim()
                    .to_string(),
            );
            start = end + 1;
        }
        self.body.drain(..start);
        out
    }
}
