// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A streaming harness for the content sanitizer. `sanitize_content_chunk`
//! holds back a tail that could still grow into a marker (at most `tag_max - 1` bytes),
//! so which call emits a given byte depends on the markers. Tests built on this harness
//! assert on the whole stream: every chunk plus the end-of-stream flush.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

use crate::api::sanitizer::sanitize_content_chunk;
use crate::api::stream_guards::flush_content_sanitizer;
use crate::tool_parser::LeakMarkers;

pub(super) struct Stream<'a> {
    markers: &'a LeakMarkers,
    tag_scan_buf: String,
    suppressing: bool,
    inside_envelope: bool,
    out: String,
}

impl<'a> Stream<'a> {
    pub(super) fn new(markers: &'a LeakMarkers) -> Self {
        Self {
            markers,
            tag_scan_buf: String::new(),
            suppressing: false,
            inside_envelope: false,
            out: String::new(),
        }
    }

    /// 2026-09-26: Feed one chunk; returns what this call alone emitted.
    pub(super) fn feed(&mut self, chunk: &str) -> String {
        let emitted = sanitize_content_chunk(
            chunk,
            &mut self.tag_scan_buf,
            &mut self.suppressing,
            &mut self.inside_envelope,
            self.markers,
        );
        self.out.push_str(&emitted);
        emitted
    }

    /// 2026-09-26: Feed `text` in slices of `size` bytes, each extended to the next char
    /// boundary.
    pub(super) fn feed_chunked(&mut self, text: &str, size: usize) {
        let mut start = 0;
        while start < text.len() {
            let mut end = (start + size).min(text.len());
            while !text.is_char_boundary(end) {
                end += 1;
            }
            self.feed(&text[start..end]);
            start = end;
        }
    }

    /// 2026-09-26: Flush the held-back tail and return everything emitted.
    pub(super) fn finish(mut self) -> String {
        let tail =
            flush_content_sanitizer(&mut self.tag_scan_buf, &mut self.suppressing, self.markers);
        self.out.push_str(&tail);
        self.out
    }

    pub(super) fn suppressing(&self) -> bool {
        self.suppressing
    }

    pub(super) fn inside_envelope(&self) -> bool {
        self.inside_envelope
    }

    pub(super) fn buffered(&self) -> &str {
        &self.tag_scan_buf
    }
}
