// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The streaming content sanitizer (`sanitize_content_chunk`),
//! which drops tool-call markup that leaks into content, and the re-exported
//! tool-call descriptors of `sanitizer/toolinfo.rs`.
//!
//! Owner: server API (sanitizer).
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code)]

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

use super::chat::chat_completions_inner;
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::completions::not_supported;
use super::inference_impl::{extract_thinking, strip_stop_sequences, tokenize_stop_sequences};
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::strip::strip_thinking_tags;

use super::inference_types::*;

/// 2026-09-26: Length of the longest suffix of `buf` that is a byte prefix of
/// some leak marker: the only bytes that a later chunk could complete into a
/// marker. At most `tag_max - 1`. The caller floors its cut to a char
/// boundary, which can only hold back more.
fn marker_prefix_hold(buf: &str, markers: &tool_parser::LeakMarkers, tag_max: usize) -> usize {
    let b = buf.as_bytes();
    let max_k = b.len().min(tag_max.saturating_sub(1));
    for k in (1..=max_k).rev() {
        let suffix = &b[b.len() - k..];
        let hit = markers
            .orphan_open
            .iter()
            .chain(markers.close.iter())
            .chain(markers.envelope_open.iter())
            .chain(markers.envelope_close.iter())
            .any(|m| m.as_bytes().starts_with(suffix));
        if hit {
            return k;
        }
    }
    0
}

pub fn sanitize_content_chunk(
    text: &str,
    tag_scan_buf: &mut String,
    suppressing_param_leak: &mut bool,
    inside_envelope: &mut bool,
    markers: &tool_parser::LeakMarkers,
) -> String {
    // 2026-09-26: With no orphan-open and no envelope-open markers (the
    // `LeakMarkers` default), text passes straight through, unbuffered.
    if markers.orphan_open.is_empty() && markers.envelope_open.is_empty() {
        return text.to_string();
    }
    // 2026-09-26: The longest marker bounds how much tail must be held for a
    // tag split across chunks.
    let tag_max: usize = markers
        .orphan_open
        .iter()
        .chain(markers.close.iter())
        .chain(markers.envelope_open.iter())
        .chain(markers.envelope_close.iter())
        .map(|t| t.len())
        .max()
        .unwrap_or(0);

    tag_scan_buf.push_str(text);
    let mut out = String::new();
    loop {
        if *suppressing_param_leak {
            let earliest = markers
                .close
                .iter()
                .filter_map(|t| tag_scan_buf.find(t).map(|p| (p, t.len())))
                .min_by_key(|(p, _)| *p);
            match earliest {
                Some((pos, len)) => {
                    tag_scan_buf.drain(..pos + len);
                    *suppressing_param_leak = false;
                }
                None => {
                    if tag_scan_buf.len() > tag_max.saturating_sub(1) {
                        let keep = tag_max.saturating_sub(1);
                        let drop_to = tag_scan_buf.len() - keep;
                        let cut = tag_scan_buf
                            .char_indices()
                            .map(|(i, _)| i)
                            .take_while(|&i| i <= drop_to)
                            .last()
                            .unwrap_or(0);
                        tag_scan_buf.drain(..cut);
                    }
                    break;
                }
            }
            continue;
        }

        let earliest_env_open = markers
            .envelope_open
            .iter()
            .filter_map(|t| tag_scan_buf.find(t).map(|p| (p, t.len())))
            .min_by_key(|(p, _)| *p);
        let earliest_env_close = markers
            .envelope_close
            .iter()
            .filter_map(|t| tag_scan_buf.find(t).map(|p| (p, t.len())))
            .min_by_key(|(p, _)| *p);
        // 2026-09-26: Inside an envelope, orphan open and close markers are
        // not matched: the inner tags belong to the tool call and pass
        // through. Close markers are dropped only outside an envelope.
        let (earliest_open, earliest_close) = if *inside_envelope {
            (None, None)
        } else {
            (
                markers
                    .orphan_open
                    .iter()
                    .filter_map(|t| tag_scan_buf.find(t).map(|p| (p, t.len())))
                    .min_by_key(|(p, _)| *p),
                markers
                    .close
                    .iter()
                    .filter_map(|t| tag_scan_buf.find(t).map(|p| (p, t.len())))
                    .min_by_key(|(p, _)| *p),
            )
        };
        // 2026-09-26: The earliest match wins. At the same position the order
        // is envelope open, envelope close, orphan open, close, so an envelope
        // open takes its bytes before orphan suppression can start.
        #[derive(Copy, Clone)]
        enum ActKind {
            EnvelopeOpen,
            EnvelopeClose,
            OrphanOpen,
            OrphanClose,
        }
        let mut best: Option<(usize, usize, ActKind)> = None;
        let consider = |cand: Option<(usize, usize)>,
                        kind: ActKind,
                        best: &mut Option<(usize, usize, ActKind)>| {
            if let Some((p, l)) = cand {
                match best {
                    None => *best = Some((p, l, kind)),
                    Some((bp, _, _)) if p < *bp => *best = Some((p, l, kind)),
                    _ => {}
                }
            }
        };
        consider(earliest_env_open, ActKind::EnvelopeOpen, &mut best);
        consider(earliest_env_close, ActKind::EnvelopeClose, &mut best);
        consider(earliest_open, ActKind::OrphanOpen, &mut best);
        consider(earliest_close, ActKind::OrphanClose, &mut best);

        match best {
            Some((pos, tag_len, kind)) => {
                let before: String = tag_scan_buf.drain(..pos).collect();
                out.push_str(&before);
                match kind {
                    ActKind::EnvelopeOpen => {
                        // 2026-09-26: Envelope markers are emitted, not dropped.
                        let env_bytes: String = tag_scan_buf.drain(..tag_len).collect();
                        out.push_str(&env_bytes);
                        *inside_envelope = true;
                    }
                    ActKind::EnvelopeClose => {
                        let env_bytes: String = tag_scan_buf.drain(..tag_len).collect();
                        out.push_str(&env_bytes);
                        *inside_envelope = false;
                    }
                    ActKind::OrphanOpen => {
                        tag_scan_buf.drain(..tag_len);
                        *suppressing_param_leak = true;
                        tracing::warn!(
                            "orphan tool-call leak in content stream; suppressing until close"
                        );
                    }
                    ActKind::OrphanClose => {
                        tag_scan_buf.drain(..tag_len);
                        // 2026-09-26: A stray close outside suppression is dropped.
                    }
                }
                continue;
            }
            None => {
                // 2026-09-26: Hold back only `marker_prefix_hold` bytes; the
                // rest cannot become part of a marker and is emitted now.
                let buf_len = tag_scan_buf.len();
                let hold = marker_prefix_hold(tag_scan_buf, markers, tag_max);
                if buf_len <= hold {
                    break;
                }
                let commit_to = buf_len - hold;
                // 2026-09-26: Floor the cut to a char boundary. It can equal
                // `buf_len`, so with nothing held the whole buffer is emitted.
                let mut cut = commit_to;
                while cut > 0 && !tag_scan_buf.is_char_boundary(cut) {
                    cut -= 1;
                }
                let emit: String = tag_scan_buf.drain(..cut).collect();
                out.push_str(&emit);
                break;
            }
        }
    }
    // 2026-09-26: No final tag scrub here. Each iteration searches the whole
    // buffer, so outside an envelope a complete marker never reaches `out`;
    // inside one, the inner tags are the tool-call payload the parser reads.
    // What is left in `tag_scan_buf` at stream end is scrubbed by
    // `stream_guards::flush_content_sanitizer`.
    out
}

pub const F7_STALL_WARN_THRESHOLD: u32 = 4;
pub const F7_STALL_REFUSE_THRESHOLD: u32 = 5;
const F7_BASH_COMMAND_PREFIX_LEN: usize = 80;
const F7_OTHER_ARG_FALLBACK_LEN: usize = 80;

pub type F7StallBuckets = std::collections::HashMap<(String, String), u32>;

mod toolinfo;

pub use toolinfo::{ToolKind, classify_tool, extract_bash_final_action, primary_arg_for_tool};
