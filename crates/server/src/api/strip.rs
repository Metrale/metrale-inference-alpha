// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Scrubs for assistant content: thinking tags in plain completions,
//! leftover thinking markers, and tool-call markup when no tool call was produced.
//!
//! Owner: server API.
//! Invariants:
//! - When the input contains none of the tags a function looks for, it returns the input
//!   byte for byte.

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
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};

use super::inference_types::*;
use super::sanitizer::*;

pub(crate) fn strip_thinking_tags(text: &str) -> String {
    let default_parser = crate::reasoning_parser::ReasoningFormat::Qwen.into_parser();
    // 2026-09-26: Return untagged text unchanged: the reasoning parser trims its content
    // even when there is no thinking block.
    if !text.contains(default_parser.start_tag()) && !text.contains(default_parser.end_tag()) {
        return text.to_string();
    }
    extract_thinking(text, false, Some(&*default_parser)).1
}

#[cfg(test)]
mod strip_thinking_tests {
    use super::strip_thinking_tags;

    #[test]
    fn no_thinking_tags_preserves_completion_bytes() {
        for text in [
            " there is no way a bee should be able to fly. Its wings are too",
            " ",
            "\n\n",
            "\t  ",
            "    return value;\n",
            "\n```python\n    print('hello')\n```\n",
            "  café 世界  ",
            " partial </thin ",
            "",
        ] {
            assert_eq!(strip_thinking_tags(text).as_bytes(), text.as_bytes());
        }
    }

    #[test]
    fn marked_thinking_keeps_existing_extraction() {
        for (text, expected) in [
            ("<think>reasoning</think> answer ", "answer"),
            ("reasoning</think> answer ", "answer"),
            ("  <think>unfinished reasoning", ""),
            ("<think>first</think>A<think>second</think>B", "AB"),
            ("</think>\n\n  answer", "answer"),
        ] {
            assert_eq!(strip_thinking_tags(text), expected);
        }
    }
}

/// 2026-09-26: Remove every complete `</think>`, `</thinking>`, `<thinking>`, `</analysis>`
/// and `<analysis>` from assistant content, trimming the whitespace that follows each.
/// The reasoning split ends reasoning at the first `</think>`, so a second one stays in
/// the content. Partial markers are left in place. Used by the blocking chat path and,
/// once thinking is done, on each streamed delta (`chat_stream/handle_token.rs`).
pub(crate) fn scrub_think_markers(text: &str) -> String {
    const MARKERS: [&str; 5] = [
        "</think>",
        "</thinking>",
        "<thinking>",
        "</analysis>",
        "<analysis>",
    ];
    let mut out = text.to_string();
    for tag in MARKERS {
        while let Some(pos) = out.find(tag) {
            out = format!("{}{}", &out[..pos], out[pos + tag.len()..].trim_start());
        }
    }
    out
}

#[cfg(test)]
mod scrub_think_tests {
    use super::scrub_think_markers;

    #[test]
    fn removes_second_close_left_by_the_split() {
        assert_eq!(
            scrub_think_markers("17 × 23 = 391.</think>17 × 23 = 391."),
            "17 × 23 = 391.17 × 23 = 391."
        );
    }

    #[test]
    fn trims_whitespace_after_a_removed_marker() {
        assert_eq!(scrub_think_markers("</think>\n\n  hello"), "hello");
    }

    #[test]
    fn removes_every_occurrence_not_just_the_first() {
        assert_eq!(scrub_think_markers("a</think>b</think>c"), "abc");
    }

    #[test]
    fn leaves_ordinary_content_untouched() {
        let s = "Merge sort splits [38, 27] then merges. O(n log n).";
        assert_eq!(scrub_think_markers(s), s);
    }

    #[test]
    fn does_not_eat_partial_or_lookalike_markers() {
        assert_eq!(
            scrub_think_markers("think about </thin"),
            "think about </thin"
        );
    }
}

/// 2026-09-26: Cut assistant content at the first `<tool_call>` or `<function=` and trim
/// the end of what is left. Callers apply it only when no tool call was produced: the
/// blocking path when the choice has no tool calls (`chat_blocking_choice.rs`), and the
/// streaming path, once thinking is done, on each delta of a request with no tools and
/// no tool detector (`chat_stream/handle_token.rs`).
pub(crate) fn strip_orphan_tool_markup(text: &str) -> String {
    const OPENERS: [&str; 2] = ["<tool_call>", "<function="];
    // 2026-09-26: Trim only after a cut. The streaming path calls this on each delta, and
    // a delta can be whitespace only (a lone space before the next token), which a trim
    // would delete.
    match OPENERS.iter().filter_map(|op| text.find(op)).min() {
        Some(cut) => text[..cut].trim_end().to_string(),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod orphan_tool_tests {
    use super::strip_orphan_tool_markup;

    #[test]
    fn cuts_the_live_leak() {
        let s = "You are the story you keep telling yourself.\
                 <tool_call>catch_error({'error': {'message': \"nope\"}})";
        assert_eq!(
            strip_orphan_tool_markup(s),
            "You are the story you keep telling yourself."
        );
    }

    #[test]
    fn cuts_at_function_opener() {
        assert_eq!(
            strip_orphan_tool_markup("Here you go.<function=foo>{}</function>"),
            "Here you go."
        );
    }

    #[test]
    fn leaves_clean_content_untouched() {
        let s = "The sky is blue due to Rayleigh scattering.";
        assert_eq!(strip_orphan_tool_markup(s), s);
    }

    #[test]
    fn does_not_trip_on_the_word_function() {
        let s = "You can call a function to do that.";
        assert_eq!(strip_orphan_tool_markup(s), s);
    }

    /// 2026-09-26: The streaming path applies this to each delta, so with no opener a
    /// delta that is whitespace, or ends in it, must pass through byte for byte.
    #[test]
    fn no_opener_passes_through_byte_identical_including_whitespace() {
        assert_eq!(strip_orphan_tool_markup(" "), " ");
        assert_eq!(strip_orphan_tool_markup("\n\n"), "\n\n");
        assert_eq!(strip_orphan_tool_markup("found "), "found ");
    }
}
