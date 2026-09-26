// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Tool-parser tests for parallel calls: several `<tool_call>`
//! blocks in one response give every call, in order, from `parse_tool_calls`,
//! and per-call events under increasing indices from the streaming detector.
//! Also covers an unterminated trailing call. Hermes and qwen3_coder bodies.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;

mod helpers;

use helpers::{
    args_for_idx, assert_json_eq, drive_chunked, hermes_call, indexed_trace, name_for_idx,
    qwen3_coder_call,
};

#[test]
fn parse_hermes_two_calls_all_returned_in_order() {
    let text = format!(
        "{}\n{}",
        hermes_call("get_weather", r#"{"city": "Paris"}"#),
        hermes_call("get_time", r#"{"tz": "CET"}"#),
    );
    let (content, calls) = parse_tool_calls(&text);
    assert!(content.is_none(), "no content expected: {content:?}");
    assert_eq!(calls.len(), 2, "BOTH calls must be returned");
    assert_eq!(calls[0].function.name, "get_weather");
    assert_json_eq(
        &calls[0].function.arguments,
        r#"{"city":"Paris"}"#,
        "call 0",
    );
    assert_eq!(calls[1].function.name, "get_time");
    assert_json_eq(&calls[1].function.arguments, r#"{"tz":"CET"}"#, "call 1");
    assert_ne!(calls[0].id, calls[1].id, "each call gets a distinct id");
}

#[test]
fn parse_hermes_three_calls_same_name_distinct_args() {
    // 2026-09-26: One function called three times with different arguments:
    // no call is merged or dropped.
    let text = ["Paris", "Berlin", "Tokyo"]
        .iter()
        .map(|c| hermes_call("get_weather", &format!(r#"{{"city": "{c}"}}"#)))
        .collect::<Vec<_>>()
        .join("\n");
    let (content, calls) = parse_tool_calls(&text);
    assert!(content.is_none());
    assert_eq!(calls.len(), 3, "all THREE same-name calls must survive");
    for (i, city) in ["Paris", "Berlin", "Tokyo"].iter().enumerate() {
        assert_eq!(calls[i].function.name, "get_weather");
        assert_json_eq(
            &calls[i].function.arguments,
            &format!(r#"{{"city":"{city}"}}"#),
            &format!("call {i}"),
        );
    }
}

#[test]
fn parse_qwen3_coder_two_calls_all_returned() {
    let text = format!(
        "{}\n{}",
        qwen3_coder_call("search", &[("query", "rust")]),
        qwen3_coder_call("read", &[("path", "/tmp/a.rs")]),
    );
    let (content, calls) = parse_tool_calls(&text);
    assert!(content.is_none(), "no content expected: {content:?}");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "search");
    assert_json_eq(
        &calls[0].function.arguments,
        r#"{"query":"rust"}"#,
        "call 0",
    );
    assert_eq!(calls[1].function.name, "read");
    assert_json_eq(
        &calls[1].function.arguments,
        r#"{"path":"/tmp/a.rs"}"#,
        "call 1",
    );
}

#[test]
fn parse_qwen3_coder_three_calls_with_content_around() {
    // 2026-09-26: Text before and after the calls is kept as content.
    let text = format!(
        "Let me check all three.\n{}\n{}\n{}\nDone.",
        qwen3_coder_call("get_weather", &[("city", "Paris")]),
        qwen3_coder_call("get_weather", &[("city", "Berlin")]),
        qwen3_coder_call("get_weather", &[("city", "Tokyo")]),
    );
    let (content, calls) = parse_tool_calls(&text);
    assert_eq!(calls.len(), 3, "all three calls parsed");
    for (i, city) in ["Paris", "Berlin", "Tokyo"].iter().enumerate() {
        assert_eq!(calls[i].function.name, "get_weather");
        assert_json_eq(
            &calls[i].function.arguments,
            &format!(r#"{{"city":"{city}"}}"#),
            &format!("call {i}"),
        );
    }
    let content = content.expect("prose around the calls is preserved");
    assert!(content.contains("Let me check all three."));
    assert!(content.contains("Done."));
}

#[test]
fn parse_single_call_regression_unchanged() {
    let text = hermes_call("get_weather", r#"{"city": "Paris"}"#);
    let (content, calls) = parse_tool_calls(&text);
    assert!(content.is_none());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    assert_json_eq(
        &calls[0].function.arguments,
        r#"{"city":"Paris"}"#,
        "call 0",
    );
}

/// 2026-09-26: Spaced tag fragments and runs of role names, which must never
/// end up inside a parsed argument.
const DRIFT_SOUP: &str = "</ parameter >userassistantusersystemsystemassistant\n \n\nusersystem";

fn assert_no_soup(calls: &[ToolCall], ctx: &str) {
    for (i, c) in calls.iter().enumerate() {
        assert!(
            !c.function.arguments.contains("userassistant"),
            "{ctx}: call {i} swallowed drift soup: {:?}",
            c.function.arguments
        );
        assert!(
            !c.function.arguments.contains("</ parameter"),
            "{ctx}: call {i} swallowed tag soup: {:?}",
            c.function.arguments
        );
    }
}

#[test]
fn parse_qwen3_coder_unterminated_tail_garbage_contained() {
    // 2026-09-26: Two complete calls, then a third whose value never closes
    // and runs into `DRIFT_SOUP`. The complete calls parse; the third keeps
    // none of the soup.
    let text = format!(
        "{}\n{}\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo {DRIFT_SOUP}",
        qwen3_coder_call("get_weather", &[("city", "Paris")]),
        qwen3_coder_call("get_weather", &[("city", "Berlin")]),
    );
    let (_content, calls) = parse_tool_calls(&text);
    assert_eq!(calls.len(), 3, "two complete + one salvaged call");
    assert_json_eq(
        &calls[0].function.arguments,
        r#"{"city":"Paris"}"#,
        "call 0",
    );
    assert_json_eq(
        &calls[1].function.arguments,
        r#"{"city":"Berlin"}"#,
        "call 1",
    );
    assert_eq!(calls[2].function.name, "get_weather");
    assert_no_soup(&calls, "qwen3_coder blocking");
}

#[test]
fn parse_qwen3_coder_unterminated_tail_keeps_closed_params() {
    // 2026-09-26: Only the unterminated value is dropped; parameters closed
    // before it are kept.
    let text = format!(
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n\
         <parameter=units>\ncelsius {DRIFT_SOUP}"
    );
    let (_content, calls) = parse_tool_calls(&text);
    assert_eq!(calls.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args.get("city").and_then(|v| v.as_str()), Some("Tokyo"));
    assert!(
        args.get("units").is_none(),
        "unterminated param must be dropped, got {args:?}"
    );
    assert_no_soup(&calls, "closed-params containment");
}

#[test]
fn parse_hermes_unterminated_tail_garbage_contained() {
    // 2026-09-26: A complete call, then a JSON body cut off inside a string
    // that runs into the soup; the second call keeps none of it.
    let text = format!(
        "{}\n<tool_call>\n{{\"name\": \"get_weather\", \"arguments\": {{\"city\": \"Tokyo {DRIFT_SOUP}",
        hermes_call("get_weather", r#"{"city": "Paris"}"#),
    );
    let (_content, calls) = parse_tool_calls(&text);
    assert_eq!(calls.len(), 2, "complete + salvaged call");
    assert_json_eq(
        &calls[0].function.arguments,
        r#"{"city":"Paris"}"#,
        "call 0",
    );
    assert_eq!(calls[1].function.name, "get_weather");
    assert_no_soup(&calls, "hermes blocking");
}

#[test]
fn streaming_blocking_parity_on_unterminated_tail() {
    // 2026-09-26: Blocking and streaming give the same call count and names
    // for the same text, and neither carries the soup into any argument.
    let text = format!(
        "{}\n{}\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo {DRIFT_SOUP}",
        qwen3_coder_call("get_weather", &[("city", "Paris")]),
        qwen3_coder_call("get_weather", &[("city", "Berlin")]),
    );

    let (_content, blocking_calls) = parse_tool_calls(&text);

    let mut det = StreamingToolDetector::new();
    let outputs = drive_chunked(&mut det, &text, 7);
    let started: Vec<usize> = indexed_trace(&outputs)
        .iter()
        .filter(|(k, _)| *k == "start" || *k == "call")
        .map(|(_, i)| *i)
        .collect();

    assert_eq!(
        blocking_calls.len(),
        started.len(),
        "blocking ({}) vs streaming ({}) call count diverged on the same emission",
        blocking_calls.len(),
        started.len(),
    );
    for (i, tc) in blocking_calls.iter().enumerate() {
        assert_eq!(
            name_for_idx(&outputs, i).as_deref(),
            Some(tc.function.name.as_str()),
            "name mismatch at idx {i}"
        );
        let streamed = args_for_idx(&outputs, i);
        assert!(
            !streamed.contains("userassistant") && !streamed.contains("</ parameter"),
            "streaming leaked soup at idx {i}: {streamed:?}"
        );
    }
    assert_no_soup(&blocking_calls, "parity blocking side");
    // 2026-09-26: Both paths give the complete calls the same arguments, as
    // JSON values.
    for (i, city) in ["Paris", "Berlin"].iter().enumerate() {
        assert_json_eq(
            &blocking_calls[i].function.arguments,
            &format!(r#"{{"city":"{city}"}}"#),
            &format!("blocking call {i}"),
        );
        assert_json_eq(
            &args_for_idx(&outputs, i),
            &format!(r#"{{"city":"{city}"}}"#),
            &format!("streamed call {i}"),
        );
    }
}

#[test]
fn streaming_hermes_two_calls_incrementing_index() {
    let text = format!(
        "{}\n{}",
        hermes_call("get_weather", r#"{"city": "Paris"}"#),
        hermes_call("get_time", r#"{"tz": "CET"}"#),
    );
    let mut det = StreamingToolDetector::new();
    let outputs = drive_chunked(&mut det, &text, 7);

    let trace = indexed_trace(&outputs);
    // 2026-09-26: Starts, and ends, each come in index order 0, 1.
    let starts: Vec<usize> = trace
        .iter()
        .filter(|(k, _)| *k == "start" || *k == "call")
        .map(|(_, i)| *i)
        .collect();
    let ends: Vec<usize> = trace
        .iter()
        .filter(|(k, _)| *k == "end" || *k == "call")
        .map(|(_, i)| *i)
        .collect();
    assert_eq!(starts, vec![0, 1], "trace: {trace:?}");
    assert_eq!(ends, vec![0, 1], "trace: {trace:?}");
    assert_eq!(name_for_idx(&outputs, 0).as_deref(), Some("get_weather"));
    assert_eq!(name_for_idx(&outputs, 1).as_deref(), Some("get_time"));
    assert_json_eq(&args_for_idx(&outputs, 0), r#"{"city":"Paris"}"#, "idx 0");
    assert_json_eq(&args_for_idx(&outputs, 1), r#"{"tz":"CET"}"#, "idx 1");
}

#[test]
fn streaming_qwen3_coder_three_calls_incrementing_index() {
    let text = format!(
        "{}\n{}\n{}",
        qwen3_coder_call("get_weather", &[("city", "Paris")]),
        qwen3_coder_call("get_weather", &[("city", "Berlin")]),
        qwen3_coder_call("get_weather", &[("city", "Tokyo")]),
    );
    let mut det = StreamingToolDetector::new();
    let outputs = drive_chunked(&mut det, &text, 5);

    let trace = indexed_trace(&outputs);
    let starts: Vec<usize> = trace
        .iter()
        .filter(|(k, _)| *k == "start" || *k == "call")
        .map(|(_, i)| *i)
        .collect();
    assert_eq!(starts, vec![0, 1, 2], "trace: {trace:?}");
    for (i, city) in ["Paris", "Berlin", "Tokyo"].iter().enumerate() {
        assert_eq!(
            name_for_idx(&outputs, i).as_deref(),
            Some("get_weather"),
            "idx {i}"
        );
        assert_json_eq(
            &args_for_idx(&outputs, i),
            &format!(r#"{{"city":"{city}"}}"#),
            &format!("idx {i}"),
        );
    }
    assert!(det.has_tool_calls());
}

#[test]
fn streaming_single_call_close_streamed_regression() {
    // 2026-09-26: A single call with a streamed `</tool_call>` gives exactly
    // one call, at idx 0.
    let text = hermes_call("get_weather", r#"{"city": "Paris"}"#);
    let mut det = StreamingToolDetector::new();
    let outputs = drive_chunked(&mut det, &text, 6);
    let trace = indexed_trace(&outputs);
    assert!(
        trace.iter().all(|(_, i)| *i == 0),
        "single call stays at idx 0: {trace:?}"
    );
    assert_eq!(
        trace
            .iter()
            .filter(|(k, _)| *k == "end" || *k == "call")
            .count(),
        1,
        "exactly one completed call: {trace:?}"
    );
    assert_json_eq(&args_for_idx(&outputs, 0), r#"{"city":"Paris"}"#, "idx 0");
}

#[test]
fn streaming_single_call_dangling_close_flush_regression() {
    // 2026-09-26: With no `</tool_call>` in the stream, `flush` recovers the
    // call at idx 0.
    let mut det = StreamingToolDetector::new();
    let mut outputs =
        det.process("<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"tz\": \"CET\"}}\n");
    outputs.extend(det.flush());
    let trace = indexed_trace(&outputs);
    assert!(
        trace.iter().all(|(_, i)| *i == 0),
        "single dangling call stays at idx 0: {trace:?}"
    );
    assert_eq!(name_for_idx(&outputs, 0).as_deref(), Some("get_time"));
    assert_json_eq(&args_for_idx(&outputs, 0), r#"{"tz":"CET"}"#, "idx 0");
}

#[test]
fn streaming_two_calls_with_interleaved_content() {
    // 2026-09-26: Text between calls goes out as content and does not shift
    // the indices.
    let text = format!(
        "Checking Paris first. {} now Berlin: {} all done.",
        hermes_call("get_weather", r#"{"city": "Paris"}"#),
        hermes_call("get_weather", r#"{"city": "Berlin"}"#),
    );
    let mut det = StreamingToolDetector::new();
    let outputs = drive_chunked(&mut det, &text, 9);
    let starts: Vec<usize> = indexed_trace(&outputs)
        .iter()
        .filter(|(k, _)| *k == "start" || *k == "call")
        .map(|(_, i)| *i)
        .collect();
    assert_eq!(starts, vec![0, 1]);
    let content: String = outputs
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::Content(c) => Some(c.as_str()),
            _ => None,
        })
        .collect();
    assert!(content.contains("Checking Paris first."), "{content:?}");
    assert!(content.contains("all done."), "{content:?}");
}
/// 2026-09-26: Two Hermes calls with no whitespace inside the envelope
/// (`<tool_call>{"name":…,"arguments":{…}}</tool_call>`): both parse paths
/// return both calls' arguments, streaming at every chunk size from 1 to 16.
#[test]
fn grammar_shaped_hermes_two_calls_all_chunk_sizes() {
    let text = "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>\n<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Berlin\"}}</tool_call>";
    let (_c, calls) = parse_tool_calls(text);
    assert_eq!(calls.len(), 2, "blocking count");
    assert_json_eq(&calls[0].function.arguments, r#"{"city":"Paris"}"#, "b0");
    assert_json_eq(&calls[1].function.arguments, r#"{"city":"Berlin"}"#, "b1");
    for chunk in 1..=16usize {
        let mut det = StreamingToolDetector::new();
        let outputs = drive_chunked(&mut det, text, chunk);
        let a0 = args_for_idx(&outputs, 0);
        let a1 = args_for_idx(&outputs, 1);
        let p0: Result<serde_json::Value, _> = serde_json::from_str(&a0);
        let p1: Result<serde_json::Value, _> = serde_json::from_str(&a1);
        assert!(p0.is_ok(), "chunk={chunk} idx0 args not JSON: {a0:?}");
        assert!(p1.is_ok(), "chunk={chunk} idx1 args not JSON: {a1:?}");
        assert_eq!(
            p0.unwrap(),
            serde_json::json!({"city":"Paris"}),
            "chunk={chunk} idx0"
        );
        assert_eq!(
            p1.unwrap(),
            serde_json::json!({"city":"Berlin"}),
            "chunk={chunk} idx1"
        );
    }
}
