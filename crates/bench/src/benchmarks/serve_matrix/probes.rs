// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The serve matrix's per-round probes: each turns its requests into
//! a [`Signal`] (or a tok/s figure); the bars that read them are in `score.rs`.
//!
//! Owner: bench, serve matrix.
//! Invariants:
//! - Every probe request is greedy (temperature 0) and carries no sampling
//!   penalty (`base_body`, and `coherence::probe` for the coherence leg). A
//!   repetition penalty would penalise the repeated `\n` that `code_shape`
//!   requires.

use std::time::Duration;

use anyhow::Result;
use serde_json::{Value, json};

use super::score::Signal;
use crate::benchmarks::{one_line, stats};
use crate::coherence;
use crate::http;
use crate::plugin::TargetEndpoint;

/// 2026-09-26: Needle for the long-context probe; a test asserts the filler never
/// contains it.
pub const NEEDLE: &str = "PURPLE-DOLPHIN-42";

/// 2026-09-26: The body every probe here starts from: streamed, greedy, no
/// penalties, with `max_tokens`. Callers add the messages and any tools.
fn base_body(model: &str, max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    })
}

fn with_user(mut body: Value, prompt: String) -> Value {
    body["messages"] = json!([{"role": "user", "content": prompt}]);
    body
}

/// 2026-09-26: What one round's coherence leg established.
pub struct Coherence {
    pub passed: usize,
    pub total: usize,
    /// 2026-09-26: The round's identity bar. It fails when a readable
    /// `/v1/models` does not list this round's model, or when a coherence
    /// request failed. An unreadable model list does not fail it
    /// (`coherence::probe_for`).
    pub identity: Signal,
}

/// 2026-09-26: Run [`crate::coherence::probe`] (the `coherence::CHECKS` questions
/// and the model-list check) and map it to the round's coherence count and
/// identity signal. The string is the probe's concern, if any.
pub async fn coherence_probe(target: &TargetEndpoint, timeout: Duration) -> (Coherence, String) {
    let report = coherence::probe(target, timeout).await;
    let identity = match (&report.transport_error, &report.served_instead) {
        (Some(e), _) => Signal::Fail(one_line(e)),
        (None, Some(served)) if served.is_empty() => {
            Signal::Fail("the endpoint has no model loaded".into())
        }
        (None, Some(served)) => Signal::Fail(format!(
            "serving {} — not {}, which this round loaded",
            served.join(", "),
            target.model
        )),
        (None, None) => Signal::Pass,
    };
    let detail = report.concern(target).unwrap_or_default();
    (
        Coherence {
            passed: report.answers.iter().filter(|a| a.passed).count(),
            total: coherence::CHECKS.len(),
            identity,
        },
        one_line(detail),
    )
}

/// 2026-09-26: Structural codegen: does the reply define `fib` with an indented
/// body (`code_shape`)? The reply is not executed.
pub async fn codegen_probe(target: &TargetEndpoint, timeout: Duration, budget: usize) -> Signal {
    let body = with_user(
        base_body(&target.model, budget),
        "Write a Python function `fib(n)` that returns the n-th Fibonacci number. \
         Reply with the code only, no explanation."
            .into(),
    );
    let outcome = match http::chat_stream(target, &body, timeout).await {
        Ok(o) => o,
        Err(e) => return Signal::Fail(one_line(format!("{e:#}"))),
    };
    match code_shape(&outcome.text) {
        Ok(()) => Signal::Pass,
        Err(why) => Signal::Fail(format!("{why}: {}", one_line(&outcome.text))),
    }
}

/// 2026-09-26: The structural test on the reply text.
pub fn code_shape(text: &str) -> Result<(), String> {
    let lines: Vec<&str> = text.lines().collect();
    let Some((start, header)) = lines.iter().enumerate().find(|(_, line)| {
        line.trim_start()
            .strip_prefix("def fib")
            .is_some_and(|rest| rest.trim_start().starts_with('('))
    }) else {
        return Err("no `def fib` in the reply".into());
    };
    if !header.contains(':') {
        return Err("`def fib` has no parameter list or colon".into());
    }
    // 2026-09-26: The body must be on its own indented line; a reply collapsed
    // onto one line has none.
    let indented = lines
        .iter()
        .skip(start + 1)
        .any(|l| !l.trim().is_empty() && (l.starts_with(' ') || l.starts_with('\t')));
    if !indented {
        return Err("the function has no indented body — the reply collapsed onto one line".into());
    }
    Ok(())
}

/// 2026-09-26: The `get_weather` tool the tool-call probe offers, in OpenAI shape.
fn weather_tool() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        },
    }])
}

/// 2026-09-26: Did the model call `get_weather` for Paris? A 4xx refusal of the
/// request is `NotApplicable` (`is_client_rejection`); any other error, or a
/// reply without that call, is a `Fail`.
pub async fn tool_call_probe(target: &TargetEndpoint, timeout: Duration, budget: usize) -> Signal {
    let mut body = with_user(
        base_body(&target.model, budget),
        "What is the weather in Paris?".into(),
    );
    body["tools"] = weather_tool();
    body["tool_choice"] = json!("auto");
    let outcome = match http::chat_stream(target, &body, timeout).await {
        Ok(o) => o,
        Err(e) => {
            // 2026-09-26: A 4xx refusal is a capability gap; a transport failure
            // is not excused as one.
            let msg = one_line(format!("{e:#}"));
            return if is_client_rejection(&msg) {
                Signal::NotApplicable(msg)
            } else {
                Signal::Fail(msg)
            };
        }
    };
    score_tool_call(&outcome)
}

fn score_tool_call(outcome: &http::ChatOutcome) -> Signal {
    let Some(call) = outcome.tool_calls.iter().find(|c| !c.name.is_empty()) else {
        return Signal::Fail("no tool call in the reply".into());
    };
    if call.name != "get_weather" {
        return Signal::Fail(format!("called {:?}, not get_weather", call.name));
    }
    match serde_json::from_str::<Value>(&call.arguments) {
        Ok(args) => match args.get("location").and_then(Value::as_str) {
            Some(location) if location.trim().eq_ignore_ascii_case("Paris") => Signal::Pass,
            Some(location) => Signal::Fail(format!("called for {location:?}, not Paris")),
            None => Signal::Fail(format!("arguments carry no location: {}", call.arguments)),
        },
        Err(e) => Signal::Fail(format!("arguments are not JSON ({e}): {}", call.arguments)),
    }
}

/// 2026-09-26: Did the server answer with a 4xx? Only [`crate::http`]'s non-200
/// wording (`endpoint returned "<status line>"`) counts, and only a 4xx code in
/// the status line: a bare substring match would also take a `400s` timeout or
/// a URL on port 8400.
fn is_client_rejection(msg: &str) -> bool {
    let Some(rest) = msg.split_once("endpoint returned \"").map(|(_, r)| r) else {
        return false;
    };
    let status = rest.split('"').next().unwrap_or_default();
    status.split_whitespace().any(|word| {
        word.len() == 3 && word.starts_with('4') && word.chars().all(|c| c.is_ascii_digit())
    })
}

/// 2026-09-26: Needle-in-a-haystack with a `tokens`-token filler.
pub async fn long_context_probe(
    target: &TargetEndpoint,
    timeout: Duration,
    tokens: usize,
    tag: &str,
) -> Signal {
    let filler = stats::make_prompt(tokens, stats::PromptMode::Natural, tag);
    // 2026-09-26: Mid-document, at the char boundary at or before half the
    // filler: a needle at either end can be found by attending only to the
    // edges.
    let split = filler.len() / 2;
    let cut = (0..=split)
        .rev()
        .find(|i| filler.is_char_boundary(*i))
        .unwrap_or(0);
    let prompt = format!(
        "{}\nThe secret code is {NEEDLE}.\n{}\n\nWhat is the secret code? Reply with the code only.",
        &filler[..cut],
        &filler[cut..]
    );
    let body = with_user(base_body(&target.model, 32), prompt);
    match http::chat_stream(target, &body, timeout).await {
        Ok(o) if o.text.contains(NEEDLE) => Signal::Pass,
        Ok(o) => Signal::Fail(format!("needle not recalled: {}", one_line(&o.text))),
        Err(e) => Signal::Fail(one_line(format!("{e:#}"))),
    }
}

/// 2026-09-26: Decode tok/s from the client's TPOT (`ChatOutcome::tpot_ms`) on a
/// `budget`-token request. `None` when TPOT is undefined (no delta arrived, or
/// fewer than two completion tokens); a failed request reports 0 with its
/// error.
pub async fn tps_probe(
    target: &TargetEndpoint,
    timeout: Duration,
    budget: usize,
    tag: &str,
) -> (Option<f64>, Option<String>) {
    let prompt = stats::make_prompt(128, stats::PromptMode::Count, tag);
    let body = with_user(base_body(&target.model, budget), prompt);
    match http::chat_stream(target, &body, timeout).await {
        Ok(o) => (o.tpot_ms.filter(|v| *v > 0.0).map(|ms| 1000.0 / ms), None),
        Err(e) => (Some(0.0), Some(one_line(format!("{e:#}")))),
    }
}

#[cfg(test)]
#[path = "probes_tests.rs"]
mod tests;
