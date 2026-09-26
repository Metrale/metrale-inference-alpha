// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Path-independence of tool calls: one target turn, whose correct
//! answer is to call nothing, reached by three predecessor paths that share the
//! same long prefix. The replay script offers no tools, and its comparison
//! passes reworded prose, so it cannot see a tool call that moves with history.
//!
//! Owner: bench, SSM poisoning gate.
//! Invariants:
//! - Only the target's tool calls are compared, raw and in order; its prose is
//!   not.
//!
//! ```text
//! A  prefix → ack → TARGET
//! B  prefix → ack → a turn that calls get_current_weather → TARGET
//! C  prefix → ack → a turn that calls web_search          → TARGET
//! ```
//!
//! The user turns of the paths differ only in the interposed turn. Path A is
//! the reference; the target's `tool_calls` must equal A's on B and C. The
//! target asks for something no offered tool answers, so a call there is wrong
//! whatever its wording.

use serde_json::{Value, json};

use crate::benchmarks::transcript::Transcript;

/// 2026-09-26: The tool schemas offered on every turn of this probe.
pub fn tools() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"}
                    },
                    "required": ["location"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the public web for a query string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"]
                }
            }
        }),
    ]
}

/// 2026-09-26: The interposed turn of path B, which asks for a
/// `get_current_weather` call.
pub const CALLS_WEATHER: &str = "What is the current weather in San Francisco, CA? Use the tool.";

/// 2026-09-26: The interposed turn of path C, which asks for a `web_search` call.
pub const CALLS_SEARCH: &str =
    "Search the web for the VirusTotal API domain report endpoint. Use the tool.";

/// 2026-09-26: The target turn. Its answer is in the document the conversation
/// already carries, and neither offered tool bears on it, so the correct reply
/// calls nothing.
pub const TARGET: &str = "From SYSTEM DOCUMENT 7741-C already in this \
     conversation, state in one sentence what closed membership means. Answer \
     from the document itself.";

/// 2026-09-26: A labelled predecessor path to [`TARGET`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    /// 2026-09-26: prefix → ack → target. The reference.
    Direct,
    /// 2026-09-26: prefix → ack → [`CALLS_WEATHER`] → target.
    AfterWeather,
    /// 2026-09-26: prefix → ack → [`CALLS_SEARCH`] → target.
    AfterSearch,
}

impl Path {
    pub const ALL: [Path; 3] = [Path::Direct, Path::AfterWeather, Path::AfterSearch];

    pub fn label(self) -> &'static str {
        match self {
            Path::Direct => "direct",
            Path::AfterWeather => "after-weather",
            Path::AfterSearch => "after-search",
        }
    }

    /// 2026-09-26: The turn to interpose between the ack and the target, if any.
    pub fn interposed(self) -> Option<&'static str> {
        match self {
            Path::Direct => None,
            Path::AfterWeather => Some(CALLS_WEATHER),
            Path::AfterSearch => Some(CALLS_SEARCH),
        }
    }
}

/// 2026-09-26: The request body for a tool-offering turn: the replay probe's
/// greedy, seed-0 body plus [`tools`] and `tool_choice: "auto"`, so the model
/// decides whether to call.
pub fn request_body(model: &str, messages: &[Value], max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": tools(),
        "tool_choice": "auto",
    })
}

/// 2026-09-26: One path's outcome: the target turn's transcript.
#[derive(Debug, Clone)]
pub struct PathResult {
    pub path: Path,
    pub target: Transcript,
}

impl PathResult {
    /// 2026-09-26: The comparison key: the tool calls alone, as raw
    /// `(name, arguments)` pairs in order. Prose is excluded.
    pub fn calls(&self) -> Vec<(String, String)> {
        self.target.tool_calls.clone()
    }
}

/// 2026-09-26: One path disagreeing with the reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divergence {
    pub path: &'static str,
    pub reference_calls: Vec<(String, String)>,
    pub path_calls: Vec<(String, String)>,
}

impl Divergence {
    /// 2026-09-26: A one-line reading for the log and the verdict.
    pub fn describe(&self) -> String {
        let fmt = |c: &Vec<(String, String)>| {
            if c.is_empty() {
                "no call".to_string()
            } else {
                c.iter()
                    .map(|(n, a)| format!("{n}({a})"))
                    .collect::<Vec<_>>()
                    .join(" + ")
            }
        };
        format!(
            "{}: reference made {}, this path made {}",
            self.path,
            fmt(&self.reference_calls),
            fmt(&self.path_calls)
        )
    }
}

/// 2026-09-26: Compare every path against the reference (`Path::Direct`).
/// Returns one [`Divergence`] per path whose tool calls differ at all; there is
/// no tolerance band as there is for replay prose. Empty when every path
/// agrees, and also when no reference path is present.
pub fn divergences(results: &[PathResult]) -> Vec<Divergence> {
    let Some(reference) = results.iter().find(|r| r.path == Path::Direct) else {
        return Vec::new();
    };
    let ref_calls = reference.calls();
    results
        .iter()
        .filter(|r| r.path != Path::Direct)
        .filter_map(|r| {
            let calls = r.calls();
            (calls != ref_calls).then(|| Divergence {
                path: r.path.label(),
                reference_calls: ref_calls.clone(),
                path_calls: calls,
            })
        })
        .collect()
}

/// 2026-09-26: Did the reference path itself call a tool on the target? A
/// different finding from [`divergences`]: the driver fails the run on a
/// divergence and only logs this one.
pub fn reference_called(results: &[PathResult]) -> bool {
    results
        .iter()
        .find(|r| r.path == Path::Direct)
        .is_some_and(|r| !r.calls().is_empty())
}

#[cfg(test)]
#[path = "toolcall_tests.rs"]
mod tests;
