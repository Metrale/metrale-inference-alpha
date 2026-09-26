// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What a request emitted, reduced to the parts the equality gates
//! compare (contamination, scheduler equivalence, KAT equality, SSM poisoning).
//!
//! Owner: bench.
//! Invariants:
//! - `canonical` holds reasoning, text, tool calls and finish reason, and no
//!   token count.

use crate::http::ChatOutcome;

/// 2026-09-26: The comparable part of one reply.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Transcript {
    /// 2026-09-26: Kept apart from `text`, as `ChatOutcome` does, so a change of
    /// reasoning is not hidden by an identical answer.
    pub reasoning: String,
    pub text: String,
    /// 2026-09-26: `(name, raw arguments)`. Arguments stay raw: re-serialising
    /// the JSON would normalise key order and whitespace.
    pub tool_calls: Vec<(String, String)>,
    pub finish_reason: Option<String>,
    /// 2026-09-26: `ChatOutcome::completion_tokens`: the server's
    /// `usage.completion_tokens` when the stream carries it, else the streamed
    /// delta count.
    pub completion_tokens: usize,
    /// 2026-09-26: Diagnostic: not in `canonical`, so no divergence check here
    /// compares it. The SSM poisoning gate reads it as its turn-1 cache count.
    pub cached_prompt_tokens: usize,
}

impl From<&ChatOutcome> for Transcript {
    fn from(o: &ChatOutcome) -> Self {
        Self {
            reasoning: o.reasoning.clone(),
            text: o.text.clone(),
            tool_calls: o
                .tool_calls
                .iter()
                .map(|t| (t.name.clone(), t.arguments.clone()))
                .collect(),
            finish_reason: o.finish_reason.clone(),
            completion_tokens: o.completion_tokens,
            cached_prompt_tokens: o.cached_prompt_tokens,
        }
    }
}

impl Transcript {
    /// 2026-09-26: The text a divergence check compares, concatenated in a fixed
    /// order with control-character separators; callers also use it for
    /// longest-common-prefix localisation. The token counts are left out:
    /// callers compare `completion_tokens` separately.
    pub fn canonical(&self) -> String {
        let mut s = String::with_capacity(self.reasoning.len() + self.text.len() + 64);
        s.push_str(&self.reasoning);
        s.push('\u{1}');
        s.push_str(&self.text);
        for (name, args) in &self.tool_calls {
            s.push('\u{2}');
            s.push_str(name);
            s.push('\u{3}');
            s.push_str(args);
        }
        s.push('\u{4}');
        s.push_str(self.finish_reason.as_deref().unwrap_or(""));
        s
    }

    /// 2026-09-26: The first canary in `all`, other than `own`, that appears in
    /// this reply's canonical text. Unlike a diff, it needs no reference.
    pub fn carries_foreign_canary<'a>(&self, own: &str, all: &[&'a str]) -> Option<&'a str> {
        let hay = self.canonical();
        all.iter().find(|c| **c != own && hay.contains(*c)).copied()
    }
}

/// 2026-09-26: One request's result. An error is its own variant, so callers
/// score it apart from a transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestOutcome {
    Ok(Box<Transcript>),
    Error(String),
}

impl RequestOutcome {
    pub fn transcript(&self) -> Option<&Transcript> {
        match self {
            Self::Ok(t) => Some(t),
            Self::Error(_) => None,
        }
    }
}
