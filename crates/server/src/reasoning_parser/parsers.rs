// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The reasoning parsers: one tag-delimited parser configured per format, and the Gemma 4 channel parser.
//!
//! Owner: server.
//! Invariants: none beyond the types.

use super::{ReasoningFormat, ReasoningParser};

/// 2026-09-26: The boxed parser for a [`ReasoningFormat`].
pub(super) fn build(fmt: ReasoningFormat) -> Box<dyn ReasoningParser> {
    match fmt {
        // 2026-09-26: The `<think>` formats assume the prompt opened the block
        // (`prompt_opens_think = true`).
        ReasoningFormat::Qwen => Box::new(TagReasoningParser::THINK_PROMPT_OPENED.named("qwen")),
        ReasoningFormat::DeepSeekR1 => {
            Box::new(TagReasoningParser::THINK_PROMPT_OPENED.named("deepseek_r1"))
        }
        ReasoningFormat::MiniMax => {
            Box::new(TagReasoningParser::THINK_PROMPT_OPENED.named("minimax"))
        }
        // 2026-09-26: Mistral output opens its own `[THINK]` block.
        ReasoningFormat::Mistral => Box::new(TagReasoningParser {
            name: "mistral",
            start: "[THINK]",
            end: "[/THINK]",
            prompt_opens_think: false,
        }),
        ReasoningFormat::Gemma4 => Box::new(Gemma4ReasoningParser),
    }
}

/// 2026-09-26: A tag-delimited reasoning parser, used for every format but
/// Gemma 4, each with its own [`ReasoningParser::name`].
pub(super) struct TagReasoningParser {
    pub(super) name: &'static str,
    pub(super) start: &'static str,
    pub(super) end: &'static str,
    /// 2026-09-26: `true` when output is taken to begin inside the reasoning
    /// block because the prompt opened it (the `<think>` formats); `false` when
    /// the model emits its own opening tag (Mistral).
    pub(super) prompt_opens_think: bool,
}

impl TagReasoningParser {
    /// 2026-09-26: `<think>` tags, opened by the prompt; renamed per format by
    /// [`Self::named`].
    const THINK_PROMPT_OPENED: Self = Self {
        name: "qwen",
        start: "<think>",
        end: "</think>",
        prompt_opens_think: true,
    };

    /// 2026-09-26: This config under another name.
    const fn named(self, name: &'static str) -> Self {
        Self { name, ..self }
    }
}

impl ReasoningParser for TagReasoningParser {
    fn name(&self) -> &str {
        self.name
    }
    fn start_tag(&self) -> &str {
        self.start
    }
    fn end_tag(&self) -> &str {
        self.end
    }
    fn extract_thinking(&self, text: &str, enable_thinking: bool) -> (Option<String>, String) {
        extract_tag_thinking(
            text,
            self.start,
            self.end,
            self.prompt_opens_think,
            enable_thinking,
        )
    }
}

/// 2026-09-26: Split `text` into `(reasoning, content)` for a tag-delimited
/// format:
///
///  * With `prompt_opens_think`, only a `start` tag at the very beginning
///    (after whitespace) is consumed. Without it, text before the first
///    `start` is content.
///  * Reasoning ends at the first `end` tag; what follows is content.
///  * With no `end` tag, the output is all reasoning if it is inside the block
///    (`prompt_opens_think && enable_thinking`, or a `start` tag was found),
///    and all content otherwise.
///  * Balanced `start..end` pairs left in the content are removed.
pub(super) fn extract_tag_thinking(
    text: &str,
    start: &str,
    end: &str,
    prompt_opens_think: bool,
    enable_thinking: bool,
) -> (Option<String>, String) {
    // 2026-09-26: `pre` is content before the block, `body` runs from the start
    // of reasoning, and `in_block` says whether `body` begins inside the block.
    let (pre, body, in_block): (&str, &str, bool) = if prompt_opens_think {
        // 2026-09-26: A `start` later in the text is not a delimiter here; the
        // pair-stripping below removes a balanced one from the content.
        match text.trim_start().strip_prefix(start) {
            Some(rest) => ("", rest, true),
            None => ("", text, enable_thinking),
        }
    } else {
        match text.find(start) {
            Some(p) => (&text[..p], &text[p + start.len()..], true),
            None => ("", text, false),
        }
    };

    let (reasoning, content) = match body.find(end) {
        Some(e) => {
            let mut c = String::with_capacity(pre.len() + body.len() - e);
            c.push_str(pre);
            c.push_str(&body[e + end.len()..]);
            (body[..e].to_string(), c)
        }
        // 2026-09-26: Inside the block with no `end`: the output stopped
        // mid-reasoning.
        None if in_block => (body.to_string(), pre.to_string()),
        None => (String::new(), text.to_string()),
    };

    let content = strip_tag_pairs(&content, start, end);

    let reasoning = reasoning.trim();
    let content = content.trim().to_string();
    if enable_thinking && !reasoning.is_empty() {
        (Some(reasoning.to_string()), content)
    } else {
        (None, content)
    }
}

/// 2026-09-26: Remove every balanced `start..end` pair from `s`; an unmatched
/// `start` and what follows it stay.
fn strip_tag_pairs(s: &str, start: &str, end: &str) -> String {
    let mut out = s.to_string();
    while let Some(a) = out.find(start) {
        match out[a + start.len()..].find(end) {
            Some(rel) => {
                let b = a + start.len() + rel + end.len();
                out.replace_range(a..b, "");
            }
            None => break,
        }
    }
    out
}

/// 2026-09-26: Gemma 4's channel format: reasoning between `<|channel>` (plus
/// an optional `thought` label) and `<channel|>`, the answer after it. Text
/// before the channel is kept as content.
pub(super) struct Gemma4ReasoningParser;

impl Gemma4ReasoningParser {
    const OPEN: &'static str = "<|channel>";
    const CLOSE: &'static str = "<channel|>";
}

impl ReasoningParser for Gemma4ReasoningParser {
    fn name(&self) -> &str {
        "gemma4"
    }
    fn start_tag(&self) -> &str {
        Self::OPEN
    }
    fn end_tag(&self) -> &str {
        Self::CLOSE
    }
    fn extract_thinking(&self, text: &str, enable_thinking: bool) -> (Option<String>, String) {
        let Some((before, after_open)) = text.split_once(Self::OPEN) else {
            return (None, text.trim().to_string());
        };
        let after_label = after_open
            .trim_start()
            .strip_prefix("thought")
            .unwrap_or(after_open);
        match after_label.split_once(Self::CLOSE) {
            Some((reasoning, answer)) => {
                let reasoning = reasoning.trim();
                let content = format!("{}{}", before.trim(), answer.trim());
                let content = content.trim().to_string();
                if enable_thinking && !reasoning.is_empty() {
                    (Some(reasoning.to_string()), content)
                } else {
                    (None, content)
                }
            }
            // 2026-09-26: No close: the output stopped mid-reasoning.
            None => {
                let reasoning = after_label.trim();
                if enable_thinking && !reasoning.is_empty() {
                    (Some(reasoning.to_string()), before.trim().to_string())
                } else {
                    (None, before.trim().to_string())
                }
            }
        }
    }
}
