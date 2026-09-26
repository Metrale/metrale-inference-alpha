// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The budget-aware grammar close, `compile_grammar_state` and
//! `StartPrefillResult`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Longest close, in bytes, that `emit_grammar_close` searches
/// for. With no close within it, nothing is emitted.
const MAX_GRAMMAR_CLOSE_BYTES: usize = 32;

/// 2026-09-25: Budget-aware close, called just before a length stop: when
/// a grammar is attached, not terminated, and stopping is not legal at its
/// position, emit the shortest grammar-legal close
/// (`GrammarState::completion_token_ids`) so the truncated output still
/// parses. The close tokens are pushed to `output_tokens` and streamed
/// through `RequestIo::emit`; a failed send stops the loop.
///
/// Emits nothing when `budget_close` (the run's
/// `SchedLevers::grammar_budget_close`) is false, inside `<think>`, or when
/// no close fits in `MAX_GRAMMAR_CLOSE_BYTES`.
pub(crate) fn emit_grammar_close(
    a: &mut ActiveSeq,
    io: &crate::scheduler::io::SchedIo,
    budget_close: bool,
) {
    if a.inside_thinking || !budget_close {
        return;
    }
    let close = {
        let Some(gs) = a.grammar_state.as_mut() else {
            return;
        };
        if gs.is_terminated() || gs.stop_legal(&a.eos_tokens) {
            return;
        }
        match gs.completion_token_ids(MAX_GRAMMAR_CLOSE_BYTES) {
            Some(tokens) if !tokens.is_empty() => tokens,
            _ => return,
        }
    };
    tracing::info!(
        close_len = close.len(),
        output_len = a.output_tokens.len(),
        "grammar budget-close: emitting graceful close so length-stop yields parseable output"
    );
    for tok in close {
        let tok = tok as u32;
        a.output_tokens.push(tok);
        if !io
            .req
            .emit(&a.sink, StreamEvent::Token(tok), "token stream")
        {
            break;
        }
    }
}

/// 2026-09-25: Compile the request's grammar state.
///
/// `None` when there is no spec or engine, when the tool parser opts out
/// (logged at debug), or when compiling or building the state fails
/// (logged as a warning). The prefill steps call it once per request.
pub fn compile_grammar_state(
    engine: &mut Option<GrammarEngine>,
    grammar_spec: &Option<GrammarSpec>,
    eos_tokens: &[u32],
) -> Option<GrammarState> {
    let spec = grammar_spec.as_ref()?;
    let engine = engine.as_mut()?;

    // 2026-09-25: the request's tool parser compiles its own grammar. A
    // parser that keeps the `ToolCallParser::compile_tool_grammar` default
    // (`None`), such as `MistralNativeParser`, opts out: no grammar.
    let compiled = match spec {
        GrammarSpec::ToolCall {
            tools,
            parser,
            use_triggers,
        } => match parser.compile_tool_grammar(engine, tools, *use_triggers) {
            Some(result) => result,
            None => {
                tracing::debug!(
                    "Grammar: parser '{}' opted out of constrained decoding for this request",
                    parser.name(),
                );
                return None;
            }
        },
        GrammarSpec::JsonObject => engine.compile_json_grammar(),
        GrammarSpec::JsonSchema { schema } => engine.compile_json_schema(schema),
    };

    let label = match spec {
        GrammarSpec::ToolCall { parser, tools, .. } => {
            format!("parser={}, tools={}", parser.name(), tools.len())
        }
        GrammarSpec::JsonObject => "response_format=json_object".to_string(),
        GrammarSpec::JsonSchema { .. } => "response_format=json_schema".to_string(),
    };

    match compiled {
        Ok(grammar) => {
            let vocab_size = engine.vocab_size();
            // 2026-09-25: `on_warm` saves the grown mask cache to disk when
            // the request's background prewarm finishes. `None` when
            // persistence is off.
            let on_warm = engine.mask_snapshot_hook();
            match GrammarState::new_with_hook(&grammar, vocab_size, on_warm) {
                Ok(state) => {
                    tracing::info!("Grammar constrained decoding active: {label}");
                    // 2026-09-25: `accept_token` exempts these stop/EOS
                    // tokens from grammar refusal.
                    Some(state.with_stop_tokens(eos_tokens))
                }
                Err(e) => {
                    tracing::warn!("Grammar state creation failed: {e}");
                    None
                }
            }
        }
        Err(e) => {
            tracing::warn!("Grammar compilation failed: {e}");
            None
        }
    }
}

/// 2026-09-25: Result of starting a chunked prefill.
pub enum StartPrefillResult {
    /// 2026-09-25: The whole prompt was prefilled; the sequence joins decode.
    Active(ActiveSeq),
    /// 2026-09-25: Chunks remain; the caller adds it to `prefilling`.
    InProgress(PrefillInProgress),
    /// 2026-09-25: The request already finished and `finish_sequence` ran:
    /// a beam-search request, a first token that is EOS, or
    /// `max_tokens <= 1`.
    Finished,
}
