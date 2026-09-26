// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A kept decoded token: record it, snapshot SSM state, stream it,
//! and apply the length, context, grammar and fuzzy-repetition stops.
//!
//! Owner: scheduler.
//! Invariants:
//! - The fuzzy-repetition stop sets `a.guard_stop` where it sets
//!   `finished`; `finish_guard_tests.rs` pins the count.

use super::*;
use crate::scheduler::io::Effect;

/// 2026-09-26: Record and stream a token that is neither a control token nor
/// an EOS, then run the stops that may end the sequence after it.
pub(super) fn emit_content_token(
    a: &mut ActiveSeq,
    tok: u32,
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    a.output_tokens.push(tok);
    // 2026-09-25: Drive the tool-body and parameter-body state, as the speculative
    // paths do through `emit_step::token`.
    crate::scheduler::emit_step::update_tool_param_state(a, tok);
    // 2026-09-25: Outside thinking: at a boundary token, snapshot the SSM state so a
    // later rollback to it can restore the recurrent state
    // (`rollback::snapshot_boundary_if_ssm`; a no-op for models without
    // SSM layers or without a decode ring).
    if !a.inside_thinking {
        rollback::snapshot_boundary_if_ssm(a, model, sched);
        // 2026-09-25: Block-aligned Marconi checkpoint of the live SSM state.
        let _ = sched
            .io
            .dev
            .apply(Effect::MarconiCheckpoint { seq: &mut a.seq });
    }
    // 2026-09-25: Thinking tokens of a request that did not enable thinking (a
    // spontaneous `<think>`) are not streamed; they stay in `output_tokens`.
    let suppress_stream = a.inside_thinking && !a.enable_thinking;
    if a.sink.is_streaming() && !suppress_stream {
        let event = if let Some(lp) = a.logprobs_data.last().cloned() {
            StreamEvent::TokenWithLogprobs(tok, lp)
        } else {
            StreamEvent::Token(tok)
        };
        if !sched.io.req.emit(&a.sink, event, "decode_logits token") {
            tracing::debug!(target: "met::scheduler::decode_logits_step", "Streaming receiver dropped (decode_logits), finishing seq");
            a.finished = true;
        }
    }
    if a.remaining == 0 {
        // 2026-09-25: Budget exhausted. Outside thinking, and unless
        // `grammar_budget_close` is off, `emit_grammar_close` first closes an
        // active grammar that cannot legally stop here, so the length-stopped
        // output still parses.
        crate::scheduler::emit_step::emit_grammar_close(
            a,
            &sched.io,
            sched.levers.grammar_budget_close,
        );
        tracing::info!(target: "met::scheduler::decode_logits_step", "process_decode_logits: remaining=0, output_tokens={}, thinking_tokens={}",
            a.output_tokens.len(),
            a.thinking_tokens
        );
        a.finished = true;
    }
    // 2026-09-25: Context ceiling: finish once `seq_len + 1` reaches `max_seq_len`
    // (never when it is 0), whatever the thinking state. No EOS is pushed, so
    // the finish reason is "length".
    if !a.finished && seqlen_force_stop(a.seq.seq_len, sched.limits.max_seq_len) {
        tracing::info!(target: "met::scheduler::decode_logits_step", seq_len = a.seq.seq_len,
            max_seq_len = sched.limits.max_seq_len,
            output_tokens = a.output_tokens.len(),
            "process_decode_logits: max_seq_len ceiling reached; force-stop (finish=length)"
        );
        a.finished = true;
    }
    if a.grammar_state
        .as_ref()
        .is_some_and(|gs| gs.is_terminated())
    {
        a.finished = true;
    }

    // 2026-09-25: Fuzzy repetition watchdog (`detect_fuzzy_repetition`: three
    // near-copies of one window at the tail). Skipped inside thinking and
    // inside an open tool call, whose parameter tags repeat. The call is open
    // when the last `<tool_call>` follows the last `</tool_call>`, so a
    // completed call does not disable the check.
    let last_tc_start = a
        .tool_call_start_token
        .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
    let last_tc_end = a
        .tool_call_end_token
        .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
    let inside_tool_call = match (last_tc_start, last_tc_end) {
        (Some(start), Some(end)) => start > end,
        (Some(_), None) => true,
        _ => false,
    };
    if sched.levers.loop_watchdog()
        && !a.finished
        && !a.inside_thinking
        && !inside_tool_call
        && let Some((pattern_len, mis_a, mis_b)) =
            detect_fuzzy_repetition(&a.output_tokens, sched.watchdog.fuzzy_repeat_tolerance_div)
    {
        // 2026-09-25: Roll back, discarding at least the three copies, and re-steer;
        // if the rollback is declined, end the response.
        let min_keep = pattern_len * 3;
        match rollback_to_boundary(a, min_keep, model, sched) {
            RollbackOutcome::RolledBack { dropped } => {
                tracing::warn!(target: "met::scheduler::decode_logits_step", pattern_len,
                    mismatches = mis_a + mis_b,
                    dropped,
                    rollback = a.rollback_count,
                    "Fuzzy repetition detected; rolled back to boundary, re-steering"
                );
            }
            RollbackOutcome::Fallback(reason) => {
                tracing::warn!(target: "met::scheduler::decode_logits_step", "Fuzzy repetition: {pattern_len}-tok pattern x3 ({mis_a}+{mis_b} \
                     mismatches), stopping at {} tokens (rollback declined: {reason:?})",
                    a.output_tokens.len()
                );
                a.guard_stop = Some("fuzzy_repetition");
                a.finished = true;
            }
        }
    }
}
