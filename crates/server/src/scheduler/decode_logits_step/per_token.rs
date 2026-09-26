// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: One decoded token's bookkeeping in
//! `process_decode_logits_skipping`: hard stops, thinking state, grammar,
//! tool-call guards and EOS handling. A token that is kept is handed to
//! `content_emit`.
//!
//! Owner: scheduler.
//! Invariants:
//! - The `<tool_response>` and stray-`</think>` hard stops set
//!   `a.guard_stop` where they set `finished`; `finish_guard_tests.rs`
//!   pins the count over this module's files.

use super::*;

/// 2026-09-26: Apply one sampled token to its sequence. Each early `return`
/// ends this token's processing; the caller moves on to the next row.
pub(super) fn process_decoded_token(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    now: std::time::Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    a.last_token = tok;
    a.last_token_time = now;

    // 2026-09-25: `<tool_response>` hard stop (`SchedLevers::tool_response_stop`, on
    // unless `METRALE_TOOL_RESPONSE_STOP` is `0` or `false`): the model must not
    // emit this control token; if it does, end the turn before the grammar and
    // EOS handling below.
    if sched.levers.tool_response_stop
        && let Some(trs) = sched.limits.tool_response_hard_stop
        && tok == trs
    {
        a.output_tokens.push(tok);
        a.finished = true;
        // 2026-09-25: Name the cut, or `derive_finish_reason` reports a plain "stop".
        a.guard_stop = Some(GUARD_STOP_TOOL_RESPONSE);
        tracing::debug!(target: "met::scheduler::decode_logits_step", "<tool_response> hard-stop fired (id={trs}); ending turn");
        return;
    }

    // 2026-09-25: `<think>` outside thinking: enter thinking mode without emitting
    // the token. The thinking budget is the spontaneous budget halved per
    // earlier thinking-watchdog fire (at most 4 times, so 1/16), floored at 8.
    // `PostCloseThinkMask` masks `a.think_start_token` while `think_ended` is
    // set.
    if !a.inside_thinking && think_start_token == Some(tok) {
        let decay_shift = a.think_watchdog_fires.min(4);
        let decayed = a.spontaneous_think_budget >> decay_shift;
        a.inside_thinking = true;
        a.think_ended = false;
        a.think_skip_count = 0;
        a.thinking_budget = Some(decayed.max(8));
        if a.think_watchdog_fires > 0 {
            tracing::debug!(target: "met::scheduler::decode_logits_step", fires = a.think_watchdog_fires,
                decayed_budget = decayed,
                "Spontaneous <think> re-entry after watchdog; decayed budget"
            );
        } else {
            tracing::debug!(target: "met::scheduler::decode_logits_step", "Spontaneous <think> detected, entering thinking mode");
        }
        return;
    }

    // 2026-09-25: Drop a `</think>` that arrives outside thinking; the 50th such token
    // (`think_skip_count`) ends the turn.
    if !a.inside_thinking && think_end_token == Some(tok) {
        a.think_skip_count += 1;
        if a.think_skip_count >= 50 {
            a.finished = true;
            // 2026-09-25: Name the cut: the stray token is not pushed, so
            // `derive_finish_reason` has no other sign that a guard ended the turn.
            a.guard_stop = Some(GUARD_STOP_THINK_SKIP);
            tracing::debug!(target: "met::scheduler::decode_logits_step", "</think> think-skip watchdog hard-stop fired (50 consecutive strays); \
                 ending turn"
            );
        }
        return;
    }
    // 2026-09-25: After `</think>`, any other token resets the stray count.
    if a.think_ended {
        a.think_skip_count = 0;
    }

    // 2026-09-25: Advance the grammar matcher only outside thinking; thinking tokens
    // (the closing `</think>` included) are not part of the constrained output.
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
    {
        gs.accept_token(tok);
    }

    // 2026-09-25: Thinking tokens, `</think>` included, draw down the same
    // `remaining` budget as content tokens (`handle_content_token`), so thinking
    // cannot run a request past `max_tokens`. `thinking_budget` is the separate
    // per-block cap armed below.
    if a.inside_thinking {
        a.consume_generation_budget();
        if think_end_token == Some(tok) {
            a.inside_thinking = false;
            a.force_end_thinking = false;
            a.sentence_defer_count = 0;
            a.consecutive_confident = 0;
            a.in_code_fence = false;
            a.think_ended = true;
            // 2026-09-25: One-shot flag for the first token after `</think>`
            // (`PinToToolCallStart` reads it); the next content token clears it.
            a.think_just_ended = true;
        } else {
            a.thinking_tokens += 1;
            // 2026-09-25: Track ``` code-fence parity inside thinking; a forced
            // `</think>` is deferred inside a fence (`should_inject_think_end`).
            // The thinking-loop watchdog below ignores fences.
            a.in_code_fence = toggle_code_fence(a.in_code_fence, tok, code_fence_token);
            // 2026-09-25: Budget exhausted: arm the forced `</think>`, which a later
            // step's pipeline injects.
            if let Some(budget) = a.thinking_budget
                && a.thinking_tokens >= budget
                && !a.force_end_thinking
            {
                a.force_end_thinking = true;
                a.sentence_defer_count = 0;
                tracing::info!(target: "met::scheduler::decode_logits_step", source = if a.enable_thinking {
                        "request (client budget/effort; scaled by --max-thinking-budget)"
                    } else {
                        "spontaneous <think> (--max-thinking-budget / MODEL.toml)"
                    },
                    "Thinking budget exhausted ({budget} tokens), arming </think>; \
                     deferring up to {MAX_SENTENCE_DEFER_TOKENS} tokens for sentence boundary"
                );
            }
            // 2026-09-25: Thinking-loop watchdog: every `THINK_LOOP_CHECK_STRIDE`
            // thinking tokens (from `THINK_LOOP_MIN_TOKENS` on), look for a
            // repeating tail and arm the forced `</think>`.
            if !sched.levers.disable_watchdogs
                && sched.watchdog.enable_think_loop_watchdog
                && !a.force_end_thinking
                && a.thinking_tokens >= THINK_LOOP_MIN_TOKENS
                && a.thinking_tokens.is_multiple_of(THINK_LOOP_CHECK_STRIDE)
                && detect_thinking_token_loop_with(
                    &a.output_tokens,
                    a.repetition_detection,
                    sched.watchdog,
                )
            {
                a.force_end_thinking = true;
                a.sentence_defer_count = 0;
                a.think_watchdog_fires = a.think_watchdog_fires.saturating_add(1);
                tracing::warn!(target: "met::scheduler::decode_logits_step", thinking_tokens = a.thinking_tokens,
                    watchdog_fires = a.think_watchdog_fires,
                    "Thinking-loop watchdog fired (period-{}…{} repeat in tail); forcing </think> early",
                    THINK_LOOP_PERIOD_MIN,
                    THINK_LOOP_PERIOD_MAX,
                );
            }
        }
    } else {
        handle_content_token(a, model, sched);
    }

    // 2026-09-25: A `<tool_call>` outside thinking satisfies `require_tool_call`; one
    // inside thinking does not.
    if a.require_tool_call && tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.require_tool_call = false;
        a.tool_call_opened = true;
    }
    // 2026-09-25: Every `<tool_call>` opened outside thinking resets the inter-tool
    // prose count.
    if tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.prose_tokens_since_last_tool = 0;
        // 2026-09-25: Tool-call repetition guard: count the `<tool_call>` opens after a
        // call has completed (`tool_call_completed`), and end the response at
        // `MAX_POST_COMPLETION_TOOL_OPENS`.
        if a.tool_call_completed {
            a.post_completion_tool_opens = a.post_completion_tool_opens.saturating_add(1);
            const MAX_POST_COMPLETION_TOOL_OPENS: u32 = 8;
            if a.post_completion_tool_opens >= MAX_POST_COMPLETION_TOOL_OPENS {
                tracing::warn!(target: "met::scheduler::decode_logits_step", opens = a.post_completion_tool_opens,
                    "tool-call repetition runaway: model re-opened {MAX_POST_COMPLETION_TOOL_OPENS}+ tool-call blocks after a completed call on a tool_choice=auto turn; ending response (was burning to max_tokens). Sanitizer keeps the first valid call(s)."
                );
                a.output_tokens.push(tok);
                a.tool_call_opened = true;
                if let Some(ref mut gs) = a.grammar_state {
                    gs.accept_token(tok);
                }
                a.finished = true;
                return;
            }
        }
    }
    // 2026-09-25: `require_tool_call` still set after 512 output tokens: clear it,
    // which lifts its hold on EOS.
    if a.require_tool_call && a.output_tokens.len() > 512 {
        tracing::warn!(target: "met::scheduler::decode_logits_step", "require_tool_call safety: no <tool_call> after 512 tokens, clearing EOS suppression"
        );
        a.require_tool_call = false;
    }

    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp);
    }

    // 2026-09-25: `</tool_call>` outside thinking. A request with an active grammar or
    // declared tools (`tools_present`) keeps generating past a closed call, so
    // the model can emit further calls; any other request ends here.
    if tool_call_end_token == Some(tok) && !a.inside_thinking {
        a.output_tokens.push(tok);
        // 2026-09-25: Read by the EOS escape below and by the repeated-open guard
        // above.
        a.tool_call_completed = true;
        if a.sink.is_streaming() {
            let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                StreamEvent::TokenWithLogprobs(tok, lp)
            } else {
                StreamEvent::Token(tok)
            };
            if !sched.io.req.emit(&a.sink, event, "tool_call_end") {
                tracing::warn!(target: "met::scheduler::decode_logits_step", "Streaming receiver dropped during tool_call_end, finishing sequence"
                );
                a.finished = true;
                return;
            }
        }
        if a.grammar_state.is_none() && !a.tools_present {
            // 2026-09-25: No grammar and no tools declared: end the turn.
            a.finished = true;
        }
        // 2026-09-25: The `continue` below skips `update_tool_param_state`; clear
        // `inside_tool_body` and advance the grammar matcher here.
        a.inside_tool_body = false;
        if let Some(ref mut gs) = a.grammar_state {
            gs.accept_token(tok);
        }
        // 2026-09-25: Clear `think_ended` at `</tool_call>`, so `PostCloseThinkMask` no
        // longer masks `<think>` and the model may think again before its next
        // call. Re-entry still shrinks the budget by `think_watchdog_fires`
        // (the `<think>` branch above).
        a.think_ended = false;
        return;
    }

    // 2026-09-25: EOS handling: each `*_suppress*` flag below can hold a sampled EOS
    // back. EOS escape (`SchedLevers::tool_eos_escape`, on unless
    // `METRALE_TOOL_EOS_ESCAPE` is `0` or `false`): once a tool call has
    // completed, outside a tool body and thinking, the grammar does not hold
    // EOS back.
    let eos_escape = sched.levers.tool_eos_escape
        && a.tool_call_completed
        && !a.inside_tool_body
        && !a.inside_thinking;
    // 2026-09-25: The grammar holds EOS back only when the response cannot legally
    // end at the matcher's position (`grammar_blocks_stop`). That fills a
    // bitmask, so it is evaluated only when the sampled token is an EOS.
    let grammar_suppresses_eos = a.eos_tokens.contains(&tok)
        && !eos_escape
        && crate::grammar::grammar_blocks_stop(a.grammar_state.as_mut(), &a.eos_tokens);
    let legacy_suppresses_eos = a.require_tool_call;
    let min_tokens_suppresses = a.output_tokens.len() < a.min_tokens;
    // 2026-09-25: Inside thinking, EOS is held back unless a hard ceiling is hit
    // (`hard_ceiling_hit`): `remaining` is 0 (it already counts this token) or
    // `seq_len` is at the `max_seq_len` ceiling.
    let hard_ceiling = hard_ceiling_hit(a.remaining, a.seq.seq_len, sched.limits.max_seq_len);
    let thinking_suppresses_eos = eos_suppressed_by_thinking(a.inside_thinking, hard_ceiling);
    // 2026-09-25: Post-think EOS guard: on a tool-armed turn (`require_tool_call` or
    // `tool_request`), after `</think>`, hold EOS back until the output holds
    // `POST_THINK_MIN_CONTENT` tokens beyond the thinking ones, so the model
    // has room to open a tool call.
    const POST_THINK_MIN_CONTENT: u32 = 16;
    let post_think_content_tokens =
        (a.output_tokens.len() as u32).saturating_sub(a.thinking_tokens);
    let tools_armed = a.require_tool_call || a.tool_request;
    let post_think_suppresses_eos =
        tools_armed && a.think_ended && post_think_content_tokens < POST_THINK_MIN_CONTENT;
    let suppress_eos = grammar_suppresses_eos
        || legacy_suppresses_eos
        || min_tokens_suppresses
        || thinking_suppresses_eos
        || post_think_suppresses_eos;

    if a.eos_tokens.contains(&tok) && !suppress_eos {
        // 2026-09-25: An EOS that is not held back: counted in `output_tokens` but not
        // streamed; the sequence finishes.
        a.output_tokens.push(tok);
        crate::scheduler::emit_step::update_tool_param_state(a, tok);
        a.finished = true;
    } else if a.eos_tokens.contains(&tok) && suppress_eos {
        // 2026-09-25: A held-back EOS is dropped (not streamed, not in
        // `output_tokens`) and decoding continues.
        //
        // When thinking is the only reason it is held back and the model sets
        // `honor_eos_inside_thinking`, the EOS closes the thinking block: the
        // same state changes as the `</think>` branch above, plus
        // `think_force_closed`. The EOS itself is still dropped, and thinking no
        // longer holds back a later one.
        let thinking_is_sole_suppressor = thinking_suppresses_eos
            && !grammar_suppresses_eos
            && !legacy_suppresses_eos
            && !post_think_suppresses_eos
            && !min_tokens_suppresses;
        // 2026-09-25: MODEL.toml `[behavior].honor_eos_inside_thinking`, false when
        // unset.
        let honor_eos_in_think = sched.watchdog.honor_eos_inside_thinking;
        if thinking_is_sole_suppressor && honor_eos_in_think {
            a.inside_thinking = false;
            a.think_force_closed = true;
            a.force_end_thinking = false;
            a.sentence_defer_count = 0;
            a.consecutive_confident = 0;
            a.in_code_fence = false;
            a.think_ended = true;
            a.think_just_ended = true;
        }
        tracing::debug!(
            target: "metrale::eos",
            tok,
            implicit_think_close = thinking_is_sole_suppressor && honor_eos_in_think,
            thinking_sole_suppressor = thinking_is_sole_suppressor,
            honor_eos_inside_thinking = honor_eos_in_think,
            inside_thinking = a.inside_thinking,
            think_ended = a.think_ended,
            thinking_tokens = a.thinking_tokens,
            by_thinking = thinking_suppresses_eos,
            by_grammar = grammar_suppresses_eos,
            by_legacy_tool = legacy_suppresses_eos,
            by_post_think = post_think_suppresses_eos,
            by_min_tokens = min_tokens_suppresses,
            "EOS suppressed; model forced to continue"
        );
    } else {
        super::content_emit::emit_content_token(a, tok, model, sched);
    }
}
