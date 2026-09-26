// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Content-phase token handling for `process_decode_logits`: runs once
//! per sampled token while the sequence is outside `<think>`. Budget
//! bookkeeping, the post-think content cap, and two degeneration watchdogs
//! (content loop, inter-tool prose). Each watchdog first tries
//! `rollback_to_boundary` and ends the response only when the rollback is
//! declined.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Slow-path diagnostic: when the content-loop watchdog fires,
/// re-scan to report which `(period, repeats)` matched. Used only on the
/// watchdog-fired branch — runs once per fire, never on the steady-state
/// stride check. Returns `None` if no period matched (caller should not
/// have invoked this).
///
/// Returns the smallest matched period. The count reported is the configured
/// `min_repeats`: the check stops once that many end-anchored windows match.
/// `params` must be the same effective thresholds the detector matched with
/// ([`WatchdogParams::content_loop_params`] output), or the re-scan reports a
/// pattern the fired detector never saw (or none at all).
fn describe_content_token_loop(
    tokens: &[u32],
    params: Option<crate::api::inference_types::RepetitionDetectionParams>,
) -> Option<(usize, usize)> {
    let (period_min, period_max, min_repeats) = match params {
        Some(p) => (
            p.min_pattern_size as usize,
            p.max_pattern_size as usize,
            p.min_count as usize,
        ),
        None => (
            CONTENT_LOOP_PERIOD_MIN,
            CONTENT_LOOP_PERIOD_MAX,
            CONTENT_LOOP_MIN_REPEATS,
        ),
    };
    let n = tokens.len();
    if n < CONTENT_LOOP_MIN_TOKENS as usize {
        return None;
    }
    if min_repeats < 2 {
        return None;
    }
    for pattern_len in period_min..=period_max {
        if pattern_len * min_repeats > n {
            return None;
        }
        // 2026-09-25: The same end-anchored check as the private
        // `helpers::detectors::has_repeating_pattern_anchored`.
        let mut all_match = true;
        'outer: for offset_in_window in 1..=pattern_len {
            let target = tokens[n - offset_in_window];
            for m in 1..min_repeats {
                let idx = n - (pattern_len * m + offset_in_window);
                if tokens[idx] != target {
                    all_match = false;
                    break 'outer;
                }
            }
        }
        if all_match {
            return Some((pattern_len, min_repeats));
        }
    }
    None
}

/// 2026-09-25: Handle one sampled token that lands in the content phase (model is
/// not inside `<think>`). Mutates `a` in place: draws down the generation
/// budget, advances content counters, and runs the post-think content cap and
/// the content-loop and inter-tool-prose watchdogs.
///
/// `model` is passed to [`super::rollback::rollback_to_boundary`], which
/// restores SSM state on hybrid models.
pub fn handle_content_token(
    a: &mut ActiveSeq,
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    a.consume_generation_budget();
    a.content_started = true;
    a.content_tokens = a.content_tokens.saturating_add(1);
    // 2026-09-25: Post-think content cap: a request with an active grammar ends once
    // its content tokens exceed `max_post_think_content_tokens`, whatever
    // `inside_tool_body` says. `emit_step::token` runs the same guard. The
    // value is MODEL.toml `[behavior].max_post_think_content_tokens`, 100_000
    // when unset; the qwen3.6-35b-a3b MODEL.toml files set 8192.
    if !sched.levers.disable_watchdogs
        && a.grammar_state.is_some()
        && a.content_tokens > sched.watchdog.max_post_think_content_tokens
    {
        tracing::warn!(
            content_tokens = a.content_tokens,
            max = sched.watchdog.max_post_think_content_tokens,
            "post-think content cap exceeded in non-MTP decode path; ending response (tool-active request would otherwise burn to max_tokens)"
        );
        a.guard_stop = Some(GUARD_STOP_POST_THINK_CAP);
        a.finished = true;
    }
    // 2026-09-25: `think_just_ended` is a one-shot flag for the first content token
    // after `</think>`; this token consumes it.
    a.think_just_ended = false;

    // 2026-09-25: Content-loop watchdog: outside tool bodies, every
    // `CONTENT_LOOP_CHECK_STRIDE` content tokens (from `CONTENT_LOOP_MIN_TOKENS`
    // on), look for a repeating tail, exactly or with digit runs normalized.
    // Thresholds (`WatchdogParams::content_loop_params`): the request's
    // `repetition_detection`, else `--content-loop-min-repeats` /
    // `METRALE_CONTENT_LOOP_MIN_REPEATS`, else the built-in constants.
    let loop_params = sched.watchdog.content_loop_params(a.repetition_detection);
    if !sched.levers.disable_watchdogs
        && sched.levers.loop_watchdog()
        && !a.inside_tool_body
        && a.content_tokens >= CONTENT_LOOP_MIN_TOKENS
        && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
        && (detect_content_token_loop_with(&a.output_tokens, loop_params)
            || sched.masks.numeric.as_deref().is_some_and(|m| {
                detect_content_token_loop_normalized_with(&a.output_tokens, m, loop_params)
            }))
    {
        // 2026-09-25: Re-scan to report the matched `(period, repeats)` in the log.
        // This runs only when the watchdog has fired, never on the stride check.
        let pattern = describe_content_token_loop(&a.output_tokens, loop_params);
        let (period, repeats) = pattern.unwrap_or((0, 0));
        // 2026-09-25: Roll back to the last boundary and re-steer, discarding at least
        // `CONTENT_LOOP_PERIOD_MAX` tokens; if the rollback is declined, end the
        // response.
        match rollback_to_boundary(a, CONTENT_LOOP_PERIOD_MAX, model, sched) {
            RollbackOutcome::RolledBack { dropped } => {
                tracing::warn!(
                    content_tokens = a.content_tokens,
                    dropped,
                    rollback = a.rollback_count,
                    matched_period = period,
                    matched_repeats = repeats,
                    "Content-loop watchdog fired (period-{}…{} repeat); rolled back to boundary, re-steering",
                    CONTENT_LOOP_PERIOD_MIN,
                    CONTENT_LOOP_PERIOD_MAX,
                );
            }
            RollbackOutcome::Fallback(reason) => {
                tracing::warn!(
                    content_tokens = a.content_tokens,
                    output_len = a.output_tokens.len(),
                    matched_period = period,
                    matched_repeats = repeats,
                    ?reason,
                    "Content-loop watchdog fired (period-{}…{} repeat); ending response early (rollback declined). \
                     Tune via --content-loop-min-repeats / METRALE_CONTENT_LOOP_MIN_REPEATS, per-request \
                     repetition_detection, or disarm via --content-loop-watchdog off / \
                     METRALE_CONTENT_LOOP_WATCHDOG=0",
                    CONTENT_LOOP_PERIOD_MIN,
                    CONTENT_LOOP_PERIOD_MAX,
                );
                // 2026-09-25: A server cut must name its guard, or
                // `derive_finish_reason` sees budget left and wires "stop".
                a.guard_stop = Some(GUARD_STOP_CONTENT_LOOP);
                a.finished = true;
            }
        }
    }

    // 2026-09-25: Inter-tool prose budget: on a tool request (`tool_request`, which
    // stays set if the grammar disengages), count content tokens outside tool
    // bodies since the last tool call; past `max_inter_tool_prose`, roll back
    // or end the response.
    if !sched.levers.disable_watchdogs && !a.inside_tool_body && a.tool_request {
        a.prose_tokens_since_last_tool = a.prose_tokens_since_last_tool.saturating_add(1);
        let max_prose = sched.watchdog.max_inter_tool_prose;
        if a.prose_tokens_since_last_tool > max_prose {
            // 2026-09-25: Roll back to the last boundary (the rollback also rewinds the
            // grammar matcher) and re-steer; if declined, end the response.
            match rollback_to_boundary(a, CONTENT_LOOP_PERIOD_MAX, model, sched) {
                RollbackOutcome::RolledBack { dropped } => {
                    tracing::warn!(
                        max = max_prose,
                        dropped,
                        rollback = a.rollback_count,
                        "Inter-tool prose budget exhausted; rolled back to boundary, re-steering"
                    );
                }
                RollbackOutcome::Fallback(reason) => {
                    tracing::warn!(
                        prose_tokens = a.prose_tokens_since_last_tool,
                        max = max_prose,
                        ?reason,
                        "Inter-tool prose budget exhausted, ending response (rollback declined); \
                         raise via --max-inter-tool-prose / METRALE_MAX_INTER_TOOL_PROSE / \
                         MODEL.toml [behavior].max_inter_tool_prose (0 disables)"
                    );
                    a.guard_stop = Some(GUARD_STOP_INTER_TOOL_PROSE);
                    a.finished = true;
                }
            }
        }
    }
}
