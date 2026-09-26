// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Handlers for the tool-call variants of the streaming detector's
//! `DetectorOutput` (`ToolCall`, `ToolCallStart`, `ToolCallDelta`,
//! `ToolCallArgsFragment`, `ToolCallEnd`), called from `handle_token` and
//! `handle_done`.
//!
//! Owner: server chat streaming.
//! Invariants: none beyond the types.

use crate::ir::StreamDelta;
use crate::tool_parser;

use super::super::stream_guards::{bump_f12_tool_call_count, flush_content_sanitizer};
use super::ctx::StreamCtx;
use super::state::{PendingRetry, StreamState};

type DeltaVec = Vec<StreamDelta>;

/// 2026-09-26: Push `delta` to `deltas`, or, when `ctx.tool_retry_enabled`,
/// to `state.buffered_tool_chunks[idx]`. The stream context is built with
/// `tool_retry_enabled: false` (`chat_stream/mod.rs`), so every delta goes
/// to `deltas`.
fn emit_or_buffer_tool_delta(
    state: &mut StreamState,
    ctx: &StreamCtx,
    idx: usize,
    delta: StreamDelta,
    deltas: &mut DeltaVec,
) {
    if ctx.tool_retry_enabled {
        state
            .buffered_tool_chunks
            .entry(idx)
            .or_default()
            .push(delta);
    } else {
        deltas.push(delta);
    }
}

/// 2026-09-26: Move the deltas buffered for tool call `idx` into `deltas`;
/// no-op when none are buffered.
fn flush_buffered_tool_chunks(state: &mut StreamState, idx: usize, deltas: &mut DeltaVec) {
    if let Some(chunks) = state.buffered_tool_chunks.remove(&idx) {
        deltas.extend(chunks);
    }
}

/// 2026-09-26: Discard the deltas buffered for tool call `idx`.
fn drop_buffered_tool_chunks(state: &mut StreamState, idx: usize) {
    state.buffered_tool_chunks.remove(&idx);
}

/// 2026-09-26: Handles `DetectorOutput::ToolCall(tc, idx)`, a complete tool call.
pub(super) fn handle_complete_tool_call(
    state: &mut StreamState,
    ctx: &StreamCtx,
    tc: &mut tool_parser::ToolCall,
    tc_idx: usize,
    deltas: &mut DeltaVec,
) {
    let pre_tool_tail = flush_content_sanitizer(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &ctx.leak_markers,
    );
    if !pre_tool_tail.is_empty() {
        deltas.push(StreamDelta::Content {
            text: pre_tool_tail,
            token_ids: state.take_ids_if(ctx.req_return_token_ids),
        });
    }
    tool_parser::backfill_required_params(std::slice::from_mut(tc), &ctx.tool_defs_for_backfill);
    if ctx.wants_typed_arguments {
        tool_parser::coerce_all(std::slice::from_mut(tc), &ctx.tool_defs_for_backfill);
    }
    if let Some(ref cwd) = ctx.cwd_for_normalize {
        tool_parser::normalize_paths(std::slice::from_mut(tc), cwd);
    }
    // 2026-09-26: `MissingParam` and `EmptyRequired` are soft: the call is still
    // emitted. Any other issue is hard: the call is replaced by a content chunk.
    let validation = tool_parser::assess_tool_call(tc, &ctx.tool_defs_for_backfill).map_err(|i| {
        (
            matches!(
                i,
                tool_parser::ToolCallIssue::MissingParam(_)
                    | tool_parser::ToolCallIssue::EmptyRequired(_)
            ),
            i.into_message(),
        )
    });
    let is_soft = validation.as_ref().err().is_some_and(|(soft, _)| *soft);
    if let Err((_, e)) = &validation
        && !is_soft
    {
        tracing::warn!(
            tool = %tc.function.name,
            "tool call validation error (hard): {e}; replacing with content and ending"
        );
        let msg = format!("[metrale] Tool call rejected: {e}");
        deltas.push(StreamDelta::Content {
            text: msg,
            token_ids: state.take_ids_if(ctx.req_return_token_ids),
        });
        state.stop_string_triggered = true;
    } else if let Err((_, e)) = &validation {
        // 2026-09-26: Soft issue: emit the call anyway and leave the error to
        // the client's own per-tool schema check.
        tracing::warn!(
            tool = %tc.function.name,
            "tool call validation error (soft): {e}; passing through to opencode"
        );
        bump_f12_tool_call_count(
            &mut state.tool_calls_emitted_count,
            ctx.max_tool_calls_per_response,
            &mut state.stop_string_triggered,
        );
        let preview: String = tc.function.arguments.chars().take(120).collect();
        let s = if tc.function.arguments.len() > preview.len() {
            "…"
        } else {
            ""
        };
        tracing::info!("Tool call: {}({preview}{s})", tc.function.name);
        crate::metrics::TOOL_CALLS_TOTAL.inc();
        deltas.push(StreamDelta::ToolCallStart {
            index: tc_idx,
            id: tc.id.clone(),
            name: tc.function.name.clone(),
        });
        deltas.push(StreamDelta::ToolCallArgs {
            index: tc_idx,
            fragment: tc.function.arguments.clone(),
            token_ids: Vec::new(),
        });
        // 2026-09-26: Once per response (`corrective_hint_sent`), when a
        // required parameter is empty or the arguments contain a garbled
        // `</parameter<parameter=` boundary, append a content chunk that names
        // the problem, so the model does not retry the call unchanged.
        if !state.corrective_hint_sent {
            let empties = tool_parser::find_empty_required_params(tc, &ctx.tool_defs_for_backfill);
            let garbled = tc.function.arguments.contains("</parameter<parameter=");
            if !empties.is_empty() || garbled {
                state.corrective_hint_sent = true;
                let msg = format!(
                    "
[metrale] The {} call above has EMPTY required parameter(s): {}.                      It will fail. Re-issue the call with real values for every                      required parameter (do not repeat it unchanged).",
                    tc.function.name,
                    if empties.is_empty() {
                        "<garbled parameter boundary>".to_string()
                    } else {
                        empties.join(", ")
                    },
                );
                deltas.push(StreamDelta::Content {
                    text: msg,
                    token_ids: state.take_ids_if(ctx.req_return_token_ids),
                });
            }
        }
    } else if state
        .tool_arg_dedup
        .check(&tc.function.name, &tc.function.arguments)
    {
        tracing::warn!(
            tool = %tc.function.name,
            "tool-arg dedup tripped: refusing redundant tool_call and ending response"
        );
        state.stop_string_triggered = true;
        state.tool_loop_capped = true;
        state
            .cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);
    } else {
        // 2026-09-26: The same-name run cap, also applied in
        // `handle_tool_call_end`, catches repeated calls whose arguments
        // differ, which `tool_arg_dedup` (keyed on name and arguments) misses.
        let run_len = advance_name_run(&mut state.name_run, &tc.function.name);
        if run_len >= MAX_CONSEC_SAME_NAME_CALLS {
            tracing::warn!(
                tool = %tc.function.name,
                run = run_len,
                "Bug-2 name-run cap tripped (complete-call path): {run_len} successive `{}` tool calls; ending response",
                tc.function.name
            );
            state.stop_string_triggered = true;
            state.tool_loop_capped = true;
        }
        bump_f12_tool_call_count(
            &mut state.tool_calls_emitted_count,
            ctx.max_tool_calls_per_response,
            &mut state.stop_string_triggered,
        );
        let preview: String = tc.function.arguments.chars().take(120).collect();
        let s = if tc.function.arguments.len() > preview.len() {
            "…"
        } else {
            ""
        };
        tracing::info!("Tool call: {}({preview}{s})", tc.function.name);
        crate::metrics::TOOL_CALLS_TOTAL.inc();
        deltas.push(StreamDelta::ToolCallStart {
            index: tc_idx,
            id: tc.id.clone(),
            name: tc.function.name.clone(),
        });
        deltas.push(StreamDelta::ToolCallArgs {
            index: tc_idx,
            fragment: tc.function.arguments.clone(),
            token_ids: Vec::new(),
        });
    }
}

/// 2026-09-26: Handles `DetectorOutput::ToolCallStart`: flushes the content
/// sanitiser, opens the argument accumulator for `idx`, counts the call
/// against `max_tool_calls_per_response`, and emits the call header.
pub(super) fn handle_tool_call_start(
    state: &mut StreamState,
    ctx: &StreamCtx,
    tc_id: String,
    name: String,
    idx: usize,
    deltas: &mut DeltaVec,
) {
    let pre_tool_tail = flush_content_sanitizer(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &ctx.leak_markers,
    );
    if !pre_tool_tail.is_empty() {
        deltas.push(StreamDelta::Content {
            text: pre_tool_tail,
            token_ids: state.take_ids_if(ctx.req_return_token_ids),
        });
    }
    state
        .streaming_tool_args
        .insert(idx, (name.clone(), String::new()));
    bump_f12_tool_call_count(
        &mut state.tool_calls_emitted_count,
        ctx.max_tool_calls_per_response,
        &mut state.stop_string_triggered,
    );
    let start = StreamDelta::ToolCallStart {
        index: idx,
        id: tc_id,
        name,
    };
    emit_or_buffer_tool_delta(state, ctx, idx, start, deltas);
}

/// 2026-09-26: Handles `DetectorOutput::ToolCallDelta`, which carries a
/// call's full canonical arguments once, at its close. Runs the same
/// `backfill_required_params`, `coerce_all`, `normalize_paths` and
/// `assess_tool_call` chain as [`handle_complete_tool_call`]. When no
/// `ToolCallStart` opened `idx`, `args` is emitted unchanged.
pub(super) fn handle_tool_call_delta(
    state: &mut StreamState,
    ctx: &StreamCtx,
    args: String,
    idx: usize,
    deltas: &mut DeltaVec,
) {
    let mut emit_args = args.clone();
    if let Some(entry) = state.streaming_tool_args.get_mut(&idx) {
        let name = entry.0.clone();
        let mut tc = tool_parser::ToolCall {
            id: format!("call_{:016x}", idx),
            call_type: "function".into(),
            function: tool_parser::FunctionCall {
                name: name.clone(),
                arguments: args.clone(),
            },
        };
        tool_parser::backfill_required_params(
            std::slice::from_mut(&mut tc),
            &ctx.tool_defs_for_backfill,
        );
        if ctx.wants_typed_arguments {
            tool_parser::coerce_all(std::slice::from_mut(&mut tc), &ctx.tool_defs_for_backfill);
        }
        if let Some(ref cwd) = ctx.cwd_for_normalize {
            tool_parser::normalize_paths(std::slice::from_mut(&mut tc), cwd);
        }
        if let Err(issue) = tool_parser::assess_tool_call(&tc, &ctx.tool_defs_for_backfill) {
            // 2026-09-26: `handle_tool_call_start` has already emitted the
            // header for `idx`. A soft issue (`MissingParam`, `EmptyRequired`)
            // still emits the arguments, leaving the error to the client's
            // per-tool schema check. A hard issue ends the response with a
            // content chunk.
            let is_soft = matches!(
                issue,
                tool_parser::ToolCallIssue::MissingParam(_)
                    | tool_parser::ToolCallIssue::EmptyRequired(_)
            );
            let e = issue.into_message();
            if is_soft {
                tracing::warn!(
                    tool = %name,
                    "tool call validation error (stream Δ, soft): {e}; passing through so opencode can surface its own per-tool schema error"
                );
                emit_args = tc.function.arguments.clone();
                entry.1.push_str(&emit_args);
            } else if ctx.tool_retry_enabled {
                // 2026-09-26: Not reached: `tool_retry_enabled` is always
                // false (`chat_stream/mod.rs`), and nothing reads `pending_retry`.
                tracing::warn!(
                    tool = %name,
                    "tool call validation error (stream Δ, hard, retry pending): {e}"
                );
                entry.1.push_str(&args);
                let errors_summary = e.to_string();
                drop_buffered_tool_chunks(state, idx);
                state.pending_retry = Some(PendingRetry {
                    errors_summary,
                    failed_idx: idx,
                });
                state.stop_string_triggered = true;
                state
                    .cancel_flag
                    .store(true, std::sync::atomic::Ordering::Release);
                return;
            } else {
                tracing::warn!(
                    tool = %name,
                    "tool call validation error (stream Δ, hard): {e}; replacing with content and ending"
                );
                let msg = format!("[metrale] Tool call rejected: {e}");
                deltas.push(StreamDelta::Content {
                    text: msg,
                    token_ids: Vec::new(),
                });
                state.stop_string_triggered = true;
                entry.1.push_str(&args);
                return;
            }
        } else {
            emit_args = tc.function.arguments.clone();
            entry.1.push_str(&emit_args);
        }
    } else if !args.is_empty() {
        // 2026-09-26: No `ToolCallStart` opened `idx`: `args` is emitted
        // unchanged below.
    }
    if !emit_args.is_empty() {
        let frag = StreamDelta::ToolCallArgs {
            index: idx,
            fragment: emit_args,
            token_ids: Vec::new(),
        };
        emit_or_buffer_tool_delta(state, ctx, idx, frag, deltas);
        if ctx.tool_retry_enabled {
            flush_buffered_tool_chunks(state, idx, deltas);
        }
    }
}

/// 2026-09-26: Handles `DetectorOutput::ToolCallArgsFragment`, a slice of
/// `function.arguments` that the detector has already coerced (XML) or sliced
/// (JSON). Appends it verbatim to the accumulated arguments and emits it as a
/// `tool_calls[idx].function.arguments` fragment, with no coercion or
/// validation here. If no `ToolCallStart` opened `idx`, the fragment is dropped.
pub(super) fn handle_tool_call_args_fragment(
    state: &mut StreamState,
    _ctx: &StreamCtx,
    fragment: String,
    idx: usize,
    deltas: &mut DeltaVec,
) {
    let Some(entry) = state.streaming_tool_args.get_mut(&idx) else {
        return;
    };
    entry.1.push_str(&fragment);
    deltas.push(StreamDelta::ToolCallArgs {
        index: idx,
        fragment,
        token_ids: Vec::new(),
    });
}

/// 2026-09-26: Advance the same-name run counter for a completed call and
/// return the new run length. A different name restarts the run at 1.
fn advance_name_run(name_run: &mut Option<(String, u32)>, name: &str) -> u32 {
    let run_len = match name_run {
        Some((prev, n)) if prev == name => *n + 1,
        _ => 1,
    };
    *name_run = Some((name.to_string(), run_len));
    run_len
}

/// 2026-09-26: Same-name calls in a row that end the response, whatever their
/// arguments. A parallel fan-out is a run of same-name calls, so the cap
/// equals the scheduler's bound on `<tool_call>` opens after a completed call
/// (`MAX_POST_COMPLETION_TOOL_OPENS` in `scheduler/decode_logits_step/per_token.rs`),
/// below the total `max_tool_calls_per_response` (`METRALE_MAX_TOOL_CALLS_PER_RESPONSE`,
/// default 12, `chat_stream/mod.rs`).
const MAX_CONSEC_SAME_NAME_CALLS: u32 = 8;

/// 2026-09-26: Handles `DetectorOutput::ToolCallEnd`: closes the argument
/// accumulator for `idx`, then ends the response (`stop_string_triggered`,
/// `tool_loop_capped`) when `tool_arg_dedup_within` matches the call or the
/// same-name run reaches [`MAX_CONSEC_SAME_NAME_CALLS`]. Otherwise logs the
/// call and counts it in `TOOL_CALLS_TOTAL`.
pub(super) fn handle_tool_call_end(state: &mut StreamState, _ctx: &StreamCtx, idx: usize) {
    if let Some((name, args_json)) = state.streaming_tool_args.remove(&idx) {
        if state.tool_arg_dedup_within.check(&name, &args_json) {
            tracing::warn!(
                tool = %name,
                "F11 within-response dedup tripped: 2+ identical streaming tool calls; ending response"
            );
            state.stop_string_triggered = true;
            state.tool_loop_capped = true;
        }
        let run_len = advance_name_run(&mut state.name_run, &name);
        if run_len >= MAX_CONSEC_SAME_NAME_CALLS && !state.stop_string_triggered {
            tracing::warn!(
                tool = %name,
                run = run_len,
                "Bug-2 name-run cap tripped: {run_len} successive `{name}` tool calls; ending response (F11 missed because args drift)"
            );
            state.stop_string_triggered = true;
            state.tool_loop_capped = true;
        }
        if !state.stop_string_triggered {
            let preview: String = args_json.chars().take(120).collect();
            let s = if args_json.len() > preview.len() {
                "…"
            } else {
                ""
            };
            tracing::info!("Tool call: {name}({preview}{s})");
            crate::metrics::TOOL_CALLS_TOTAL.inc();
        }
    }
}

#[cfg(test)]
mod name_run_cap_tests {
    //! 2026-09-26: The same-name run cap: a three-call parallel fan-out stays
    //! under it, and a same-name runaway trips it before the total cap of 12.
    use super::{MAX_CONSEC_SAME_NAME_CALLS, advance_name_run};

    #[test]
    fn three_call_parallel_fanout_stays_under_cap() {
        let mut run = None;
        for i in 1..=3u32 {
            let len = advance_name_run(&mut run, "get_weather");
            assert_eq!(len, i);
            assert!(
                len < MAX_CONSEC_SAME_NAME_CALLS,
                "a 3-call same-name parallel fan-out must not trip the doom-loop cap"
            );
        }
    }

    #[test]
    fn runaway_same_name_run_still_trips() {
        let mut run = None;
        let mut tripped_at = None;
        for i in 1..=12u32 {
            if advance_name_run(&mut run, "bash") >= MAX_CONSEC_SAME_NAME_CALLS {
                tripped_at = Some(i);
                break;
            }
        }
        assert_eq!(
            tripped_at,
            Some(MAX_CONSEC_SAME_NAME_CALLS),
            "cap must fire below the F12 total cap (12)"
        );
    }

    #[test]
    fn different_name_resets_run() {
        let mut run = None;
        assert_eq!(advance_name_run(&mut run, "get_weather"), 1);
        assert_eq!(advance_name_run(&mut run, "get_weather"), 2);
        assert_eq!(advance_name_run(&mut run, "get_time"), 1);
        assert_eq!(advance_name_run(&mut run, "get_weather"), 1);
    }
}
