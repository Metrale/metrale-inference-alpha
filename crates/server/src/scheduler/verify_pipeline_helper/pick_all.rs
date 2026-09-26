// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `verify_pick_all_with_pipeline`: picks every verify position
//! of one sequence, through a GPU-argmax fast path when one applies and the
//! host pipeline otherwise.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: pick a token for each of one sequence's verify positions.
///
/// `argmax_ids` holds the GPU argmax of each position and sets K. The fast
/// paths, tried in order, return `argmax_ids` unchanged without copying the
/// logits rows: the masked chat path (`fast_masked`), the grammar path and
/// the grammarless path below. Otherwise the K rows are copied to host and
/// `pick_positions_from_host` picks them; it returns fewer than K picks when
/// a speculative grammar advance is refused. If that copy fails,
/// `argmax_ids` is returned.
///
/// `row_base` is the sequence's first row in the shared logits buffer: 0 on
/// the single-sequence paths, the prefix sum of the earlier sequences' row
/// counts on `step_verify_k4_batched`.
pub fn verify_pick_all_with_pipeline(
    model: &dyn Model,
    argmax_ids: &[u32],
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    row_base: usize,
) -> Vec<u32> {
    use crate::scheduler::mtp_timing::Phase;
    let k = argmax_ids.len();
    if k == 0 {
        return Vec::new();
    }

    // 2026-09-25: the masked chat path; its gates are in `fast_masked.rs`.
    if let Some(picks) = fast_masked::try_chat_fast_path(model, argmax_ids, a, ctx, row_base) {
        return picks;
    }

    // 2026-09-25: grammar fast path. Eligible when `fast_greedy_grammar` is on
    // (`METRALE_DISABLE_FAST_GREEDY=1` turns it off), a grammar is active,
    // the sequence is outside thinking, decoding is greedy (temperature 0 or
    // `force_temp_zero`), and the penalties are not `Blocked`. Each position's
    // GPU argmax must be grammar-allowed (a grammar-allowed global maximum is
    // the maximum of the allowed set) and, for `ReduceOnly` penalties,
    // penalty-immune (`fast_greedy`). The matcher is advanced speculatively
    // between positions and rolled back afterwards, as the host path does.
    //
    // The other pipeline stages are not consulted: `MinTokensEosMask`,
    // `PostCloseThinkMask`, `PinToToolCallStart` and the tool-call bias can
    // still apply outside thinking, so this path can emit a token the host
    // path would have masked. Temperature above 0 always takes the host path,
    // where the sampling branch runs.
    let fast_penalty_gate = if ctx.sampling.fast_greedy_grammar
        && a.grammar_state.is_some()
        && !a.inside_thinking
        && (a.temperature == 0.0 || ctx.sampling.force_temp_zero)
    {
        crate::scheduler::fast_greedy::classify_penalties(
            &crate::scheduler::sample_step::penalty_params_for(
                a,
                crate::scheduler::sample_step::PositionKind::Verify,
                0.0,
                None,
                Vec::new(),
                ctx.watchdog.min_reasoning_floor,
            ),
        )
    } else {
        crate::scheduler::fast_greedy::PenaltyGate::Blocked
    };
    if fast_penalty_gate != crate::scheduler::fast_greedy::PenaltyGate::Blocked {
        let t_fast = ctx.clock.now();
        let vocab = model.vocab_size();
        let logits_base = model.logits_buffer_ptr();
        // 2026-09-25: the same history scope the host path penalises
        // (`penalty_history_scope`), copied before `a.grammar_state` is
        // borrowed mutably.
        let scoped_history: Vec<u32> =
            if fast_penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly {
                crate::scheduler::sample_step::penalty_history_scope(
                    &a.output_tokens,
                    ctx.tool_call_end_token,
                )
                .to_vec()
            } else {
                Vec::new()
            };
        let before = a.grammar_state.as_ref().map(|gs| gs.num_history_steps());
        let mut fast: Vec<u32> = Vec::with_capacity(k);
        let mut all_allowed = true;
        // 2026-09-25: the block ends `gs`'s mutable borrow before the
        // rollback below borrows `a.grammar_state` again.
        {
            let Some(gs) = a.grammar_state.as_mut() else {
                unreachable!("grammar_state present (gated by is_some above)")
            };
            for (i, &tok) in argmax_ids.iter().enumerate() {
                // 2026-09-25: `ReduceOnly`: the argmax must be absent from the
                // scoped history and have a raw logit above 0.
                if fast_penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly
                    && !crate::scheduler::fast_greedy::argmax_immune(tok, &scoped_history, || {
                        crate::scheduler::fast_greedy::logit_is_positive(
                            model,
                            logits_base,
                            row_base + i,
                            vocab,
                            tok,
                        )
                    })
                {
                    all_allowed = false;
                    break;
                }
                let allowed = if gs.is_terminated() {
                    true
                } else {
                    gs.fill_bitmask();
                    gs.is_token_allowed(tok)
                };
                if !allowed {
                    all_allowed = false;
                    break;
                }
                fast.push(tok);
                // 2026-09-25: advance speculatively so position i+1 is
                // checked against the matcher state after pick i.
                if i + 1 < k && !gs.is_terminated() {
                    let _ = gs.accept_token(tok);
                }
            }
        }
        // 2026-09-25: roll back by the history delta, not by the number of
        // `accept_token` calls: `GrammarState::accept_token` returns true
        // for stop tokens and in the terminated state without adding a
        // history step.
        if let (Some(b), Some(gs)) = (before, a.grammar_state.as_mut()) {
            let adv = gs.num_history_steps().saturating_sub(b);
            if adv > 0 {
                gs.rollback(adv);
            }
        }
        ctx.tel.mark(Phase::FastGreedy, t_fast);
        if all_allowed && fast.len() == k {
            return fast;
        }
    }

    // 2026-09-25: grammarless fast path. Eligible when `fast_greedy_chat` is
    // on (`METRALE_NO_FAST_GREEDY_CHAT=1` turns it off), no grammar is
    // active, the sequence is outside thinking, decoding is greedy, and the
    // penalties are `Neutral`, or `ReduceOnly` with every argmax
    // penalty-immune. No mask stage is consulted (see the grammar path
    // above). The GPU argmax can also break near-ties differently from the
    // host scan, so the picks are not always the host path's.
    let chat_fast_gate = if ctx.sampling.fast_greedy_chat
        && a.grammar_state.is_none()
        && !a.inside_thinking
        && (a.temperature == 0.0 || ctx.sampling.force_temp_zero)
    {
        crate::scheduler::fast_greedy::classify_penalties(
            &crate::scheduler::sample_step::penalty_params_for(
                a,
                crate::scheduler::sample_step::PositionKind::Verify,
                0.0,
                None,
                Vec::new(),
                ctx.watchdog.min_reasoning_floor,
            ),
        )
    } else {
        crate::scheduler::fast_greedy::PenaltyGate::Blocked
    };
    if chat_fast_gate != crate::scheduler::fast_greedy::PenaltyGate::Blocked {
        let t_fast = ctx.clock.now();
        let vocab = model.vocab_size();
        let logits_base = model.logits_buffer_ptr();
        let scoped_history: Vec<u32> =
            if chat_fast_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly {
                crate::scheduler::sample_step::penalty_history_scope(
                    &a.output_tokens,
                    ctx.tool_call_end_token,
                )
                .to_vec()
            } else {
                Vec::new()
            };
        let all_immune = argmax_ids.iter().enumerate().all(|(i, &tok)| {
            chat_fast_gate == crate::scheduler::fast_greedy::PenaltyGate::Neutral
                || crate::scheduler::fast_greedy::argmax_immune(tok, &scoped_history, || {
                    crate::scheduler::fast_greedy::logit_is_positive(
                        model,
                        logits_base,
                        row_base + i,
                        vocab,
                        tok,
                    )
                })
        });
        ctx.tel.mark(Phase::FastGreedy, t_fast);
        if all_immune {
            return argmax_ids.to_vec();
        }
    }

    let vocab = model.vocab_size();
    // 2026-09-25: the rows are read as BF16; this path does not check
    // `logits_ptr_is_fp32`.
    let elem_bytes = 2usize;
    let total = k * vocab * elem_bytes;
    let t_d2h = ctx.clock.now();
    let mut buf = vec![0u8; total];
    if model
        .copy_logits_to_host(
            model
                .logits_buffer_ptr()
                .offset(row_base * vocab * elem_bytes),
            &mut buf,
        )
        .is_err()
    {
        return argmax_ids.to_vec();
    }
    ctx.tel.mark(Phase::D2h, t_d2h);

    pick_positions::pick_positions_from_host(&buf, vocab, elem_bytes, k, a, ctx)
}
