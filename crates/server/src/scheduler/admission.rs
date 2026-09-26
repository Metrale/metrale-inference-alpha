// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: KV admission: a request is admitted only if the pool can hold its
//! prompt plus its decode depth, on top of what is already in flight.
//!
//! Policy: a request reserves `prompt + min(max_tokens, watermark)` tokens of
//! blocks, clamped to `max_seq_len`, against the total pool minus the same
//! reservation for every active, prefilling, spilled and requeued sequence.
//! Requests that do not fit go back to the front of the pending queue and are
//! gated again on the next tick. A model that reports no KV pool (0 total
//! blocks) admits everything. Before the KV check, requests whose adapter
//! differs from the in-flight cohort are held back the same way.
//!
//! Watermark: `METRALE_KV_ADMIT_WATERMARK` (tokens). Unset or unparsable, it is
//! `max_seq_len` (unbounded when `max_seq_len` is 0), so a request reserves its
//! whole `max_tokens`. A lower value logs a warning; `0` reserves the prompt
//! only.
//!
//! Owner: scheduler.
//! Invariants:
//! - Among the requests that pass the adapter filter, KV admission stops at
//!   the first that does not fit; no later request is admitted past it.
//! - Held-back requests return to the front of the pending queue, each group
//!   (adapter-deferred, KV overflow) in its original order.

use super::*;

/// 2026-09-25: Resolve the admission watermark once at scheduler start.
pub(super) fn resolve_admit_watermark(max_seq_len: usize) -> usize {
    let default = if max_seq_len > 0 {
        max_seq_len
    } else {
        usize::MAX
    };
    match std::env::var("METRALE_KV_ADMIT_WATERMARK") {
        Err(_) => default,
        Ok(v) => match v.parse::<usize>() {
            Ok(w) => {
                if w < default {
                    tracing::warn!(
                        "METRALE_KV_ADMIT_WATERMARK={w} < the honest bound ({default}): \
                         admission may OVERCOMMIT the KV pool; sequences past the \
                         watermark depth will hit decode-time preemption (resume, \
                         not kill — but pure overhead). 0 = legacy prompt-only \
                         reservation."
                    );
                }
                w
            }
            Err(_) => {
                tracing::warn!(
                    "METRALE_KV_ADMIT_WATERMARK={v:?} is not an integer; using default {default}"
                );
                default
            }
        },
    }
}

/// 2026-09-25: Blocks for `tokens` tokens: `tokens / block_size + 1` (a `block_size`
/// of 0 counts as 1). Every reservation in this module goes through it.
pub(super) fn blocks_for_tokens(tokens: usize, block_size: usize) -> usize {
    tokens / block_size.max(1) + 1
}

/// 2026-09-25: One in-flight sequence's KV demand, in tokens.
pub(super) struct SeqDemand {
    /// 2026-09-25: Tokens whose KV exists or must exist at resume (prompt + processed).
    pub current_tokens: usize,
    /// 2026-09-25: Generation still owed (`remaining` / `max_tokens`).
    pub budget_tokens: usize,
}

/// 2026-09-25: Blocks to reserve for one sequence: current + min(budget, watermark)
/// tokens. When `max_seq_len` is nonzero the depth is clamped to it, or to
/// `current_tokens` if that is already larger.
pub(super) fn seq_commitment_blocks(
    d: &SeqDemand,
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> usize {
    let mut depth = d
        .current_tokens
        .saturating_add(d.budget_tokens.min(watermark));
    if max_seq_len > 0 {
        depth = depth.min(max_seq_len.max(d.current_tokens));
    }
    blocks_for_tokens(depth, block_size)
}

/// 2026-09-25: Total reserved blocks for everything already in flight.
pub(super) fn committed_blocks(
    demands: &[SeqDemand],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> usize {
    demands
        .iter()
        .map(|d| seq_commitment_blocks(d, watermark, max_seq_len, block_size))
        .sum()
}

/// 2026-09-25: How many of `reqs` (`(prompt_len, max_tokens)`, in admission order) fit.
///
/// Returns `(admit_count, forced_oversize)`. Admission stops at the first
/// request that does not fit, so a small request never overtakes a large one
/// that arrived first. Liveness: when `committed` is 0 and the first request
/// alone does not fit, it is admitted anyway (`forced_oversize = true`)
/// rather than queued forever.
pub(super) fn admit_count(
    total_blocks: usize,
    committed: usize,
    reqs: &[(usize, usize)],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> (usize, bool) {
    let mut used = committed;
    let mut n = 0usize;
    for &(prompt, max_tokens) in reqs {
        let need = seq_commitment_blocks(
            &SeqDemand {
                current_tokens: prompt,
                budget_tokens: max_tokens,
            },
            watermark,
            max_seq_len,
            block_size,
        );
        if used.saturating_add(need) <= total_blocks {
            used += need;
            n += 1;
        } else if n == 0 && committed == 0 {
            return (1, true);
        } else {
            break;
        }
    }
    (n, false)
}

/// 2026-09-25: Defer any new request whose adapter differs from the in-flight cohort's.
///
/// Returns the requests that may join the current batch; the rest are pushed
/// back to the front of the pending queue, in their relative order. The
/// cohort is the adapter of the first active (else prefilling) sequence; with
/// nothing in flight it is the first new request's. A model without LoRA
/// support maps every slot to the base id, so every request passes.
fn filter_adapter_cohort(
    model: &dyn Model,
    pending: &mut PendingQueue,
    new_reqs: Vec<InferenceRequest>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
) -> Vec<InferenceRequest> {
    let cohort = active
        .first()
        .map(|a| a.seq.adapter_slot)
        .or_else(|| prefilling.first().map(|p| p.seq.adapter_slot));
    // 2026-09-25: Nothing in flight: the first new request defines the cohort, so two
    // requests for different adapters arriving together are not batched.
    let cohort_slot = match cohort {
        Some(s) => s,
        None => match new_reqs.first() {
            Some(r) => r.adapter_slot(),
            None => return new_reqs,
        },
    };
    let cohort_id = model.adapter_id_for(cohort_slot);
    let (admitted, deferred): (Vec<_>, Vec<_>) = new_reqs
        .into_iter()
        .partition(|r| model.adapter_id_for(r.adapter_slot()) == cohort_id);
    if !deferred.is_empty() {
        tracing::debug!(
            "adapter cohort: holding {} request(s) for a different adapter until \
             the current batch drains (v0 is single-active)",
            deferred.len()
        );
        for (i, req) in deferred.into_iter().enumerate() {
            pending.requests.insert(i, req);
        }
    }
    admitted
}

/// 2026-09-25: Split this tick's drained requests into the admitted ones (returned)
/// and the held-back ones (pushed back to the front of the pending queue, in
/// arrival order): first by adapter cohort, then by KV fit.
#[allow(clippy::too_many_arguments)]
pub(super) fn gate_admissions(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    model: &dyn Model,
    pending: &mut PendingQueue,
    new_reqs: Vec<InferenceRequest>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    swapped: &[SwappedSeq],
    preempted: &[PreemptedSeq],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> Vec<InferenceRequest> {
    if new_reqs.is_empty() {
        return new_reqs;
    }
    // 2026-09-25: Adapter cohort: a batch may only hold sequences sharing one adapter.
    // A decode batch with a row routed to a non-active adapter fails as a whole
    // (`decode_batch_compute_main` calls `ensure_decode_route_servable`), so a
    // mismatched request waits at the head of the pending queue instead.
    //
    // Identity comes from `adapter_id_for`, which resolves the `-1` "defer to
    // active" slot; comparing raw slot indices would treat `-1` and the active
    // slot as different cohorts.
    let new_reqs = filter_adapter_cohort(model, pending, new_reqs, active, prefilling);
    if new_reqs.is_empty() {
        return new_reqs;
    }
    let total_blocks = model.num_total_blocks();
    if total_blocks == 0 {
        // 2026-09-25: No KV pool reported: nothing to reserve against.
        return new_reqs;
    }
    let mut demands: Vec<SeqDemand> =
        Vec::with_capacity(active.len() + prefilling.len() + swapped.len() + preempted.len());
    demands.extend(active.iter().map(|a| SeqDemand {
        // 2026-09-25: +1: the pending `last_token` decode input not yet in seq_len.
        current_tokens: a.seq.seq_len + 1,
        budget_tokens: a.remaining,
    }));
    demands.extend(prefilling.iter().map(|p| SeqDemand {
        current_tokens: p.prompt_tokens.len(),
        budget_tokens: p.max_tokens,
    }));
    demands.extend(swapped.iter().map(|s| SeqDemand {
        current_tokens: s.seq_len + 1,
        budget_tokens: s.remaining,
    }));
    demands.extend(preempted.iter().map(|p| SeqDemand {
        current_tokens: p.tokens.len() + 1,
        budget_tokens: p.a.remaining,
    }));
    let committed = committed_blocks(&demands, watermark, max_seq_len, block_size);
    let infos: Vec<(usize, usize)> = new_reqs
        .iter()
        .map(|r| (r.prompt_len(), r.max_tokens()))
        .collect();
    let (admit, forced) = admit_count(
        total_blocks,
        committed,
        &infos,
        watermark,
        max_seq_len,
        block_size,
    );
    if forced {
        tracing::warn!(
            "admitting a request whose reservation exceeds the whole KV pool \
             ({} prompt + min({}, watermark {}) tokens vs {} blocks); the block \
             allocator will back-pressure at runtime",
            infos[0].0,
            infos[0].1,
            watermark,
            total_blocks,
        );
    }
    // 2026-09-25: Overflow requests are re-gated every tick; log only when the
    // queued count changes.
    if admit >= new_reqs.len() {
        sched.admit_last_queued.set(0);
        return new_reqs;
    }
    let mut admitted = new_reqs;
    let overflow = admitted.split_off(admit);
    if sched.admit_last_queued.replace(overflow.len()) != overflow.len() {
        tracing::info!(
            "KV admission: {} of {} request(s) fit ({} blocks committed of {}); \
             {} queued until capacity frees",
            admitted.len(),
            admitted.len() + overflow.len(),
            committed,
            total_blocks,
            overflow.len(),
        );
    }
    for (i, req) in overflow.into_iter().enumerate() {
        pending.requests.insert(i, req);
    }
    admitted
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2026-09-25: Fixture: a 102k-token pool at block_size 16, and requests of 202
    // prompt + 1024 max tokens (128 of them demand about 157k tokens).
    const BS: usize = 16;
    const POOL_BLOCKS: usize = 102_000 / BS;
    const PROMPT: usize = 202;
    const MAX_TOK: usize = 1024;
    const MAX_SEQ_LEN: usize = 8192;

    fn req_blocks() -> usize {
        blocks_for_tokens(PROMPT + MAX_TOK, BS)
    }

    #[test]
    fn block_math_matches_legacy_formula() {
        assert_eq!(blocks_for_tokens(0, 16), 1);
        assert_eq!(blocks_for_tokens(15, 16), 1);
        assert_eq!(blocks_for_tokens(16, 16), 2);
        assert_eq!(blocks_for_tokens(1226, 16), 77);
    }

    #[test]
    fn watermark_caps_the_decode_reservation() {
        let d = SeqDemand {
            current_tokens: 200,
            budget_tokens: 4096,
        };
        assert_eq!(
            seq_commitment_blocks(&d, 512, MAX_SEQ_LEN, BS),
            blocks_for_tokens(200 + 512, BS)
        );
        assert_eq!(
            seq_commitment_blocks(&d, usize::MAX, MAX_SEQ_LEN, BS),
            blocks_for_tokens(200 + 4096, BS)
        );
        assert_eq!(
            seq_commitment_blocks(&d, 0, MAX_SEQ_LEN, BS),
            blocks_for_tokens(200, BS)
        );
        // 2026-09-25: max_seq_len clamps the depth.
        let long = SeqDemand {
            current_tokens: 8000,
            budget_tokens: 4096,
        };
        assert_eq!(
            seq_commitment_blocks(&long, usize::MAX, MAX_SEQ_LEN, BS),
            blocks_for_tokens(MAX_SEQ_LEN, BS)
        );
    }

    #[test]
    fn fits_everything_admission_unchanged() {
        // 2026-09-25: 64 requests need 64 x 77 = 4928 of the 6375 blocks: all fit.
        let reqs = vec![(PROMPT, MAX_TOK); 64];
        let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, MAX_SEQ_LEN, MAX_SEQ_LEN, BS);
        assert_eq!(n, 64);
        assert!(!forced);
    }

    #[test]
    fn c128_overflow_queues_instead_of_admit_then_shoot() {
        // 2026-09-25: 128 requests exceed the pool: the gate admits what fits in full
        // and leaves the rest queued.
        let reqs = vec![(PROMPT, MAX_TOK); 128];
        let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, MAX_SEQ_LEN, MAX_SEQ_LEN, BS);
        assert_eq!(n, POOL_BLOCKS / req_blocks());
        assert!(n < 128);
        assert!(!forced);
        assert!(n * req_blocks() <= POOL_BLOCKS);
    }

    #[test]
    fn in_flight_commitments_reduce_capacity() {
        // 2026-09-25: 40 active sequences mid-decode still owe their remaining budget.
        let demands: Vec<SeqDemand> = (0..40)
            .map(|_| SeqDemand {
                current_tokens: 600,
                budget_tokens: 700,
            })
            .collect();
        let committed = committed_blocks(&demands, MAX_SEQ_LEN, MAX_SEQ_LEN, BS);
        assert_eq!(committed, 40 * blocks_for_tokens(1300, BS));
        let reqs = vec![(PROMPT, MAX_TOK); 128];
        let (n, _) = admit_count(POOL_BLOCKS, committed, &reqs, MAX_SEQ_LEN, MAX_SEQ_LEN, BS);
        assert_eq!(n, (POOL_BLOCKS - committed) / req_blocks());
    }

    #[test]
    fn admission_stops_at_first_misfit_no_head_of_line_bypass() {
        // 2026-09-25: A huge request mid-queue blocks later small ones from jumping it.
        let reqs = vec![
            (PROMPT, MAX_TOK),
            // 2026-09-25: 6315 blocks: does not fit beside the first request's 77.
            (100_000, MAX_TOK),
            (PROMPT, MAX_TOK),
        ];
        let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, usize::MAX, 0, BS);
        assert_eq!(n, 1);
        assert!(!forced);
    }

    #[test]
    fn liveness_oversized_lone_request_still_admits() {
        // 2026-09-25: Nothing in flight and a request bigger than the whole pool: admit
        // it rather than queue it forever.
        let reqs = vec![(200_000, MAX_TOK)];
        let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, usize::MAX, 0, BS);
        assert_eq!(n, 1);
        assert!(forced);
        // 2026-09-25: But with anything in flight it waits its turn.
        let (n, forced) = admit_count(POOL_BLOCKS, 10, &reqs, usize::MAX, 0, BS);
        assert_eq!(n, 0);
        assert!(!forced);
    }

    #[test]
    fn watermark_zero_reserves_prompt_only_legacy() {
        // 2026-09-25: Watermark 0 reserves the prompt only: 128 x 13 = 1664 blocks fit.
        let reqs = vec![(PROMPT, MAX_TOK); 128];
        let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, 0, MAX_SEQ_LEN, BS);
        assert_eq!(n, 128);
        assert!(!forced);
    }
}
