// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Scoring for the contamination benchmark: classify every prompt at
//! every rung and in the post-check against its own solo reference. The module
//! reads no handle, network or clock; `score_tests.rs` tests it with
//! in-memory outcomes.
//!
//! Why more than "one solo reference, then diff each rung against it":
//!
//! 1. A single reference cannot tell a concurrency defect from a prompt that
//!    is not reproducible alone. The reference runs twice; a prompt whose two
//!    solo runs disagree is `AloneUnstable`, is excluded from the per-rung
//!    comparison, and still fails the verdict under its own name.
//! 2. State that survives the concurrent episode shows only in later solo
//!    work, so a solo post-check leg after the rungs classifies a divergence
//!    there as `Persistent`.
//! 3. The driver primes every prompt once before any measured leg, so the
//!    measured legs start cache-warm unless a prime request failed (the driver
//!    then logs a warning). `cached_prompt_tokens` is not part of the
//!    comparison.
//!
//! Owner: bench (contamination).
//! Invariants:
//! - Each cell inserted into `Score::cells` increments exactly one of the
//!   class counters.
//! - `verdict` passes only when at least one cell was compared and every
//!   compared cell is `Identical`.

use std::collections::BTreeMap;

use super::transcript::{RequestOutcome, Transcript};

/// 2026-09-26: What happened to one prompt in one leg: a rung, the
/// post-check, or (as `"ref"`) its reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Class {
    Identical,
    /// 2026-09-26: Streams differ, or the streams match and the completion-token
    /// counts differ. `at` is the longest common prefix in characters.
    Diverged {
        at: usize,
        detail: String,
    },
    /// 2026-09-26: Another request's canary appeared in this reply: leakage on
    /// its own evidence, no reference required.
    Contaminated {
        foreign: String,
    },
    /// 2026-09-26: Diverged in the solo post-check leg that runs after the
    /// rungs.
    Persistent {
        at: usize,
    },
    /// 2026-09-26: The prompt's two solo reference runs disagreed, so
    /// contamination cannot be attributed for it.
    AloneUnstable,
    /// 2026-09-26: The request failed, a solo reference failed, or the reply or a
    /// reference has fewer completion tokens than the floor.
    Unmeasured {
        why: String,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Score {
    pub prompts: usize,
    pub rungs: usize,
    pub compared: usize,
    pub identical: usize,
    pub diverged: usize,
    pub contaminated: usize,
    pub persistent: usize,
    pub alone_unstable: usize,
    pub unmeasured: usize,
    pub foreign_canaries: usize,
    pub tokens_compared: usize,
    pub earliest_divergence: Option<usize>,
    /// 2026-09-26: `(prompt_idx, leg_label) -> Class`, for the report table. The
    /// leg label is `"ref"` when the prompt's reference could not be used.
    pub cells: BTreeMap<(usize, String), Class>,
}

/// 2026-09-26: Longest common prefix, in characters.
fn lcp(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// 2026-09-26: Classify one measured outcome against its own solo reference.
fn classify(
    reference: &Transcript,
    got: &RequestOutcome,
    own_canary: &str,
    all_canaries: &[&str],
    min_completion_tokens: usize,
    persistent_leg: bool,
) -> Class {
    let t = match got {
        RequestOutcome::Error(e) => {
            return Class::Unmeasured {
                why: format!("request failed: {e}"),
            };
        }
        RequestOutcome::Ok(t) => t,
    };
    // 2026-09-26: Liveness before equality: two replies that both stopped after
    // three tokens are equal and prove nothing.
    if t.completion_tokens < min_completion_tokens {
        return Class::Unmeasured {
            why: format!(
                "only {} completion tokens (floor {min_completion_tokens})",
                t.completion_tokens
            ),
        };
    }
    // 2026-09-26: The canary check runs before the comparison: leakage is a
    // stronger statement than "differs from its reference".
    if let Some(foreign) = t.carries_foreign_canary(own_canary, all_canaries) {
        return Class::Contaminated {
            foreign: foreign.to_string(),
        };
    }
    let (a, b) = (reference.canonical(), t.canonical());
    if a != b {
        let at = lcp(&a, &b);
        return if persistent_leg {
            Class::Persistent { at }
        } else {
            Class::Diverged {
                at,
                detail: "stream differs from its solo reference".into(),
            }
        };
    }
    // 2026-09-26: Equal streams with a different server-side count is still a
    // divergence: the server accounted for the same text differently.
    if reference.completion_tokens != t.completion_tokens {
        return Class::Diverged {
            at: a.chars().count(),
            detail: format!(
                "identical stream but completion_tokens {} vs {}",
                reference.completion_tokens, t.completion_tokens
            ),
        };
    }
    Class::Identical
}

/// 2026-09-26: Inputs for one scoring pass. Every field is recorded data.
pub struct Legs<'a> {
    /// 2026-09-26: Solo reference, run twice. Indexed by prompt.
    pub ref_a: &'a [RequestOutcome],
    pub ref_b: &'a [RequestOutcome],
    /// 2026-09-26: `(label, outcomes)` per concurrency rung.
    pub rungs: &'a [(String, Vec<RequestOutcome>)],
    /// 2026-09-26: Solo again, after the rungs.
    pub post: &'a [RequestOutcome],
    pub canaries: &'a [String],
    pub min_completion_tokens: usize,
}

pub fn score(legs: &Legs) -> Score {
    let all: Vec<&str> = legs.canaries.iter().map(String::as_str).collect();
    let mut s = Score {
        prompts: legs.ref_a.len(),
        rungs: legs.rungs.len(),
        ..Default::default()
    };

    for i in 0..legs.ref_a.len() {
        let own = legs.canaries.get(i).map(String::as_str).unwrap_or("");
        // 2026-09-26: The reference itself must be measurable and reproducible,
        // or this prompt cannot speak to contamination at all.
        let (a, b) = (
            legs.ref_a[i].transcript(),
            legs.ref_b.get(i).and_then(RequestOutcome::transcript),
        );
        let reference = match (a, b) {
            (Some(a), Some(b))
                if a.completion_tokens < legs.min_completion_tokens
                    || b.completion_tokens < legs.min_completion_tokens =>
            {
                s.unmeasured += 1;
                s.cells.insert(
                    (i, "ref".into()),
                    Class::Unmeasured {
                        why: format!(
                            "a solo reference was below the {} completion-token floor",
                            legs.min_completion_tokens
                        ),
                    },
                );
                continue;
            }
            (Some(a), Some(b))
                if a.canonical() == b.canonical() && a.completion_tokens == b.completion_tokens =>
            {
                a
            }
            (Some(_), Some(_)) => {
                s.alone_unstable += 1;
                s.cells.insert((i, "ref".into()), Class::AloneUnstable);
                continue;
            }
            _ => {
                s.unmeasured += 1;
                s.cells.insert(
                    (i, "ref".into()),
                    Class::Unmeasured {
                        why: "a solo reference leg failed".into(),
                    },
                );
                continue;
            }
        };

        let mut legs_for_prompt: Vec<(String, &RequestOutcome, bool)> = legs
            .rungs
            .iter()
            .filter_map(|(label, outs)| outs.get(i).map(|o| (label.clone(), o, false)))
            .collect();
        if let Some(p) = legs.post.get(i) {
            legs_for_prompt.push(("post".into(), p, true));
        }

        for (label, got, persistent) in legs_for_prompt {
            let c = classify(
                reference,
                got,
                own,
                &all,
                legs.min_completion_tokens,
                persistent,
            );
            s.compared += 1;
            match &c {
                Class::Identical => {
                    s.identical += 1;
                    s.tokens_compared += reference.completion_tokens;
                }
                Class::Diverged { at, .. } => {
                    s.diverged += 1;
                    s.earliest_divergence =
                        Some(s.earliest_divergence.map_or(*at, |e: usize| e.min(*at)));
                }
                Class::Persistent { at } => {
                    s.persistent += 1;
                    s.earliest_divergence =
                        Some(s.earliest_divergence.map_or(*at, |e: usize| e.min(*at)));
                }
                Class::Contaminated { .. } => {
                    s.contaminated += 1;
                    s.foreign_canaries += 1;
                }
                Class::Unmeasured { .. } => s.unmeasured += 1,
                Class::AloneUnstable => s.alone_unstable += 1,
            }
            s.cells.insert((i, label), c);
        }
    }
    s
}

/// 2026-09-26: Zero tolerance: any unmeasured, alone-unstable, contaminated,
/// persistent or diverged cell fails, as does a score with nothing compared.
///
/// The comparison is of stream text and token counts, not a timing statistic,
/// so there is no noise term. Batch-width numerics can flip a near-tie argmax
/// with no cross-request leak; that case is classified `Diverged`, not
/// `Contaminated`, and still fails, because no principled bound says how many
/// flipped tokens make a leak.
pub fn verdict(s: &Score) -> crate::result::Verdict {
    use crate::result::Verdict;
    if s.compared == 0 {
        return Verdict::fail("nothing measured: 0 comparisons (every leg failed?)");
    }
    let mut bad = Vec::new();
    if s.unmeasured > 0 {
        bad.push(format!("{} unmeasured", s.unmeasured));
    }
    if s.alone_unstable > 0 {
        bad.push(format!(
            "{} not reproducible ALONE at temp 0 (#435 class — contamination \
             unattributable for them)",
            s.alone_unstable
        ));
    }
    if s.contaminated > 0 {
        bad.push(format!(
            "{} CONTAMINATED ({} foreign canaries)",
            s.contaminated, s.foreign_canaries
        ));
    }
    if s.persistent > 0 {
        bad.push(format!("{} PERSISTENT (survived into solo)", s.persistent));
    }
    if s.diverged > 0 {
        bad.push(format!("{} diverged", s.diverged));
    }
    if bad.is_empty() {
        return Verdict::pass(format!(
            "{} prompts x {} rungs + post-check: all {} streams identical to \
             their solo reference ({} tokens compared)",
            s.prompts, s.rungs, s.compared, s.tokens_compared
        ));
    }
    let where_ = s
        .earliest_divergence
        .map(|c| format!("; earliest divergence at char {c}"))
        .unwrap_or_default();
    Verdict::fail(format!("{}{where_}", bad.join(" · ")))
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod score_tests;
