// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Given what each order produced, decide whether every reply is a
//! function of its own request. The driver does the requests; this module does
//! no I/O, so the tests construct each failure without a server.
//!
//! Owner: bench, kat_equality.
//! Invariants:
//! - `verdict` passes only with at least two orders, at least one sample, not
//!   every reference reply empty, and every sample `Equal`.

use crate::benchmarks::transcript::RequestOutcome;

/// 2026-09-26: The order a sample set is issued in, as indices into it.
///
/// Deterministic and RNG-free, so a failing run can be re-run to reproduce it.
/// Order 0 is the order `dataset::load_shard` returns, which is also the
/// order the BFCL benchmark issues its draw in.
/// Order 1 is the reverse, which changes what preceded every sample. Further
/// orders are rotations; for `len >= 2` none of them is order 0.
pub fn permutation(index: usize, len: usize) -> Vec<usize> {
    match index {
        0 => (0..len).collect(),
        1 => (0..len).rev().collect(),
        k if len == 0 => {
            let _ = k;
            Vec::new()
        }
        k => {
            // 2026-09-26: The shift is forced into 1..len, so for len >= 2 no
            // rotation is order 0.
            let shift = (k * len / (k + 1)).max(1) % len.max(1);
            let shift = if shift == 0 { 1 } else { shift };
            (0..len).map(|i| (i + shift) % len).collect()
        }
    }
}

/// 2026-09-26: What one sample produced under one order.
#[derive(Debug, Clone)]
pub struct Observation {
    pub sample_id: String,
    pub outcome: RequestOutcome,
}

/// 2026-09-26: One full pass over the sample set, in one order.
#[derive(Debug, Clone)]
pub struct OrderRun {
    /// 2026-09-26: The order's name (`canonical`, `reversed`, `rotation-<k>`),
    /// for the failure message.
    pub label: String,
    /// 2026-09-26: In issue order. Lookups go through `sample_id`, so the
    /// storage order is not an index.
    pub observations: Vec<Observation>,
}

impl OrderRun {
    fn find(&self, sample_id: &str) -> Option<&Observation> {
        self.observations.iter().find(|o| o.sample_id == sample_id)
    }
}

/// 2026-09-26: What happened to one `sample_id` across every order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampleVerdict {
    /// 2026-09-26: Byte-identical everywhere, with equal completion-token
    /// counts. The only passing outcome.
    Equal,
    /// 2026-09-26: Differs from the reference order in `other_order`, the
    /// first order that differs.
    Diverged {
        other_order: String,
        /// 2026-09-26: Bytes of `canonical()` that matched before the first
        /// difference.
        common_prefix: usize,
    },
    /// 2026-09-26: A request failed, or the sample is missing from an order. A
    /// failure is evidence of neither equality nor inequality, so it is never
    /// counted as agreement.
    Unmeasured(String),
}

/// 2026-09-26: The reduction the verdict and the report both read.
#[derive(Debug, Clone, Default)]
pub struct Score {
    pub orders: usize,
    pub samples: usize,
    pub equal: usize,
    pub diverged: usize,
    pub unmeasured: usize,
    /// 2026-09-26: At most `NAMED_DIVERGENCES` ids, for the verdict string.
    pub diverged_ids: Vec<String>,
    /// 2026-09-26: Reference-order replies whose `canonical()` has no
    /// character at or above U+0020: nothing was said. A server that answers
    /// every request with nothing is equal across orders, so `verdict` fails a
    /// run where every reply is empty.
    pub empty_replies: usize,
}

/// 2026-09-26: How many diverged ids a verdict names before it stops.
const NAMED_DIVERGENCES: usize = 8;

/// 2026-09-26: Compare every sample of order 0 against every other order.
/// Order 0 is the reference.
pub fn score(runs: &[OrderRun]) -> Score {
    let mut s = Score {
        orders: runs.len(),
        ..Default::default()
    };
    let Some(reference) = runs.first() else {
        return s;
    };
    s.samples = reference.observations.len();
    for obs in &reference.observations {
        match verdict_for(&obs.sample_id, reference, &runs[1..]) {
            SampleVerdict::Equal => s.equal += 1,
            SampleVerdict::Diverged { .. } => {
                s.diverged += 1;
                if s.diverged_ids.len() < NAMED_DIVERGENCES {
                    s.diverged_ids.push(obs.sample_id.clone());
                }
            }
            SampleVerdict::Unmeasured(_) => s.unmeasured += 1,
        }
        if let RequestOutcome::Ok(t) = &obs.outcome
            && t.canonical().chars().all(|c| (c as u32) < 0x20)
        {
            // 2026-09-26: Only separators and other control characters.
            s.empty_replies += 1;
        }
    }
    s
}

/// 2026-09-26: One sample's verdict across the non-reference orders.
pub fn verdict_for(sample_id: &str, reference: &OrderRun, others: &[OrderRun]) -> SampleVerdict {
    let Some(base) = reference.find(sample_id) else {
        return SampleVerdict::Unmeasured(format!("{sample_id} missing from the reference order"));
    };
    let RequestOutcome::Ok(base_t) = &base.outcome else {
        let RequestOutcome::Error(e) = &base.outcome else {
            unreachable!("RequestOutcome has two variants")
        };
        return SampleVerdict::Unmeasured(format!("reference order: {e}"));
    };
    let base_c = base_t.canonical();
    for other in others {
        let Some(obs) = other.find(sample_id) else {
            return SampleVerdict::Unmeasured(format!("{sample_id} missing from {}", other.label));
        };
        let RequestOutcome::Ok(t) = &obs.outcome else {
            let RequestOutcome::Error(e) = &obs.outcome else {
                unreachable!("RequestOutcome has two variants")
            };
            return SampleVerdict::Unmeasured(format!("{}: {e}", other.label));
        };
        let c = t.canonical();
        // 2026-09-26: `completion_tokens` is compared too: a reply can be
        // byte-identical as text while the server disagrees about how many
        // tokens it emitted, and that is a real difference in what ran.
        if c != base_c || t.completion_tokens != base_t.completion_tokens {
            return SampleVerdict::Diverged {
                other_order: other.label.clone(),
                common_prefix: common_prefix_len(&base_c, &c),
            };
        }
    }
    SampleVerdict::Equal
}

fn common_prefix_len(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

/// 2026-09-26: The gate's rule, in order. Anything not proven equal fails.
pub fn verdict(s: &Score) -> crate::result::Verdict {
    use crate::result::Verdict;
    if s.orders < 2 {
        return Verdict::fail(format!(
            "EQUALITY UNPROVEN: {} order(s) ran. Equality is a claim about TWO orders; \
             one order compares with nothing.",
            s.orders
        ));
    }
    if s.samples == 0 {
        return Verdict::fail(
            "EQUALITY UNPROVEN: the reference order issued no samples, so nothing was compared."
                .to_string(),
        );
    }
    // 2026-09-26: Before believing any equality result, refuse a run that could
    // not have seen a difference: a server answering every request with
    // nothing is byte-identical across every order.
    if s.empty_replies == s.samples {
        return Verdict::fail(format!(
            "EQUALITY VACUOUS: all {} replies were empty, so every order agrees trivially. \
             This proves the harness ran, not that the server is order-independent.",
            s.samples
        ));
    }
    if s.unmeasured > 0 {
        return Verdict::fail(format!(
            "EQUALITY UNPROVEN: {} of {} samples were unmeasured (a failed request is not \
             evidence of agreement).",
            s.unmeasured, s.samples
        ));
    }
    if s.diverged > 0 {
        let named = s.diverged_ids.join(", ");
        let more = s.diverged.saturating_sub(s.diverged_ids.len());
        let tail = if more > 0 {
            format!(" (+{more} more)")
        } else {
            String::new()
        };
        return Verdict::fail(format!(
            "ORDER-DEPENDENT: {} of {} samples changed when the request ORDER changed — \
             {named}{tail}. At temperature 0 a sample's reply must be a function of that \
             sample. A benchmark built on this cannot be sharded, and a score taken under \
             one order does not describe another.",
            s.diverged, s.samples
        ));
    }
    crate::result::Verdict::pass(format!(
        "{} samples byte-identical across {} request orders",
        s.samples, s.orders
    ))
}
