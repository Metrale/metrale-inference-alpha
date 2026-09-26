// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Given what each pass produced, decide whether the synchronous
//! and asynchronous routers are the same function of a request. No I/O, so
//! the tests build each failure without a server.
//!
//! Owner: bench, scheduler-equivalence gate.
//! Invariants: `verdict` passes only when every cell has samples, not every
//! reference reply is empty, nothing is unmeasured, no control diverged and
//! no async reply diverged.

use std::collections::BTreeMap;
use std::time::Duration;

use super::host::{Lane, Router};
use super::unmeasured::{self, Unmeasured};
use crate::benchmarks::transcript::Transcript;
use crate::http::RequestFailure;

/// 2026-09-26: Which of the three passes over a lane a leg is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Pass {
    /// 2026-09-26: The reference: the synchronous serve.
    Sync,
    /// 2026-09-26: The candidate: the asynchronous serve.
    Async,
    /// 2026-09-26: The synchronous serve measured a second time: what "the
    /// same router answers the same twice" looks like on this workload.
    Control,
}

impl Pass {
    pub fn label(self) -> &'static str {
        match self {
            Pass::Sync => "sync",
            Pass::Async => "async",
            Pass::Control => "sync-control",
        }
    }
    pub fn router(self) -> Router {
        match self {
            Pass::Sync | Pass::Control => Router::Sync,
            Pass::Async => Router::Async,
        }
    }
}

/// 2026-09-26: What one sample produced in one leg.
#[derive(Debug, Clone)]
pub struct Reply {
    pub sample_id: String,
    /// 2026-09-26: A failure is its own variant, so two failed requests can
    /// never compare equal and read as agreement.
    pub outcome: Result<Box<Transcript>, RequestFailure>,
    /// 2026-09-26: Issue to last byte or failure, on the gate's clock.
    pub elapsed: Duration,
}

/// 2026-09-26: One full pass over the draw: a lane, a pass, a concurrency.
#[derive(Debug, Clone)]
pub struct Leg {
    pub lane: Lane,
    pub pass: Pass,
    pub concurrency: usize,
    pub replies: Vec<Reply>,
}

impl Leg {
    pub(super) fn find(&self, sample_id: &str) -> Option<&Reply> {
        self.replies.iter().find(|r| r.sample_id == sample_id)
    }
}

/// 2026-09-26: One sample, one leg against the reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampleVerdict {
    /// 2026-09-26: Byte-identical: text, reasoning, tool calls, finish reason
    /// and the server's completion token count.
    Equal,
    /// 2026-09-26: The replies differ. Against the async leg this is the
    /// finding; against the control it makes an async difference
    /// unattributable.
    Diverged {
        /// 2026-09-26: Bytes of the reference's canonical form that matched
        /// before the first difference.
        common_prefix: usize,
    },
    /// 2026-09-26: A request failed or a sample is missing: evidence of
    /// nothing, never counted as agreement.
    Unmeasured(String),
}

/// 2026-09-26: The reduction of one leg against the reference leg of its
/// cell.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Diff {
    pub equal: usize,
    pub diverged: usize,
    pub unmeasured: usize,
    /// 2026-09-26: At most `NAMED_DIVERGENCES` ids, for the verdict and the
    /// report table.
    pub diverged_ids: Vec<String>,
}

/// 2026-09-26: One (lane, concurrency) cell: the reference and what was held
/// against it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cell {
    pub lane: Lane,
    pub concurrency: usize,
    pub samples: usize,
    /// 2026-09-26: Reference replies whose canonical form has no character at
    /// or above U+0020. The vacuity guard: an engine answering nothing agrees
    /// with itself.
    pub empty_replies: usize,
    pub async_vs_sync: Diff,
    /// 2026-09-26: `None` when no control leg was run for the cell.
    pub control_vs_sync: Option<Diff>,
    /// 2026-09-26: One entry per request of this cell that failed or never
    /// came back, in any leg. A failed sync reference is one entry, though it
    /// leaves the sample unmeasured in both the async and the control diff.
    pub unmeasured: Vec<Unmeasured>,
}

#[derive(Debug, Clone, Default)]
pub struct Score {
    pub cells: Vec<Cell>,
}

/// 2026-09-26: How many diverged ids a verdict names before it stops.
const NAMED_DIVERGENCES: usize = 8;

/// 2026-09-26: The comparison rule, the same as `kat_equality::verdict_for`:
/// the canonical transcript and the server's token count.
pub fn verdict_for(sample_id: &str, reference: &Leg, other: &Leg) -> SampleVerdict {
    let Some(base) = reference.find(sample_id) else {
        return SampleVerdict::Unmeasured(format!("{sample_id} missing from the reference"));
    };
    let base_t = match &base.outcome {
        Ok(t) => t,
        Err(e) => return SampleVerdict::Unmeasured(format!("reference: {e}")),
    };
    let Some(obs) = other.find(sample_id) else {
        return SampleVerdict::Unmeasured(format!(
            "{sample_id} missing from {}",
            other.pass.label()
        ));
    };
    let t = match &obs.outcome {
        Ok(t) => t,
        Err(e) => {
            return SampleVerdict::Unmeasured(format!("{}: {e}", other.pass.label()));
        }
    };
    let (a, b) = (base_t.canonical(), t.canonical());
    if a != b || base_t.completion_tokens != t.completion_tokens {
        return SampleVerdict::Diverged {
            common_prefix: a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count(),
        };
    }
    SampleVerdict::Equal
}

fn diff(reference: &Leg, other: &Leg) -> Diff {
    let mut d = Diff::default();
    for r in &reference.replies {
        match verdict_for(&r.sample_id, reference, other) {
            SampleVerdict::Equal => d.equal += 1,
            SampleVerdict::Diverged { .. } => {
                d.diverged += 1;
                if d.diverged_ids.len() < NAMED_DIVERGENCES {
                    d.diverged_ids.push(r.sample_id.clone());
                }
            }
            SampleVerdict::Unmeasured(_) => d.unmeasured += 1,
        }
    }
    d
}

fn empty(t: &Transcript) -> bool {
    t.canonical().chars().all(|c| (c as u32) < 0x20)
}

/// 2026-09-26: Group the legs into cells and reduce each against its sync
/// reference. A cell with no sync reference has zero samples, which the
/// verdict refuses; a cell with no async leg counts every sample unmeasured.
pub fn score(legs: &[Leg]) -> Score {
    let mut keys: Vec<(Lane, usize)> = legs.iter().map(|l| (l.lane, l.concurrency)).collect();
    keys.sort();
    keys.dedup();
    let find = |lane: Lane, c: usize, pass: Pass| {
        legs.iter()
            .find(|l| l.lane == lane && l.concurrency == c && l.pass == pass)
    };
    let mut cells = Vec::new();
    for (lane, c) in keys {
        let reference = find(lane, c, Pass::Sync);
        let candidate = find(lane, c, Pass::Async);
        let control = find(lane, c, Pass::Control);
        let unmeasured = reference
            .map(|r| unmeasured::collect(r, candidate, control))
            .unwrap_or_default();
        let (samples, empty_replies, async_vs_sync, control_vs_sync) = match reference {
            Some(r) => (
                r.replies.len(),
                r.replies
                    .iter()
                    .filter(|x| matches!(&x.outcome, Ok(t) if empty(t)))
                    .count(),
                // 2026-09-26: A missing candidate leg is every sample
                // unmeasured, never an empty diff that reads as agreement.
                candidate.map(|a| diff(r, a)).unwrap_or(Diff {
                    unmeasured: r.replies.len(),
                    ..Diff::default()
                }),
                control.map(|k| diff(r, k)),
            ),
            None => (0, 0, Diff::default(), None),
        };
        cells.push(Cell {
            lane,
            concurrency: c,
            samples,
            empty_replies,
            async_vs_sync,
            control_vs_sync,
            unmeasured,
        });
    }
    Score { cells }
}

/// 2026-09-26: The gate's rule, in order. Anything not proven equal fails,
/// and a control that diverged makes the router's difference unattributable.
pub fn verdict(s: &Score) -> crate::result::Verdict {
    use crate::result::Verdict;
    if s.cells.is_empty() {
        return Verdict::fail(
            "EQUIVALENCE UNPROVEN: no (lane, concurrency) cell ran, so nothing was compared."
                .to_string(),
        );
    }
    for c in &s.cells {
        let where_ = format!("{} C={}", c.lane.label(), c.concurrency);
        if c.samples == 0 {
            return Verdict::fail(format!(
                "EQUIVALENCE UNPROVEN: the sync reference at {where_} issued no samples."
            ));
        }
        if c.empty_replies == c.samples {
            return Verdict::fail(format!(
                "EQUIVALENCE VACUOUS: all {} replies at {where_} were empty, so the routers \
                 agree trivially. This proves the harness ran, not that the routers are \
                 equivalent.",
                c.samples
            ));
        }
        if c.async_vs_sync.unmeasured > 0 {
            return Verdict::fail(format!(
                "EQUIVALENCE UNPROVEN: {} of {} samples at {where_} were unmeasured in the \
                 async pass (a failed request is not evidence of agreement). Causes: {}.",
                c.async_vs_sync.unmeasured,
                c.samples,
                unmeasured::breakdown(&c.unmeasured)
            ));
        }
        if let Some(k) = &c.control_vs_sync {
            if k.unmeasured > 0 {
                return Verdict::fail(format!(
                    "EQUIVALENCE UNPROVEN: {} of {} samples at {where_} were unmeasured in \
                     the sync control pass. Causes: {}.",
                    k.unmeasured,
                    c.samples,
                    unmeasured::breakdown(&c.unmeasured)
                ));
            }
            if k.diverged > 0 {
                return Verdict::fail(format!(
                    "CONTROL DIVERGED: the synchronous router answered {} of {} samples \
                     differently on a second pass at {where_} — the workload is not \
                     deterministic under this lane, so an async difference there would be \
                     unattributable ({}{}). Pin the lane harder or find the nondeterminism \
                     before reading the async column.",
                    k.diverged,
                    c.samples,
                    k.diverged_ids.join(", "),
                    more(k)
                ));
            }
        }
    }
    let failing: Vec<&Cell> = s
        .cells
        .iter()
        .filter(|c| c.async_vs_sync.diverged > 0)
        .collect();
    if !failing.is_empty() {
        let per_cell: Vec<String> = failing
            .iter()
            .map(|c| {
                format!(
                    "{} C={}: {} of {} ({}{})",
                    c.lane.label(),
                    c.concurrency,
                    c.async_vs_sync.diverged,
                    c.samples,
                    c.async_vs_sync.diverged_ids.join(", "),
                    more(&c.async_vs_sync)
                )
            })
            .collect();
        return Verdict::fail(format!(
            "ROUTER-DEPENDENT: samples changed between the synchronous and the asynchronous \
             router while the control held — {}. At temperature 0 a reply must be a function \
             of its request, whichever router served it.",
            per_cell.join("; ")
        ));
    }
    let lanes: Vec<&str> = {
        let mut v: Vec<&str> = s.cells.iter().map(|c| c.lane.label()).collect();
        v.dedup();
        v
    };
    let cs: Vec<String> = {
        let mut v: Vec<usize> = s.cells.iter().map(|c| c.concurrency).collect();
        v.sort_unstable();
        v.dedup();
        v.iter().map(|c| c.to_string()).collect()
    };
    let controlled = s.cells.iter().all(|c| c.control_vs_sync.is_some());
    Verdict::pass(format!(
        "{} samples byte-identical between sync and async at C={{{}}} under {} ({})",
        s.cells.first().map(|c| c.samples).unwrap_or(0),
        cs.join(","),
        lanes.join(" and "),
        if controlled {
            "sync-vs-sync control held in every cell"
        } else {
            "no control leg"
        }
    ))
}

fn more(d: &Diff) -> String {
    let left = d.diverged.saturating_sub(d.diverged_ids.len());
    if left > 0 {
        format!(" +{left} more")
    } else {
        String::new()
    }
}

/// 2026-09-26: Raw numbers for the record. Every class is a key even at zero.
pub fn metrics(s: &Score, diagnostics: &BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    let mut m = BTreeMap::new();
    let sum = |f: &dyn Fn(&Cell) -> usize| s.cells.iter().map(f).sum::<usize>() as f64;
    m.insert("cells".into(), s.cells.len() as f64);
    m.insert(
        "samples".into(),
        s.cells.first().map(|c| c.samples).unwrap_or(0) as f64,
    );
    m.insert("diverged".into(), sum(&|c| c.async_vs_sync.diverged));
    m.insert(
        "unmeasured".into(),
        sum(&|c| {
            c.async_vs_sync.unmeasured
                + c.control_vs_sync
                    .as_ref()
                    .map(|k| k.unmeasured)
                    .unwrap_or(0)
        }),
    );
    m.insert(
        "control_diverged".into(),
        sum(&|c| c.control_vs_sync.as_ref().map(|k| k.diverged).unwrap_or(0)),
    );
    m.insert("empty_replies".into(), sum(&|c| c.empty_replies));
    m.extend(unmeasured::metrics(
        s.cells.iter().flat_map(|c| c.unmeasured.iter()),
    ));
    for c in &s.cells {
        m.insert(
            format!("diverged_{}_c{}", c.lane.label(), c.concurrency),
            c.async_vs_sync.diverged as f64,
        );
        m.insert(
            format!("control_diverged_{}_c{}", c.lane.label(), c.concurrency),
            c.control_vs_sync.as_ref().map(|k| k.diverged).unwrap_or(0) as f64,
        );
    }
    for (k, v) in diagnostics {
        m.insert(format!("async_{k}"), *v);
    }
    m
}
