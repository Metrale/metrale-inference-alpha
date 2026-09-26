// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Adaptive MTP draft count for batch widths 9..=16: one or two
//! drafts, chosen from the observed accept statistics, with the static ladder
//! (`metrale_model_layers::speculative::mtp_ladder_drafts`) as the floor.
//!
//! Owner: speculative.
//! Invariants:
//! - `drafts_for` never returns fewer drafts than the static ladder, and
//!   returns exactly the ladder's count outside 9..=16, when
//!   `RungParams::disabled`, when `METRALE_NO_MTP_K_LADDER` is set, or when
//!   `num_drafts < 2`.
//! - `observe` changes no state for widths outside 9..=16 or when disabled.
//!
//! With `p1` the first-draft accept rate and `p2_cond` the second-draft accept
//! rate given an accepted first draft, two drafts yield `token_ratio(p1,
//! p2_cond) = 1 + p1*p2_cond/(1 + p1)` times the tokens per verify step of one
//! draft. The controller moves to two drafts when the smoothed ratio reaches
//! `enter` and back to one when it falls below `leave` (`next_state`). The
//! estimates are EWMAs seeded with their first sample. With the default
//! tiered verify pools the scheduler clamps the lift back to one draft
//! (`spec_capacity::clamp_drafts_to_slot_capacity`; see that module).
//!
//! At one draft nothing proposes a second token, so `p2_cond` cannot be
//! observed there. The controller then runs two drafts for a flush (a probe)
//! only on evidence or after a long backstop; see `drafts_for`. Measured
//! 2026-08-01 on dgx1 at C=16 from flush timestamps: a steady flush took
//! 1.16 s and the flush that entered a probe 2.90 s, so a probe cost about
//! 1.74 s.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// 2026-09-25: The batch widths the controller adapts. Every other width
/// keeps the static ladder's count.
const BAND: std::ops::RangeInclusive<usize> = 9..=16;

/// 2026-09-25: Move to two drafts when the smoothed token ratio is at or
/// above this. Overridden by `METRALE_MTP_RUNG_ENTER`.
const ENTER: f64 = 1.32;
/// 2026-09-25: Move back to one draft when the smoothed token ratio falls
/// below this. Overridden by `METRALE_MTP_RUNG_LEAVE`.
const LEAVE: f64 = 1.30;
/// 2026-09-25: EWMA weight of the `p1` and `p2_cond` estimates the decision
/// reads (effective window `1/ALPHA` = 2 flushes). Overridden by
/// `METRALE_MTP_RUNG_ALPHA`, capped at 1.
const ALPHA: f64 = 0.5;
/// 2026-09-25: EWMA weight of the `p1` estimate the probe trigger reads
/// (effective window about 7 flushes). It is slower than [`ALPHA`] so that
/// flush-to-flush noise in `p1` does not buy probes. Overridden by
/// `METRALE_MTP_RUNG_ALPHA_SLOW`, capped at 1.
const ALPHA_SLOW: f64 = 0.15;
/// 2026-09-25: Backstop probe interval in flushes, for a `p2_cond` drift at
/// constant `p1` that [`P1_TRIGGER`] cannot see. With the probe and flush
/// times measured 2026-08-01 (1.74 s and 1.16 s, module doc), 2048 flushes is
/// one probe per about 40 minutes, about 0.07% of the time. Overridden by
/// `METRALE_MTP_RUNG_PROBE_TICKS`.
const PROBE_TICKS: u64 = 2048;
/// 2026-09-25: Probe when the slow `p1` EWMA has risen this far above its
/// value at the last depth flush. Overridden by `METRALE_MTP_RUNG_P1_TRIGGER`.
const P1_TRIGGER: f64 = 0.08;

/// 2026-09-25: The controller's thresholds, fixed for the life of one
/// `AdaptiveRung`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RungParams {
    /// 2026-09-25: Pins the static ladder. `from_env` sets it when
    /// `METRALE_MTP_STATIC_RUNG` or `METRALE_MTP_K_LADDER` is present, whatever
    /// the value (`0` included): an operator who spells out the rungs gets
    /// exactly those rungs.
    pub disabled: bool,
    pub enter: f64,
    pub leave: f64,
    pub alpha: f64,
    pub alpha_slow: f64,
    pub probe_ticks: u64,
    pub p1_trigger: f64,
}

impl RungParams {
    /// 2026-09-25: The compiled constants, with adaptation on.
    pub const DEFAULTS: Self = Self {
        disabled: false,
        enter: ENTER,
        leave: LEAVE,
        alpha: ALPHA,
        alpha_slow: ALPHA_SLOW,
        probe_ticks: PROBE_TICKS,
        p1_trigger: P1_TRIGGER,
    };

    /// 2026-09-25: Reads the environment. A tunable whose value does not
    /// parse as a finite positive number keeps its constant (`tunable`).
    pub fn from_env() -> Self {
        Self {
            disabled: std::env::var_os("METRALE_MTP_STATIC_RUNG").is_some()
                || std::env::var_os("METRALE_MTP_K_LADDER").is_some(),
            enter: tunable("METRALE_MTP_RUNG_ENTER", ENTER),
            leave: tunable("METRALE_MTP_RUNG_LEAVE", LEAVE),
            alpha: tunable("METRALE_MTP_RUNG_ALPHA", ALPHA).min(1.0),
            alpha_slow: tunable("METRALE_MTP_RUNG_ALPHA_SLOW", ALPHA_SLOW).min(1.0),
            probe_ticks: tunable("METRALE_MTP_RUNG_PROBE_TICKS", PROBE_TICKS as f64) as u64,
            p1_trigger: tunable("METRALE_MTP_RUNG_P1_TRIGGER", P1_TRIGGER),
        }
    }
}

/// 2026-09-25: `var` parsed as `f64` when it is finite and positive,
/// otherwise `default`.
fn tunable(var: &str, default: f64) -> f64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default)
}

/// 2026-09-25: Expected tokens per verify step at two drafts relative to one:
/// `(1 + p1 + p1*p2_cond) / (1 + p1)`. Returns 1.0 when `p1 <= 0` or either
/// input is not finite.
pub fn token_ratio(p1: f64, p2_cond: f64) -> f64 {
    if p1 <= 0.0 || !p1.is_finite() || !p2_cond.is_finite() {
        return 1.0;
    }
    1.0 + p1 * p2_cond / (1.0 + p1)
}

/// 2026-09-25: The second-draft conditional accept implied by a flush at two
/// drafts, where `mean_na = p1 + p1*p2_cond`: `(mean_na - p1) / p1`, clamped
/// to `[0, 1]`. Deeper drafts are not modelled. `None` when `p1 <= 0`.
pub fn p2_cond_from(p1: f64, mean_na: f64) -> Option<f64> {
    (p1 > 0.0).then(|| ((mean_na - p1) / p1).clamp(0.0, 1.0))
}

/// 2026-09-25: The next state from the current one and the smoothed token
/// ratio; `true` means two drafts.
pub fn next_state(at_depth: bool, tr: f64, p: &RungParams) -> bool {
    if at_depth {
        tr >= p.leave
    } else {
        tr >= p.enter
    }
}

struct Ctl {
    p1: AtomicU64,
    /// 2026-09-25: Slow-EWMA `p1`, read only by the probe trigger in
    /// `drafts_for`.
    p1_slow: AtomicU64,
    p2: AtomicU64,
    seeded_p1: AtomicBool,
    seeded_p1_slow: AtomicBool,
    seeded_p2: AtomicBool,
    at_depth: AtomicBool,
    tick: AtomicU64,
    last_probe: AtomicU64,
    /// 2026-09-25: Slow-EWMA `p1` at the last depth flush (`k_drafts >= 2`).
    p1_at_decision: AtomicU64,
    flips: AtomicU64,
}

/// 2026-09-25: The controller. Each scheduler context owns one
/// (`SchedCtx::rung`), so accept history is not shared between contexts.
pub struct AdaptiveRung {
    params: RungParams,
    ctl: Ctl,
    /// 2026-09-25: The last `engaged` passed to `note_width_regime`; starts
    /// `true`.
    width_engaged: AtomicBool,
    /// 2026-09-25: Number of changes of `width_engaged`.
    width_flips: AtomicU64,
}

fn ewma_a(cell: &AtomicU64, seeded: &AtomicBool, sample: f64, a: f64) -> f64 {
    let prev = f64::from_bits(cell.load(Ordering::Relaxed));
    let next = if seeded.swap(true, Ordering::Relaxed) {
        a * sample + (1.0 - a) * prev
    } else {
        sample
    };
    cell.store(next.to_bits(), Ordering::Relaxed);
    next
}

impl AdaptiveRung {
    pub const fn new(params: RungParams) -> Self {
        Self {
            params,
            ctl: Ctl {
                p1: AtomicU64::new(0),
                p1_slow: AtomicU64::new(0),
                p2: AtomicU64::new(0),
                seeded_p1: AtomicBool::new(false),
                seeded_p1_slow: AtomicBool::new(false),
                seeded_p2: AtomicBool::new(false),
                at_depth: AtomicBool::new(false),
                tick: AtomicU64::new(0),
                last_probe: AtomicU64::new(0),
                p1_at_decision: AtomicU64::new(0),
                flips: AtomicU64::new(0),
            },
            width_engaged: AtomicBool::new(true),
            width_flips: AtomicU64::new(0),
        }
    }

    pub fn from_env() -> Self {
        Self::new(RungParams::from_env())
    }

    pub fn params(&self) -> &RungParams {
        &self.params
    }

    /// 2026-09-25: `true` while the controller's state is two drafts.
    pub fn at_depth(&self) -> bool {
        self.ctl.at_depth.load(Ordering::Relaxed)
    }

    /// 2026-09-25: Number of state changes so far.
    pub fn flips(&self) -> u64 {
        self.ctl.flips.load(Ordering::Relaxed)
    }

    pub fn width_engaged(&self) -> bool {
        self.width_engaged.load(Ordering::Relaxed)
    }

    pub fn width_flips(&self) -> u64 {
        self.width_flips.load(Ordering::Relaxed)
    }

    /// 2026-09-25: Feeds one accept-statistics flush at batch width `n`:
    /// `k_drafts` is the flush's largest draft depth, `p1` its first-draft
    /// accept rate, `mean_na` its mean accepted drafts per verify. The
    /// scheduler's `AcceptBuckets::record` is the caller; this type keeps no
    /// accept counters of its own.
    pub fn observe(&self, n: usize, k_drafts: usize, p1: f64, mean_na: f64) {
        let (p, c) = (&self.params, &self.ctl);
        if p.disabled || !BAND.contains(&n) {
            return;
        }
        let p1_e = ewma_a(&c.p1, &c.seeded_p1, p1, p.alpha);
        let p1_slow = ewma_a(&c.p1_slow, &c.seeded_p1_slow, p1, p.alpha_slow);
        let tick = c.tick.fetch_add(1, Ordering::Relaxed) + 1;
        if k_drafts >= 2 {
            // 2026-09-25: A depth flush observes p2_cond and resets both probe
            // triggers read by `drafts_for` (`last_probe`, `p1_at_decision`).
            c.last_probe.store(tick, Ordering::Relaxed);
            c.p1_at_decision.store(p1_slow.to_bits(), Ordering::Relaxed);
            if let Some(p2) = p2_cond_from(p1, mean_na) {
                let p2_e = ewma_a(&c.p2, &c.seeded_p2, p2, p.alpha);
                let tr = token_ratio(p1_e, p2_e);
                let was = c.at_depth.load(Ordering::Relaxed);
                let now = next_state(was, tr, p);
                if now != was {
                    c.at_depth.store(now, Ordering::Relaxed);
                    let flips = c.flips.fetch_add(1, Ordering::Relaxed) + 1;
                    tracing::info!(
                        "MTP rung n={n} -> k_drafts={} (token_ratio={tr:.4} p1={p1_e:.3} \
                         p2_cond={p2_e:.3} enter={:.3} leave={:.3} tick={tick} flips={flips})",
                        if now { 2 } else { 1 },
                        p.enter,
                        p.leave,
                    );
                }
            }
        }
    }

    /// 2026-09-25: The draft count for a batch of `n_active`: the static
    /// ladder's count, raised to `min(2, num_drafts)` inside the adapted band
    /// while the state is two drafts or a probe is due.
    pub fn drafts_for(&self, n_active: usize, num_drafts: usize) -> usize {
        let (p, c) = (&self.params, &self.ctl);
        let base = metrale_model_layers::speculative::mtp_ladder_drafts(n_active, num_drafts);
        if p.disabled
            || metrale_model_layers::speculative::mtp_ladder_disabled()
            || num_drafts < 2
            || !BAND.contains(&n_active)
        {
            return base;
        }
        // 2026-09-25: A probe is due when p2_cond has never been observed,
        // when the slow p1 has risen by `p1_trigger` since the last depth
        // flush, or when `probe_ticks` flushes have passed since it. Only a
        // rise counts: `token_ratio` increases with p1, so a fall cannot make
        // two drafts newly worthwhile.
        let tick = c.tick.load(Ordering::Relaxed);
        let p1_moved = f64::from_bits(c.p1_slow.load(Ordering::Relaxed))
            - f64::from_bits(c.p1_at_decision.load(Ordering::Relaxed))
            >= p.p1_trigger;
        let probing = !c.seeded_p2.load(Ordering::Relaxed)
            || p1_moved
            || tick.saturating_sub(c.last_probe.load(Ordering::Relaxed)) >= p.probe_ticks;
        if c.at_depth.load(Ordering::Relaxed) || probing {
            return 2.min(num_drafts).max(base);
        }
        base
    }

    /// 2026-09-25: Records the width decision the scheduler already took:
    /// `engaged` is `n_active <= cap`, where `cap` is the run's
    /// `mtp_max_seqs` lever (32 unless overridden). It logs one INFO line per
    /// change and counts changes in `width_flips`; it decides nothing.
    pub fn note_width_regime(&self, n_active: usize, engaged: bool, cap: usize) {
        if self.width_engaged.swap(engaged, Ordering::Relaxed) == engaged {
            return;
        }
        let flips = self.width_flips.fetch_add(1, Ordering::Relaxed) + 1;
        if engaged {
            tracing::info!(
                "speculation ENGAGED at width n={n_active} (dispatch cap {cap}) — flips={flips}"
            );
        } else {
            tracing::info!(
                "speculation DISENGAGED at width n={n_active} > dispatch cap {cap}: this width \
                 plain-decodes (METRALE_MTP_MAX_SEQS raises the cap; the verify pools grow with it) \
                 — flips={flips}"
            );
        }
    }
}

#[cfg(test)]
#[path = "adaptive_rung_tests.rs"]
mod tests;
