// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The concurrency sweep's run verdict, as pure functions of the
//! metrics map and the counts.
//!
//! - Any request error fails the verdict, gating or not.
//! - With every floor at 0.0 the verdict is info.
//! - When a floor is set, a vacuous or cache-uncontrolled cell makes the run
//!   INCONCLUSIVE (a fail), and so does a set floor with no comparable cell
//!   to judge it on.
//!
//! Owner: bench (concurrency).
//! Invariants: [`sweep_verdict`] returns PASS only when some floor is set.

use std::collections::BTreeMap;

use crate::result::Verdict;

/// 2026-09-26: Run-verdict floors on aggregate tok/s. 0.0 is the off state
/// for each.
#[derive(Clone, Debug, Default)]
pub(crate) struct Floors {
    pub per_c: Vec<(usize, f64)>,
    pub peak: f64,
}

impl Floors {
    pub(crate) fn gating(&self) -> bool {
        self.peak > 0.0 || self.per_c.iter().any(|(_, f)| *f > 0.0)
    }
}

/// 2026-09-26: Flagged cells, counted per reason: a vacuous cell did not deliver its tokens, a cache-uncontrolled cell's
/// requests disagree about the cache state, and a non-MTP cell ran below the
/// 1.5 accept depth. Only the first two affect the verdict.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Exclusions {
    pub(crate) vacuous: usize,
    pub(crate) cache_uncontrolled: usize,
    pub(crate) non_mtp_arm: usize,
}

/// 2026-09-26: Compares the raw metric against each floor as handed in. When
/// the gate fills the floors, the server's `apply_threshold_params` has
/// already subtracted each bound's noise, so this agrees with
/// `gate::scoring::compare`.
pub(crate) fn sweep_verdict(
    metrics: &BTreeMap<String, f64>,
    cells: usize,
    errors: usize,
    excl: Exclusions,
    vacuity_floor_pct: f64,
    floors: &Floors,
) -> Verdict {
    let Exclusions {
        vacuous,
        cache_uncontrolled,
        non_mtp_arm,
    } = excl;
    if errors > 0 {
        return Verdict::fail(format!(
            "{errors} request(s) failed — affected rows are not comparable"
        ));
    }
    if !floors.gating() {
        return if vacuous > 0 || cache_uncontrolled > 0 {
            Verdict::info(format!(
                "{cells} cells, {vacuous} below the vacuity floor ({vacuity_floor_pct:.0}% \
                 of osl), {cache_uncontrolled} without sufficient observed warm-cache use — flagged \
                 rows' tok/s are not comparable"
            ))
        } else {
            Verdict::info(format!(
                "{cells} cells, no request errors, all above the vacuity floor"
            ))
        };
    }
    if vacuous > 0 {
        return Verdict::fail(format!(
            "INCONCLUSIVE: {vacuous} of {cells} cells below the vacuity floor \
             ({vacuity_floor_pct:.0}% of osl) — undelivered tokens cannot clear a \
             throughput floor, whatever the aggregate prints"
        ));
    }
    if cache_uncontrolled > 0 {
        return Verdict::fail(format!(
            "INCONCLUSIVE: {cache_uncontrolled} of {cells} cells requested warm-up but a \
             measured request did not report a material cached-prompt fraction"
        ));
    }
    // 2026-09-26: `non_mtp_arm` is published but does not decide the verdict
    // (see `CellRow::arm_is_not_mtp`).
    let _ = non_mtp_arm;
    let mut basis = Vec::new();
    for (c, floor) in floors.per_c.iter().filter(|(_, f)| *f > 0.0) {
        let key = format!("c{c}_aggregate_tok_s");
        let Some(value) = metrics.get(&key) else {
            return Verdict::fail(format!(
                "INCONCLUSIVE: the C={c} floor is set ({floor:.1} tok/s) but the sweep \
                 produced no comparable C={c} cell to judge it on"
            ));
        };
        if *value < *floor {
            return Verdict::fail(format!(
                "BELOW THE C={c} FLOOR — {value:.1} aggregate tok/s vs the {floor:.1} floor"
            ));
        }
        basis.push(format!("C{c} {value:.1}/{floor:.1}"));
    }
    if floors.peak > 0.0 {
        let Some(value) = metrics.get("peak_aggregate_tok_s") else {
            return Verdict::fail(format!(
                "INCONCLUSIVE: the peak floor is set ({:.1} tok/s) but no comparable cell \
                 produced a peak to judge it on",
                floors.peak
            ));
        };
        if *value < floors.peak {
            return Verdict::fail(format!(
                "BELOW THE PEAK FLOOR — {value:.1} aggregate tok/s vs the {:.1} floor",
                floors.peak
            ));
        }
        basis.push(format!("peak {value:.1}/{:.1}", floors.peak));
    }
    Verdict::pass(format!(
        "{cells} cells, zero errors, zero vacuous — every populated floor met \
         ({})",
        basis.join(" · ")
    ))
}
