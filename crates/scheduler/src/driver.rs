// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The scheduler loop. A [`Core`] plans a tick, executes its lanes and
//! ends it; [`run`] only sequences those calls and never touches a device.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

/// 2026-09-25: What one tick will do, decided by [`Core::plan`].
pub struct TickPlan<L> {
    /// 2026-09-25: The lanes to execute, in order.
    pub lanes: Vec<L>,
    /// 2026-09-25: Leave the loop without running `lanes`; [`run`] then calls
    /// [`Core::finish`].
    pub shutdown: bool,
}

/// 2026-09-25: What a lane says about the rest of its tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneVerdict {
    /// 2026-09-25: Run the remaining lanes and end the tick.
    Proceed,
    /// 2026-09-25: Skip the remaining lanes and [`Core::end_tick`]; the next tick
    /// starts.
    SkipRest,
}

pub trait Core {
    type Lane;
    /// 2026-09-25: Decide the tick's lanes, or that the loop ends.
    fn plan(&mut self) -> TickPlan<Self::Lane>;
    /// 2026-09-25: Run one lane; its verdict decides whether the tick goes on.
    fn execute_lane(&mut self, lane: Self::Lane) -> LaneVerdict;
    /// 2026-09-25: The work after the lanes; skipped when a lane returns `SkipRest`.
    fn end_tick(&mut self);
    /// 2026-09-25: Called once, after a plan with `shutdown` set.
    fn finish(self);
}

pub fn run<C: Core>(mut core: C) {
    loop {
        let plan = core.plan();
        if plan.shutdown {
            break;
        }
        let mut skip_rest = false;
        for lane in plan.lanes {
            if core.execute_lane(lane) == LaneVerdict::SkipRest {
                skip_rest = true;
                break;
            }
        }
        if skip_rest {
            continue;
        }
        core.end_tick();
    }
    core.finish();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: A core that records the driver's call order and scripts its verdicts.
    struct Scripted {
        ticks: Vec<(Vec<LaneVerdict>, bool)>,
        log: Vec<String>,
        finished: std::rc::Rc<std::cell::Cell<bool>>,
    }

    impl Core for Scripted {
        type Lane = LaneVerdict;
        fn plan(&mut self) -> TickPlan<LaneVerdict> {
            let (lanes, shutdown) = self.ticks.remove(0);
            self.log
                .push(format!("plan{}", if shutdown { " shutdown" } else { "" }));
            TickPlan { lanes, shutdown }
        }
        fn execute_lane(&mut self, lane: LaneVerdict) -> LaneVerdict {
            self.log.push(format!("lane {lane:?}"));
            lane
        }
        fn end_tick(&mut self) {
            self.log.push("end_tick".into());
        }
        fn finish(self) {
            self.finished.set(true);
            assert_eq!(
                self.log,
                [
                    "plan",
                    "lane Proceed",
                    "lane Proceed",
                    "end_tick",
                    "plan",
                    "lane Proceed",
                    "lane SkipRest",
                    "plan",
                    "end_tick",
                    "plan shutdown",
                ]
            );
        }
    }

    #[test]
    fn the_driver_runs_lanes_in_order_skips_the_tick_end_on_an_idle_lane_and_finishes_once() {
        use LaneVerdict::*;
        let finished = std::rc::Rc::new(std::cell::Cell::new(false));
        run(Scripted {
            ticks: vec![
                (vec![Proceed, Proceed], false),
                // 2026-09-25: SkipRest drops the third lane and the end of the tick.
                (vec![Proceed, SkipRest, Proceed], false),
                (vec![], false),
                (vec![Proceed], true),
            ],
            log: Vec::new(),
            finished: finished.clone(),
        });
        assert!(finished.get());
    }
}
