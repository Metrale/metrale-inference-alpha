// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Cool-down: a box that runs hot mid-campaign is parked, and the rest of the fleet keeps working.
//!
//! Owner: server CLI (`met benchmark certify`).
//! - Before a worker takes another unit it reads the node's hottest chassis
//!   zone and the driver's thermal-throttle flag. At the class's
//!   `chassis_park_c` or above, or with the throttle asserted, the node is
//!   parked: it takes nothing and re-reads every [`RECHECK`] until the zone is
//!   at or below `chassis_resume_c` and the throttle is clear, or until
//!   [`MAX_PARK`] has passed.
//! - Both lines are absolute temperatures from `kernels/<hw>/HARDWARE.toml`
//!   `[benchmarks.limits.thermal]` (`hardware::limits::ThermalEnvelope`), not
//!   offsets from the box's temperature at plan time.
//! - Pending units go to whichever node is free; a parked box that hosts the
//!   bundled Speed class delays that class.
//! - A node with no temperature reading and no throttle flag is not parked;
//!   that is said once per worker.
//! - `--dangerous-ignore-thermals` turns every park into a warning and lets the
//!   box keep taking units; `gate::agreement` still judges the records.
//!
//! Invariants: none beyond the types.

use std::time::Duration;

use super::node::Node;
use metrale_bench::hardware::limits::ThermalEnvelope;

/// 2026-09-26: How often a parked node is re-read.
pub const RECHECK: Duration = Duration::from_secs(60);
/// 2026-09-26: The longest a node stays parked. A box that will not cool resumes with
/// a warning; the records it then writes are still judged by the equivalence
/// policy.
pub const MAX_PARK: Duration = Duration::from_secs(30 * 60);

/// 2026-09-26: What the reading says about taking another unit.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// 2026-09-26: Take one.
    Ready,
    /// 2026-09-26: Wait: the reading (NaN when absent), and whether the driver reports a
    /// thermal throttle.
    Park { now_c: f64, throttled: bool },
    /// 2026-09-26: No temperature reading and no throttle; take one, and say so.
    Blind,
}

/// 2026-09-26: One live reading of a node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reading {
    /// 2026-09-26: The hottest chassis zone, °C.
    pub chassis_c: Option<f64>,
    /// 2026-09-26: The driver's thermal-slowdown flag, when it could be read.
    pub throttled: Option<bool>,
}

/// 2026-09-26: Pure: the hysteresis rule against the class's envelope. Unparked, park at
/// or above `chassis_park_c`; parked, stay parked above `chassis_resume_c`. A
/// throttle flag parks regardless of the temperature; a missing temperature
/// parks nothing unless the flag is set.
#[must_use]
pub fn judge(r: Reading, parked: bool, env: &ThermalEnvelope) -> Verdict {
    let throttled = r.throttled == Some(true);
    let Some(now_c) = r.chassis_c else {
        return if throttled {
            Verdict::Park {
                now_c: f64::NAN,
                throttled,
            }
        } else {
            Verdict::Blind
        };
    };
    let hold = throttled
        || if parked {
            now_c > env.chassis_resume_c
        } else {
            now_c >= env.chassis_park_c
        };
    if hold {
        Verdict::Park { now_c, throttled }
    } else {
        Verdict::Ready
    }
}

/// 2026-09-26: Where a node's live chassis reading comes from.
pub trait Probe: Send + Sync {
    /// 2026-09-26: The node's chassis temperature and throttle flag now.
    fn read(&self, node: &Node) -> Reading;
}

/// 2026-09-26: The real probe: this box through `HardwareState`, a remote node through
/// `metralectl bench nodes` (a failed call reads as no reading).
pub struct FleetProbe {
    pub metralectl: std::sync::Arc<dyn super::metralectl::Metralectl>,
}

impl Probe for FleetProbe {
    fn read(&self, node: &Node) -> Reading {
        if node.local {
            let s = metrale_bench::hardware::HardwareState::collect();
            return Reading {
                chassis_c: s.hottest_chassis_c(),
                throttled: s.throttle_active.thermal(),
            };
        }
        let blind = Reading {
            chassis_c: None,
            throttled: None,
        };
        let Ok(rows) = self.metralectl.nodes(std::slice::from_ref(&node.addr)) else {
            return blind;
        };
        let Some(info) = rows
            .iter()
            .find(|r| r.node == node.addr)
            .and_then(|r| r.info.as_ref())
        else {
            return blind;
        };
        let fp = super::node::fingerprint_of(info);
        Reading {
            chassis_c: fp.hottest_chassis_c,
            throttled: fp.thermal_alert,
        }
    }
}

/// 2026-09-26: One node's cool-down state across a worker's loop.
#[derive(Debug, Default)]
pub struct Gate {
    parked_since: Option<std::time::Instant>,
    said_blind: bool,
    /// 2026-09-26: Under `ignore`: whether the last reading would have parked, so the
    /// warning is said on the way in and the all-clear on the way out, not
    /// on every unit.
    warned_hot: bool,
}

impl Gate {
    /// 2026-09-26: Whether the node may take a unit now, reading the probe. `false`
    /// means the caller should sleep [`RECHECK`] and ask again. Parking,
    /// resuming and the first blind reading are reported through `say`. With
    /// `ignore` (`--dangerous-ignore-thermals`) the answer is always `true` and
    /// a park becomes a warning. With no envelope the answer is `true` and the
    /// probe is not read.
    pub fn may_take(
        &mut self,
        node: &Node,
        probe: &dyn Probe,
        envelope: Option<ThermalEnvelope>,
        ignore: bool,
        say: &dyn Fn(&str),
    ) -> bool {
        // 2026-09-26: No envelope happens only under `--dangerous-ignore-thermals`
        // (`certify_cmd` refuses a class without limits), and `certify_cmd`
        // warned about it at plan time.
        let Some(envelope) = envelope else {
            return true;
        };
        let r = probe.read(node);
        let park_c = envelope.chassis_park_c;
        let resume_c = envelope.chassis_resume_c;
        let why = |now_c: f64, throttled: bool| {
            if throttled {
                format!("the driver reports a thermal slowdown (chassis {now_c:.0} °C)")
            } else {
                format!("chassis {now_c:.0} °C (park at {park_c:.0}, resume at {resume_c:.0})")
            }
        };
        if ignore {
            match judge(r, self.warned_hot, &envelope) {
                Verdict::Park { now_c, throttled } => {
                    if !self.warned_hot {
                        self.warned_hot = true;
                        say(&format!(
                            "WARNING --dangerous-ignore-thermals: {} would be parked — {}; \
                             continuing on the operator's say-so — its records are still judged \
                             by the equivalence policy",
                            node.addr,
                            why(now_c, throttled)
                        ));
                    }
                }
                Verdict::Ready => {
                    if self.warned_hot {
                        self.warned_hot = false;
                        say(&format!(
                            "{} is back at or below {resume_c:.0} °C",
                            node.addr
                        ));
                    }
                }
                Verdict::Blind => {}
            }
            return true;
        }
        match judge(r, self.parked_since.is_some(), &envelope) {
            Verdict::Ready => {
                if let Some(since) = self.parked_since.take() {
                    say(&format!(
                        "cool-down: {} is back at or below {resume_c:.0} °C after {} s; resuming",
                        node.addr,
                        since.elapsed().as_secs()
                    ));
                }
                true
            }
            Verdict::Blind => {
                if !self.said_blind {
                    self.said_blind = true;
                    say(&format!(
                        "cool-down: {} reports no chassis temperature; it is never parked \
                         (the records' own captures still decide equivalence)",
                        node.addr
                    ));
                }
                self.parked_since = None;
                true
            }
            Verdict::Park { now_c, throttled } => {
                let since = *self.parked_since.get_or_insert_with(|| {
                    say(&format!(
                        "cool-down: {} parked — {}; nothing more until it is at or below \
                         {resume_c:.0} °C with no throttle — the other boxes keep working",
                        node.addr,
                        why(now_c, throttled)
                    ));
                    std::time::Instant::now()
                });
                if since.elapsed() >= MAX_PARK {
                    say(&format!(
                        "cool-down: {} still {} after {} s parked; resuming anyway — its records \
                         are judged by the equivalence policy like any other",
                        node.addr,
                        why(now_c, throttled),
                        since.elapsed().as_secs()
                    ));
                    self.parked_since = None;
                    return true;
                }
                false
            }
        }
    }
}

#[cfg(test)]
#[path = "thermal_tests.rs"]
mod thermal_tests;
