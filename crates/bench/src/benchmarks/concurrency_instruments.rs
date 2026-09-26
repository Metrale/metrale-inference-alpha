// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The concurrency sweep's per-rung instrument keys: both ITL
//! clocks, the arrival-gap (jitter) distribution, and the GPU-rail energy of
//! the measured batch. `c{C}_tpot_*` is the client clock and
//! `c{C}_server_tpot_*` the server clock (the `server_` prefix `quick_speed`
//! also uses); no separate `itl_*` key exists. A key gates nothing unless a
//! BENCH.toml bound names it.
//!
//! Owner: bench (concurrency).
//! Invariants: an unmeasured clock emits no key, never a zero.

use std::collections::BTreeMap;

use super::CellRow;
use crate::hardware::energy::EnergyWindow;

impl CellRow {
    /// 2026-09-26: The instrument keys for one rung, under `prefix` (`"c8_"`).
    /// Without `usage.decode_time_ms` from the server there is no
    /// `server_tpot` key.
    pub(super) fn instrument_metrics(
        &self,
        prefix: &str,
        idle: Option<&EnergyWindow>,
        m: &mut BTreeMap<String, f64>,
    ) {
        let pairs = [
            ("tpot_p50_ms", self.tpot.p50),
            ("tpot_p90_ms", self.tpot.p90),
            ("server_tpot_p50_ms", self.server_tpot.p50),
            ("server_tpot_p90_ms", self.server_tpot.p90),
        ];
        for (k, v) in pairs {
            if let Some(v) = v {
                m.insert(format!("{prefix}{k}"), v);
            }
        }
        if let Some(g) = &self.gaps {
            g.metrics(prefix, m);
        }
        if let Some(e) = &self.energy {
            e.metrics(prefix, self.tokens, idle, m);
        }
    }
}
