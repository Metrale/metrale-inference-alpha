// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `/v1/events` stream: one JSON object per SSE `data:` line
//! (JSONL); the server sends one per device-sample period.
//!
//! Schema (version [`EVENT_SCHEMA_VERSION`]); every key is always present,
//! and a quantity the device does not report is `null`, never 0:
//!
//! ```text
//! { "v": 1, "seq": <u64, per-stream>, "kind": "snapshot",
//!   "level": "off"|"basic"|"kernel", "at_ns": <telemetry clock>,
//!   "device_state": "not_started"|"available"|"unavailable",
//!   "device": null | { "power_mw", "energy_counter_mj", "graphics_clock_mhz",
//!       "sm_clock_mhz", "mem_clock_mhz", "temperature_c",
//!       "clocks_event_reasons", "mem_used_bytes", "mem_total_bytes",
//!       "pcie_tx_kbps", "pcie_rx_kbps", "nvlink_active_links",
//!       "samples", "age_ns", "read_errors" },
//!   "energy": { "gpu_mj_total", "gpu_counter_mj", "tokens_total",
//!       "joules_per_token_live", "joules_per_token_cumulative",
//!       "counter_resets", "counter_wraps" },
//!   "sched": { "pending", "active", "prefilling", "swapped",
//!       "kv_blocks_free", "kv_blocks_total", "ssm_slots_used",
//!       "ssm_slots_total", "ticks", "stream_syncs", "blocking_d2h",
//!       "graph_replays", "graph_captures" },
//!   "spec": [ { "drafts", "steps", "acceptance_by_depth": [f64] } ],
//!   "requests": { "finished", "ttft_mean_ms", "tpot_mean_ms", "e2e_mean_ms",
//!       "energy_mean_j", "energy_unattributed" } }
//! ```
//!
//! Owner: telemetry.
//! Invariants: an event line holds no newline.

use serde::Serialize;

use crate::snapshot::TelemetrySnapshot;

pub const EVENT_SCHEMA_VERSION: u32 = 1;

#[derive(Serialize)]
struct Event<'a> {
    v: u32,
    seq: u64,
    kind: &'static str,
    #[serde(flatten)]
    snap: &'a TelemetrySnapshot,
}

/// 2026-09-26: One event line, with no trailing newline (SSE framing adds it).
pub fn snapshot_line(seq: u64, snap: &TelemetrySnapshot) -> String {
    serde_json::to_string(&Event {
        v: EVENT_SCHEMA_VERSION,
        seq,
        kind: "snapshot",
        snap,
    })
    .expect("a snapshot is plain data and always serialises")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::MonotonicClock;
    use crate::hub::Telemetry;
    use crate::level::{Level, TelemetryConfig};

    #[test]
    fn a_line_is_one_json_object_carrying_every_schema_key() {
        static CLOCK: MonotonicClock = MonotonicClock;
        let t = Telemetry::new(&CLOCK);
        t.configure(&TelemetryConfig::serving(Level::Basic, 0));
        t.tokens(3);
        let line = snapshot_line(7, &t.snapshot());
        assert!(!line.contains('\n'), "one line per event");
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["v"], 1);
        assert_eq!(v["seq"], 7);
        assert_eq!(v["kind"], "snapshot");
        assert_eq!(v["level"], "basic");
        assert_eq!(v["device_state"], "not_started");
        assert!(v["device"].is_null(), "no sample yet is null, not zeros");
        assert!(v["energy"]["joules_per_token_live"].is_null());
        for k in ["pending", "kv_blocks_free", "graph_replays", "stream_syncs"] {
            assert!(v["sched"].get(k).is_some(), "sched.{k}");
        }
        for k in ["finished", "tpot_mean_ms", "energy_unattributed"] {
            assert!(v["requests"].get(k).is_some(), "requests.{k}");
        }
    }
}
