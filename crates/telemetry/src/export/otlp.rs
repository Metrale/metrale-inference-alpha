// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The OTLP/HTTP metrics body in the JSON encoding, which the
//! server's pusher POSTs to `<endpoint>/v1/metrics` as `application/json`.
//!
//! Behind the `otlp` cargo feature. 64-bit integers (`asInt`, `*UnixNano`)
//! are JSON strings, and every sum is cumulative (`aggregationTemporality`
//! 2) and monotonic.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use serde_json::{Value, json};

use crate::snapshot::TelemetrySnapshot;

fn gauge(name: &str, unit: &str, now: &str, v: f64) -> Value {
    json!({"name": name, "unit": unit,
           "gauge": {"dataPoints": [{"timeUnixNano": now, "asDouble": v}]}})
}

fn sum(name: &str, unit: &str, start: &str, now: &str, v: u64) -> Value {
    json!({"name": name, "unit": unit,
           "sum": {"aggregationTemporality": 2, "isMonotonic": true,
                   "dataPoints": [{"startTimeUnixNano": start, "timeUnixNano": now,
                                   "asInt": v.to_string()}]}})
}

/// 2026-09-26: The `ExportMetricsServiceRequest` for one snapshot.
/// `start_unix_ns` is when the cumulative series began. A J/token, request
/// or device gauge whose value is `None` is left out.
pub fn metrics_body(snap: &TelemetrySnapshot, start_unix_ns: u64, now_unix_ns: u64) -> String {
    let (start, now) = (start_unix_ns.to_string(), now_unix_ns.to_string());
    let e = &snap.energy;
    let s = &snap.sched;
    let mut metrics = vec![
        sum("metrale.gpu.energy", "mJ", &start, &now, e.gpu_mj_total),
        sum(
            "metrale.energy.tokens",
            "{token}",
            &start,
            &now,
            e.tokens_total,
        ),
        sum(
            "metrale.requests.finished",
            "{request}",
            &start,
            &now,
            snap.requests.finished,
        ),
        sum(
            "metrale.gpu.stream_syncs",
            "{sync}",
            &start,
            &now,
            s.stream_syncs,
        ),
        sum(
            "metrale.gpu.blocking_d2h",
            "{copy}",
            &start,
            &now,
            s.blocking_d2h,
        ),
        sum(
            "metrale.cuda_graph.replays",
            "{replay}",
            &start,
            &now,
            s.graph_replays,
        ),
        sum(
            "metrale.cuda_graph.captures",
            "{capture}",
            &start,
            &now,
            s.graph_captures,
        ),
        gauge(
            "metrale.sched.active",
            "{sequence}",
            &now,
            s.shape.active as f64,
        ),
        gauge(
            "metrale.sched.pending",
            "{request}",
            &now,
            s.shape.pending as f64,
        ),
        gauge(
            "metrale.kv.blocks_free",
            "{block}",
            &now,
            s.shape.kv_blocks_free as f64,
        ),
        gauge(
            "metrale.kv.blocks_total",
            "{block}",
            &now,
            s.shape.kv_blocks_total as f64,
        ),
    ];
    let optional = [
        (
            "metrale.energy.joules_per_token",
            "J/{token}",
            e.joules_per_token_live,
        ),
        ("metrale.request.tpot", "ms", snap.requests.tpot_mean_ms),
        ("metrale.request.e2e", "ms", snap.requests.e2e_mean_ms),
    ];
    for (name, unit, v) in optional {
        if let Some(v) = v {
            metrics.push(gauge(name, unit, &now, v));
        }
    }
    if let Some(d) = &snap.device {
        let r = &d.reading;
        let device = [
            ("metrale.gpu.power", "W", r.power_mw.map(|v| v as f64 / 1e3)),
            (
                "metrale.gpu.temperature",
                "Cel",
                r.temperature_c.map(|v| v as f64),
            ),
            (
                "metrale.gpu.clock.sm",
                "MHz",
                r.sm_clock_mhz.map(|v| v as f64),
            ),
        ];
        for (name, unit, v) in device {
            if let Some(v) = v {
                metrics.push(gauge(name, unit, &now, v));
            }
        }
    }
    json!({"resourceMetrics": [{
        "resource": {"attributes": [
            {"key": "service.name", "value": {"stringValue": "metrale-engine"}}]},
        "scopeMetrics": [{
            "scope": {"name": "metrale-telemetry", "version": env!("CARGO_PKG_VERSION")},
            "metrics": metrics}]}]})
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::MonotonicClock;
    use crate::hub::Telemetry;
    use crate::level::{Level, TelemetryConfig};

    #[test]
    fn the_body_encodes_int64_as_strings_and_cumulative_sums() {
        static CLOCK: MonotonicClock = MonotonicClock;
        let t = Telemetry::new(&CLOCK);
        t.configure(&TelemetryConfig::serving(Level::Basic, 0));
        t.tokens(5);
        let body = metrics_body(&t.snapshot(), 10, 20);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let metrics = v["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
            .as_array()
            .unwrap();
        let energy = metrics
            .iter()
            .find(|m| m["name"] == "metrale.gpu.energy")
            .unwrap();
        let p = &energy["sum"]["dataPoints"][0];
        assert_eq!(energy["sum"]["aggregationTemporality"], 2);
        assert_eq!(p["asInt"], "0", "int64 is a JSON string");
        assert_eq!(p["startTimeUnixNano"], "10");
        assert_eq!(p["timeUnixNano"], "20");
        assert!(
            metrics.iter().all(|m| m["name"] != "metrale.gpu.power"),
            "no device sample, no power gauge"
        );
    }
}
