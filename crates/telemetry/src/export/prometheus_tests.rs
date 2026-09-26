// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the telemetry `/metrics` text: every sample line is a
//! declared metric with a numeric value, histograms are cumulative, and level
//! `Kernel` exports lanes and escaped kernel names.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use super::*;
use crate::clock::MonotonicClock;
use crate::device::DeviceReading;
use crate::energy::CounterTracker;
use crate::level::TelemetryConfig;
use crate::request::RequestTiming;

fn hub(level: Level) -> &'static Telemetry {
    static CLOCK: MonotonicClock = MonotonicClock;
    let t: &'static Telemetry = Box::leak(Box::new(Telemetry::new(&CLOCK)));
    t.configure(&TelemetryConfig::serving(level, 0));
    t
}

/// 2026-09-26: Every sample line, as `name{labels} value`.
fn samples(text: &str) -> Vec<&str> {
    text.lines().filter(|l| !l.starts_with('#')).collect()
}

#[test]
fn every_sample_line_is_a_declared_metric_with_a_numeric_value() {
    let t = hub(Level::Basic);
    let mut tr = CounterTracker::default();
    let reading = DeviceReading {
        power_mw: Some(61_500),
        energy_counter_mj: Some(1_000),
        clocks_event_reasons: Some(0x4 | 0x80),
        ..Default::default()
    };
    t.absorb_device(&mut tr, &reading, 0);
    t.tokens(4);
    t.absorb_device(
        &mut tr,
        &DeviceReading {
            energy_counter_mj: Some(3_000),
            ..reading
        },
        0,
    );
    t.request_finished(&RequestTiming {
        ttft_ms: 5.0,
        decode_ms: 30.0,
        output_tokens: 4,
    });
    let text = render(t, &|_| None);
    let declared: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("# TYPE "))
        .map(|l| l.split(' ').next().unwrap())
        .collect();
    for line in samples(&text) {
        let (series, value) = line.rsplit_once(' ').unwrap();
        let name = series.split('{').next().unwrap();
        let base = ["_bucket", "_sum", "_count"]
            .iter()
            .find_map(|s| name.strip_suffix(s).filter(|b| declared.contains(b)))
            .unwrap_or(name);
        assert!(declared.contains(&base), "undeclared series {name}");
        assert!(
            value.parse::<f64>().is_ok(),
            "non-numeric value in {line:?}"
        );
    }
    assert!(text.contains("metrale_gpu_power_watts 61.5\n"));
    assert!(text.contains("metrale_gpu_energy_millijoules_total 2000\n"));
    assert!(text.contains("metrale_gpu_energy_counter_millijoules 3000\n"));
    assert!(text.contains("metrale_energy_tokens_total 4\n"));
    assert!(text.contains("metrale_energy_joules_per_token 0.5\n"));
    assert!(text.contains("metrale_gpu_clocks_event_reason_active{reason=\"sw_power_cap\"} 1\n"));
    assert!(text.contains(
        "metrale_gpu_clocks_event_reason_active{reason=\"hw_power_brake_slowdown\"} 1\n"
    ));
    assert!(text.contains("metrale_gpu_clocks_event_reason_active{reason=\"gpu_idle\"} 0\n"));
    assert!(text.contains("metrale_telemetry_level{level=\"basic\"} 1\n"));
    assert!(
        !text.contains("metrale_gpu_lane_seconds"),
        "L1 is Kernel-only"
    );
    assert!(
        !text.contains("metrale_gpu_temperature_celsius"),
        "unreported is absent"
    );
}

#[test]
fn a_histogram_is_cumulative_and_its_inf_bucket_is_its_count() {
    let t = hub(Level::Basic);
    for (d, n) in [(10.0, 3), (1_000.0, 2), (1e9, 2)] {
        t.request_finished(&RequestTiming {
            ttft_ms: 0.0,
            decode_ms: d,
            output_tokens: n,
        });
    }
    let text = render(t, &|_| None);
    let buckets: Vec<u64> = samples(&text)
        .into_iter()
        .filter(|l| l.starts_with("metrale_request_e2e_seconds_bucket"))
        .map(|l| l.rsplit_once(' ').unwrap().1.parse().unwrap())
        .collect();
    assert!(
        buckets.windows(2).all(|w| w[1] >= w[0]),
        "cumulative: {buckets:?}"
    );
    assert_eq!(*buckets.last().unwrap(), 3, "+Inf is the count");
    assert_eq!(
        buckets[buckets.len() - 2],
        2,
        "a 1e6-second request is past every bound"
    );
    assert!(text.contains("metrale_request_e2e_seconds_count 3\n"));
}

#[test]
fn kernel_level_exports_lanes_and_named_kernels_with_escaped_labels() {
    let t = hub(Level::Kernel);
    t.kernel.record(crate::kernel::SpanKey::Lane(2), 2_000_000);
    t.kernel.record(crate::kernel::SpanKey::Kernel(0x40), 1_500);
    let text = render(t, &|h| (h == 0x40).then(|| "mod::\"q\"".to_string()));
    assert!(text.contains("metrale_gpu_lane_seconds_count{lane=\"decode\"} 1\n"));
    assert!(
        text.contains("metrale_gpu_kernel_seconds_total{kernel=\"mod::\\\"q\\\"\"} 0.0000015\n"),
        "{text}"
    );
    assert!(text.contains("metrale_gpu_kernel_samples_total{kernel=\"mod::\\\"q\\\"\"} 1\n"));
}
