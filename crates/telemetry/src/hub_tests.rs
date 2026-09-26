// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The hub end to end over the scripted NVML driver, plus the
//! counter's wrap and reset, a missing driver, the `Off` level and racing
//! readers.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::device::{DeviceSource, NvmlSource};
use crate::nvml_script::{self, Script};

/// 2026-09-26: `now_ns` is set by hand; `elapsed_ns` is always 1 µs.
struct ManualClock(AtomicU64);

impl Clock for ManualClock {
    fn now_ns(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
    fn elapsed_ns(&self, _since: Instant) -> u64 {
        1_000
    }
}

fn hub(level: Level) -> (&'static Telemetry, &'static ManualClock) {
    let clock: &'static ManualClock = Box::leak(Box::new(ManualClock(AtomicU64::new(0))));
    let t: &'static Telemetry = Box::leak(Box::new(Telemetry::new(clock)));
    t.configure(&TelemetryConfig::serving(level, 0));
    (t, clock)
}

const SEC: u64 = 1_000_000_000;

/// 2026-09-26: A known counter delta over a known token count, read through
/// the production NVML wrapper, yields the exact J/token: cumulative, live
/// and per request.
#[test]
fn a_scripted_counter_delta_yields_the_exact_joules_per_token() {
    nvml_script::install(Script {
        energy_mj: [1_000_000, 1_050_000].into(),
        ..Script::default()
    });
    let (t, clock) = hub(Level::Basic);
    let mut src = NvmlSource::with_fns(nvml_script::table(), 0).unwrap();
    let mut tracker = CounterTracker::default();

    clock.0.store(SEC, Ordering::Relaxed);
    t.absorb_device(&mut tracker, &src.read(), src.read_errors());
    t.tokens(100);
    clock.0.store(2 * SEC, Ordering::Relaxed);
    t.absorb_device(&mut tracker, &src.read(), src.read_errors());

    assert_eq!(t.energy_mj.get(), 50_000, "50 J over the window");
    assert_eq!(t.tokens_since_first_sample(), 100);
    assert_eq!(t.joules_per_token_cumulative(), Some(0.5));
    assert_eq!(t.joules_per_token_live.get(), Some(0.5));
    assert_eq!(
        t.energy_counter_raw.get(),
        1_050_000,
        "the raw counter is kept"
    );

    // 2026-09-26: A request that emitted 40 of the window's 100 tokens over
    // exactly that window is charged 40 × 0.5 J.
    t.request_finished(&RequestTiming {
        ttft_ms: 200.0,
        decode_ms: 800.0,
        output_tokens: 40,
    });
    let e = t.requests.energy.snapshot();
    assert_eq!((e.count, e.sum), (1, 20_000));
    assert_eq!(t.requests.energy_unattributed.get(), 0);
}

/// 2026-09-26: Tokens counted before the first energy sample are not charged
/// to the energy that sample starts counting.
#[test]
fn tokens_before_the_first_sample_are_outside_the_denominator() {
    nvml_script::install(Script {
        energy_mj: [10, 4_010].into(),
        ..Script::default()
    });
    let (t, _) = hub(Level::Basic);
    let mut src = NvmlSource::with_fns(nvml_script::table(), 0).unwrap();
    let mut tracker = CounterTracker::default();
    t.tokens(1_000);
    t.absorb_device(&mut tracker, &src.read(), 0);
    t.tokens(8);
    t.absorb_device(&mut tracker, &src.read(), 0);
    assert_eq!(t.joules_per_token_cumulative(), Some(0.5), "4 J / 8 tokens");
}

/// 2026-09-26: A wrap is charged the elapsed energy, a reset the energy since
/// the restart; each is counted, and the total never goes backwards.
#[test]
fn a_counter_wrap_and_a_reset_are_both_charged_and_counted() {
    nvml_script::install(Script {
        energy_mj: [u64::MAX - 500, 700, 5_000_000, 3_000].into(),
        ..Script::default()
    });
    let (t, _) = hub(Level::Basic);
    let mut src = NvmlSource::with_fns(nvml_script::table(), 0).unwrap();
    let mut tracker = CounterTracker::default();
    let mut totals = Vec::new();
    for _ in 0..4 {
        t.absorb_device(&mut tracker, &src.read(), 0);
        totals.push(t.energy_mj.get());
    }
    // 2026-09-26: The wrap is 500 to reach MAX, 1 to reach 0 and 700 more:
    // 1201. Then +(5_000_000 - 700), then a reset charging 3_000.
    assert_eq!(
        totals,
        vec![0, 1_201, 1_201 + 4_999_300, 1_201 + 4_999_300 + 3_000]
    );
    assert_eq!(t.energy_wraps.get(), 1);
    assert_eq!(t.energy_resets.get(), 1);
    assert!(totals.windows(2).all(|w| w[1] >= w[0]));
}

/// 2026-09-26: NVML missing, through the real loader and the real start path:
/// the reason is reported, requests are still measured, and none is charged
/// energy.
#[test]
fn a_missing_driver_is_reported_and_requests_stay_unattributed() {
    let (t, clock) = hub(Level::Off);
    let r = crate::sampler::start_with(t, &TelemetryConfig::serving(Level::Basic, 0), |_| {
        metrale_gpu_sys::nvml::NvmlLibrary::open_from(&["/nonexistent/libnvidia-ml.so.1"])
            .map(|_| -> Box<dyn DeviceSource> { unreachable!("the path does not exist") })
    });
    let why = r.err().expect("no library, no sampler");
    assert!(why.contains("not loadable"), "{why}");
    assert_eq!(t.device_state(), DeviceState::Unavailable);
    assert_eq!(t.device_unavailable_reason(), Some(why.as_str()));

    clock.0.store(5 * SEC, Ordering::Relaxed);
    t.tokens(10);
    t.request_finished(&RequestTiming {
        ttft_ms: 10.0,
        decode_ms: 90.0,
        output_tokens: 10,
    });
    assert_eq!(
        t.requests.finished.get(),
        1,
        "the request itself is still measured"
    );
    assert_eq!(t.requests.energy_unattributed.get(), 1);
    assert_eq!(t.joules_per_token_cumulative(), None, "no energy, no ratio");
    let text = crate::export::prometheus::render(t, &|_| None);
    assert!(text.contains("metrale_gpu_device_available 0"), "{text}");
    assert!(
        !text.contains("metrale_energy_joules_per_token "),
        "an unmeasured ratio is absent"
    );
}

/// 2026-09-26: At `Off` nothing records and nothing exports.
#[test]
fn level_off_records_nothing_and_exports_nothing() {
    let (t, _) = hub(Level::Off);
    t.tokens(5);
    t.spec_verified(3, 2);
    t.stream_sync();
    t.request_finished(&RequestTiming {
        ttft_ms: 1.0,
        decode_ms: 1.0,
        output_tokens: 2,
    });
    assert_eq!(t.tokens.get(), 0);
    assert!(t.spec.widths().is_empty());
    assert_eq!(t.sched.stream_syncs.get(), 0);
    assert_eq!(t.requests.finished.get(), 0);
    assert!(!t.step_begin(), "no kernel spans below level Kernel");
    assert_eq!(crate::export::prometheus::render(t, &|_| None), "");
}

/// 2026-09-26: Readers racing the sampler never see two samples mixed.
#[test]
fn readers_racing_the_sampler_see_whole_samples() {
    struct Uniform(u64);
    impl DeviceSource for Uniform {
        fn read(&mut self) -> DeviceReading {
            self.0 += 1;
            let v = Some(self.0);
            DeviceReading {
                power_mw: v,
                energy_counter_mj: v,
                graphics_clock_mhz: v,
                sm_clock_mhz: v,
                mem_clock_mhz: v,
                temperature_c: v,
                clocks_event_reasons: v,
                mem_used_bytes: v,
                mem_total_bytes: v,
                pcie_tx_kbps: v,
                pcie_rx_kbps: v,
                nvlink_active_links: v,
            }
        }
        fn read_errors(&self) -> u64 {
            0
        }
    }
    let (t, _) = hub(Level::Basic);
    let writer = std::thread::spawn(move || {
        let mut src = Uniform(0);
        let mut tracker = CounterTracker::default();
        for _ in 0..50_000 {
            t.absorb_device(&mut tracker, &src.read(), 0);
        }
    });
    let mut seen = 0u64;
    while !writer.is_finished() {
        if let Some((r, _, _)) = t.device_reading() {
            let v = r.power_mw;
            assert!(
                [
                    r.energy_counter_mj,
                    r.sm_clock_mhz,
                    r.temperature_c,
                    r.nvlink_active_links
                ]
                .iter()
                .all(|x| *x == v),
                "torn sample: {r:?}"
            );
            seen += 1;
        }
    }
    writer.join().unwrap();
    assert!(seen > 0);
    assert_eq!(
        t.energy_mj.get(),
        49_999,
        "one mJ per sample after the first"
    );
}

#[test]
fn kernel_level_arms_per_kernel_spans_on_one_step_in_n() {
    let clock: &'static ManualClock = Box::leak(Box::new(ManualClock(AtomicU64::new(0))));
    let t: &'static Telemetry = Box::leak(Box::new(Telemetry::new(clock)));
    t.configure(
        &TelemetryConfig::new(Level::Kernel, crate::level::DEVICE_SAMPLE_PERIOD, 4, 0).unwrap(),
    );
    let armed: Vec<bool> = (0..9).map(|_| t.step_begin()).collect();
    assert_eq!(
        armed,
        vec![true, false, false, false, true, false, false, false, true]
    );
    assert!(t.kernel_spans_armed(), "the ninth step is armed");
    t.configure(&TelemetryConfig::serving(Level::Basic, 0));
    assert!(!t.kernel_spans_armed(), "leaving Kernel disarms");
}
