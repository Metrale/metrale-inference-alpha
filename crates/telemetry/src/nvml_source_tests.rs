// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `NvmlSource` over the scripted driver: every reported field
//! lands, an unsupported query is absent (not zero), a failing query is
//! counted, a missing ordinal fails at open; plus the reading's word-form
//! round trip.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use super::*;
use crate::nvml_script::{self, Script};

#[test]
fn every_reported_field_lands_in_the_reading() {
    nvml_script::install(Script {
        energy_mj: [42_000].into(),
        ..Script::default()
    });
    let mut src = NvmlSource::with_fns(nvml_script::table(), 0).unwrap();
    let r = src.read();
    assert_eq!(r.power_mw, Some(60_000));
    assert_eq!(r.energy_counter_mj, Some(42_000));
    assert_eq!(r.graphics_clock_mhz, Some(1000));
    assert_eq!(r.sm_clock_mhz, Some(1001));
    assert_eq!(r.mem_clock_mhz, Some(1002));
    assert_eq!(r.temperature_c, Some(55));
    assert_eq!(r.clocks_event_reasons, Some(0x4));
    assert_eq!(r.mem_used_bytes, Some(2 << 30));
    assert_eq!(r.mem_total_bytes, Some(8 << 30));
    assert_eq!(r.pcie_tx_kbps, None, "an unresolved entry point is absent");
    assert_eq!(r.nvlink_active_links, None);
    assert_eq!(src.read_errors(), 0);
}

#[test]
fn unsupported_memory_is_absent_and_a_failing_query_is_counted() {
    nvml_script::install(Script {
        memory_supported: false,
        temperature_rc: 999,
        ..Script::default()
    });
    let mut src = NvmlSource::with_fns(nvml_script::table(), 0).unwrap();
    let r = src.read();
    assert_eq!(r.mem_used_bytes, None);
    assert_eq!(r.mem_total_bytes, None);
    assert_eq!(r.temperature_c, None, "a failed read is None, not 0");
    assert_eq!(
        src.read_errors(),
        1,
        "only the non-NOT_SUPPORTED failure counts"
    );
    assert_eq!(
        r.power_mw,
        Some(60_000),
        "one failure does not blank the rest"
    );
}

#[test]
fn a_device_ordinal_the_driver_lacks_is_unavailable_at_open() {
    nvml_script::install(Script::default());
    let err = NvmlSource::with_fns(nvml_script::table(), 5)
        .err()
        .expect("ordinal 5 does not exist");
    assert!(matches!(
        err,
        metrale_gpu_sys::nvml::NvmlUnavailable::NoDevice { index: 5, .. }
    ));
}

#[test]
fn a_reading_round_trips_through_its_word_form() {
    let r = DeviceReading {
        power_mw: Some(1),
        energy_counter_mj: Some(u64::MAX),
        temperature_c: Some(0),
        ..DeviceReading::default()
    };
    let (back, at) = DeviceReading::decode(&r.encode(77));
    assert_eq!(back, r, "Some(0) stays Some(0), absent stays absent");
    assert_eq!(at, 77);
}
