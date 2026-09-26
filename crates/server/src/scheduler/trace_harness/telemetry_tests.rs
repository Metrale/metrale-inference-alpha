// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Every scenario again with telemetry at `Basic`, fed into an instrument set of its own.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! Two claims. The trace (every model call and every byte a client got)
//! equals the golden recorded with telemetry `Off`. And the instruments
//! agree with what the clients received: the emitted-token tally equals
//! the output tokens the clients report (`watchdog_rollback`, which rolls
//! tokens back, may only tally more), one finished request per client,
//! and the verify scenarios record steps at the width they draft.

use metrale_telemetry::clock::MonotonicClock;
use metrale_telemetry::{Level, Telemetry, TelemetryConfig};

use super::runner::run_scenario;
use super::scenarios;
use super::tests::{SERIAL, golden_dir};

fn basic() -> &'static Telemetry {
    static CLOCK: MonotonicClock = MonotonicClock;
    let t: &'static Telemetry = Box::leak(Box::new(Telemetry::new(&CLOCK)));
    t.configure(&TelemetryConfig::serving(Level::Basic, 0));
    t
}

/// 2026-09-25: Output tokens a client line reports: `n=` of a stream's Done frame, or
/// the length of a blocking Response's token list.
fn reported_tokens(line: &str) -> u64 {
    if let Some(rest) = line.split("n=").nth(1) {
        return rest.split(',').next().unwrap().parse().unwrap();
    }
    let list = line
        .split("tokens=[")
        .nth(1)
        .and_then(|r| r.split(']').next())
        .unwrap_or_else(|| panic!("no Done or Response in {line:?}"));
    list.split(',').filter(|s| !s.trim().is_empty()).count() as u64
}

#[test]
fn basic_telemetry_leaves_every_trace_golden_and_counts_what_clients_got() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for mut sc in scenarios::all() {
        let t = basic();
        sc.opts.telemetry = t;
        let live = run_scenario(&sc).join("\n") + "\n";
        let golden =
            std::fs::read_to_string(golden_dir().join(format!("{}.trace", sc.name))).unwrap();
        assert!(
            live == golden,
            "{}: the trace moved under telemetry",
            sc.name
        );

        let outs: Vec<&str> = golden.lines().filter(|l| l.starts_with("out s")).collect();
        let reported: u64 = outs.iter().map(|l| reported_tokens(l)).sum();
        let tallied = t.tokens.get();
        if sc.name == "watchdog_rollback" {
            assert!(tallied >= reported, "{}: {tallied} < {reported}", sc.name);
        } else {
            assert_eq!(tallied, reported, "{}: token tally", sc.name);
        }
        assert_eq!(
            t.requests.finished.get(),
            outs.len() as u64,
            "{}: one finished request per client",
            sc.name
        );
        assert!(t.sched.ticks.get() > 0, "{}: ticks published", sc.name);
        assert!(t.sched.decode_rows.count() > 0, "{}: batch shape", sc.name);
        let drain = crate::scheduler::mtp_timing::Phase::LoopDrain as usize;
        assert!(
            t.sched.phase[drain].count() > 0,
            "{}: loop phases timed",
            sc.name
        );
        let width = match sc.name {
            "verify_k2" => Some(1),
            "verify_k3" => Some(2),
            "verify_k4" | "batched_verify_k4" => Some(3),
            _ => None,
        };
        if let Some(w) = width {
            assert!(
                t.spec.widths().contains(&w),
                "{}: no width-{w} verify step recorded (have {:?})",
                sc.name,
                t.spec.widths()
            );
        }
    }
}
