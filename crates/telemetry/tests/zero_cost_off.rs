// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: At level `Off`, no hot-path entry point of `Telemetry` reads
//! the clock or allocates.
//!
//! A test binary of its own because it installs a global allocator, which
//! counts per thread, so the harness's other threads cannot move this test's
//! count. On these paths `Telemetry` reads time only through its injected
//! [`Clock`]. The `Basic` test is the control: the same calls read the clock
//! there.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use metrale_telemetry::clock::Clock;
use metrale_telemetry::request::RequestTiming;
use metrale_telemetry::sched::SchedShape;
use metrale_telemetry::{Level, Telemetry, TelemetryConfig};

struct CountingAlloc;

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

fn bump() {
    // 2026-09-26: `try_with`: the TLS slot may already be gone during thread teardown.
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

// 2026-09-26: SAFETY: forwards every call to the system allocator unchanged.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}

struct CountingClock(AtomicU64);

impl Clock for CountingClock {
    fn now_ns(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed);
        1
    }
    fn elapsed_ns(&self, _: Instant) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed);
        1
    }
}

/// 2026-09-26: Every hot-path entry point, `n` times.
fn drive(t: &Telemetry, since: Instant, n: usize) {
    let shape = SchedShape {
        active: 4,
        pending: 1,
        ..SchedShape::default()
    };
    let timing = RequestTiming {
        ttft_ms: 12.0,
        decode_ms: 340.0,
        output_tokens: 17,
    };
    for i in 0..n {
        t.tokens(1);
        t.phase(i % 8, since);
        t.sched_tick(&shape);
        t.stream_sync();
        t.blocking_d2h();
        t.graph_replay();
        t.graph_capture();
        t.spec_verified(3, i % 4);
        t.request_finished(&timing);
        std::hint::black_box(t.step_begin());
        std::hint::black_box(t.kernel_spans_armed());
        std::hint::black_box(t.lane_spans());
    }
}

#[test]
fn the_off_path_reads_no_clock_and_allocates_nothing() {
    static CLOCK: CountingClock = CountingClock(AtomicU64::new(0));
    static T: Telemetry = Telemetry::new(&CLOCK);
    assert_eq!(T.level(), Level::Off, "a new instrument set starts Off");
    let since = Instant::now();

    let a0 = allocs();
    drive(&T, since, 10_000);
    let a1 = allocs();

    assert_eq!(CLOCK.0.load(Ordering::Relaxed), 0, "Off read the clock");
    assert_eq!(a1 - a0, 0, "Off allocated");
    assert_eq!(T.tokens.get(), 0, "and recorded nothing");
    assert_eq!(T.requests.finished.get(), 0);
}

#[test]
fn the_same_calls_at_basic_read_the_clock_and_still_never_allocate() {
    static CLOCK: CountingClock = CountingClock(AtomicU64::new(0));
    static T: Telemetry = Telemetry::new(&CLOCK);
    // 2026-09-26: Configuring allocates the energy ring, once, before counting.
    T.configure(&TelemetryConfig::serving(Level::Basic, 0));
    let since = Instant::now();

    let a0 = allocs();
    drive(&T, since, 10_000);
    let a1 = allocs();

    // 2026-09-26: Per iteration, `phase` reads it once and `request_finished` once.
    assert_eq!(CLOCK.0.load(Ordering::Relaxed), 20_000, "the control moves");
    assert_eq!(a1 - a0, 0, "the Basic hot path allocates nothing either");
    assert_eq!(T.tokens.get(), 10_000);
    assert_eq!(T.requests.finished.get(), 10_000);
}

#[test]
fn the_counting_allocator_sees_this_threads_allocations() {
    let a0 = allocs();
    let v = std::hint::black_box(vec![0u8; 64]);
    assert!(allocs() > a0, "a Vec allocation is counted");
    drop(v);
}
