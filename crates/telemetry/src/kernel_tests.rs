// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The span ring over a scripted event source, and the per-kernel
//! table.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use std::sync::Mutex;

use super::*;

#[derive(Default)]
struct Scripted {
    /// 2026-09-26: (slot, edge, stream) in record order.
    recorded: Mutex<Vec<(usize, Edge, u64)>>,
    fail_record: Mutex<bool>,
    /// 2026-09-26: What `poll` answers per slot.
    polls: Mutex<Vec<SpanPoll>>,
}

impl EventOps for Scripted {
    fn record(&self, slot: usize, edge: Edge, stream: u64) -> bool {
        if *self.fail_record.lock().unwrap() {
            return false;
        }
        self.recorded.lock().unwrap().push((slot, edge, stream));
        true
    }
    fn poll(&self, slot: usize) -> SpanPoll {
        self.polls.lock().unwrap()[slot]
    }
}

fn ring(slots: usize) -> SpanRing<Scripted> {
    let ops = Scripted {
        polls: Mutex::new(vec![SpanPoll::Pending; slots]),
        ..Default::default()
    };
    SpanRing::new(ops, slots)
}

#[test]
fn a_completed_span_lands_once_and_a_pending_one_waits() {
    let sink = KernelInstruments::new();
    let r = ring(4);
    let a = r.begin(SpanKey::Lane(2), 7, &sink).unwrap();
    r.end(a, 7, &sink);
    let b = r.begin(SpanKey::Kernel(0xabc0), 7, &sink).unwrap();
    r.end(b, 7, &sink);
    assert_eq!(
        *r.ops().recorded.lock().unwrap(),
        vec![
            (0, Edge::Start, 7),
            (0, Edge::End, 7),
            (1, Edge::Start, 7),
            (1, Edge::End, 7)
        ]
    );

    r.ops().polls.lock().unwrap()[0] = SpanPoll::Ready(5_000);
    assert_eq!(r.harvest(&sink), 1, "only the completed span lands");
    assert_eq!(sink.lane[2].count(), 1);
    assert!(
        sink.kernels().is_empty(),
        "the kernel span is still pending"
    );

    r.ops().polls.lock().unwrap()[1] = SpanPoll::Ready(700);
    assert_eq!(r.harvest(&sink), 1);
    assert_eq!(r.harvest(&sink), 0, "a harvested span is not landed twice");
    assert_eq!(
        sink.kernels(),
        vec![KernelTime {
            func: 0xabc0,
            total_ns: 700,
            samples: 1
        }]
    );
}

#[test]
fn a_full_ring_drops_and_counts_instead_of_waiting() {
    let sink = KernelInstruments::new();
    let r = ring(2);
    let a = r.begin(SpanKey::Lane(0), 0, &sink).unwrap();
    let _b = r.begin(SpanKey::Lane(1), 0, &sink).unwrap();
    assert!(
        r.begin(SpanKey::Lane(0), 0, &sink).is_none(),
        "both slots in flight"
    );
    assert_eq!(sink.spans_dropped.get(), 1);
    r.end(a, 0, &sink);
    r.ops().polls.lock().unwrap()[0] = SpanPoll::Ready(1);
    r.harvest(&sink);
    // 2026-09-26: Slot 0 is free again. The next begin lands on slot 1, still
    // open; the one after reaches slot 0.
    let _ = r.begin(SpanKey::Lane(0), 0, &sink);
    assert!(r.begin(SpanKey::Lane(0), 0, &sink).is_some());
}

#[test]
fn failed_records_and_unreadable_events_free_the_slot_and_count() {
    let sink = KernelInstruments::new();
    let r = ring(1);
    *r.ops().fail_record.lock().unwrap() = true;
    assert!(r.begin(SpanKey::Lane(0), 0, &sink).is_none());
    assert_eq!(sink.span_failures.get(), 1);
    *r.ops().fail_record.lock().unwrap() = false;
    let t = r
        .begin(SpanKey::Lane(0), 0, &sink)
        .expect("the slot was freed");
    r.end(t, 0, &sink);
    r.ops().polls.lock().unwrap()[0] = SpanPoll::Failed;
    assert_eq!(r.harvest(&sink), 0);
    assert_eq!(sink.span_failures.get(), 2);
    assert!(r.begin(SpanKey::Lane(0), 0, &sink).is_some(), "freed again");
}

#[test]
fn the_kernel_table_accumulates_per_handle_and_reports_overflow() {
    let sink = KernelInstruments::new();
    sink.record(SpanKey::Kernel(0x1000), 10);
    sink.record(SpanKey::Kernel(0x1000), 30);
    sink.record(SpanKey::Kernel(0x2000), 5);
    let mut k = sink.kernels();
    k.sort_by_key(|k| k.func);
    assert_eq!(
        k[0],
        KernelTime {
            func: 0x1000,
            total_ns: 40,
            samples: 2
        }
    );
    assert_eq!(k[1].samples, 1);
    for f in 1..=(KERNEL_TABLE_SLOTS as u64 + 5) {
        sink.record(SpanKey::Kernel(0x10_0000 + f * 16), 1);
    }
    assert!(
        sink.table_full.get() >= 5,
        "handles past capacity are counted, not dropped silently"
    );
}

#[cfg(feature = "nvtx")]
#[test]
fn the_nvtx_range_names_are_the_lane_names() {
    let names: Vec<&str> = LANES_NVTX.iter().map(|c| c.to_str().unwrap()).collect();
    assert_eq!(names, LANES);
}
