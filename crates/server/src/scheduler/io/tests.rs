// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The router contracts the scheduler loop's drain logic depends on, and the
//! seam guard: outside `SEAM_ALLOW`, no scheduler source line contains a
//! `SEAM_FORBIDDEN` pattern (clock reads, channel sends, the spill store,
//! metrics), except the `SEAM_COMPOSITION` pairs.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::time::Duration;

use super::{Arrivals, RequestIo, TokioRequestIo, WaitPolicy};
use crate::scheduler::types::PendingQueue;

fn inbox() -> (
    std::sync::Arc<TokioRequestIo>,
    tokio::sync::mpsc::Sender<crate::api::InferenceRequest>,
    tokio::sync::mpsc::Sender<crate::scheduler::LoraRotation>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let (rtx, rrx) = tokio::sync::mpsc::channel(8);
    (TokioRequestIo::new(rx, rrx), tx, rtx)
}

fn request() -> crate::api::InferenceRequest {
    crate::scheduler::test_support::blocking_request(None)
}

fn settle() {
    std::thread::sleep(Duration::from_millis(20));
}

#[test]
fn no_wait_hands_over_what_is_queued_and_nothing_else() {
    let (io, tx, _rtx) = inbox();
    assert!(io.recv(WaitPolicy::NoWait).requests.is_empty());
    tx.blocking_send(request()).unwrap();
    tx.blocking_send(request()).unwrap();
    settle();
    let got = io.recv(WaitPolicy::NoWait);
    assert_eq!(got.requests.len(), 2);
    assert!(!got.closed);
    assert!(
        io.recv(WaitPolicy::NoWait).requests.is_empty(),
        "a recv empties the inbox"
    );
}

#[test]
fn bounded_returns_empty_after_the_slice_and_early_on_arrival() {
    let (io, tx, _rtx) = inbox();
    let t0 = std::time::Instant::now();
    let got = io.recv(WaitPolicy::Bounded(Duration::from_millis(30)));
    assert!(got.requests.is_empty() && !got.closed);
    assert!(t0.elapsed() >= Duration::from_millis(30));

    let io2 = io.clone();
    let h = std::thread::spawn(move || io2.recv(WaitPolicy::Bounded(Duration::from_secs(5))));
    settle();
    tx.blocking_send(request()).unwrap();
    let t1 = std::time::Instant::now();
    let got = h.join().unwrap();
    assert_eq!(got.requests.len(), 1);
    assert!(
        t1.elapsed() < Duration::from_secs(1),
        "woken by the arrival, not the deadline"
    );
}

#[test]
fn block_wakes_on_a_rotation_alone_and_on_close() {
    let (io, tx, rtx) = inbox();
    let io2 = io.clone();
    let h = std::thread::spawn(move || io2.recv(WaitPolicy::Block));
    settle();
    let (ack, _rx) = tokio::sync::oneshot::channel();
    rtx.blocking_send((crate::scheduler::LoraCommand::Rotate("a".into()), ack))
        .unwrap();
    let got = h.join().unwrap();
    assert!(got.requests.is_empty());
    assert_eq!(got.rotations.len(), 1);
    assert!(!got.closed);

    drop(tx);
    let got = io.recv(WaitPolicy::Block);
    assert!(got.closed, "a closed channel ends a blocking wait");
    assert!(io.recv(WaitPolicy::Block).closed, "and stays closed");
}

#[test]
fn a_closed_inbox_never_waits() {
    let io = TokioRequestIo::closed();
    let t0 = std::time::Instant::now();
    assert!(io.recv(WaitPolicy::Block).closed);
    assert!(io.recv(WaitPolicy::Bounded(Duration::from_secs(5))).closed);
    assert!(t0.elapsed() < Duration::from_secs(1));
}

#[test]
fn the_pending_queue_keeps_arrival_order_and_latches_closed() {
    let mut q = PendingQueue::new();
    assert!(q.is_idle());
    q.absorb(Arrivals {
        requests: vec![request()],
        rotations: Vec::new(),
        closed: false,
    });
    assert!(!q.is_idle());
    q.requests.clear();
    q.absorb(Arrivals {
        requests: Vec::new(),
        rotations: Vec::new(),
        closed: true,
    });
    assert!(
        q.closed && !q.is_idle(),
        "closed is never idle: the loop must notice"
    );
    q.absorb(Arrivals::default());
    assert!(q.closed, "a later batch cannot un-close");
}

/// 2026-09-25: Files that may read the clock, touch a channel or a file, or publish a
/// metric: the routers, their implementations, the sink types and the tests.
const SEAM_ALLOW: &[&str] = &[
    "/io/",
    "/mod_helpers/send.rs",
    "/mtp_timing.rs",
    "/mtp_accept_debug.rs",
    "/dumps.rs",
    "/trace_harness/",
    "/test_support.rs",
    "_tests.rs",
    "/tests.rs",
];

/// 2026-09-25: A business file must reach these through `SchedIo` / `LogitsContext`.
const SEAM_FORBIDDEN: &[&str] = &[
    "Instant::now()",
    ".elapsed()",
    "blocking_send(",
    ".try_send(",
    "KvSpillManager",
    "crate::metrics::",
    "SnapshotCell",
    "spawn_terminal_send(",
    "bounded_stream_send(",
];

/// 2026-09-25: Where a router's input is built before being handed over.
const SEAM_COMPOSITION: &[(&str, &str)] = &[
    ("/core/mod.rs", "KvSpillManager"),
    ("/config.rs", "SnapshotCell"),
];

fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(dir).expect("scheduler dir") {
        let p = e.expect("entry").path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn no_scheduler_business_file_does_its_own_io() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/scheduler");
    let mut files = Vec::new();
    walk(&root, &mut files);
    assert!(files.len() > 50, "scan found {} files", files.len());
    let mut hits = Vec::new();
    for f in files {
        let rel = f.strip_prefix(&root).unwrap().to_string_lossy().to_string();
        let rel = format!("/{rel}");
        if SEAM_ALLOW.iter().any(|a| rel.contains(a)) {
            continue;
        }
        let src = std::fs::read_to_string(&f).unwrap();
        for (n, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            // 2026-09-25: The composition root builds the spill store it hands the
            // router; the config carries the snapshot cell it hands to it.
            if SEAM_COMPOSITION
                .iter()
                .any(|(f, t)| rel == *f && code.contains(t))
            {
                continue;
            }
            for tok in SEAM_FORBIDDEN {
                if code.contains(tok) {
                    hits.push(format!("{rel}:{}: {tok}", n + 1));
                }
            }
        }
    }
    assert!(
        hits.is_empty(),
        "raw I/O outside the routers:\n{}",
        hits.join("\n")
    );
}
