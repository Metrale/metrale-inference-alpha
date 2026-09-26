// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batch-width tests for the MTP gate: plain decode steps are
//! charged the active width, and a width-bucket change stales both estimates.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: A plain decode step over n sequences is charged n tokens.
#[test]
fn decode_steps_charge_the_batch_width() {
    let mut g = MtpGate::new(1);
    run_mtp_until_probe(&mut g, 2, ms(50));
    // 2026-09-25: Serial probe at width 4: 4 tokens per 10 ms step, 400 tok/s.
    // The width-1 to width-4 change lands on the probe's still-empty window,
    // so all `WINDOW_STEPS` steps are measured in the new bucket.
    for _ in 0..WINDOW_STEPS {
        assert_eq!(g.next_step(), GateStep::MeasureDecode);
        g.record_decode(ms(10), 4);
    }
    let serial = g.serial_tps_debug().expect("probe closed a window");
    assert!(
        (serial - 400.0).abs() < 1.0,
        "serial EWMA must read the delivered 400 tok/s, not the pre-fix \
         one-token 100 tok/s (got {serial:.1})"
    );
}

/// 2026-09-26: Drain-tail graph borrowing (metrale-model-engine
/// `model/trait_impl/graph_borrow.rs`) can replay a wider captured CUDA graph
/// for a shrinking batch, so a step's wall includes padding lanes. The gate
/// must still be charged the tokens delivered: every `gate.record_decode` call
/// in a non-test file under the server crate's `src/scheduler/` must pass
/// `active.len()`, which this test reads from the source. The decode lane's
/// call in `scheduler/core/lane_decode.rs` is checked in full.
#[test]
fn arbiter_charges_active_width_never_a_padded_width() {
    let scheduler =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../server/src/scheduler");
    let src = std::fs::read_to_string(scheduler.join("core/lane_decode.rs")).unwrap();
    let flat: String = src.split_whitespace().collect();
    assert!(
        flat.contains(
            "gate.record_decode(sched.io.clock.now().saturating_duration_since(t0),active.len(),)"
        ),
        "decode steps must be charged at the active batch width"
    );
    let mut calls = 0usize;
    let mut pending = vec![scheduler];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let path = entry.expect("readable scheduler entry").path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            if !name.ends_with(".rs") || name.ends_with("tests.rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            for call in text.split("gate.record_decode(").skip(1) {
                calls += 1;
                let args = call.split(';').next().unwrap_or("");
                assert!(
                    args.contains("active.len()"),
                    "every decode charge must use active.len(), got \
                     `record_decode({args})` in {}",
                    path.display()
                );
            }
        }
    }
    assert!(calls >= 1, "the scan found no `gate.record_decode(` call");
}

/// 2026-09-25: A width-bucket change discards the partial window and makes a
/// probe due, as a depth-regime change does. The first step in the new bucket
/// closes a one-step window that replaces the current mode's estimate, and the
/// next step probes the other mode at the new width.
#[test]
fn width_regime_change_stales_and_remeasures_in_the_new_regime() {
    let mut g = MtpGate::new(1);
    // 2026-09-25: Both estimates at width 1; the gate stays Mtp (40 against
    // 25 tok/s).
    run_mtp_until_probe(&mut g, 2, ms(50));
    drive_serial(&mut g, WINDOW_STEPS, ms(40));
    assert!(!g.in_serial_mode());
    // 2026-09-25: Half a window at width 1, then the batch widens into bucket
    // 8.
    drive_mtp(&mut g, WINDOW_STEPS / 2, 2, ms(50));
    g.record_verify_step(ms(5), 6, 8);
    assert!(
        g.serial.stale,
        "off-mode baseline is stale after the change"
    );
    let mtp = g.mtp_tps_debug().expect("one-step window closed");
    assert!(
        (mtp - 1200.0).abs() < 1.0,
        "Mtp EWMA must be REPLACED by the in-regime window (6 tok / 5 ms = \
         1200 tok/s), not blended with the width-1 40 tok/s (got {mtp:.0})"
    );
    // 2026-09-25: The early probe opens at once and measures serial at the
    // new width.
    assert_eq!(
        g.next_step(),
        GateStep::MeasureDecode,
        "probe pulled forward"
    );
    for _ in 0..WINDOW_STEPS {
        g.record_decode(ms(10), 8);
    }
    assert!(
        !g.serial.stale,
        "probe re-measured serial in the new regime"
    );
    // 2026-09-25: Serial 800 tok/s against MTP 1200: stays Mtp.
    assert!(!g.in_serial_mode());
    assert_eq!(g.take_fresh_decision(), None);
}

/// 2026-09-25: Width changes inside one power-of-two bucket are not a regime
/// change: nothing goes stale and the window keeps its steps.
#[test]
fn width_jitter_inside_a_bucket_does_not_stale() {
    let mut g = MtpGate::new(1);
    for w in [5usize, 6, 7, 8, 6, 5, 7] {
        g.record_verify_step(ms(10), 2 * w, w);
    }
    assert!(
        !g.mtp.stale && !g.serial.stale,
        "same-bucket churn is noise"
    );
    assert_eq!(g.win_steps, 7, "nothing discarded inside the bucket");
    // 2026-09-25: Dropping to bucket 4 is a regime change: the partial window
    // is discarded, the step's own window (closed early by the due probe)
    // re-measures Mtp, and serial stays stale until its probe.
    g.record_verify_step(ms(10), 8, 4);
    assert!(g.serial.stale && !g.mtp.stale);
    assert_eq!(
        g.win_steps, 0,
        "mixed-width window discarded, fresh one closed"
    );
    assert_eq!(
        g.next_step(),
        GateStep::MeasureDecode,
        "probe pulled forward"
    );
}
