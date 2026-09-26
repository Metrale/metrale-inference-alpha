// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests that every reachable site in `chat_stream` that stores `true`
//! into `cancel_flag` also sets `guard_stop`, `tool_loop_capped` or `stop_string_matched`, which
//! `handle_done::resolve_wire_finish_reason` turns into the wire finish reason. A bare
//! cancel ends the sequence with budget left and no scheduler guard, which
//! `derive_finish_reason` reports as "stop". The property is over all sites, so the
//! test reads the source text.
//!
//! Owner: server streaming API.
//! Invariants: none beyond the types.

/// 2026-09-26: The `chat_stream` files with a `.store(true` within three lines of a
/// `cancel_flag`. A file missing here is not scanned at all.
const CANCEL_SITE_FILES: &[(&str, &str)] = &[
    ("handle_token.rs", include_str!("handle_token.rs")),
    (
        "handle_token/detector_content.rs",
        include_str!("handle_token/detector_content.rs"),
    ),
    ("tool_handlers.rs", include_str!("tool_handlers.rs")),
    ("state.rs", include_str!("state.rs")),
];

/// 2026-09-26: Any one of these within the window above a `cancel_flag` store names the
/// cut. `guard_stop` and `tool_loop_capped` reach the wire as `"length"`,
/// `stop_string_matched` as `"stop"`.
const GUARD_MARKERS: &[&str] = &[
    "guard_stop = Some(",
    "tool_loop_capped = true",
    "stop_string_matched = true",
];

/// 2026-09-26: Lines searched above a store. Every current site sets its marker within
/// ten lines of the store.
const WINDOW: usize = 25;

/// 2026-09-26: True for the store that cannot run.
fn is_dead_path(preceding: &str) -> bool {
    // 2026-09-26: The retry cut in `tool_handlers.rs` runs only under
    // `ctx.tool_retry_enabled`, which `mod.rs` always sets to `false`. Match the
    // assignment, not the name: `state.rs` initialises `pending_retry: None` 15 lines
    // above a live store.
    preceding.contains("pending_retry = Some(")
}

#[test]
fn every_cancel_flag_store_names_its_guard() {
    let mut bare: Vec<String> = Vec::new();
    let mut checked = 0usize;

    for (name, src) in CANCEL_SITE_FILES {
        let lines: Vec<&str> = src.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !line.contains(".store(true") {
                continue;
            }
            let lo = i.saturating_sub(3);
            if !lines[lo..=i].iter().any(|l| l.contains("cancel_flag")) {
                continue;
            }
            checked += 1;
            let from = i.saturating_sub(WINDOW);
            let preceding = lines[from..=i].join("\n");
            if is_dead_path(&preceding) {
                continue;
            }
            if !GUARD_MARKERS.iter().any(|m| preceding.contains(m)) {
                bare.push(format!("{name}:{}", i + 1));
            }
        }
    }

    // 2026-09-26: If the scan stops matching, the floor fails the test instead of
    // passing over nothing.
    assert!(
        checked >= 4,
        "found only {checked} cancel_flag stores — the scan stopped matching. \
         Fix the detection before trusting a green here."
    );

    assert!(
        bare.is_empty(),
        "cancel_flag flipped without naming a guard at: {bare:?}\n\
         A bare cancel reaches the client as finish_reason \"stop\", claiming \
         the model finished when it was truncated. Set guard_stop (or \
         tool_loop_capped / stop_string_matched) at the site so the wire \
         reason is \"length\"."
    );
}

#[test]
fn the_scan_would_notice_a_bare_store() {
    // 2026-09-26: The detector flags a store with no marker, and adding a marker
    // clears it.
    let synthetic = "\
        if streak > MAX {\n\
        \x20   tracing::warn!(\"ending stream\");\n\
        \x20   state.loop_watchdog_triggered = true;\n\
        \x20   state.cancel_flag.store(true, Ordering::Release);\n\
        }\n";
    let lines: Vec<&str> = synthetic.lines().collect();
    let idx = lines
        .iter()
        .position(|l| l.contains(".store(true"))
        .expect("fixture must contain a store");
    let preceding = lines[..=idx].join("\n");
    assert!(
        !GUARD_MARKERS.iter().any(|m| preceding.contains(m)),
        "the detector failed to flag a store with no guard marker — it \
         would pass the real invariant test vacuously"
    );
    let fixed = preceding.replace(
        "state.loop_watchdog_triggered = true;",
        "state.guard_stop = Some(\"suppress_streak\");",
    );
    assert!(GUARD_MARKERS.iter().any(|m| fixed.contains(m)));
}
