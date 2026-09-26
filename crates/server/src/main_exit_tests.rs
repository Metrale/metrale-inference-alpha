// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Source-scanning tests that both of `main`'s exits map a latched GPU fault onto the exit status.
//!
//! `metrale_core::fault::exit_code` has its own unit tests; these check that
//! `main` calls it. Like `api/chat_stream/cancel_guard_tests.rs`, they read the
//! source as text. `main` has two exits, the startup escape and the normal
//! tail, and a fault can latch before either.
//!
//! Owner: server.
//! Invariants: none beyond the types.

const MAIN_RS: &str = include_str!("main.rs");

/// 2026-09-26: `main` reads the fault latch. Both exits contain the read, so
/// this passes while either one does; the count test below catches one exit
/// left unmapped.
#[test]
fn main_consults_the_fault_latch_before_exiting() {
    assert!(
        MAIN_RS.contains("fault::global().fault()"),
        "main does not consult the fault latch on the way out — a poisoned \
         context will exit 0 and `restart: on-failure` will not restart it"
    );
}

/// 2026-09-26: Both exits map through `exit_code`: the call must appear
/// exactly twice, so a single uncovered exit fails.
#[test]
fn both_of_mains_exit_paths_map_the_fault_onto_the_status() {
    let sites = MAIN_RS.matches("fault::exit_code(").count();
    assert_eq!(
        sites, 2,
        "expected both of main's exits (startup escape + normal tail) to map \
         through fault::exit_code, found {sites}"
    );
}

/// 2026-09-26: Negative control: the healthy arm still returns the run's own
/// result, so a clean shutdown does not exit nonzero.
#[test]
fn a_healthy_run_still_returns_its_own_result() {
    assert!(
        MAIN_RS.contains("None => result"),
        "main no longer has a healthy arm that returns the run's own status; a \
         clean shutdown would then exit nonzero and restart-loop"
    );
}
